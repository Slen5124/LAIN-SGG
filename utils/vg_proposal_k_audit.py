"""Audit VG SGDet proposal coverage for several proposal K values.

This diagnostic runs only the EGTR detector and LAIN proposal filtering.  It
does not create directed HO tokens or run IA/LA relation scoring, so K=200 can
be inspected without constructing 39,800 relation tokens.
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pocket.ops
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from datasets import DataFactory, custom_collate
from detr.util.misc import nested_tensor_from_tensor_list
from models.LAIN import build_detector
from utils.vg_ovr_audit import recover_target_boxes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-args", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help=(
            "Temporarily override LAIN's proposal score threshold for this "
            "read-only audit. The checkpoint and saved run arguments are "
            "not modified."
        ),
    )
    parser.add_argument(
        "--proposal-k",
        type=int,
        nargs="+",
        default=[30, 50, 100, 200],
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def average(values):
    return sum(values) / len(values) if values else 0.0


def pair_is_covered(subject_hits, object_hits):
    """Return true when two distinct retained proposals cover a directed pair."""
    subject_indices = torch.nonzero(subject_hits, as_tuple=True)[0]
    object_indices = torch.nonzero(object_hits, as_tuple=True)[0]
    if len(subject_indices) == 0 or len(object_indices) == 0:
        return False
    return bool((subject_indices[:, None] != object_indices[None, :]).any())


def main():
    cli = parse_args()
    proposal_k = tuple(sorted(set(cli.proposal_k)))
    if not proposal_k or proposal_k[0] <= 1:
        raise ValueError("--proposal-k values must be greater than one")

    with open(cli.run_args, "r", encoding="utf-8") as handle:
        run_args = SimpleNamespace(**json.load(handle))

    run_args.resume = cli.checkpoint
    run_args.eval = True
    run_args.debug = True
    run_args.local_rank = 0
    run_args.world_size = 1
    run_args.num_workers = cli.num_workers

    seed = int(run_args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    clip_name = Path(run_args.clip_dir_vit).stem
    if clip_name == "ViT-B-16":
        run_args.clip_model_name = "ViT-B/16"
    elif clip_name == "ViT-L-14-336px":
        run_args.clip_model_name = "ViT-L/14@336px"
    else:
        raise ValueError(f"Unsupported CLIP checkpoint name: {clip_name}")

    trainset = DataFactory(
        name="vg",
        partition="train",
        data_root=run_args.data_root,
        clip_model_name=run_args.clip_model_name,
        num_classes=run_args.num_classes,
        args=run_args,
    )
    test_transform_source = DataFactory(
        name="vg",
        partition="test",
        data_root=run_args.data_root,
        clip_model_name=run_args.clip_model_name,
        num_classes=run_args.num_classes,
        args=run_args,
    )

    # [Proposal-K audit]
    # Use the same train image IDs as the overfit run but deterministic test
    # transforms, so all K values see exactly the same boxes and annotations.
    trainset = copy.copy(trainset)
    trainset.transforms = test_transform_source.transforms
    trainset.clip_transforms = test_transform_source.clip_transforms
    limit = min(cli.num_images, len(trainset))

    class _AuditSubset(type(trainset)):
        def __len__(self):
            return limit

    trainset.__class__ = _AuditSubset
    loader = DataLoader(
        trainset,
        batch_size=1,
        shuffle=False,
        num_workers=cli.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=custom_collate,
    )

    run_args.human_idx = 0
    model = build_detector(
        run_args,
        trainset.dataset.object_class_to_target_class,
        trainset.dataset.object_n_verb_to_interaction,
        run_args.clip_dir_vit,
    )
    checkpoint = torch.load(
        cli.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.cuda().eval()

    # [Proposal score-threshold audit]
    # Override only the runtime proposal filter. Detector weights, NMS, the
    # relation model, and the checkpoint remain unchanged.
    if cli.score_threshold is not None:
        if not 0.0 <= cli.score_threshold <= 1.0:
            raise ValueError("--score-threshold must be between 0 and 1")
        model.box_score_thresh = cli.score_threshold

    stats = {
        k: {
            "images": 0,
            "gt": 0,
            "subject_covered": 0,
            "object_covered": 0,
            "pair_covered": 0,
            "macro_pair_recall": [],
            "retained_proposals": [],
        }
        for k in proposal_k
    }

    with torch.no_grad():
        for batch in tqdm(loader, total=limit):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            target = batch[1][0]
            images_orig = [item[0].float() for item in inputs]
            images_clip = [item[1] for item in inputs]
            image_sizes = torch.as_tensor(
                [image.size()[-2:] for image in images_clip],
                device=images_orig[0].device,
            )
            nested = nested_tensor_from_tensor_list(images_orig)
            if nested.mask is None:
                raise RuntimeError("EGTR detector requires an image padding mask")

            detector_outputs = model.detector(
                pixel_values=nested.tensors,
                pixel_mask=(~nested.mask).long(),
            )
            raw_results = {
                "pred_logits": detector_outputs.logits,
                "pred_boxes": detector_outputs.pred_boxes,
                "feats": detector_outputs.last_hidden_state,
            }
            postprocessed = model.postprocessor(raw_results, image_sizes)

            gt_boxes_h = recover_target_boxes(
                target["boxes_h"], target["size"]
            )
            gt_boxes_o = recover_target_boxes(
                target["boxes_o"], target["size"]
            )
            gt_count = len(target["verb"])

            for k in proposal_k:
                # [Proposal-K audit]
                # Reuse the same detector output and vary only LAIN's retained
                # proposal cap. Relation tokens and relation logits are skipped.
                model.max_instances = k
                proposal = model.prepare_region_proposals(postprocessed)[0]
                boxes = proposal["boxes"].cpu()
                labels = proposal["labels"].cpu()
                image_pair_covered = 0

                stats[k]["images"] += 1
                stats[k]["gt"] += gt_count
                stats[k]["retained_proposals"].append(len(boxes))

                for gt_index in range(gt_count):
                    subject_hits = (
                        box_iou(
                            boxes,
                            gt_boxes_h[gt_index:gt_index + 1],
                        ).squeeze(1)
                        >= cli.iou_threshold
                    ) & (labels == int(target["subject"][gt_index]))
                    object_hits = (
                        box_iou(
                            boxes,
                            gt_boxes_o[gt_index:gt_index + 1],
                        ).squeeze(1)
                        >= cli.iou_threshold
                    ) & (labels == int(target["object"][gt_index]))

                    stats[k]["subject_covered"] += int(subject_hits.any())
                    stats[k]["object_covered"] += int(object_hits.any())
                    covered = pair_is_covered(subject_hits, object_hits)
                    stats[k]["pair_covered"] += int(covered)
                    image_pair_covered += int(covered)

                if gt_count:
                    stats[k]["macro_pair_recall"].append(
                        image_pair_covered / gt_count
                    )

    summary = {
        "checkpoint": cli.checkpoint,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "images": limit,
        "iou_threshold": cli.iou_threshold,
        "score_threshold": float(model.box_score_thresh),
        "proposal_k": {},
    }
    for k in proposal_k:
        current = stats[k]
        retained = current["retained_proposals"]
        summary["proposal_k"][str(k)] = {
            "average_retained_proposals": average(retained),
            "minimum_retained_proposals": min(retained) if retained else 0,
            "maximum_retained_proposals": max(retained) if retained else 0,
            "subject_recall": ratio(
                current["subject_covered"], current["gt"]
            ),
            "object_recall": ratio(
                current["object_covered"], current["gt"]
            ),
            "micro_directed_pair_recall": ratio(
                current["pair_covered"], current["gt"]
            ),
            "macro_directed_pair_recall": average(
                current["macro_pair_recall"]
            ),
        }

    output_path = Path(cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\n===== VG PROPOSAL-K AUDIT =====")
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"checkpoint_epoch: {summary['checkpoint_epoch']}")
    print(f"images: {summary['images']}")
    print(f"score_threshold: {summary['score_threshold']}")
    for k, metrics in summary["proposal_k"].items():
        print(f"\nK={k}")
        for name, value in metrics.items():
            print(f"  {name}: {value}")
    print(f"\nsaved: {output_path}")


if __name__ == "__main__":
    main()
