"""Measure the SGDet proposal ceiling on a deterministic VG train subset.

This diagnostic ignores predicate scores when measuring proposal coverage.  A
GT relation is oracle-covered when the retained proposals contain a directed
subject/object pair whose two boxes pass IoU and whose two object classes match.
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
from models.LAIN import build_detector
from utils.vg_evaluator import evaluate_vg_image_recall
from utils.vg_ovr_audit import (
    build_box_labels,
    recover_target_boxes,
    unique_pair_groups,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-args", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    cli = parse_args()
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

    # Match --overfit-debug: same train IDs, deterministic evaluation transform.
    trainset = copy.copy(trainset)
    trainset.transforms = test_transform_source.transforms
    trainset.clip_transforms = test_transform_source.clip_transforms

    limit = min(cli.num_images, len(trainset))

    class _OracleSubset(type(trainset)):
        def __len__(self):
            return limit

    trainset.__class__ = _OracleSubset
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

    base = tuple(trainset.dataset.base_predicate_indices)
    novel = tuple(trainset.dataset.novel_predicate_indices)
    base_tensor = torch.as_tensor(base, dtype=torch.long)
    novel_tensor = torch.as_tensor(novel, dtype=torch.long)
    recall_k = (20, 50, 100)

    stats = {
        "images": 0,
        "gt": 0,
        "base_gt": 0,
        "novel_gt": 0,
        "subject_covered": 0,
        "object_covered": 0,
        "directed_pair_covered": 0,
        "base_pair_covered": 0,
        "novel_pair_covered": 0,
        "oracle_recall_per_image": [],
        "oracle_base_recall_per_image": [],
        "oracle_novel_recall_per_image": [],
        "actual_matched": {k: 0 for k in recall_k},
        "actual_base_matched": {k: 0 for k in recall_k},
        "actual_novel_matched": {k: 0 for k in recall_k},
        "actual_recall": {k: [] for k in recall_k},
        "actual_base_recall": {k: [] for k in recall_k},
        "actual_novel_recall": {k: [] for k in recall_k},
    }

    with torch.no_grad():
        for batch in tqdm(loader, total=limit):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            target = batch[1][0]
            gt_predicates = target["verb"].long()
            base_mask = torch.isin(gt_predicates, base_tensor)
            novel_mask = torch.isin(gt_predicates, novel_tensor)

            outputs = model(inputs, batch[1])
            stats["images"] += 1
            stats["gt"] += len(gt_predicates)
            stats["base_gt"] += int(base_mask.sum())
            stats["novel_gt"] += int(novel_mask.sum())
            image_pair_covered = 0
            image_base_pair_covered = 0
            image_novel_pair_covered = 0

            if outputs is None or len(outputs) == 0:
                if len(gt_predicates):
                    stats["oracle_recall_per_image"].append(0.0)
                if base_mask.any():
                    stats["oracle_base_recall_per_image"].append(0.0)
                if novel_mask.any():
                    stats["oracle_novel_recall_per_image"].append(0.0)
                for k in recall_k:
                    if len(gt_predicates):
                        stats["actual_recall"][k].append(0.0)
                    if base_mask.any():
                        stats["actual_base_recall"][k].append(0.0)
                    if novel_mask.any():
                        stats["actual_novel_recall"][k].append(0.0)
                continue

            detection = pocket.ops.relocate_to_cpu(outputs[0], ignore=True)
            groups = unique_pair_groups(detection)
            box_labels = build_box_labels(detection)
            proposal_boxes = detection["boxes"]
            gt_boxes_h = recover_target_boxes(target["boxes_h"], target["size"])
            gt_boxes_o = recover_target_boxes(target["boxes_o"], target["size"])

            for gt_index in range(len(gt_predicates)):
                subject_hits = (
                    box_iou(
                        proposal_boxes,
                        gt_boxes_h[gt_index:gt_index + 1],
                    ).squeeze(1)
                    >= cli.iou_threshold
                ) & (box_labels == int(target["subject"][gt_index]))
                object_hits = (
                    box_iou(
                        proposal_boxes,
                        gt_boxes_o[gt_index:gt_index + 1],
                    ).squeeze(1)
                    >= cli.iou_threshold
                ) & (box_labels == int(target["object"][gt_index]))

                stats["subject_covered"] += int(subject_hits.any())
                stats["object_covered"] += int(object_hits.any())

                pair_covered = any(
                    bool(subject_hits[group["subject_box_index"]])
                    and bool(object_hits[group["object_box_index"]])
                    for group in groups
                )
                if pair_covered:
                    image_pair_covered += 1
                    stats["directed_pair_covered"] += 1
                    if base_mask[gt_index]:
                        image_base_pair_covered += 1
                        stats["base_pair_covered"] += 1
                    if novel_mask[gt_index]:
                        image_novel_pair_covered += 1
                        stats["novel_pair_covered"] += 1

            if len(gt_predicates):
                stats["oracle_recall_per_image"].append(
                    image_pair_covered / len(gt_predicates)
                )
            if base_mask.any():
                stats["oracle_base_recall_per_image"].append(
                    image_base_pair_covered / int(base_mask.sum())
                )
            if novel_mask.any():
                stats["oracle_novel_recall_per_image"].append(
                    image_novel_pair_covered / int(novel_mask.sum())
                )

            actual = evaluate_vg_image_recall(
                detection=detection,
                target=target,
                recall_k=recall_k,
                iou_threshold=cli.iou_threshold,
                base_predicate_indices=base,
                novel_predicate_indices=novel,
            )
            for k in recall_k:
                stats["actual_matched"][k] += actual["matched"][k]
                stats["actual_base_matched"][k] += actual[
                    "base_matched"
                ][k]
                stats["actual_novel_matched"][k] += actual[
                    "novel_matched"
                ][k]
                stats["actual_recall"][k].append(actual["recall"][k])
                if actual["base_recall"][k] is not None:
                    stats["actual_base_recall"][k].append(
                        actual["base_recall"][k]
                    )
                if actual["novel_recall"][k] is not None:
                    stats["actual_novel_recall"][k].append(
                        actual["novel_recall"][k]
                    )

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    def average(values):
        return sum(values) / len(values) if values else 0.0

    summary = {
        "checkpoint": cli.checkpoint,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "images": stats["images"],
        "gt": stats["gt"],
        "base_gt": stats["base_gt"],
        "novel_gt": stats["novel_gt"],
        "oracle_subject_recall": ratio(stats["subject_covered"], stats["gt"]),
        "oracle_object_recall": ratio(stats["object_covered"], stats["gt"]),
        "oracle_directed_pair_recall": ratio(
            stats["directed_pair_covered"], stats["gt"]
        ),
        "oracle_base_pair_recall": ratio(
            stats["base_pair_covered"], stats["base_gt"]
        ),
        "oracle_novel_pair_recall": ratio(
            stats["novel_pair_covered"], stats["novel_gt"]
        ),
        "oracle_macro_pair_recall": average(
            stats["oracle_recall_per_image"]
        ),
        "oracle_macro_base_pair_recall": average(
            stats["oracle_base_recall_per_image"]
        ),
        "oracle_macro_novel_pair_recall": average(
            stats["oracle_novel_recall_per_image"]
        ),
    }
    for k in recall_k:
        summary[f"R@{k}"] = average(stats["actual_recall"][k])
        summary[f"bR@{k}"] = average(stats["actual_base_recall"][k])
        summary[f"nR@{k}"] = average(stats["actual_novel_recall"][k])
        summary[f"micro_R@{k}"] = ratio(
            stats["actual_matched"][k], stats["gt"]
        )
        summary[f"micro_bR@{k}"] = ratio(
            stats["actual_base_matched"][k], stats["base_gt"]
        )
        summary[f"micro_nR@{k}"] = ratio(
            stats["actual_novel_matched"][k], stats["novel_gt"]
        )

    output_path = Path(cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\n===== VG TRAIN ORACLE AUDIT =====")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
