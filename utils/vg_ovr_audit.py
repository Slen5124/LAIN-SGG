"""Audit where zero-shot VG predicate recall is lost.

This script is read-only with respect to model weights. It loads an existing
OvR checkpoint, runs the normal LAIN inference path, and measures:

1. Novel-GT subject/object and directed-pair proposal coverage.
2. The rank of the correct Novel predicate within a covered pair.
3. Base-versus-Novel score margins.
4. Graph-constrained versus no-graph Novel Recall@K.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

# [Repository-local diagnostic]
# This file is intended to live in ``~/LAIN/utils``. Add the repository root
# explicitly so running ``python utils/vg_ovr_audit.py`` resolves LAIN modules
# consistently, independent of Python's script-directory behaviour.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

import pocket.ops

from datasets import DataFactory, custom_collate
from detr.util import box_ops
from models.LAIN import build_detector
from utils.vg_evaluator import evaluate_vg_image_recall


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-args", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--base-score-scales",
        default="1.0,0.75,0.5,0.25,0.1,0.05",
        help=(
            "Comma-separated post-hoc scales applied only to Base-predicate "
            "candidate scores. This is a diagnostic sweep and never changes "
            "the checkpoint."
        ),
    )
    return parser.parse_args()


def recover_target_boxes(boxes, size):
    boxes = box_ops.box_cxcywh_to_xyxy(boxes)
    height, width = size
    scale = torch.stack([width, height, width, height])
    return boxes * scale


def build_box_labels(detection):
    num_boxes = len(detection["boxes"])
    labels = torch.full((num_boxes,), -1, dtype=torch.long)
    pairing = detection["pairing"].long()

    if pairing.shape[1] > 0:
        labels[pairing[0]] = detection["subjects"].long()
        labels[pairing[1]] = detection["objects"].long()

    return labels


def unique_pair_groups(detection):
    pairing = detection["pairing"].long()
    num_boxes = len(detection["boxes"])

    if pairing.shape[1] == 0:
        return []

    pair_keys = pairing[0] * num_boxes + pairing[1]
    groups = []

    for pair_key in torch.unique(pair_keys, sorted=False):
        indices = torch.nonzero(
            pair_keys == pair_key,
            as_tuple=False,
        ).squeeze(1)
        first = indices[0]
        groups.append(
            {
                "indices": indices,
                "subject_box_index": int(pairing[0, first]),
                "object_box_index": int(pairing[1, first]),
                "subject_class": int(detection["subjects"][first]),
                "object_class": int(detection["objects"][first]),
            }
        )

    return groups


def predicate_score_vector(detection, group, num_predicates=50):
    scores = torch.full(
        (num_predicates,),
        -torch.inf,
        dtype=detection["scores"].dtype,
    )
    indices = group["indices"]
    labels = detection["labels"][indices].long()
    scores[labels] = detection["scores"][indices]
    return scores


def prediction_match_matrix(detection, indices, target, iou_threshold):
    num_gt = len(target["verb"])

    if len(indices) == 0:
        return torch.zeros(0, num_gt, dtype=torch.bool)

    pairing = detection["pairing"][:, indices].long()
    boxes_h = detection["boxes"][pairing[0]]
    boxes_o = detection["boxes"][pairing[1]]
    gt_boxes_h = recover_target_boxes(
        target["boxes_h"],
        target["size"],
    )
    gt_boxes_o = recover_target_boxes(
        target["boxes_o"],
        target["size"],
    )

    matches = box_iou(boxes_h, gt_boxes_h) >= iou_threshold
    matches &= box_iou(boxes_o, gt_boxes_o) >= iou_threshold
    matches &= (
        detection["subjects"][indices, None].long()
        == target["subject"][None, :].long()
    )
    matches &= (
        detection["labels"][indices, None].long()
        == target["verb"][None, :].long()
    )
    matches &= (
        detection["objects"][indices, None].long()
        == target["object"][None, :].long()
    )
    return matches


def no_graph_novel_recall(
    detection,
    target,
    novel_mask,
    recall_k,
    iou_threshold,
):
    order = detection["scores"].argsort(descending=True)
    result = {}

    for k in recall_k:
        indices = order[:k]
        matches = prediction_match_matrix(
            detection,
            indices,
            target,
            iou_threshold,
        )
        matched_gt = (
            matches.any(dim=0)
            if len(matches) > 0
            else torch.zeros(len(target["verb"]), dtype=torch.bool)
        )
        result[k] = float(
            (matched_gt & novel_mask).sum().item()
            / max(int(novel_mask.sum().item()), 1)
        )

    return result


def mean_or_zero(values):
    return float(sum(values) / len(values)) if values else 0.0


def median_or_zero(values):
    return float(np.median(values)) if values else 0.0


def parse_score_scales(raw_value):
    scales = tuple(float(item) for item in raw_value.split(","))
    if not scales:
        raise ValueError("At least one Base score scale is required")
    if any(scale < 0.0 for scale in scales):
        raise ValueError("Base score scales must be non-negative")
    return scales


def calibrate_base_scores(detection, base_predicates, scale):
    """Return a shallow detection copy with scaled Base scores.

    [Post-hoc Base calibration]
    The original graph-constrained evaluator compares the final candidate
    scores without Seen/Novel correction. For diagnosis only, multiply scores
    whose predicate belongs to the Base split. Boxes, pairs, labels, and Novel
    scores stay untouched, and no model parameter is modified.
    """
    calibrated = dict(detection)
    calibrated["scores"] = detection["scores"].clone()
    base_mask = torch.isin(
        detection["labels"].long(),
        base_predicates,
    )
    calibrated["scores"][base_mask] *= scale
    return calibrated


def append_recall_value(container, metric_name, k, value):
    if value is not None:
        container[metric_name][k].append(float(value))


def main():
    cli = parse_args()
    score_scales = parse_score_scales(cli.base_score_scales)

    with open(cli.run_args, "r", encoding="utf-8") as handle:
        run_args = SimpleNamespace(**json.load(handle))

    run_args.resume = cli.checkpoint
    run_args.eval = True
    run_args.debug = True
    run_args.local_rank = 0
    run_args.world_size = 1
    run_args.num_workers = cli.num_workers

    if not getattr(run_args, "vg_ovr", False):
        raise ValueError("The supplied run args do not enable --vg-ovr")
    if getattr(run_args, "CSC", False):
        raise ValueError("VG OvR audit expects the shared prompt, CSC=False")

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

    testset = DataFactory(
        name="vg",
        partition="test",
        data_root=run_args.data_root,
        clip_model_name=run_args.clip_model_name,
        num_classes=run_args.num_classes,
        args=run_args,
    )
    loader = DataLoader(
        testset,
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
        testset.dataset.object_class_to_target_class,
        testset.dataset.object_n_verb_to_interaction,
        run_args.clip_dir_vit,
    )
    checkpoint = torch.load(
        cli.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.cuda().eval()

    base = tuple(testset.dataset.base_predicate_indices)
    novel = tuple(testset.dataset.novel_predicate_indices)
    if len(base) != 35 or len(novel) != 15:
        raise ValueError(
            f"Unexpected OvR split sizes: base={len(base)}, novel={len(novel)}"
        )
    base_tensor = torch.as_tensor(base, dtype=torch.long)
    novel_tensor = torch.as_tensor(novel, dtype=torch.long)

    recall_k = (20, 50, 100)
    stats = {
        "images": 0,
        "images_with_novel_gt": 0,
        "novel_gt": 0,
        "novel_subject_covered": 0,
        "novel_object_covered": 0,
        "novel_pair_covered": 0,
        "pair_count": 0,
        "pair_top1_novel": 0,
        "pair_top5_has_novel": 0,
        "correct_novel_ranks": [],
        "correct_novel_only_ranks": [],
        "all_pair_margins": [],
        "covered_novel_pair_margins": [],
        "graph_zr": {k: [] for k in recall_k},
        "no_graph_zr": {k: [] for k in recall_k},
        "calibration": {
            scale: {
                "recall": {k: [] for k in recall_k},
                "base_recall": {k: [] for k in recall_k},
                "novel_recall": {k: [] for k in recall_k},
            }
            for scale in score_scales
        },
    }

    limit = min(cli.num_images, len(testset))
    for batch_index, batch in enumerate(tqdm(loader, total=limit)):
        if batch_index >= limit:
            break

        inputs = pocket.ops.relocate_to_cuda(batch[0])
        target = batch[1][0]
        gt_predicates = target["verb"].long()
        base_mask = torch.isin(gt_predicates, base_tensor)
        novel_mask = torch.isin(gt_predicates, novel_tensor)
        outputs = model(inputs, batch[1])
        stats["images"] += 1

        # [Zero-output accounting]
        # An image with Novel GT but no prediction is a genuine zero-recall
        # case. Count it instead of silently removing it from the audit.
        if outputs is None or len(outputs) == 0:
            if novel_mask.any():
                num_novel = int(novel_mask.sum().item())
                stats["images_with_novel_gt"] += 1
                stats["novel_gt"] += num_novel
                for k in recall_k:
                    stats["graph_zr"][k].append(0.0)
                    stats["no_graph_zr"][k].append(0.0)
            # A prediction-free image contributes zero to every applicable
            # calibration metric; it must not disappear from the mean.
            for scale in score_scales:
                for k in recall_k:
                    if len(gt_predicates) > 0:
                        stats["calibration"][scale]["recall"][k].append(0.0)
                    if base_mask.any():
                        stats["calibration"][scale]["base_recall"][k].append(0.0)
                    if novel_mask.any():
                        stats["calibration"][scale]["novel_recall"][k].append(0.0)
            continue

        detection = pocket.ops.relocate_to_cpu(outputs[0], ignore=True)
        groups = unique_pair_groups(detection)
        box_labels = build_box_labels(detection)

        pair_vectors = []
        for group in groups:
            vector = predicate_score_vector(detection, group)
            pair_vectors.append(vector)
            stats["pair_count"] += 1
            stats["pair_top1_novel"] += int(
                int(vector.argmax()) in novel
            )
            top5 = vector.topk(min(5, len(vector))).indices
            stats["pair_top5_has_novel"] += int(
                torch.isin(top5, novel_tensor).any().item()
            )
            stats["all_pair_margins"].append(
                float(vector[base_tensor].max() - vector[novel_tensor].max())
            )

        # [Calibration sweep]
        # Re-rank the same predictions after scaling Base scores. This tests
        # whether zR=0 is mainly a Seen-class score bias before changing or
        # retraining any LAIN component.
        for scale in score_scales:
            calibrated = calibrate_base_scores(
                detection,
                base_tensor,
                scale,
            )
            calibrated_result = evaluate_vg_image_recall(
                detection=calibrated,
                target=target,
                recall_k=recall_k,
                iou_threshold=cli.iou_threshold,
                base_predicate_indices=base,
                novel_predicate_indices=novel,
            )
            for k in recall_k:
                append_recall_value(
                    stats["calibration"][scale],
                    "recall",
                    k,
                    calibrated_result["recall"][k],
                )
                append_recall_value(
                    stats["calibration"][scale],
                    "base_recall",
                    k,
                    calibrated_result["base_recall"][k],
                )
                append_recall_value(
                    stats["calibration"][scale],
                    "novel_recall",
                    k,
                    calibrated_result["novel_recall"][k],
                )

        if not novel_mask.any():
            continue

        stats["images_with_novel_gt"] += 1
        novel_gt_indices = torch.nonzero(
            novel_mask,
            as_tuple=False,
        ).squeeze(1)
        stats["novel_gt"] += len(novel_gt_indices)

        gt_boxes_h = recover_target_boxes(
            target["boxes_h"], target["size"]
        )
        gt_boxes_o = recover_target_boxes(
            target["boxes_o"], target["size"]
        )
        proposal_boxes = detection["boxes"]

        for gt_index in novel_gt_indices.tolist():
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

            stats["novel_subject_covered"] += int(subject_hits.any())
            stats["novel_object_covered"] += int(object_hits.any())

            matching_group_indices = []
            for group_index, group in enumerate(groups):
                if (
                    bool(subject_hits[group["subject_box_index"]].item())
                    and bool(object_hits[group["object_box_index"]].item())
                ):
                    matching_group_indices.append(group_index)

            if not matching_group_indices:
                continue

            stats["novel_pair_covered"] += 1
            predicate = int(gt_predicates[gt_index])
            best_rank = 51
            best_group_index = None

            for group_index in matching_group_indices:
                vector = pair_vectors[group_index]
                rank = int((vector > vector[predicate]).sum().item()) + 1
                if rank < best_rank:
                    best_rank = rank
                    best_group_index = group_index

            stats["correct_novel_ranks"].append(best_rank)
            vector = pair_vectors[best_group_index]
            # [Novel-only semantic rank]
            # Separate Base-vs-Novel competition from ordering inside the 15
            # Novel predicates. A low overall rank but good Novel-only rank
            # indicates calibration bias; poor ranks in both spaces indicate
            # failed zero-shot semantic alignment.
            novel_only_rank = int(
                (
                    vector[novel_tensor]
                    > vector[predicate]
                ).sum().item()
            ) + 1
            stats["correct_novel_only_ranks"].append(novel_only_rank)
            stats["covered_novel_pair_margins"].append(
                float(vector[base_tensor].max() - vector[novel_tensor].max())
            )

        graph_result = evaluate_vg_image_recall(
            detection=detection,
            target=target,
            recall_k=recall_k,
            iou_threshold=cli.iou_threshold,
            base_predicate_indices=base,
            novel_predicate_indices=novel,
        )
        no_graph_result = no_graph_novel_recall(
            detection,
            target,
            novel_mask,
            recall_k,
            cli.iou_threshold,
        )

        for k in recall_k:
            graph_value = graph_result["novel_recall"][k]
            if graph_value is not None:
                stats["graph_zr"][k].append(float(graph_value))
            stats["no_graph_zr"][k].append(no_graph_result[k])

    novel_gt = max(stats["novel_gt"], 1)
    pair_count = max(stats["pair_count"], 1)
    ranks = stats["correct_novel_ranks"]
    novel_only_ranks = stats["correct_novel_only_ranks"]

    summary = {
        "checkpoint": cli.checkpoint,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "images": stats["images"],
        "images_with_novel_gt": stats["images_with_novel_gt"],
        "novel_gt": stats["novel_gt"],
        "novel_subject_class_aware_recall": (
            stats["novel_subject_covered"] / novel_gt
        ),
        "novel_object_class_aware_recall": (
            stats["novel_object_covered"] / novel_gt
        ),
        "novel_directed_pair_recall": (
            stats["novel_pair_covered"] / novel_gt
        ),
        "pair_top1_novel_fraction": (
            stats["pair_top1_novel"] / pair_count
        ),
        "pair_top5_has_novel_fraction": (
            stats["pair_top5_has_novel"] / pair_count
        ),
        "covered_novel_gt_count": len(ranks),
        "correct_novel_rank_mean": mean_or_zero(ranks),
        "correct_novel_rank_median": median_or_zero(ranks),
        "correct_novel_rank_at_1": mean_or_zero([r <= 1 for r in ranks]),
        "correct_novel_rank_at_5": mean_or_zero([r <= 5 for r in ranks]),
        "correct_novel_rank_at_10": mean_or_zero([r <= 10 for r in ranks]),
        "correct_novel_only_rank_mean": mean_or_zero(novel_only_ranks),
        "correct_novel_only_rank_median": median_or_zero(novel_only_ranks),
        "correct_novel_only_rank_at_1": mean_or_zero(
            [r <= 1 for r in novel_only_ranks]
        ),
        "correct_novel_only_rank_at_3": mean_or_zero(
            [r <= 3 for r in novel_only_ranks]
        ),
        "correct_novel_only_rank_at_5": mean_or_zero(
            [r <= 5 for r in novel_only_ranks]
        ),
        "all_pair_base_minus_novel_margin_mean": mean_or_zero(
            stats["all_pair_margins"]
        ),
        "covered_novel_pair_margin_mean": mean_or_zero(
            stats["covered_novel_pair_margins"]
        ),
    }

    for k in recall_k:
        summary[f"graph_zR@{k}"] = mean_or_zero(stats["graph_zr"][k])
        summary[f"no_graph_zR@{k}"] = mean_or_zero(
            stats["no_graph_zr"][k]
        )

    summary["base_score_calibration"] = {}
    for scale in score_scales:
        scale_summary = {}
        for k in recall_k:
            scale_summary[f"R@{k}"] = mean_or_zero(
                stats["calibration"][scale]["recall"][k]
            )
            scale_summary[f"bR@{k}"] = mean_or_zero(
                stats["calibration"][scale]["base_recall"][k]
            )
            scale_summary[f"zR@{k}"] = mean_or_zero(
                stats["calibration"][scale]["novel_recall"][k]
            )
        summary["base_score_calibration"][str(scale)] = scale_summary

    output_path = Path(cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n===== VG OvR AUDIT =====")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
