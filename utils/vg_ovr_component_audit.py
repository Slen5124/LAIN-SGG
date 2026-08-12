"""Counterfactual component audit for LAIN VG open-vocabulary relations.

This script never trains or saves the model. It evaluates the same checkpoint
under five temporary text-path configurations to locate where Novel-predicate
semantic alignment is lost:

1. trained prompt + trained composer (the current model),
2. trained prompt + no composer,
3. initial prompt + trained composer,
4. initial prompt + no composer,
5. trained prompt + initial composer.

The initial parameters are reconstructed by building the model with the saved
training seed immediately before loading the checkpoint.
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pocket.ops
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from datasets import DataFactory, custom_collate
from models.LAIN import build_detector
from utils.vg_evaluator import evaluate_vg_image_recall
from utils.vg_ovr_audit import (
    build_box_labels,
    predicate_score_vector,
    recover_target_boxes,
    unique_pair_groups,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-args", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def mean_or_zero(values):
    return float(sum(values) / len(values)) if values else 0.0


def median_or_zero(values):
    return float(np.median(values)) if values else 0.0


def clone_state_dict(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def anchor_only_compose(
    self,
    subject_labels,
    object_labels,
    predicate_features,
):
    """Bypass pair FiLM while preserving the trained predicate anchor.

    [Counterfactual diagnostic]
    The current composer applies pair-conditioned scale and shift. This
    replacement broadcasts the normalized predicate features to every pair,
    isolating the prompt/visual alignment from composer modulation.
    """
    if len(subject_labels) != len(object_labels):
        raise ValueError("VG subject/object pair counts do not match")
    anchors = F.normalize(
        predicate_features.float(),
        dim=-1,
        eps=1e-6,
    )
    return anchors.unsqueeze(0).expand(
        len(subject_labels),
        -1,
        -1,
    )


def configure_mode(
    model,
    mode,
    original_compose,
    initial_ctx,
    trained_ctx,
    initial_composer,
    trained_composer,
):
    """Temporarily configure one read-only counterfactual text path."""
    use_initial_prompt = mode.startswith("initial_prompt")
    use_initial_composer = mode.endswith("initial_composer")
    bypass_composer = mode.endswith("no_composer")

    selected_ctx = initial_ctx if use_initial_prompt else trained_ctx
    with torch.no_grad():
        model.clip_head.prompt_learner.ctx.copy_(
            selected_ctx.to(
                device=model.clip_head.prompt_learner.ctx.device,
                dtype=model.clip_head.prompt_learner.ctx.dtype,
            )
        )

    composer_state = (
        initial_composer
        if use_initial_composer
        else trained_composer
    )
    model.triplet_pair_composer.load_state_dict(composer_state)
    model.tp = None

    if bypass_composer:
        model.compose_vg_text_features = MethodType(
            anchor_only_compose,
            model,
        )
    else:
        model.compose_vg_text_features = original_compose


def novel_rank_statistics(
    detection,
    target,
    novel_tensor,
    iou_threshold,
):
    """Return correct Novel-only ranks for class-aware covered GT pairs."""
    groups = unique_pair_groups(detection)
    if not groups:
        return []

    pair_vectors = [
        predicate_score_vector(detection, group)
        for group in groups
    ]
    box_labels = build_box_labels(detection)
    proposal_boxes = detection["boxes"]
    gt_boxes_h = recover_target_boxes(
        target["boxes_h"],
        target["size"],
    )
    gt_boxes_o = recover_target_boxes(
        target["boxes_o"],
        target["size"],
    )
    gt_predicates = target["verb"].long()
    novel_mask = torch.isin(gt_predicates, novel_tensor)
    ranks = []

    for gt_index in torch.nonzero(
        novel_mask,
        as_tuple=False,
    ).squeeze(1).tolist():
        subject_hits = (
            box_iou(
                proposal_boxes,
                gt_boxes_h[gt_index:gt_index + 1],
            ).squeeze(1)
            >= iou_threshold
        ) & (box_labels == int(target["subject"][gt_index]))
        object_hits = (
            box_iou(
                proposal_boxes,
                gt_boxes_o[gt_index:gt_index + 1],
            ).squeeze(1)
            >= iou_threshold
        ) & (box_labels == int(target["object"][gt_index]))

        predicate = int(gt_predicates[gt_index])
        best_rank = None

        for group, vector in zip(groups, pair_vectors):
            if not bool(
                subject_hits[group["subject_box_index"]].item()
            ):
                continue
            if not bool(
                object_hits[group["object_box_index"]].item()
            ):
                continue

            rank = int(
                (
                    vector[novel_tensor]
                    > vector[predicate]
                ).sum().item()
            ) + 1
            best_rank = rank if best_rank is None else min(best_rank, rank)

        if best_rank is not None:
            ranks.append(best_rank)

    return ranks


def evaluate_mode(
    model,
    dataset,
    mode,
    num_images,
    num_workers,
    iou_threshold,
    base,
    novel,
):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=custom_collate,
    )
    recall_k = (20, 50, 100)
    values = {
        "recall": {k: [] for k in recall_k},
        "base_recall": {k: [] for k in recall_k},
        "novel_recall": {k: [] for k in recall_k},
    }
    novel_ranks = []
    novel_tensor = torch.as_tensor(novel, dtype=torch.long)
    limit = min(num_images, len(dataset))

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            tqdm(loader, total=limit, desc=mode)
        ):
            if batch_index >= limit:
                break

            inputs = pocket.ops.relocate_to_cuda(batch[0])
            target = batch[1][0]
            outputs = model(inputs, batch[1])

            if outputs is None or len(outputs) == 0:
                gt_predicates = target["verb"].long()
                base_present = torch.isin(
                    gt_predicates,
                    torch.as_tensor(base),
                ).any()
                novel_present = torch.isin(
                    gt_predicates,
                    novel_tensor,
                ).any()
                for k in recall_k:
                    if len(gt_predicates) > 0:
                        values["recall"][k].append(0.0)
                    if base_present:
                        values["base_recall"][k].append(0.0)
                    if novel_present:
                        values["novel_recall"][k].append(0.0)
                continue

            detection = pocket.ops.relocate_to_cpu(
                outputs[0],
                ignore=True,
            )
            result = evaluate_vg_image_recall(
                detection=detection,
                target=target,
                recall_k=recall_k,
                iou_threshold=iou_threshold,
                base_predicate_indices=base,
                novel_predicate_indices=novel,
            )
            for metric_name in values:
                for k in recall_k:
                    metric_value = result[metric_name][k]
                    if metric_value is not None:
                        values[metric_name][k].append(
                            float(metric_value)
                        )

            novel_ranks.extend(
                novel_rank_statistics(
                    detection,
                    target,
                    novel_tensor,
                    iou_threshold,
                )
            )

    summary = {}
    prefixes = {
        "recall": "R",
        "base_recall": "bR",
        "novel_recall": "zR",
    }
    for metric_name, prefix in prefixes.items():
        for k in recall_k:
            summary[f"{prefix}@{k}"] = mean_or_zero(
                values[metric_name][k]
            )

    summary.update(
        {
            "covered_novel_gt": len(novel_ranks),
            "novel_only_rank_mean": mean_or_zero(novel_ranks),
            "novel_only_rank_median": median_or_zero(novel_ranks),
            "novel_only_rank_at_1": mean_or_zero(
                [rank <= 1 for rank in novel_ranks]
            ),
            "novel_only_rank_at_3": mean_or_zero(
                [rank <= 3 for rank in novel_ranks]
            ),
            "novel_only_rank_at_5": mean_or_zero(
                [rank <= 5 for rank in novel_ranks]
            ),
        }
    )
    return summary


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

    if not getattr(run_args, "vg_ovr", False):
        raise ValueError("The supplied run args do not enable --vg-ovr")
    if getattr(run_args, "CSC", False):
        raise ValueError("VG OvR component audit expects CSC=False")

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
    run_args.human_idx = 0
    model = build_detector(
        run_args,
        testset.dataset.object_class_to_target_class,
        testset.dataset.object_n_verb_to_interaction,
        run_args.clip_dir_vit,
    )

    # Snapshot the seed-reconstructed initialization before loading training.
    initial_ctx = model.clip_head.prompt_learner.ctx.detach().cpu().clone()
    initial_composer = clone_state_dict(model.triplet_pair_composer)

    checkpoint = torch.load(
        cli.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trained_ctx = model.clip_head.prompt_learner.ctx.detach().cpu().clone()
    trained_composer = clone_state_dict(model.triplet_pair_composer)
    model.cuda().eval()
    original_compose = model.compose_vg_text_features

    base = tuple(testset.dataset.base_predicate_indices)
    novel = tuple(testset.dataset.novel_predicate_indices)
    modes = (
        "trained_prompt_trained_composer",
        "trained_prompt_no_composer",
        "initial_prompt_trained_composer",
        "initial_prompt_no_composer",
        "trained_prompt_initial_composer",
    )

    summary = {
        "checkpoint": cli.checkpoint,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "num_images": min(cli.num_images, len(testset)),
        "modes": {},
    }

    for mode in modes:
        configure_mode(
            model,
            mode,
            original_compose,
            initial_ctx,
            trained_ctx,
            initial_composer,
            trained_composer,
        )
        summary["modes"][mode] = evaluate_mode(
            model,
            testset,
            mode,
            cli.num_images,
            cli.num_workers,
            cli.iou_threshold,
            base,
            novel,
        )

    output_path = Path(cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n===== VG OvR COMPONENT AUDIT =====")
    for mode, metrics in summary["modes"].items():
        print(
            f"{mode}: "
            f"R@100={metrics['R@100']:.4f}, "
            f"bR@100={metrics['bR@100']:.4f}, "
            f"zR@100={metrics['zR@100']:.4f}, "
            f"novel-rank-median={metrics['novel_only_rank_median']:.2f}, "
            f"novel-rank@5={metrics['novel_only_rank_at_5']:.4f}"
        )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
