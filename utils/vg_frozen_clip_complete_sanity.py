"""Comprehensive sanity audit for frozen-CLIP VG predicate classification.

This diagnostic deliberately removes detector and pair-generation errors by
using GT boxes, GT object labels, and GT directed pairs.  It validates four
independent contracts:

1. GT object crops must retain enough visual signal for CLIP object ranking.
2. Frozen CLIP predicate Acc@K is measured on raw and S/O-marked union crops.
3. A matched image/text pair should score above a shuffled image/text pair.
4. Qualitative Top-5 predictions are saved for direct visual inspection.

No parameter is trained or loaded from a LAIN relation checkpoint.
"""

import argparse
import html
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from CLIP.customCLIP import tokenize
from datasets.vg import VGDataset
from utils.vg_frozen_clip_gt_pair_audit import (
    build_union_boxes,
    load_literal_cache,
    square_crop_box,
)


CLIP_MEAN = torch.tensor(
    [0.48145466, 0.4578275, 0.40821073],
    dtype=torch.float32,
)[:, None, None]
CLIP_STD = torch.tensor(
    [0.26862954, 0.26130258, 0.27577711],
    dtype=torch.float32,
)[:, None, None]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete frozen-CLIP VG GT-contract sanity audit",
    )
    parser.add_argument("--run-args", required=True)
    parser.add_argument("--literal-cache", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--qualitative-count", type=int, default=50)
    parser.add_argument(
        "--marked-template",
        default=(
            "a photo of a {subject} outlined in red {predicate} "
            "a {object} outlined in blue"
        ),
    )
    parser.add_argument("--vg-ovsgtr-protocol", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--qualitative-dir", required=True)
    return parser.parse_args()


def load_run_args(path, cli):
    with open(path, "r", encoding="utf-8") as handle:
        args = SimpleNamespace(**json.load(handle))
    if getattr(args, "dataset", None) != "vg":
        raise ValueError("The supplied run args must use dataset='vg'")
    if not getattr(args, "vg_ovr", False):
        raise ValueError("The supplied run args must enable VG OvR")
    args.eval = True
    args.debug = True
    args.local_rank = 0
    args.world_size = 1
    args.num_workers = cli.num_workers
    args.vg_ovsgtr_protocol = bool(cli.vg_ovsgtr_protocol)
    args.vg_ovsgtr_num_val_images = int(
        getattr(args, "vg_ovsgtr_num_val_images", 5000)
    )
    return args


def to_clip_tensor(image, resolution=224):
    image = image.convert("RGB").resize(
        (resolution, resolution),
        resample=Image.Resampling.BICUBIC,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - CLIP_MEAN) / CLIP_STD


def local_box(box, crop_box):
    left, top, _, _ = crop_box
    x1, y1, x2, y2 = [float(value) for value in box]
    return (x1 - left, y1 - top, x2 - left, y2 - top)


def make_pair_crop(image, crop_box, subject_box, object_box, marked):
    crop = image.crop(crop_box).convert("RGB")
    if marked:
        draw = ImageDraw.Draw(crop)
        line_width = max(2, int(max(crop.size) / 80))
        draw.rectangle(
            local_box(subject_box, crop_box),
            outline=(255, 0, 0),
            width=line_width,
        )
        draw.rectangle(
            local_box(object_box, crop_box),
            outline=(0, 80, 255),
            width=line_width,
        )
    return crop


@torch.inference_mode()
def encode_images(clip_model, crops, batch_size, device):
    features = []
    for start in range(0, len(crops), batch_size):
        end = min(start + batch_size, len(crops))
        images = torch.stack(
            [to_clip_tensor(crop) for crop in crops[start:end]]
        ).to(device)
        encoded = clip_model.encode_image(images)
        if isinstance(encoded, (tuple, list)):
            encoded = encoded[0]
        features.append(F.normalize(encoded.float(), dim=-1).cpu())
    return torch.cat(features)


@torch.inference_mode()
def encode_texts(clip_model, texts, batch_size, device):
    features = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        tokens = torch.cat(
            [tokenize(text) for text in texts[start:end]],
            dim=0,
        ).to(device)
        encoded = clip_model.encode_text(tokens)
        features.append(F.normalize(encoded.float(), dim=-1).cpu())
    return torch.cat(features)


def rank_correct(logits, labels):
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    return (logits > correct[:, None]).sum(dim=1) + 1


def rank_summary(ranks, mask=None):
    if mask is not None:
        ranks = ranks[mask]
    if len(ranks) == 0:
        return {
            "count": 0,
            "acc@1": 0.0,
            "acc@3": 0.0,
            "acc@5": 0.0,
            "median_rank": 0.0,
            "mrr": 0.0,
        }
    ranks = ranks.float()
    return {
        "count": int(len(ranks)),
        "acc@1": float((ranks <= 1).float().mean()),
        "acc@3": float((ranks <= 3).float().mean()),
        "acc@5": float((ranks <= 5).float().mean()),
        "median_rank": float(ranks.median()),
        "mrr": float((1.0 / ranks).mean()),
    }


def find_cross_image_shift(image_ids):
    """Find a deterministic roll that never retains the source image."""
    count = len(image_ids)
    for shift in range(1, count):
        shifted = torch.roll(image_ids, shifts=shift)
        if not torch.any(shifted == image_ids):
            return shift
    raise RuntimeError("Could not construct a cross-image shuffle")


def metric_line(name, metric):
    return (
        f"{name}: count={metric['count']}, Acc@1/3/5="
        f"{metric['acc@1'] * 100:.2f}/"
        f"{metric['acc@3'] * 100:.2f}/"
        f"{metric['acc@5'] * 100:.2f}, "
        f"median-rank={metric['median_rank']:.1f}, "
        f"MRR={metric['mrr']:.4f}"
    )


def main():
    cli = parse_args()
    if cli.num_images <= 0 or cli.batch_size <= 0:
        raise ValueError("Image count and batch size must be positive")
    if cli.qualitative_count < 0:
        raise ValueError("Qualitative count must be non-negative")
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    run_args = load_run_args(cli.run_args, cli)
    seed = int(getattr(run_args, "seed", 66))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = VGDataset(
        root=run_args.data_root,
        split="test",
        num_relations=50,
        args=run_args,
    )
    limit = min(cli.num_images, len(dataset))
    device = torch.device(cli.device)
    clip_model = torch.jit.load(
        str(Path(cli.clip_checkpoint).resolve()),
        map_location=device,
    ).eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)

    literal_cache, metadata = load_literal_cache(cli.literal_cache)
    object_names = metadata.get("objects", [])
    predicate_names = metadata.get("predicates", [])
    if len(object_names) != 150 or len(predicate_names) != 50:
        raise ValueError("Cache metadata vocabulary is incomplete")

    object_text = encode_texts(
        clip_model,
        [f"a photo of a {name}" for name in object_names],
        cli.batch_size,
        device,
    )
    base = torch.as_tensor(dataset.base_predicate_indices)
    novel = torch.as_tensor(dataset.novel_predicate_indices)

    relation_labels = []
    relation_subjects = []
    relation_objects = []
    relation_image_ids = []
    raw_logits_parts = []
    raw_ranks = []
    marked_ranks = []
    role_aware_marked_ranks = []
    raw_relation_features = []
    correct_text_features = []
    subject_object_ranks = []
    object_object_ranks = []
    invalid_boxes = 0
    qualitative = []
    qualitative_dir = Path(cli.qualitative_dir).resolve()
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    for image_index in tqdm(range(limit), total=limit):
        (image, target), filename = dataset[image_index]
        predicates = target["verb"].long().cpu()
        subjects = target["subject"].long().cpu()
        objects = target["object"].long().cpu()
        subject_boxes = target["boxes_h"].float().cpu()
        object_boxes = target["boxes_o"].float().cpu()
        if len(predicates) == 0:
            continue
        lengths = {
            len(predicates),
            len(subjects),
            len(objects),
            len(subject_boxes),
            len(object_boxes),
        }
        if len(lengths) != 1:
            raise ValueError(f"Misaligned GT fields in {filename}: {lengths}")

        width, height = image.size
        for boxes in (subject_boxes, object_boxes):
            valid = (
                (boxes[:, 0] >= 0)
                & (boxes[:, 1] >= 0)
                & (boxes[:, 2] <= width)
                & (boxes[:, 3] <= height)
                & (boxes[:, 2] > boxes[:, 0])
                & (boxes[:, 3] > boxes[:, 1])
            )
            invalid_boxes += int((~valid).sum())

        union_boxes = build_union_boxes(
            subject_boxes,
            object_boxes,
            height,
            width,
        )
        crop_boxes = [
            square_crop_box(box, width, height)
            for box in union_boxes
        ]
        raw_crops = [
            make_pair_crop(
                image,
                crop_box,
                subject_box,
                object_box,
                marked=False,
            )
            for crop_box, subject_box, object_box in zip(
                crop_boxes,
                subject_boxes,
                object_boxes,
            )
        ]
        marked_crops = [
            make_pair_crop(
                image,
                crop_box,
                subject_box,
                object_box,
                marked=True,
            )
            for crop_box, subject_box, object_box in zip(
                crop_boxes,
                subject_boxes,
                object_boxes,
            )
        ]
        raw_features = encode_images(
            clip_model,
            raw_crops,
            cli.batch_size,
            device,
        )
        marked_features = encode_images(
            clip_model,
            marked_crops,
            cli.batch_size,
            device,
        )
        pair_text = literal_cache[subjects, :, objects].float()
        pair_text = F.normalize(pair_text, dim=-1)
        raw_logits = torch.einsum("rd,rcd->rc", raw_features, pair_text)
        marked_logits = torch.einsum(
            "rd,rcd->rc",
            marked_features,
            pair_text,
        )
        role_aware_texts = []
        for subject_index, object_index in zip(subjects, objects):
            for predicate_name in predicate_names:
                role_aware_texts.append(
                    cli.marked_template.format(
                        subject=object_names[int(subject_index)],
                        predicate=predicate_name,
                        object=object_names[int(object_index)],
                    )
                )
        role_aware_text = encode_texts(
            clip_model,
            role_aware_texts,
            cli.batch_size,
            device,
        ).view(len(predicates), 50, -1)
        role_aware_marked_logits = torch.einsum(
            "rd,rcd->rc",
            marked_features,
            role_aware_text,
        )
        raw_rank = rank_correct(raw_logits, predicates)
        marked_rank = rank_correct(marked_logits, predicates)
        role_aware_marked_rank = rank_correct(
            role_aware_marked_logits,
            predicates,
        )

        row_indices = torch.arange(len(predicates))
        correct_text = pair_text[row_indices, predicates]
        raw_relation_features.append(raw_features)
        correct_text_features.append(correct_text)
        relation_labels.append(predicates)
        relation_subjects.append(subjects)
        relation_objects.append(objects)
        relation_image_ids.append(
            torch.full((len(predicates),), image_index, dtype=torch.long)
        )
        raw_ranks.append(raw_rank)
        raw_logits_parts.append(raw_logits)
        marked_ranks.append(marked_rank)
        role_aware_marked_ranks.append(role_aware_marked_rank)

        subject_crops = [
            image.crop(square_crop_box(box, width, height)).convert("RGB")
            for box in subject_boxes
        ]
        object_crops = [
            image.crop(square_crop_box(box, width, height)).convert("RGB")
            for box in object_boxes
        ]
        subject_features = encode_images(
            clip_model,
            subject_crops,
            cli.batch_size,
            device,
        )
        object_features = encode_images(
            clip_model,
            object_crops,
            cli.batch_size,
            device,
        )
        subject_object_ranks.append(
            rank_correct(subject_features @ object_text.T, subjects)
        )
        object_object_ranks.append(
            rank_correct(object_features @ object_text.T, objects)
        )

        remaining = cli.qualitative_count - len(qualitative)
        for relation_index in range(min(remaining, len(predicates))):
            top_values, top_indices = raw_logits[relation_index].topk(5)
            stem = (
                f"{len(qualitative):04d}_"
                f"{Path(str(filename)).stem}_r{relation_index}"
            )
            raw_name = stem + "_raw.jpg"
            marked_name = stem + "_marked.jpg"
            raw_crops[relation_index].save(
                qualitative_dir / raw_name,
                quality=95,
            )
            marked_crops[relation_index].save(
                qualitative_dir / marked_name,
                quality=95,
            )
            predicate_index = int(predicates[relation_index])
            qualitative.append(
                {
                    "image": str(filename),
                    "saved_raw_crop": raw_name,
                    "saved_marked_crop": marked_name,
                    "subject": object_names[int(subjects[relation_index])],
                    "object": object_names[int(objects[relation_index])],
                    "ground_truth_predicate": predicate_names[predicate_index],
                    "split": (
                        "base"
                        if predicate_index in dataset.base_predicate_indices
                        else "novel"
                    ),
                    "ground_truth_rank": int(raw_rank[relation_index]),
                    "top1_correct": bool(raw_rank[relation_index] == 1),
                    "top5_contains_ground_truth": bool(
                        raw_rank[relation_index] <= 5
                    ),
                    "top5": [
                        {
                            "predicate": predicate_names[int(index)],
                            "cosine": float(value),
                        }
                        for value, index in zip(top_values, top_indices)
                    ],
                }
            )

    labels = torch.cat(relation_labels)
    relation_subjects = torch.cat(relation_subjects)
    relation_objects = torch.cat(relation_objects)
    image_ids = torch.cat(relation_image_ids)
    raw_logits = torch.cat(raw_logits_parts)
    raw_ranks = torch.cat(raw_ranks)
    marked_ranks = torch.cat(marked_ranks)
    role_aware_marked_ranks = torch.cat(role_aware_marked_ranks)
    visual_features = torch.cat(raw_relation_features)
    text_features = torch.cat(correct_text_features)
    subject_ranks = torch.cat(subject_object_ranks)
    object_ranks = torch.cat(object_object_ranks)
    base_mask = torch.isin(labels, base)
    novel_mask = torch.isin(labels, novel)

    shuffle_shift = find_cross_image_shift(image_ids)
    shuffled_visual = torch.roll(visual_features, shifts=shuffle_shift, dims=0)
    matched_scores = (visual_features * text_features).sum(dim=-1)
    shuffled_scores = (shuffled_visual * text_features).sum(dim=-1)
    paired_delta = matched_scores - shuffled_scores
    top1_predictions = raw_logits.argmax(dim=1)
    per_predicate = []
    for predicate_index, predicate_name in enumerate(predicate_names):
        predicate_mask = labels == predicate_index
        predicate_ranks = raw_ranks[predicate_mask]
        predicted_counts = torch.bincount(
            top1_predictions[predicate_mask],
            minlength=50,
        )
        count = int(predicate_mask.sum())
        top_count = min(5, int((predicted_counts > 0).sum()))
        if top_count:
            values, indices = predicted_counts.topk(top_count)
            common_predictions = [
                {
                    "predicate": predicate_names[int(index)],
                    "count": int(value),
                    "fraction": float(value / count),
                }
                for value, index in zip(values, indices)
            ]
        else:
            common_predictions = []
        per_predicate.append(
            {
                "predicate": predicate_name,
                "split": (
                    "base"
                    if predicate_index in dataset.base_predicate_indices
                    else "novel"
                ),
                **rank_summary(predicate_ranks),
                "common_top1_predictions": common_predictions,
            }
        )

    confusion = torch.zeros((50, 50), dtype=torch.long)
    flat_confusion = labels * 50 + top1_predictions
    confusion.view(-1).copy_(
        torch.bincount(flat_confusion, minlength=2500)
    )
    confusion.fill_diagonal_(0)
    confusion_values, confusion_indices = confusion.flatten().topk(50)
    common_confusions = []
    for value, flat_index in zip(confusion_values, confusion_indices):
        if value == 0:
            break
        gt_index = int(flat_index // 50)
        prediction_index = int(flat_index % 50)
        common_confusions.append(
            {
                "ground_truth": predicate_names[gt_index],
                "prediction": predicate_names[prediction_index],
                "count": int(value),
            }
        )

    predicted_histogram = torch.bincount(
        top1_predictions,
        minlength=50,
    )
    histogram_values, histogram_indices = predicted_histogram.sort(
        descending=True
    )
    top1_histogram = [
        {
            "predicate": predicate_names[int(index)],
            "count": int(value),
            "fraction": float(value / len(labels)),
        }
        for value, index in zip(histogram_values, histogram_indices)
    ]

    summary = {
        "method": "frozen_clip_complete_gt_contract_sanity",
        "trained": False,
        "uses_detector": False,
        "uses_gt_subject_box": True,
        "uses_gt_object_box": True,
        "uses_gt_subject_label": True,
        "uses_gt_object_label": True,
        "uses_gt_directed_pair": True,
        "predicts_only_predicate": True,
        "images": limit,
        "relations": int(len(labels)),
        "invalid_boxes": invalid_boxes,
        "vg_ovsgtr_protocol": bool(cli.vg_ovsgtr_protocol),
        "random_baseline": {"acc@1": 0.02, "acc@3": 0.06, "acc@5": 0.10},
        "object_crop_classification": {
            "subject": rank_summary(subject_ranks),
            "object": rank_summary(object_ranks),
            "combined": rank_summary(
                torch.cat([subject_ranks, object_ranks])
            ),
            "candidate_classes": 150,
            "text_template": "a photo of a {object}",
        },
        "predicate_raw_union": {
            "all": rank_summary(raw_ranks),
            "base": rank_summary(raw_ranks, base_mask),
            "novel": rank_summary(raw_ranks, novel_mask),
            "top1_prediction_histogram": top1_histogram,
            "per_ground_truth_predicate": per_predicate,
            "common_top1_confusions": common_confusions,
        },
        "predicate_marked_union": {
            "description": (
                "red subject box, blue object box, literal cache text"
            ),
            "all": rank_summary(marked_ranks),
            "base": rank_summary(marked_ranks, base_mask),
            "novel": rank_summary(marked_ranks, novel_mask),
        },
        "predicate_role_aware_marked_union": {
            "description": (
                "red subject box, blue object box, and matching color-role "
                "text"
            ),
            "text_template": cli.marked_template,
            "all": rank_summary(role_aware_marked_ranks),
            "base": rank_summary(role_aware_marked_ranks, base_mask),
            "novel": rank_summary(role_aware_marked_ranks, novel_mask),
        },
        "matched_vs_cross_image_shuffled": {
            "shuffle_shift": shuffle_shift,
            "same_image_pairs": int(
                (
                    image_ids
                    == torch.roll(image_ids, shifts=shuffle_shift)
                ).sum()
            ),
            "matched_cosine_mean": float(matched_scores.mean()),
            "shuffled_cosine_mean": float(shuffled_scores.mean()),
            "paired_delta_mean": float(paired_delta.mean()),
            "matched_greater_fraction": float(
                (paired_delta > 0).float().mean()
            ),
        },
        "qualitative_count": len(qualitative),
        "qualitative_dir": str(qualitative_dir),
        "qualitative": qualitative,
    }

    rows = []
    for item in qualitative:
        top5 = "<br>".join(
            f"{rank}. {html.escape(candidate['predicate'])} "
            f"({candidate['cosine']:.4f})"
            for rank, candidate in enumerate(item["top5"], start=1)
        )
        status = (
            "TOP-1"
            if item["top1_correct"]
            else "TOP-5"
            if item["top5_contains_ground_truth"]
            else "MISS"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['image'])}<br>"
            f"{html.escape(item['split'])}</td>"
            f"<td><img src='{html.escape(item['saved_raw_crop'])}'></td>"
            f"<td><img src='{html.escape(item['saved_marked_crop'])}'></td>"
            f"<td>{html.escape(item['subject'])} → "
            f"{html.escape(item['object'])}<br>"
            f"GT: <b>{html.escape(item['ground_truth_predicate'])}</b><br>"
            f"rank={item['ground_truth_rank']} / {status}</td>"
            f"<td>{top5}</td>"
            "</tr>"
        )
    report_html = """<!doctype html>
<html><head><meta charset="utf-8"><title>VG Frozen CLIP Pair Audit</title>
<style>
body { font-family: sans-serif; margin: 24px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #bbb; padding: 8px; vertical-align: top; }
th { position: sticky; top: 0; background: #eee; }
img { width: 224px; height: 224px; object-fit: contain; }
</style></head><body>
<h1>VG Frozen CLIP GT-pair Top-5 Audit</h1>
<p>Red = subject, blue = object. Predictions use the unmarked raw crop.</p>
<table><thead><tr><th>Source</th><th>Raw</th><th>Marked</th>
<th>GT triplet</th><th>CLIP Top-5</th></tr></thead><tbody>
""" + "\n".join(rows) + """
</tbody></table></body></html>"""
    report_path = qualitative_dir / "index.html"
    report_path.write_text(report_html, encoding="utf-8")

    output_path = Path(cli.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\n===== FROZEN CLIP COMPLETE VG SANITY AUDIT =====")
    print("images:", limit)
    print("relations:", len(labels))
    print("invalid boxes:", invalid_boxes)
    print("\n[OBJECT CROPS, 150-WAY]")
    print(metric_line("subject", summary["object_crop_classification"]["subject"]))
    print(metric_line("object", summary["object_crop_classification"]["object"]))
    print("\n[PREDICATE, RAW UNION]")
    for name in ("all", "base", "novel"):
        print(metric_line(name, summary["predicate_raw_union"][name]))
    print("\n[PREDICATE, MARKED UNION]")
    for name in ("all", "base", "novel"):
        print(metric_line(name, summary["predicate_marked_union"][name]))
    print("\n[PREDICATE, ROLE-AWARE MARKED UNION]")
    for name in ("all", "base", "novel"):
        print(
            metric_line(
                name,
                summary["predicate_role_aware_marked_union"][name],
            )
        )
    print("\n[MATCHED VS CROSS-IMAGE SHUFFLED]")
    for key, value in summary["matched_vs_cross_image_shuffled"].items():
        print(f"{key}: {value}")
    print("qualitative crops:", qualitative_dir)
    print("qualitative HTML:", report_path)
    print("saved:", output_path)
    print("Frozen CLIP complete GT-contract sanity: OK")


if __name__ == "__main__":
    main()
