"""Audit LAIN VG OvR data selection against the official OvSGTR loader.

This script is read-only.  It does not import either project and does not
modify the dataset.  Instead, it applies both selection contracts directly
to the shared VG-SGG HDF5 annotations and reports where their sample sets
diverge.

The official behavior mirrored here comes from gpt4vision/OvSGTR:

* use ``split_GLIPunseen``;
* reserve the first 5,000 training candidates as validation data;
* retain only base-predicate relations for OvR training;
* require overlapping subject/object boxes during training;
* sample one predicate for every duplicate directed pair in ``__getitem__``;
* merge same-class GT entities whose IoU is greater than 0.9;
* truncate GT objects to at most 100 (reported, but not randomly simulated).

The audit keeps predicate identifiers 1-based while reading VG-SGG.h5,
matching the official loader.  LAIN's 0-based public target convention is
checked separately through predicate names.
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import h5py
import numpy as np


OVSGTR_BASE_PREDICATES = (
    "between",
    "to",
    "made of",
    "looking at",
    "along",
    "laying on",
    "using",
    "carrying",
    "against",
    "mounted on",
    "sitting on",
    "flying in",
    "covering",
    "from",
    "over",
    "near",
    "hanging from",
    "across",
    "at",
    "above",
    "watching",
    "covered in",
    "wearing",
    "holding",
    "and",
    "standing on",
    "lying on",
    "growing on",
    "under",
    "on back of",
    "with",
    "has",
    "in front of",
    "behind",
    "parked on",
)

OVSGTR_NOVEL_PREDICATES = (
    "belonging to",
    "part of",
    "riding",
    "walking in",
    "in",
    "of",
    "painted on",
    "playing",
    "for",
    "walking on",
    "says",
    "attached to",
    "eating",
    "on",
    "wears",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="vg",
        help="LAIN VG root containing vg_data/stanford_filtered.",
    )
    parser.add_argument(
        "--num-val-images",
        type=int,
        default=5000,
        help="Official OvSGTR validation prefix removed from training.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON summary path.",
    )
    return parser.parse_args()


def box_iou_one_to_one(boxes_a, boxes_b):
    left_top = np.maximum(boxes_a[:, :2], boxes_b[:, :2])
    right_bottom = np.minimum(boxes_a[:, 2:], boxes_b[:, 2:])
    size = np.clip(right_bottom - left_top, a_min=0.0, a_max=None)
    intersection = size[:, 0] * size[:, 1]

    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None)
    area_a *= np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0.0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None)
    area_b *= np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0.0, None)
    union = area_a + area_b - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def pairwise_iou(boxes):
    left_top = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    right_bottom = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    size = np.clip(right_bottom - left_top, a_min=0.0, a_max=None)
    intersection = size[..., 0] * size[..., 1]

    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    areas *= np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    union = areas[:, None] + areas[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def official_merge_entities(boxes, labels, relations):
    """Mirror OvSGTR get_groundtruth's same-class IoU>0.9 merge."""
    if len(boxes) == 0:
        return boxes, labels, relations, 0, 0

    match = pairwise_iou(boxes) > 0.9
    match &= labels[:, None] == labels[None, :]

    keep_ids = []
    old_to_new = {}
    for entity_id in range(len(match)):
        matched_ids = np.flatnonzero(match[entity_id]).tolist()
        if not matched_ids:
            continue
        keep_ids.append(entity_id)
        new_id = len(keep_ids) - 1
        for matched_id in matched_ids:
            old_to_new[matched_id] = new_id
        match[:, matched_ids] = False

    remapped = relations.copy()
    endpoints_changed = 0
    for row in remapped:
        old_subject = int(row[0])
        old_object = int(row[1])
        row[0] = old_to_new[old_subject]
        row[1] = old_to_new[old_object]
        endpoints_changed += int(
            row[0] != old_subject or row[1] != old_object
        )

    removed = len(boxes) - len(keep_ids)
    return boxes[keep_ids], labels[keep_ids], remapped, removed, endpoints_changed


def to_xyxy(boxes_xcycwh):
    boxes = boxes_xcycwh.astype(np.float64, copy=True)
    boxes[:, :2] -= boxes[:, 2:] / 2.0
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def load_names(dict_path):
    with open(dict_path, "r", encoding="utf-8") as handle:
        dictionary = json.load(handle)

    predicates = dictionary.get("idx_to_predicate")
    labels = dictionary.get("idx_to_label")
    if predicates is None or labels is None:
        raise KeyError("VG dictionary lacks idx_to_predicate/idx_to_label")

    predicate_names = [predicates[str(index)] for index in range(1, 51)]
    object_names = [labels[str(index)] for index in range(1, 151)]
    return predicate_names, object_names


def validate_vocabulary(predicate_names):
    base = set(OVSGTR_BASE_PREDICATES)
    novel = set(OVSGTR_NOVEL_PREDICATES)
    active = set(predicate_names)

    if base & novel:
        raise ValueError("Official Base and Novel predicate names overlap")
    if base | novel != active:
        raise ValueError(
            "Active VG predicates do not match the official OvSGTR split. "
            f"missing={sorted((base | novel) - active)}, "
            f"extra={sorted(active - (base | novel))}"
        )

    name_to_h5 = {
        name: index + 1
        for index, name in enumerate(predicate_names)
    }
    return (
        {name_to_h5[name] for name in OVSGTR_BASE_PREDICATES},
        {name_to_h5[name] for name in OVSGTR_NOVEL_PREDICATES},
    )


def relation_slice(first_rel, last_rel, relationships, predicates, image_id):
    first = int(first_rel[image_id])
    last = int(last_rel[image_id])
    if first < 0 or last < first:
        return np.zeros((0, 3), dtype=np.int64)
    rel = relationships[first:last + 1]
    pred = predicates[first:last + 1].reshape(-1, 1)
    return np.concatenate([rel.astype(np.int64), pred.astype(np.int64)], axis=1)


def localize_relations(relations, first_box):
    localized = relations.copy()
    localized[:, :2] -= int(first_box)
    return localized


def duplicate_pair_counts(relations):
    pair_counts = Counter((int(row[0]), int(row[1])) for row in relations)
    duplicate_pairs = sum(count > 1 for count in pair_counts.values())
    removed_by_sampling = sum(count - 1 for count in pair_counts.values())
    return duplicate_pairs, removed_by_sampling


def main():
    args = parse_args()
    stanford_root = os.path.join(
        args.data_root,
        "vg_data",
        "stanford_filtered",
    )
    h5_path = os.path.join(stanford_root, "VG-SGG.h5")
    dict_path = os.path.join(stanford_root, "VG-SGG-dicts.json")

    for path in (h5_path, dict_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    predicate_names, object_names = load_names(dict_path)
    base_h5, novel_h5 = validate_vocabulary(predicate_names)

    with h5py.File(h5_path, "r") as handle:
        required = {
            "split_GLIPunseen",
            "boxes_1024",
            "labels",
            "relationships",
            "predicates",
            "img_to_first_box",
            "img_to_last_box",
            "img_to_first_rel",
            "img_to_last_rel",
        }
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise KeyError(f"VG HDF5 is missing: {missing}")

        split = handle["split_GLIPunseen"][:]
        boxes = to_xyxy(handle["boxes_1024"][:])
        labels = handle["labels"][:, 0].astype(np.int64)
        relationships = handle["relationships"][:]
        predicates = handle["predicates"][:, 0]
        first_box = handle["img_to_first_box"][:]
        last_box = handle["img_to_last_box"][:]
        first_rel = handle["img_to_first_rel"][:]
        last_rel = handle["img_to_last_rel"][:]

    valid = (first_box >= 0) & (first_rel >= 0)
    train_candidates = np.flatnonzero((split == 0) & valid)
    test_candidates = np.flatnonzero((split == 2) & valid)

    if args.num_val_images < 0:
        raise ValueError("--num-val-images must be non-negative")
    official_candidates = train_candidates[args.num_val_images:]

    lain_train = []
    official_train = []
    official_removed_non_overlap = []
    official_duplicate_pair_images = 0
    official_duplicate_pairs = 0
    official_relations_removed_by_duplicate_sampling = 0
    images_over_100_objects = 0
    relation_counts = defaultdict(int)

    for image_id in train_candidates:
        rel = relation_slice(
            first_rel,
            last_rel,
            relationships,
            predicates,
            image_id,
        )
        base_rel = rel[np.isin(rel[:, 2], tuple(base_h5))]
        if len(base_rel) > 0:
            lain_train.append(int(image_id))
            relation_counts["lain_base_relations"] += len(base_rel)

    for image_id in official_candidates:
        rel = relation_slice(
            first_rel,
            last_rel,
            relationships,
            predicates,
            image_id,
        )
        rel = rel[np.isin(rel[:, 2], tuple(base_h5))]
        if len(rel) == 0:
            continue

        local_rel = localize_relations(rel, first_box[image_id])
        image_boxes = boxes[first_box[image_id]:last_box[image_id] + 1]
        if len(image_boxes) > 100:
            images_over_100_objects += 1

        subject_boxes = image_boxes[local_rel[:, 0]]
        object_boxes = image_boxes[local_rel[:, 1]]
        overlap = box_iou_one_to_one(subject_boxes, object_boxes) > 0.0
        rel = rel[overlap]
        local_rel = local_rel[overlap]
        if len(rel) == 0:
            official_removed_non_overlap.append(int(image_id))
            continue

        image_labels = labels[first_box[image_id]:last_box[image_id] + 1]
        (
            image_boxes,
            image_labels,
            local_rel,
            merged_boxes,
            changed_endpoints,
        ) = official_merge_entities(image_boxes, image_labels, local_rel)
        if merged_boxes:
            relation_counts["official_train_images_with_entity_merge"] += 1
            relation_counts["official_train_entities_merged"] += merged_boxes
            relation_counts["official_train_relations_with_remapped_endpoints"] += changed_endpoints

        duplicate_pairs, removed = duplicate_pair_counts(local_rel)
        if duplicate_pairs:
            official_duplicate_pair_images += 1
            official_duplicate_pairs += duplicate_pairs
            official_relations_removed_by_duplicate_sampling += removed

        official_train.append(int(image_id))
        relation_counts["official_base_overlap_relations"] += len(rel)

    lain_train_set = set(lain_train)
    official_train_set = set(official_train)
    validation_prefix = set(int(index) for index in train_candidates[:args.num_val_images])

    # Test is not base-filtered and does not use the train-only overlap or
    # duplicate-relation filters.  Both loaders should select these image IDs.
    test_base_relations = 0
    test_novel_relations = 0
    test_images_over_100_objects = 0
    for image_id in test_candidates:
        rel = relation_slice(
            first_rel,
            last_rel,
            relationships,
            predicates,
            image_id,
        )
        test_base_relations += int(np.isin(rel[:, 2], tuple(base_h5)).sum())
        test_novel_relations += int(np.isin(rel[:, 2], tuple(novel_h5)).sum())

        image_boxes = boxes[first_box[image_id]:last_box[image_id] + 1]
        image_labels = labels[first_box[image_id]:last_box[image_id] + 1]
        if len(image_boxes) > 100:
            test_images_over_100_objects += 1
            # The official loader randomly truncates these images.  Avoid
            # pretending that a deterministic audit can reproduce that draw.
            continue
        local_rel = localize_relations(rel, first_box[image_id])
        (
            image_boxes,
            image_labels,
            local_rel,
            merged_boxes,
            changed_endpoints,
        ) = official_merge_entities(image_boxes, image_labels, local_rel)
        if merged_boxes:
            relation_counts["official_test_images_with_entity_merge"] += 1
            relation_counts["official_test_entities_merged"] += merged_boxes
            relation_counts["official_test_relations_with_remapped_endpoints"] += changed_endpoints

    summary = {
        "contract": {
            "split_key": "split_GLIPunseen",
            "num_val_images": args.num_val_images,
            "official_train_filter_non_overlap": True,
            "official_train_filter_duplicate_relations": True,
            "official_same_class_entity_merge_iou": 0.9,
            "official_max_gt_objects": 100,
        },
        "vocabulary": {
            "objects": len(object_names),
            "predicates": len(predicate_names),
            "base_predicates": len(base_h5),
            "novel_predicates": len(novel_h5),
            "base_names": list(OVSGTR_BASE_PREDICATES),
            "novel_names": list(OVSGTR_NOVEL_PREDICATES),
        },
        "images": {
            "raw_train_with_boxes_and_relations": len(train_candidates),
            "official_validation_prefix": len(validation_prefix),
            "lain_ovr_train": len(lain_train_set),
            "official_ovsgtr_train": len(official_train_set),
            "only_in_lain": len(lain_train_set - official_train_set),
            "only_in_official": len(official_train_set - lain_train_set),
            "lain_extra_from_validation_prefix": len(lain_train_set & validation_prefix),
            "official_removed_after_non_overlap": len(official_removed_non_overlap),
            "test": len(test_candidates),
            "official_train_images_over_100_gt_objects": images_over_100_objects,
            "official_test_images_over_100_gt_objects": test_images_over_100_objects,
        },
        "relations": {
            "lain_train_base_before_overlap": relation_counts["lain_base_relations"],
            "official_train_base_after_overlap": relation_counts["official_base_overlap_relations"],
            "official_train_images_with_duplicate_directed_pairs": official_duplicate_pair_images,
            "official_train_duplicate_directed_pairs": official_duplicate_pairs,
            "official_relations_removed_by_one_predicate_sampling": official_relations_removed_by_duplicate_sampling,
            "official_train_images_with_entity_merge": relation_counts["official_train_images_with_entity_merge"],
            "official_train_entities_merged": relation_counts["official_train_entities_merged"],
            "official_train_relations_with_remapped_endpoints": relation_counts["official_train_relations_with_remapped_endpoints"],
            "official_test_images_with_entity_merge": relation_counts["official_test_images_with_entity_merge"],
            "official_test_entities_merged": relation_counts["official_test_entities_merged"],
            "official_test_relations_with_remapped_endpoints": relation_counts["official_test_relations_with_remapped_endpoints"],
            "test_base": test_base_relations,
            "test_novel": test_novel_relations,
        },
        "examples": {
            "only_in_lain_first_20": sorted(lain_train_set - official_train_set)[:20],
            "only_in_official_first_20": sorted(official_train_set - lain_train_set)[:20],
            "removed_non_overlap_first_20": official_removed_non_overlap[:20],
        },
    }

    print("\n===== LAIN vs OFFICIAL OvSGTR DATA CONTRACT =====")
    print("Vocabulary: 150 objects, 35 base + 15 novel predicates: OK")
    for key, value in summary["images"].items():
        print(f"{key}: {value}")
    for key, value in summary["relations"].items():
        print(f"{key}: {value}")

    if official_train_set - lain_train_set:
        raise AssertionError(
            "Official train contains images missing from the LAIN-style "
            "selection; inspect predicate indexing or overlap handling."
        )

    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"saved: {args.output}")

    print("VG OvSGTR data contract audit: OK")


if __name__ == "__main__":
    main()
