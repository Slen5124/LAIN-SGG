"""Visual Genome SGDet Recall@K evaluation utilities.

The evaluator follows the graph-constrained Visual Genome protocol used by
EGTR: keep one predicate per directed subject-object pair, rank predictions by
their final triplet score, and match subject/predicate/object classes together
with subject and object box IoU.
"""

from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import Tensor
from torchvision.ops import box_iou


DEFAULT_RECALL_K = (20, 50, 100)


def _recover_target_boxes(boxes: Tensor, size: Tensor) -> Tensor:
    """Convert normalized cxcywh target boxes to pixel-space xyxy boxes."""
    center_x, center_y, width, height = boxes.unbind(-1)
    boxes_xyxy = torch.stack(
        [
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ],
        dim=-1,
    )

    image_height, image_width = size.to(
        device=boxes.device,
        dtype=boxes.dtype,
    )
    scale = torch.stack(
        [
            image_width,
            image_height,
            image_width,
            image_height,
        ]
    )
    return boxes_xyxy * scale


def _validate_detection(detection: Dict[str, Tensor]) -> None:
    required = {
        "boxes",
        "pairing",
        "scores",
        "labels",
        "subjects",
        "objects",
    }
    missing = sorted(required.difference(detection))
    if missing:
        raise KeyError(
            "VG detection is missing required fields: "
            + ", ".join(missing)
        )

    pairing = detection["pairing"]
    if pairing.ndim != 2 or pairing.shape[0] != 2:
        raise ValueError(
            "VG detection pairing must have shape [2, num_predictions]"
        )

    num_predictions = pairing.shape[1]
    for key in ("scores", "labels", "subjects", "objects"):
        if len(detection[key]) != num_predictions:
            raise ValueError(
                f"VG detection field '{key}' has length "
                f"{len(detection[key])}, expected {num_predictions}"
            )


def select_graph_constrained_predictions(
    detection: Dict[str, Tensor],
    max_predictions: int = 100,
) -> Tensor:
    """Return ranked indices after keeping one predicate per directed pair."""
    _validate_detection(detection)
    if max_predictions <= 0:
        raise ValueError("max_predictions must be positive")

    scores = detection["scores"]
    pairing = detection["pairing"].long()
    num_predictions = pairing.shape[1]

    if num_predictions == 0:
        return torch.empty(
            0,
            dtype=torch.long,
            device=scores.device,
        )

    num_boxes = len(detection["boxes"])
    if num_boxes == 0:
        raise ValueError("Non-empty VG predictions require proposal boxes")
    if pairing.min().item() < 0 or pairing.max().item() >= num_boxes:
        raise ValueError("VG pairing index is outside the proposal box range")

    # [VG graph constraint]
    # A directed subject-object pair contributes only its highest-scoring
    # predicate. The final LAIN score already includes the detector prior.
    pair_keys = pairing[0] * num_boxes + pairing[1]
    unique_pair_keys = torch.unique(pair_keys, sorted=False)
    best_indices = []

    for pair_key in unique_pair_keys:
        pair_indices = torch.nonzero(
            pair_keys == pair_key,
            as_tuple=False,
        ).squeeze(1)
        local_best = scores[pair_indices].argmax()
        best_indices.append(pair_indices[local_best])

    best_indices = torch.stack(best_indices)
    order = scores[best_indices].argsort(descending=True)
    return best_indices[order[:max_predictions]]


def evaluate_vg_image_recall(
    detection: Dict[str, Tensor],
    target: Dict[str, Tensor],
    recall_k: Sequence[int] = DEFAULT_RECALL_K,
    iou_threshold: float = 0.5,
) -> Optional[Dict[str, object]]:
    """Evaluate graph-constrained SGDet Recall@K for one VG image."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    recall_k = tuple(sorted(set(int(k) for k in recall_k)))
    if not recall_k or recall_k[0] <= 0:
        raise ValueError("recall_k must contain positive integers")

    required_target = {
        "boxes_h",
        "boxes_o",
        "subject",
        "object",
        "verb",
        "size",
    }
    missing_target = sorted(required_target.difference(target))
    if missing_target:
        raise KeyError(
            "VG target is missing required fields: "
            + ", ".join(missing_target)
        )

    num_ground_truth = len(target["verb"])
    if num_ground_truth == 0:
        return None

    _validate_detection(detection)
    ranked_indices = select_graph_constrained_predictions(
        detection,
        max_predictions=max(recall_k),
    )

    boxes = detection["boxes"]
    pairing = detection["pairing"].long()
    device = boxes.device

    gt_boxes_h = _recover_target_boxes(
        target["boxes_h"].to(device),
        target["size"].to(device),
    )
    gt_boxes_o = _recover_target_boxes(
        target["boxes_o"].to(device),
        target["size"].to(device),
    )
    gt_subject = target["subject"].to(device).long()
    gt_object = target["object"].to(device).long()
    gt_predicate = target["verb"].to(device).long()

    if len(ranked_indices) == 0:
        matched_by_prediction = torch.zeros(
            0,
            num_ground_truth,
            dtype=torch.bool,
            device=device,
        )
    else:
        ranked_pairing = pairing[:, ranked_indices]
        predicted_boxes_h = boxes[ranked_pairing[0]]
        predicted_boxes_o = boxes[ranked_pairing[1]]

        subject_iou = box_iou(predicted_boxes_h, gt_boxes_h)
        object_iou = box_iou(predicted_boxes_o, gt_boxes_o)

        triplet_class_match = (
            detection["subjects"][ranked_indices, None].long()
            == gt_subject[None, :]
        )
        triplet_class_match &= (
            detection["labels"][ranked_indices, None].long()
            == gt_predicate[None, :]
        )
        triplet_class_match &= (
            detection["objects"][ranked_indices, None].long()
            == gt_object[None, :]
        )

        matched_by_prediction = triplet_class_match
        matched_by_prediction &= subject_iou >= iou_threshold
        matched_by_prediction &= object_iou >= iou_threshold

    recall = {}
    matched = {}
    for k in recall_k:
        top_matches = matched_by_prediction[:k]
        matched_gt = (
            top_matches.any(dim=0)
            if len(top_matches) > 0
            else torch.zeros(
                num_ground_truth,
                dtype=torch.bool,
                device=device,
            )
        )
        matched_count = int(matched_gt.sum().item())
        matched[k] = matched_count
        recall[k] = matched_count / num_ground_truth

    return {
        "recall": recall,
        "matched": matched,
        "num_gt": num_ground_truth,
        "num_ranked_predictions": len(ranked_indices),
    }


def summarize_vg_recall(
    image_results: Iterable[Dict[str, object]],
    recall_k: Sequence[int] = DEFAULT_RECALL_K,
) -> Dict[str, float]:
    """Average per-image Recall@K, matching the standard VG protocol."""
    recall_k = tuple(sorted(set(int(k) for k in recall_k)))
    results = list(image_results)
    if not results:
        return {f"R@{k}": 0.0 for k in recall_k}

    return {
        f"R@{k}": sum(result["recall"][k] for result in results)
        / len(results)
        for k in recall_k
    }
