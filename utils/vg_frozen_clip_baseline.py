"""Evaluate a training-free frozen-CLIP relation baseline on VG OvR.

The detector is used only to produce object boxes, object labels, and object
scores.  For every directed subject-object proposal pair, this script crops
the pair union from the raw VG PIL image, encodes that crop with
the original frozen CLIP image encoder, and compares it with the 50 cached
literal S-P-O text embeddings for the predicted endpoint classes.

No LAIN checkpoint, PromptLearner, IA/LA block, relation head, adapter, MLP,
optimizer, or backward pass is constructed or used.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pocket.ops

from datasets import DataFactory, custom_collate
from detr.util.misc import nested_tensor_from_tensor_list
from models.egtr_detector import EgtrPostProcess, load_egtr_vg_detector
from utils.vg_evaluator import evaluate_vg_image_recall, summarize_vg_recall


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training-free frozen CLIP baseline for VG OvR",
    )
    parser.add_argument("--run-args", required=True)
    parser.add_argument("--literal-cache", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-instances", type=int, default=50)
    parser.add_argument("--box-score-thresh", type=float, default=0.0)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--clip-batch-size", type=int, default=128)
    parser.add_argument(
        "--score-mode",
        choices=("clip_only", "clip_x_detector"),
        default="clip_only",
        help=(
            "Use only frozen-CLIP predicate probabilities, or multiply "
            "them by the two frozen EGTR endpoint confidences."
        ),
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def validate_cli(cli):
    if cli.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if cli.max_instances < 2:
        raise ValueError("--max-instances must be at least 2")
    if cli.clip_batch_size <= 0:
        raise ValueError("--clip-batch-size must be positive")
    for name, value in (
        ("--box-score-thresh", cli.box_score_thresh),
        ("--nms-threshold", cli.nms_threshold),
        ("--iou-threshold", cli.iou_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def load_run_args(path, cli):
    with open(path, "r", encoding="utf-8") as handle:
        args = SimpleNamespace(**json.load(handle))

    if not getattr(args, "vg_ovr", False):
        raise ValueError("The supplied args.txt must enable VG OvR")
    if getattr(args, "dataset", None) != "vg":
        raise ValueError("The supplied args.txt must use dataset='vg'")
    if int(getattr(args, "num_classes", -1)) != 50:
        raise ValueError("VG OvR requires 50 predicates")
    if not getattr(args, "egtr_detector_dir", ""):
        raise ValueError("args.txt is missing egtr_detector_dir")

    args.eval = True
    args.debug = True
    args.local_rank = 0
    args.world_size = 1
    args.num_workers = cli.num_workers

    clip_name = Path(cli.clip_checkpoint).stem
    if clip_name == "ViT-B-16":
        args.clip_model_name = "ViT-B/16"
    elif clip_name == "ViT-L-14-336px":
        args.clip_model_name = "ViT-L/14@336px"
    else:
        raise ValueError(f"Unsupported CLIP checkpoint: {clip_name}")
    return args


def load_literal_cache(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "text_features" not in payload:
        raise ValueError("Literal cache must contain 'text_features'")
    features = payload["text_features"]
    if tuple(features.shape) != (150, 50, 150, 512):
        raise ValueError(
            "Unexpected literal-cache shape: "
            f"{tuple(features.shape)}"
        )
    if not torch.isfinite(features).all():
        raise ValueError("Literal cache contains non-finite values")
    return features.contiguous(), payload.get("metadata", {})


def prepare_proposals(results, score_threshold, max_instances, nms_threshold):
    """Apply the same class-aware NMS and Top-K policy as LAIN-SGG."""
    proposals = []
    for result in results:
        scores = result["scores"]
        labels = result["labels"].long()
        boxes = result["boxes"]

        keep = batched_nms(boxes, scores, labels, nms_threshold)
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        keep = torch.nonzero(
            scores >= score_threshold,
            as_tuple=False,
        ).squeeze(1)
        if len(keep) > max_instances:
            local_order = scores[keep].argsort(descending=True)
            keep = keep[local_order[:max_instances]]

        proposals.append(
            {
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep],
            }
        )
    return proposals


def directed_pairs(num_boxes, device):
    indices = torch.arange(num_boxes, device=device)
    subject = indices[:, None].expand(num_boxes, num_boxes).reshape(-1)
    obj = indices[None, :].expand(num_boxes, num_boxes).reshape(-1)
    keep = subject != obj
    return subject[keep], obj[keep]


def union_boxes(boxes, subject, obj, image_height, image_width):
    subject_boxes = boxes[subject]
    object_boxes = boxes[obj]
    unions = torch.stack(
        [
            torch.minimum(subject_boxes[:, 0], object_boxes[:, 0]),
            torch.minimum(subject_boxes[:, 1], object_boxes[:, 1]),
            torch.maximum(subject_boxes[:, 2], object_boxes[:, 2]),
            torch.maximum(subject_boxes[:, 3], object_boxes[:, 3]),
        ],
        dim=1,
    )
    unions[:, 0::2].clamp_(0, image_width)
    unions[:, 1::2].clamp_(0, image_height)
    unions[:, 2] = torch.maximum(unions[:, 2], unions[:, 0] + 1.0)
    unions[:, 3] = torch.maximum(unions[:, 3], unions[:, 1] + 1.0)
    return unions


def square_crop_box(box, image_width, image_height):
    """Expand a pair union to a valid square in original-image pixels."""
    x1, y1, x2, y2 = [float(value) for value in box]
    side = max(x2 - x1, y2 - y1, 1.0)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = center_x - side / 2.0
    top = center_y - side / 2.0
    right = center_x + side / 2.0
    bottom = center_y + side / 2.0

    if left < 0:
        right -= left
        left = 0.0
    if top < 0:
        bottom -= top
        top = 0.0
    if right > image_width:
        left -= right - image_width
        right = float(image_width)
    if bottom > image_height:
        top -= bottom - image_height
        bottom = float(image_height)

    left = max(left, 0.0)
    top = max(top, 0.0)
    right = min(max(right, left + 1.0), float(image_width))
    bottom = min(max(bottom, top + 1.0), float(image_height))
    return (
        int(np.floor(left)),
        int(np.floor(top)),
        int(np.ceil(right)),
        int(np.ceil(bottom)),
    )


def preprocess_clip_crop(image, crop_box, resolution=224):
    """Resize and normalize a raw PIL crop for original CLIP ViT-B/16."""
    crop = image.crop(crop_box).convert("RGB")
    crop = crop.resize(
        (resolution, resolution),
        resample=Image.Resampling.BICUBIC,
    )
    array = np.asarray(crop, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.as_tensor(
        [0.48145466, 0.4578275, 0.40821073],
        dtype=tensor.dtype,
    )[:, None, None]
    std = torch.as_tensor(
        [0.26862954, 0.26130258, 0.27577711],
        dtype=tensor.dtype,
    )[:, None, None]
    return (tensor - mean) / std


def build_raw_evaluation_target(target, image_height, image_width):
    """Convert raw VG xyxy boxes to the evaluator's normalized cxcywh."""
    scale = torch.as_tensor(
        [image_width, image_height, image_width, image_height],
        dtype=torch.float32,
    )

    def convert(boxes):
        boxes = boxes.float()
        x1, y1, x2, y2 = boxes.unbind(-1)
        converted = torch.stack(
            [
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                x2 - x1,
                y2 - y1,
            ],
            dim=-1,
        )
        return converted / scale

    evaluation_target = dict(target)
    evaluation_target["boxes_h"] = convert(target["boxes_h"])
    evaluation_target["boxes_o"] = convert(target["boxes_o"])
    evaluation_target["size"] = torch.as_tensor(
        [image_height, image_width],
        dtype=torch.long,
    )
    return evaluation_target


@torch.inference_mode()
def detect(detector, postprocessor, images_orig, output_sizes):
    nested = nested_tensor_from_tensor_list(images_orig)
    if nested.mask is None:
        raise RuntimeError("EGTR requires an image padding mask")
    output = detector(
        pixel_values=nested.tensors,
        pixel_mask=(~nested.mask).long(),
    )
    raw = {
        "pred_logits": output.logits,
        "pred_boxes": output.pred_boxes,
        "feats": output.last_hidden_state,
    }
    return postprocessor(raw, output_sizes)


@torch.inference_mode()
def score_image(
    clip_model,
    literal_cache,
    raw_image,
    proposal,
    clip_batch_size,
    score_mode,
):
    """Return evaluator-format predictions for one image.

    The union crop is pair-local. Direction is supplied by the ordered
    detector labels used to choose ``subject-predicate-object`` prototypes.
    This avoids reusing one whole-image CLIP vector for every pair.
    """
    boxes = proposal["boxes"]
    labels = proposal["labels"]
    detector_scores = proposal["scores"]
    num_boxes = len(boxes)
    image_width, image_height = raw_image.size

    if num_boxes < 2:
        empty = torch.empty(0, device=boxes.device)
        return {
            "boxes": boxes,
            "pairing": torch.empty(2, 0, dtype=torch.long, device=boxes.device),
            "scores": empty,
            "labels": empty.long(),
            "subjects": empty.long(),
            "objects": empty.long(),
            "size": torch.as_tensor(
                [image_height, image_width],
                device=boxes.device,
            ),
        }

    subject, obj = directed_pairs(num_boxes, boxes.device)
    unions = union_boxes(
        boxes,
        subject,
        obj,
        image_height,
        image_width,
    )

    pair_probabilities = []
    for start in range(0, len(subject), clip_batch_size):
        end = min(start + clip_batch_size, len(subject))
        crops = torch.stack(
            [
                preprocess_clip_crop(
                    raw_image,
                    square_crop_box(box, image_width, image_height),
                )
                for box in unions[start:end].cpu()
            ]
        ).to(boxes.device)
        image_features = clip_model.encode_image(crops)
        if isinstance(image_features, (tuple, list)):
            image_features = image_features[0]
        image_features = F.normalize(image_features.float(), dim=-1)

        subject_labels = labels[subject[start:end]].cpu()
        object_labels = labels[obj[start:end]].cpu()
        text_features = literal_cache[
            subject_labels,
            :,
            object_labels,
        ].to(device=boxes.device, dtype=torch.float32)
        text_features = F.normalize(text_features, dim=-1)

        cosine = torch.einsum(
            "pd,pcd->pc",
            image_features,
            text_features,
        )
        logits = cosine * clip_model.logit_scale.exp().float()
        pair_probabilities.append(logits.softmax(dim=-1))

    predicate_probability = torch.cat(pair_probabilities, dim=0)
    endpoint_prior = (
        detector_scores[subject] * detector_scores[obj]
    ).float()
    if score_mode == "clip_only":
        final_scores = predicate_probability
    elif score_mode == "clip_x_detector":
        final_scores = predicate_probability * endpoint_prior[:, None]
    else:
        raise ValueError(f"Unsupported score mode: {score_mode}")

    num_predicates = final_scores.shape[1]
    predicate_labels = torch.arange(
        num_predicates,
        device=boxes.device,
    ).repeat(len(subject))
    expanded_subject = subject.repeat_interleave(num_predicates)
    expanded_object = obj.repeat_interleave(num_predicates)

    return {
        "boxes": boxes,
        "pairing": torch.stack(
            [expanded_subject, expanded_object],
            dim=0,
        ),
        "scores": final_scores.reshape(-1),
        "labels": predicate_labels,
        "subjects": labels[subject].repeat_interleave(num_predicates),
        "objects": labels[obj].repeat_interleave(num_predicates),
        "size": torch.as_tensor(
            [image_height, image_width],
            device=boxes.device,
        ),
    }


def main():
    cli = parse_args()
    validate_cli(cli)
    run_args = load_run_args(cli.run_args, cli)

    seed = int(getattr(run_args, "seed", 66))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(cli.device)
    dataset = DataFactory(
        name="vg",
        partition="test",
        data_root=run_args.data_root,
        clip_model_name=run_args.clip_model_name,
        num_classes=50,
        args=run_args,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cli.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=custom_collate,
    )

    detector = load_egtr_vg_detector(
        run_args.egtr_detector_dir
    ).to(device).eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    postprocessor = EgtrPostProcess()

    # [Original frozen CLIP]
    # Load the TorchScript model directly.  This baseline therefore does not
    # construct LAIN adapters or a trainable prompt path.
    clip_model = torch.jit.load(
        str(Path(cli.clip_checkpoint).resolve()),
        map_location=device,
    ).eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)

    literal_cache, cache_metadata = load_literal_cache(cli.literal_cache)
    base = tuple(dataset.dataset.base_predicate_indices)
    novel = tuple(dataset.dataset.novel_predicate_indices)
    if len(base) != 35 or len(novel) != 15:
        raise ValueError(
            f"Unexpected OvR split: base={len(base)}, novel={len(novel)}"
        )

    print("Visible GPU:", torch.cuda.get_device_name(0))
    print("Training/checkpoint load: NONE")
    print("Relation scorer: frozen CLIP union crop vs literal S-P-O")
    print("Crop source: raw VG PIL image")
    print("Score mode:", cli.score_mode)
    print("Proposal K:", cli.max_instances)
    print("Box threshold:", cli.box_score_thresh)

    results = []
    proposal_counts = []
    pair_counts = []
    limit = min(cli.num_images, len(dataset))

    for index, batch in enumerate(tqdm(loader, total=limit)):
        if index >= limit:
            break
        inputs = pocket.ops.relocate_to_cuda(batch[0])
        # [Shared raw-image coordinate system]
        # DataFactory supplies the detector tensor.  Its wrapped VGDataset
        # supplies the untransformed PIL image and pixel-space GT boxes for
        # pair cropping and evaluation.  The iteration order is identical.
        (raw_image, raw_target), _ = dataset.dataset[index]
        raw_width, raw_height = raw_image.size
        target = build_raw_evaluation_target(
            raw_target,
            raw_height,
            raw_width,
        )
        images_orig = [item[0].float() for item in inputs]
        output_sizes = torch.as_tensor(
            [[raw_height, raw_width]],
            device=device,
        )

        detector_results = detect(
            detector,
            postprocessor,
            images_orig,
            output_sizes,
        )
        proposals = prepare_proposals(
            detector_results,
            cli.box_score_thresh,
            cli.max_instances,
            cli.nms_threshold,
        )
        proposal = proposals[0]
        detection = score_image(
            clip_model,
            literal_cache,
            raw_image,
            proposal,
            cli.clip_batch_size,
            cli.score_mode,
        )

        detection_cpu = pocket.ops.relocate_to_cpu(
            detection,
            ignore=True,
        )
        image_result = evaluate_vg_image_recall(
            detection_cpu,
            target,
            recall_k=(20, 50, 100),
            iou_threshold=cli.iou_threshold,
            base_predicate_indices=base,
            novel_predicate_indices=novel,
        )
        if image_result is not None:
            results.append(image_result)
        proposal_counts.append(len(proposal["boxes"]))
        pair_counts.append(
            len(proposal["boxes"]) * (len(proposal["boxes"]) - 1)
        )

    summary = summarize_vg_recall(
        results,
        recall_k=(20, 50, 100),
        include_ovr=True,
    )
    # Use Novel Recall naming in new reports while retaining the existing
    # evaluator's internal compatibility key.
    for k in (20, 50, 100):
        summary[f"nR@{k}"] = summary.pop(f"zR@{k}")
    summary.update(
        {
            "method": "frozen_clip_union_literal_spo",
            "trained": False,
            "score_mode": cli.score_mode,
            "images": len(results),
            "proposal_k": cli.max_instances,
            "box_score_threshold": cli.box_score_thresh,
            "crop_source": "raw_vg_pil_proposal_union_square",
            "mean_proposals": float(np.mean(proposal_counts)),
            "mean_directed_pairs": float(np.mean(pair_counts)),
            "literal_cache": str(Path(cli.literal_cache).resolve()),
            "literal_template": cache_metadata.get("template"),
        }
    )

    output_path = Path(cli.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n===== TRAINING-FREE FROZEN CLIP VG OvR =====")
    print(
        "R@20/50/100: "
        f"{summary['R@20'] * 100:.2f}/"
        f"{summary['R@50'] * 100:.2f}/"
        f"{summary['R@100'] * 100:.2f}"
    )
    print(
        "bR@20/50/100: "
        f"{summary['bR@20'] * 100:.2f}/"
        f"{summary['bR@50'] * 100:.2f}/"
        f"{summary['bR@100'] * 100:.2f}"
    )
    print(
        "nR@20/50/100: "
        f"{summary['nR@20'] * 100:.2f}/"
        f"{summary['nR@50'] * 100:.2f}/"
        f"{summary['nR@100'] * 100:.2f}"
    )
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
