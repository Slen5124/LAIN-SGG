"""Measure frozen CLIP predicate ranking on ground-truth VG relation pairs.

This is a diagnostic oracle, not an SGDet result.  Ground-truth subject and
object boxes/classes select a pair-local union crop and the corresponding 50
literal S-P-O text prototypes.  The original frozen CLIP image encoder ranks
the predicates.  No detector, LAIN module, checkpoint, or training is used.
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
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from datasets.vg import VGDataset
from CLIP.customCLIP import tokenize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen CLIP GT-pair predicate oracle for VG OvR",
    )
    parser.add_argument("--run-args", required=True)
    parser.add_argument("--literal-cache", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument(
        "--text-mode",
        choices=[
            "literal_cache",
            "lain_sentence",
            "lain_predicate",
            "lain_spo",
        ],
        default="literal_cache",
        help=(
            "Text representation to rank: frozen literal S-P-O cache, a "
            "training-free LAIN-style relation sentence, LAIN learned-context "
            "predicate prompts, or LAIN learned-context S-P-O prompts."
        ),
    )
    parser.add_argument(
        "--lain-template",
        default="a photo of a {subject} {predicate} a {object}",
        help=(
            "Pair-specific sentence used by lain_sentence and lain_spo. "
            "It must contain subject, predicate, and object placeholders."
        ),
    )
    parser.add_argument(
        "--prompt-checkpoint",
        default="",
        help=(
            "LAIN checkpoint containing prompt_learner.ctx. Required for "
            "lain_predicate and lain_spo."
        ),
    )
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--clip-batch-size", type=int, default=128)
    parser.add_argument(
        "--vg-ovsgtr-protocol",
        action="store_true",
        help=(
            "Use the official OvSGTR entity-merged test ground truth. "
            "Without this flag the legacy LAIN OvR test target is used."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_run_args(path, cli):
    with open(path, "r", encoding="utf-8") as handle:
        args = SimpleNamespace(**json.load(handle))
    if getattr(args, "dataset", None) != "vg":
        raise ValueError("The supplied args.txt must use dataset='vg'")
    if not getattr(args, "vg_ovr", False):
        raise ValueError("The supplied args.txt must enable VG OvR")

    args.eval = True
    args.debug = True
    args.local_rank = 0
    args.world_size = 1
    args.num_workers = cli.num_workers
    args.vg_ovsgtr_protocol = bool(cli.vg_ovsgtr_protocol)
    args.vg_ovsgtr_num_val_images = int(
        getattr(args, "vg_ovsgtr_num_val_images", 5000)
    )

    clip_name = Path(cli.clip_checkpoint).stem
    if clip_name == "ViT-B-16":
        args.clip_model_name = "ViT-B/16"
    elif clip_name == "ViT-L-14-336px":
        raise ValueError(
            "This GT-pair baseline currently requires ViT-B/16 because "
            "the literal cache and crop preprocessing are 512-D/224px."
        )
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


def load_lain_context(path):
    """Load the learned LAIN context without loading any relation model."""
    if not path:
        raise ValueError(
            "--prompt-checkpoint is required for a LAIN prompt text mode"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("LAIN checkpoint does not contain a model state dict")

    matches = [
        (key, value)
        for key, value in state.items()
        if key.endswith("prompt_learner.ctx")
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one prompt_learner.ctx tensor, found "
            f"{[key for key, _ in matches]}"
        )
    key, context = matches[0]
    if context.ndim not in (2, 3):
        raise ValueError(
            f"Unsupported LAIN context shape: {tuple(context.shape)}"
        )
    if context.shape[-1] != 512:
        raise ValueError(
            "This audit expects ViT-B/16 512-D context vectors, got "
            f"{tuple(context.shape)}"
        )
    if context.ndim == 3 and context.shape[0] != 50:
        raise ValueError(
            "Class-specific VG context must contain 50 predicates, got "
            f"{tuple(context.shape)}"
        )
    return context.contiguous(), key, checkpoint.get("epoch")


def encode_prompt_embeddings(clip_model, prompts, tokenized_prompts):
    """Run the original causal CLIP text path from prompt embeddings."""
    dtype = clip_model.token_embedding.weight.dtype
    x = prompts.to(dtype) + clip_model.positional_embedding.to(dtype)
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x).to(dtype)
    eot = tokenized_prompts.argmax(dim=-1)
    rows = torch.arange(x.shape[0], device=x.device)
    projection = clip_model.text_projection
    # TorchScript CLIP may expose token embeddings in FP32 while keeping the
    # final text projection in FP16. Match the projection dtype explicitly,
    # as done by the original CLIP encode_text path.
    return x[rows, eot].to(projection.dtype) @ projection


@torch.inference_mode()
def encode_lain_prompts(
    clip_model,
    classnames,
    predicate_indices,
    context,
    batch_size,
    device,
):
    """Encode ``[SOS] + learned context + class text + [EOS]`` prompts."""
    if len(classnames) != len(predicate_indices):
        raise ValueError("Prompt names and predicate indices are misaligned")
    n_ctx = int(context.shape[-2])
    prompt_prefix = " ".join(["X"] * n_ctx)
    outputs = []

    for start in range(0, len(classnames), batch_size):
        end = min(start + batch_size, len(classnames))
        prompt_strings = [
            f"{prompt_prefix} {name.replace('_', ' ')}."
            for name in classnames[start:end]
        ]
        tokens = torch.cat(
            [tokenize(prompt) for prompt in prompt_strings],
            dim=0,
        ).to(device)
        embeddings = clip_model.token_embedding(tokens).clone()
        if context.ndim == 2:
            selected_context = context.unsqueeze(0).expand(
                end - start,
                -1,
                -1,
            )
        else:
            indices = torch.as_tensor(
                predicate_indices[start:end],
                dtype=torch.long,
            )
            selected_context = context.index_select(0, indices)
        embeddings[:, 1:1 + n_ctx] = selected_context.to(
            device=device,
            dtype=embeddings.dtype,
        )
        features = encode_prompt_embeddings(
            clip_model,
            embeddings,
            tokens,
        )
        outputs.append(F.normalize(features.float(), dim=-1).cpu())

    return torch.cat(outputs, dim=0)


def build_union_boxes(subject_boxes, object_boxes, height, width):
    unions = torch.stack(
        [
            torch.minimum(subject_boxes[:, 0], object_boxes[:, 0]),
            torch.minimum(subject_boxes[:, 1], object_boxes[:, 1]),
            torch.maximum(subject_boxes[:, 2], object_boxes[:, 2]),
            torch.maximum(subject_boxes[:, 3], object_boxes[:, 3]),
        ],
        dim=-1,
    )
    unions[:, 0::2].clamp_(0, width)
    unions[:, 1::2].clamp_(0, height)
    unions[:, 2] = torch.maximum(unions[:, 2], unions[:, 0] + 1.0)
    unions[:, 3] = torch.maximum(unions[:, 3], unions[:, 1] + 1.0)
    return unions


def square_crop_box(box, image_width, image_height):
    """Expand a union box to a valid square without changing its centre."""
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
    """Apply the original CLIP resize and normalization to a raw PIL crop."""
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


@torch.inference_mode()
def rank_image_relations(
    clip_model,
    literal_cache,
    cache_metadata,
    text_mode,
    lain_context,
    predicate_text_features,
    lain_template,
    image,
    target,
    clip_batch_size,
    device,
):
    predicates = target["verb"].long().to(device)
    subjects = target["subject"].long().to(device)
    objects = target["object"].long().to(device)
    if len(predicates) == 0:
        return None

    if not (
        len(predicates)
        == len(subjects)
        == len(objects)
        == len(target["boxes_h"])
        == len(target["boxes_o"])
    ):
        raise ValueError("VG GT relation fields are not aligned")

    # [Raw-image GT crop]
    # VGDataset returns the untouched PIL image and pixel-space xyxy boxes.
    # This avoids projecting detector-transform coordinates onto the
    # independently preprocessed CLIP image.
    width, height = image.size
    subject_boxes = target["boxes_h"].to(device)
    object_boxes = target["boxes_o"].to(device)
    unions = build_union_boxes(
        subject_boxes,
        object_boxes,
        height,
        width,
    )

    all_logits = []
    for start in range(0, len(predicates), clip_batch_size):
        end = min(start + clip_batch_size, len(predicates))
        crops = torch.stack(
            [
                preprocess_clip_crop(
                    image,
                    square_crop_box(box, width, height),
                )
                for box in unions[start:end].cpu()
            ]
        ).to(device)
        image_features = clip_model.encode_image(crops)
        if isinstance(image_features, (tuple, list)):
            image_features = image_features[0]
        image_features = F.normalize(image_features.float(), dim=-1)

        if text_mode == "literal_cache":
            text_features = literal_cache[
                subjects[start:end].cpu(),
                :,
                objects[start:end].cpu(),
            ].to(device=device, dtype=torch.float32)
        elif text_mode == "lain_sentence":
            object_names = cache_metadata["objects"]
            predicate_names = cache_metadata["predicates"]
            names = []
            subject_indices = subjects[start:end].detach().cpu().tolist()
            object_indices = objects[start:end].detach().cpu().tolist()
            for subject_index, object_index in zip(
                subject_indices,
                object_indices,
            ):
                subject_name = object_names[subject_index]
                object_name = object_names[object_index]
                names.extend(
                    lain_template.format(
                        subject=subject_name,
                        predicate=predicate_name,
                        object=object_name,
                    )
                    for predicate_name in predicate_names
                )
            frozen_parts = []
            for text_start in range(0, len(names), clip_batch_size):
                text_end = min(text_start + clip_batch_size, len(names))
                tokens = torch.cat(
                    [tokenize(name) for name in names[text_start:text_end]],
                    dim=0,
                ).to(device)
                frozen_parts.append(
                    F.normalize(
                        clip_model.encode_text(tokens).float(),
                        dim=-1,
                    ).cpu()
                )
            text_features = torch.cat(frozen_parts).view(
                end - start,
                50,
                -1,
            ).to(device=device, dtype=torch.float32)
        elif text_mode == "lain_predicate":
            text_features = predicate_text_features.unsqueeze(0).expand(
                end - start,
                -1,
                -1,
            ).to(device=device, dtype=torch.float32)
        elif text_mode == "lain_spo":
            object_names = cache_metadata["objects"]
            predicate_names = cache_metadata["predicates"]
            names = []
            predicate_indices = []
            subject_indices = subjects[start:end].detach().cpu().tolist()
            object_indices = objects[start:end].detach().cpu().tolist()
            for subject_index, object_index in zip(
                subject_indices,
                object_indices,
            ):
                subject_name = object_names[subject_index]
                object_name = object_names[object_index]
                for predicate_index, predicate_name in enumerate(
                    predicate_names
                ):
                    names.append(
                        lain_template.format(
                            subject=subject_name,
                            predicate=predicate_name,
                            object=object_name,
                        )
                    )
                    predicate_indices.append(predicate_index)
            flat_features = encode_lain_prompts(
                clip_model,
                names,
                predicate_indices,
                lain_context,
                clip_batch_size,
                device,
            )
            text_features = flat_features.view(
                end - start,
                50,
                -1,
            ).to(device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported text mode: {text_mode}")
        text_features = F.normalize(text_features, dim=-1)
        cosine = torch.einsum(
            "rd,rcd->rc",
            image_features,
            text_features,
        )
        all_logits.append(cosine * clip_model.logit_scale.exp().float())

    logits = torch.cat(all_logits, dim=0)
    correct = logits.gather(1, predicates[:, None]).squeeze(1)
    ranks = (logits > correct[:, None]).sum(dim=1) + 1
    masked = logits.clone()
    masked.scatter_(1, predicates[:, None], -torch.inf)
    margins = correct - masked.max(dim=1).values
    return {
        "predicates": predicates.cpu(),
        "ranks": ranks.cpu(),
        "margins": margins.cpu(),
    }


def ratio(mask):
    return float(mask.float().mean()) if len(mask) else 0.0


def subset_metrics(ranks, margins, mask):
    selected_ranks = ranks[mask]
    selected_margins = margins[mask]
    if len(selected_ranks) == 0:
        return {
            "count": 0,
            "acc@1": 0.0,
            "acc@3": 0.0,
            "acc@5": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "mrr": 0.0,
            "mean_correct_minus_best_other_logit": 0.0,
        }
    return {
        "count": int(len(selected_ranks)),
        # [GT-pair predicate classification]
        # Acc@K is the fraction of annotated GT triplets whose predicate
        # appears among frozen CLIP's K highest-scoring predicates.
        "acc@1": ratio(selected_ranks <= 1),
        "acc@3": ratio(selected_ranks <= 3),
        "acc@5": ratio(selected_ranks <= 5),
        "mean_rank": float(selected_ranks.float().mean()),
        "median_rank": float(selected_ranks.float().median()),
        "mrr": float((1.0 / selected_ranks.float()).mean()),
        "mean_correct_minus_best_other_logit": float(
            selected_margins.float().mean()
        ),
    }


def main():
    cli = parse_args()
    if cli.num_images <= 0 or cli.clip_batch_size <= 0:
        raise ValueError("Image and CLIP batch sizes must be positive")
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    run_args = load_run_args(cli.run_args, cli)
    seed = int(getattr(run_args, "seed", 66))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Use the raw VG dataset rather than DataFactory so the GT boxes and PIL
    # pixels remain in exactly the same coordinate system.
    dataset = VGDataset(
        root=run_args.data_root,
        split="test",
        num_relations=50,
        args=run_args,
    )

    device = torch.device(cli.device)
    clip_model = torch.jit.load(
        str(Path(cli.clip_checkpoint).resolve()),
        map_location=device,
    ).eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)

    literal_cache, metadata = load_literal_cache(cli.literal_cache)
    if len(metadata.get("objects", [])) != 150:
        raise ValueError("Literal-cache metadata must contain 150 objects")
    if len(metadata.get("predicates", [])) != 50:
        raise ValueError("Literal-cache metadata must contain 50 predicates")

    lain_context = None
    context_key = None
    context_epoch = None
    predicate_text_features = None
    required_fields = {"subject", "predicate", "object"}
    if not all("{" + field + "}" in cli.lain_template for field in required_fields):
        raise ValueError(
            "--lain-template must contain {subject}, {predicate}, and {object}"
        )

    if cli.text_mode in {"lain_predicate", "lain_spo"}:
        lain_context, context_key, context_epoch = load_lain_context(
            cli.prompt_checkpoint
        )
        if cli.text_mode == "lain_predicate":
            predicate_names = metadata["predicates"]
            predicate_text_features = encode_lain_prompts(
                clip_model,
                predicate_names,
                list(range(50)),
                lain_context,
                cli.clip_batch_size,
                device,
            )
    base = torch.as_tensor(
        dataset.base_predicate_indices,
        dtype=torch.long,
    )
    novel = torch.as_tensor(
        dataset.novel_predicate_indices,
        dtype=torch.long,
    )

    predicate_parts = []
    rank_parts = []
    margin_parts = []
    limit = min(cli.num_images, len(dataset))

    for index in tqdm(range(limit), total=limit):
        (image, target), _ = dataset[index]
        result = rank_image_relations(
            clip_model,
            literal_cache,
            metadata,
            cli.text_mode,
            lain_context,
            predicate_text_features,
            cli.lain_template,
            image,
            target,
            cli.clip_batch_size,
            device,
        )
        if result is None:
            continue
        predicate_parts.append(result["predicates"])
        rank_parts.append(result["ranks"])
        margin_parts.append(result["margins"])

    predicates = torch.cat(predicate_parts)
    ranks = torch.cat(rank_parts)
    margins = torch.cat(margin_parts)
    base_mask = torch.isin(predicates, base)
    novel_mask = torch.isin(predicates, novel)
    if not torch.all(base_mask | novel_mask):
        raise ValueError("GT predicate lies outside the OvR partition")

    summary = {
        "method": f"clip_gt_pair_{cli.text_mode}",
        "metric": "predicate_classification_acc_at_k",
        "metric_k": [1, 3, 5],
        "metric_definition": (
            "fraction of GT triplets whose annotated predicate is within "
            "the top-k of 50 frozen-CLIP predicate candidates"
        ),
        "trained": cli.text_mode in {"lain_predicate", "lain_spo"},
        "clip_parameters_frozen": True,
        "text_mode": cli.text_mode,
        "lain_template": cli.lain_template,
        "prompt_checkpoint": (
            str(Path(cli.prompt_checkpoint).resolve())
            if cli.prompt_checkpoint
            else None
        ),
        "prompt_context_key": context_key,
        "prompt_checkpoint_epoch": context_epoch,
        "prompt_context_shape": (
            list(lain_context.shape) if lain_context is not None else None
        ),
        "uses_detector": False,
        "crop_source": "raw_vg_pil_gt_union_square",
        "images": limit,
        "vg_ovsgtr_protocol": bool(cli.vg_ovsgtr_protocol),
        "literal_cache": str(Path(cli.literal_cache).resolve()),
        "literal_template": metadata.get("template"),
        "all": subset_metrics(ranks, margins, torch.ones_like(base_mask)),
        "base": subset_metrics(ranks, margins, base_mask),
        "novel": subset_metrics(ranks, margins, novel_mask),
    }

    output_path = Path(cli.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n===== FROZEN CLIP VG GT-PAIR PREDICATE ORACLE =====")
    print("Text mode:", cli.text_mode)
    print("CLIP parameters: frozen")
    if cli.text_mode in {"lain_predicate", "lain_spo"}:
        print("Learned LAIN context:", cli.prompt_checkpoint)
    print(
        "GT protocol:",
        (
            "official OvSGTR entity-merged"
            if cli.vg_ovsgtr_protocol
            else "legacy LAIN OvR"
        ),
    )
    for name in ("all", "base", "novel"):
        metric = summary[name]
        print(
            f"{name}: count={metric['count']}, "
            f"Acc@1/3/5="
            f"{metric['acc@1'] * 100:.2f}/"
            f"{metric['acc@3'] * 100:.2f}/"
            f"{metric['acc@5'] * 100:.2f}, "
            f"median-rank={metric['median_rank']:.1f}, "
            f"MRR={metric['mrr']:.4f}"
        )
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
