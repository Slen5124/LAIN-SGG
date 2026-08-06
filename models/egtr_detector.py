"""EGTR VG-pretrained Deformable DETR loader.

The official EGTR detector checkpoint was saved with transformers 4.18.0.
This module converts only the parameter names changed by transformers 5.14.1
and requires a strict state-dict load.

It does not implement relation pairing, predicate prediction, or SGG
post-processing. Those responsibilities remain outside the detector.
"""

import re
from pathlib import Path
from typing import Dict, Union

import torch
from torch import Tensor, nn
from transformers import (
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
)


PathLike = Union[str, Path]

_OLD_BACKBONE_PREFIX = "model.backbone.conv_encoder.model."
_NEW_BACKBONE_PREFIX = "model.backbone.model."

_EXPECTED_DROPPED_KEYS = {
    "model.backbone.model.layer1.0.downsample.1.num_batches_tracked",
    "model.backbone.model.layer2.0.downsample.1.num_batches_tracked",
    "model.backbone.model.layer3.0.downsample.1.num_batches_tracked",
    "model.backbone.model.layer4.0.downsample.1.num_batches_tracked",
}


def _convert_checkpoint_key(key: str) -> str:
    """Convert a transformers 4.18 Deformable DETR key to 5.14."""

    if key.startswith(_OLD_BACKBONE_PREFIX):
        key = (
            _NEW_BACKBONE_PREFIX
            + key[len(_OLD_BACKBONE_PREFIX):]
        )

    key = re.sub(
        r"^(model\.(?:encoder|decoder)\.layers\.\d+)\.fc1\.",
        r"\1.mlp.fc1.",
        key,
    )
    key = re.sub(
        r"^(model\.(?:encoder|decoder)\.layers\.\d+)\.fc2\.",
        r"\1.mlp.fc2.",
        key,
    )
    key = key.replace(
        ".self_attn.out_proj.",
        ".self_attn.o_proj.",
    )

    return key


def _convert_checkpoint_state_dict(
    source: Dict[str, Tensor],
    target: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Convert and validate all checkpoint tensors."""

    converted: Dict[str, Tensor] = {}
    dropped = []

    for old_key, value in source.items():
        new_key = _convert_checkpoint_key(old_key)

        if new_key not in target:
            dropped.append(new_key)
            continue

        if new_key in converted:
            raise RuntimeError(
                "Checkpoint key collision after conversion: "
                f"{new_key}"
            )

        converted[new_key] = value

    if set(dropped) != _EXPECTED_DROPPED_KEYS:
        unexpected_dropped = sorted(
            set(dropped) ^ _EXPECTED_DROPPED_KEYS
        )
        raise RuntimeError(
            "Unexpected checkpoint keys were dropped:\n"
            + "\n".join(unexpected_dropped)
        )

    missing = sorted(set(target) - set(converted))
    if missing:
        raise RuntimeError(
            "Converted checkpoint is missing model keys:\n"
            + "\n".join(missing)
        )

    shape_mismatches = []

    for key, value in converted.items():
        source_shape = tuple(value.shape)
        target_shape = tuple(target[key].shape)

        if source_shape != target_shape:
            shape_mismatches.append(
                f"{key}: checkpoint={source_shape}, "
                f"model={target_shape}"
            )

    if shape_mismatches:
        raise RuntimeError(
            "Checkpoint tensor shape mismatches:\n"
            + "\n".join(shape_mismatches)
        )

    return converted


def load_egtr_vg_detector(
    checkpoint_dir: PathLike,
) -> DeformableDetrForObjectDetection:
    """Load the official EGTR VG150 object detector strictly."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    config_path = checkpoint_dir / "config.json"
    weight_path = checkpoint_dir / "pytorch_model.bin"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"EGTR config was not found: {config_path}"
        )

    if not weight_path.is_file():
        raise FileNotFoundError(
            f"EGTR detector weights were not found: {weight_path}"
        )

    config = DeformableDetrConfig.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    )

    # Prevent an unnecessary backbone download during construction.
    # The checkpoint supplies the complete trained backbone afterward.
    if hasattr(config, "use_pretrained_backbone"):
        config.use_pretrained_backbone = False

    expected_config = {
        "num_labels": 150,
        "num_queries": 200,
        "d_model": 256,
        "encoder_layers": 6,
        "decoder_layers": 6,
        "num_feature_levels": 4,
        "with_box_refine": False,
        "two_stage": False,
    }

    for name, expected in expected_config.items():
        actual = getattr(config, name, None)

        if actual != expected:
            raise ValueError(
                "Unexpected EGTR detector configuration: "
                f"{name}={actual!r}, expected {expected!r}"
            )

    model = DeformableDetrForObjectDetection(config)
    target_state = model.state_dict()

    source_state = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(source_state, dict):
        raise TypeError(
            "Expected pytorch_model.bin to contain a state dict, "
            f"but found {type(source_state).__name__}"
        )

    converted_state = _convert_checkpoint_state_dict(
        source=source_state,
        target=target_state,
    )

    load_result = model.load_state_dict(
        converted_state,
        strict=True,
    )

    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Strict EGTR load produced incompatible keys: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )

    return model


class EgtrPostProcess(nn.Module):
    """Convert raw EGTR detector outputs to LAIN proposal format."""

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, Tensor],
        target_sizes: Tensor,
    ):
        logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        features = outputs["feats"]

        if logits.ndim != 3:
            raise ValueError(
                f"Expected logits [B, Q, C], got {tuple(logits.shape)}"
            )

        if pred_boxes.shape[:2] != logits.shape[:2]:
            raise ValueError(
                "Box/query dimensions do not match logits: "
                f"logits={tuple(logits.shape)}, "
                f"boxes={tuple(pred_boxes.shape)}"
            )

        if features.shape[:2] != logits.shape[:2]:
            raise ValueError(
                "Feature/query dimensions do not match logits: "
                f"logits={tuple(logits.shape)}, "
                f"features={tuple(features.shape)}"
            )

        if target_sizes.ndim != 2 or target_sizes.shape[1] != 2:
            raise ValueError(
                "target_sizes must have shape [B, 2], "
                f"got {tuple(target_sizes.shape)}"
            )

        # EGTR uses 150 sigmoid object logits without a separate
        # softmax background column.
        probabilities = logits.sigmoid()
        scores, labels = probabilities.max(dim=-1)

        center_x, center_y, width, height = pred_boxes.unbind(-1)

        boxes = torch.stack(
            [
                center_x - 0.5 * width,
                center_y - 0.5 * height,
                center_x + 0.5 * width,
                center_y + 0.5 * height,
            ],
            dim=-1,
        )

        image_height, image_width = target_sizes.unbind(1)
        scale = torch.stack(
            [
                image_width,
                image_height,
                image_width,
                image_height,
            ],
            dim=1,
        )

        boxes = boxes * scale[:, None, :]

        return [
            {
                "scores": image_scores,
                "labels": image_labels,
                "boxes": image_boxes,
                "feats": image_features,
            }
            for (
                image_scores,
                image_labels,
                image_boxes,
                image_features,
            ) in zip(scores, labels, boxes, features)
        ]