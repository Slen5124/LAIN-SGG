"""Compare the original TorchScript CLIP text path with LAIN's rebuilt CLIP.

This diagnostic is read-only.  It uses identical tokens for both encoders and
reports whether prototype collapse originates in tokenization, the original
checkpoint, or LAIN's ``build_model`` reconstruction path.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CLIP.clip import build_model
from CLIP.customCLIP import tokenize
from utils.vg_list import VG150_PREDICATES, get_vg_object_names


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit LAIN and TorchScript CLIP text encoding paths",
    )
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--data-root", default="vg")
    parser.add_argument("--subject", default="man")
    parser.add_argument("--object", dest="object_name", default="horse")
    parser.add_argument(
        "--template",
        default="a photo of {subject} {predicate} {object}",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def stats(values):
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def off_diagonal_stats(features):
    similarity = features @ features.T
    mask = ~torch.eye(len(features), dtype=torch.bool)
    return stats(similarity[mask])


def encode(model, tokens, device):
    model = model.to(device).eval()
    with torch.inference_mode():
        output = model.encode_text(tokens.to(device)).float()
    raw_norms = output.norm(dim=-1).cpu()
    normalized = F.normalize(output, dim=-1).cpu()
    return normalized, raw_norms


def unique_row_count(tensor):
    return int(torch.unique(tensor.cpu(), dim=0).shape[0])


def main():
    cli = parse_args()
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    checkpoint_path = Path(cli.clip_checkpoint).resolve()
    object_names = list(get_vg_object_names(cli.data_root))
    if cli.subject not in object_names:
        raise ValueError(f"Unknown VG subject name: {cli.subject!r}")
    if cli.object_name not in object_names:
        raise ValueError(f"Unknown VG object name: {cli.object_name!r}")

    predicates = list(VG150_PREDICATES)
    forward_prompts = [
        cli.template.format(
            subject=cli.subject,
            predicate=predicate,
            object=cli.object_name,
        )
        for predicate in predicates
    ]
    reverse_prompts = [
        cli.template.format(
            subject=cli.object_name,
            predicate=predicate,
            object=cli.subject,
        )
        for predicate in predicates
    ]
    prompts = forward_prompts + reverse_prompts
    tokens = torch.cat([tokenize(prompt) for prompt in prompts], dim=0)

    # EOT is the largest token id in OpenAI CLIP's vocabulary.
    eot_positions = tokens.argmax(dim=-1)
    forward_tokens = tokens[:len(predicates)]
    reverse_tokens = tokens[len(predicates):]
    changed_token_counts = (forward_tokens != reverse_tokens).sum(dim=-1)

    # Path A: execute the original serialized TorchScript model directly.
    original_model = torch.jit.load(
        str(checkpoint_path),
        map_location="cpu",
    )
    original_features, original_norms = encode(
        original_model,
        tokens,
        cli.device,
    )
    del original_model
    if cli.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Path B: reproduce the path used by LAIN after extracting state_dict.
    checkpoint_model = torch.jit.load(
        str(checkpoint_path),
        map_location="cpu",
    )
    state_dict = checkpoint_model.state_dict()
    del checkpoint_model
    rebuilt_model = build_model(
        state_dict,
        use_adapter=False,
        adapter_pos="all",
        args=None,
    )
    rebuilt_features, rebuilt_norms = encode(
        rebuilt_model,
        tokens,
        cli.device,
    )

    count = len(predicates)
    original_forward = original_features[:count]
    original_reverse = original_features[count:]
    rebuilt_forward = rebuilt_features[:count]
    rebuilt_reverse = rebuilt_features[count:]

    original_direction_cosine = (
        original_forward * original_reverse
    ).sum(dim=-1)
    rebuilt_direction_cosine = (
        rebuilt_forward * rebuilt_reverse
    ).sum(dim=-1)
    cross_path_cosine = (
        original_features * rebuilt_features
    ).sum(dim=-1)
    cross_path_max_abs_error = (
        original_features - rebuilt_features
    ).abs().amax(dim=-1)

    result = {
        "clip_checkpoint": str(checkpoint_path),
        "template": cli.template,
        "subject": cli.subject,
        "object": cli.object_name,
        "num_predicates": count,
        "token_shape": list(tokens.shape),
        "unique_token_rows_all": unique_row_count(tokens),
        "unique_token_rows_forward": unique_row_count(forward_tokens),
        "unique_token_rows_reverse": unique_row_count(reverse_tokens),
        "forward_reverse_changed_token_count": stats(changed_token_counts),
        "eot_position": stats(eot_positions),
        "original_raw_feature_norm": stats(original_norms),
        "rebuilt_raw_feature_norm": stats(rebuilt_norms),
        "original_within_pair_predicate_cosine": off_diagonal_stats(
            original_forward
        ),
        "rebuilt_within_pair_predicate_cosine": off_diagonal_stats(
            rebuilt_forward
        ),
        "original_forward_reverse_cosine": stats(
            original_direction_cosine
        ),
        "rebuilt_forward_reverse_cosine": stats(
            rebuilt_direction_cosine
        ),
        "original_vs_rebuilt_cosine": stats(cross_path_cosine),
        "original_vs_rebuilt_max_abs_error": stats(
            cross_path_max_abs_error
        ),
        "sample_token_rows": [
            {
                "predicate": predicates[index],
                "prompt": forward_prompts[index],
                "nonzero_tokens": tokens[index][
                    tokens[index] != 0
                ].tolist(),
                "eot_position": int(eot_positions[index]),
            }
            for index in range(min(5, count))
        ],
    }

    print("\n===== VG CLIP TEXT PATH AUDIT =====")
    print("checkpoint:", result["clip_checkpoint"])
    print("template:", result["template"])
    print("pair:", f"{cli.subject} -> {cli.object_name}")
    print("unique forward token rows:", result["unique_token_rows_forward"], "/", count)
    print("unique reverse token rows:", result["unique_token_rows_reverse"], "/", count)
    print("forward/reverse changed tokens:", result["forward_reverse_changed_token_count"])
    print("EOT positions:", result["eot_position"])
    print("original raw norms:", result["original_raw_feature_norm"])
    print("rebuilt raw norms:", result["rebuilt_raw_feature_norm"])
    print(
        "original predicate cosine:",
        result["original_within_pair_predicate_cosine"],
    )
    print(
        "rebuilt predicate cosine:",
        result["rebuilt_within_pair_predicate_cosine"],
    )
    print(
        "original direction cosine:",
        result["original_forward_reverse_cosine"],
    )
    print(
        "rebuilt direction cosine:",
        result["rebuilt_forward_reverse_cosine"],
    )
    print(
        "original/rebuilt matched cosine:",
        result["original_vs_rebuilt_cosine"],
    )
    print(
        "original/rebuilt max abs error:",
        result["original_vs_rebuilt_max_abs_error"],
    )

    if cli.output:
        output_path = Path(cli.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print("saved:", output_path.resolve())

    print("VG CLIP text path audit: OK")


if __name__ == "__main__":
    main()
