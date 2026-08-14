"""Read-only integrity audit for the frozen VG literal S-P-O cache.

This script verifies the cache independently of LAIN training.  It checks the
stored tensor/metadata, reproduces sampled entries with the frozen CLIP text
encoder, verifies the [subject, predicate, object] lookup contract, and
measures how well predicates are separated inside a fixed subject-object pair.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow direct execution as ``python utils/vg_literal_cache_audit.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CLIP.clip import build_model
from CLIP.customCLIP import tokenize
from models.LAIN import LAIN
from utils.vg_list import VG150_PREDICATES, get_vg_object_names


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit a frozen VG literal S-P-O CLIP cache",
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--data-root", default="vg")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def require_metadata_list(metadata, *names):
    for name in names:
        value = metadata.get(name)
        if value is not None:
            return list(value), name
    raise ValueError(
        "Cache metadata is missing all accepted keys: "
        + ", ".join(names)
    )


def load_clip(checkpoint_path, device):
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = payload.state_dict() if hasattr(payload, "state_dict") else payload
    if not isinstance(state_dict, dict):
        raise TypeError("CLIP checkpoint did not contain a state dictionary")

    # [Cache audit]
    # The literal cache was produced by the frozen CLIP text encoder.  Visual
    # adapters are irrelevant here and must not alter the text-side reference.
    model = build_model(
        state_dict,
        use_adapter=False,
        adapter_pos="all",
        args=None,
    )
    model = model.to(device).eval()
    return model


def encode_prompts(model, prompts, batch_size, device):
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            tokens = torch.cat(
                [tokenize(prompt) for prompt in prompts[start:start + batch_size]],
                dim=0,
            ).to(device)
            features = model.encode_text(tokens).float()
            encoded.append(F.normalize(features, dim=-1).cpu())
    return torch.cat(encoded, dim=0)


def tensor_stats(values):
    values = torch.as_tensor(values, dtype=torch.float32)
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def main():
    cli = parse_args()
    if cli.num_samples <= 0 or cli.num_pairs <= 0:
        raise ValueError("--num-samples and --num-pairs must be positive")
    if cli.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    cache_path = Path(cli.cache).resolve()
    clip_path = Path(cli.clip_checkpoint).resolve()
    payload = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise TypeError("Literal cache payload must be a dictionary")
    if "text_features" not in payload or "metadata" not in payload:
        raise ValueError("Literal cache requires text_features and metadata")

    features = payload["text_features"]
    metadata = payload["metadata"]
    if not isinstance(features, torch.Tensor):
        raise TypeError("text_features must be a tensor")
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")

    cache_objects, object_key = require_metadata_list(
        metadata,
        "objects",
        "object_names",
    )
    cache_predicates, predicate_key = require_metadata_list(
        metadata,
        "predicates",
        "predicate_names",
    )
    active_objects = list(get_vg_object_names(cli.data_root))
    active_predicates = list(VG150_PREDICATES)

    expected_shape = (
        len(active_objects),
        len(active_predicates),
        len(active_objects),
        512,
    )
    if tuple(features.shape) != expected_shape:
        raise ValueError(
            f"Cache shape mismatch: expected={expected_shape}, "
            f"actual={tuple(features.shape)}"
        )
    if features.dtype != torch.float16:
        raise ValueError(f"Cache dtype must be float16, got {features.dtype}")
    if not torch.isfinite(features).all():
        raise ValueError("Cache contains NaN or infinity")
    if cache_objects != active_objects:
        raise ValueError("Cache object vocabulary/order differs from active VG")
    if cache_predicates != active_predicates:
        raise ValueError("Cache predicate vocabulary/order differs from active VG")

    template = metadata.get("template")
    if not isinstance(template, str):
        raise ValueError("Cache metadata does not contain a string template")
    try:
        template.format(
            subject=active_objects[0],
            predicate=active_predicates[0],
            object=active_objects[1],
        )
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError(f"Invalid literal template: {template!r}") from error

    # Process the 1.1 GB cache in subject chunks.  Converting the entire cache
    # to FP32 at once would otherwise create an unnecessary multi-GB copy.
    norm_chunks = []
    for start in range(0, len(active_objects), 10):
        norm_chunks.append(
            features[start:start + 10].float().norm(dim=-1).reshape(-1)
        )
    norms = torch.cat(norm_chunks)

    rng = random.Random(cli.seed)
    triples = []
    for _ in range(cli.num_samples):
        subject = rng.randrange(len(active_objects))
        predicate = rng.randrange(len(active_predicates))
        object_index = rng.randrange(len(active_objects) - 1)
        if object_index >= subject:
            object_index += 1
        triples.append((subject, predicate, object_index))

    prompts = [
        template.format(
            subject=active_objects[subject],
            predicate=active_predicates[predicate],
            object=active_objects[object_index],
        )
        for subject, predicate, object_index in triples
    ]
    clip_model = load_clip(clip_path, cli.device)
    live = encode_prompts(
        clip_model,
        prompts,
        cli.batch_size,
        cli.device,
    )

    cached = torch.stack([
        F.normalize(
            features[subject, predicate, object_index].float(),
            dim=-1,
        )
        for subject, predicate, object_index in triples
    ])
    correct_cosine = (live * cached).sum(dim=-1)
    maximum_absolute_error = (live - cached).abs().amax(dim=-1)

    # [Cache axis contract]
    # Verify LAIN's actual lookup helper, rather than only indexing the tensor
    # independently inside this audit.
    shell = LAIN.__new__(LAIN)
    torch.nn.Module.__init__(shell)
    shell.dataset = "vg"
    shell.vg_text_mode = "literal_cache"
    shell.register_buffer(
        "vg_literal_text_features",
        features,
        persistent=False,
    )
    subject_indices = torch.tensor([item[0] for item in triples])
    object_indices = torch.tensor([item[2] for item in triples])
    lookup = shell.lookup_vg_literal_text_features(
        subject_indices,
        object_indices,
    )
    expected_lookup = features[subject_indices, :, object_indices, :].float()
    lookup_exact = bool(torch.equal(lookup, expected_lookup))
    if not lookup_exact:
        raise AssertionError("LAIN literal-cache lookup uses an unexpected axis order")

    ranks = []
    correct_minus_best_other = []
    correct_minus_swapped = []
    for row, (subject, predicate, object_index) in enumerate(triples):
        pair_table = F.normalize(
            features[subject, :, object_index].float(),
            dim=-1,
        )
        similarity = pair_table @ live[row]
        target = similarity[predicate]
        ranks.append(int((similarity > target).sum().item()) + 1)
        other = similarity.clone()
        other[predicate] = -torch.inf
        correct_minus_best_other.append(float(target - other.max()))
        swapped = F.normalize(
            features[object_index, predicate, subject].float(),
            dim=-1,
        )
        correct_minus_swapped.append(
            float(correct_cosine[row] - torch.dot(live[row], swapped))
        )

    pair_geometries = []
    for _ in range(cli.num_pairs):
        subject = rng.randrange(len(active_objects))
        object_index = rng.randrange(len(active_objects) - 1)
        if object_index >= subject:
            object_index += 1
        table = F.normalize(
            features[subject, :, object_index].float(),
            dim=-1,
        )
        similarity = table @ table.T
        mask = ~torch.eye(len(active_predicates), dtype=torch.bool)
        pair_geometries.append(similarity[mask])
    off_diagonal = torch.cat(pair_geometries)

    result = {
        "cache": str(cache_path),
        "clip_checkpoint": str(clip_path),
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "template": template,
        "object_metadata_key": object_key,
        "predicate_metadata_key": predicate_key,
        "object_order_matches": True,
        "predicate_order_matches": True,
        "all_finite": True,
        "norms": tensor_stats(norms),
        "num_reencoded_samples": len(triples),
        "live_to_cache_cosine": tensor_stats(correct_cosine),
        "live_to_cache_max_abs_error": tensor_stats(maximum_absolute_error),
        "lookup_axis_order": "subject,predicate,object,embedding",
        "lain_lookup_exact": lookup_exact,
        "predicate_self_retrieval_top1_fraction": sum(rank == 1 for rank in ranks) / len(ranks),
        "predicate_self_retrieval_rank_mean": sum(ranks) / len(ranks),
        "correct_minus_best_other_cosine": tensor_stats(correct_minus_best_other),
        "correct_minus_swapped_endpoints_cosine": tensor_stats(correct_minus_swapped),
        "within_pair_predicate_cosine": tensor_stats(off_diagonal),
        "samples": [
            {
                "subject_index": subject,
                "subject": active_objects[subject],
                "predicate_index": predicate,
                "predicate": active_predicates[predicate],
                "object_index": object_index,
                "object": active_objects[object_index],
                "prompt": prompts[row],
                "live_to_cache_cosine": float(correct_cosine[row]),
                "predicate_rank_within_pair": ranks[row],
            }
            for row, (subject, predicate, object_index) in enumerate(triples[:10])
        ],
    }

    print("\n===== VG LITERAL CACHE AUDIT =====")
    print("cache:", result["cache"])
    print("shape:", tuple(result["shape"]))
    print("dtype:", result["dtype"])
    print("template:", result["template"])
    print("object order matches: YES")
    print("predicate order matches: YES")
    print("LAIN lookup axis order: OK")
    print("norm range:", result["norms"]["minimum"], result["norms"]["maximum"])
    print("live/cache cosine:", result["live_to_cache_cosine"])
    print("live/cache max abs error:", result["live_to_cache_max_abs_error"])
    print(
        "predicate self-retrieval top-1 fraction:",
        result["predicate_self_retrieval_top1_fraction"],
    )
    print(
        "correct - best other predicate cosine:",
        result["correct_minus_best_other_cosine"],
    )
    print(
        "correct - swapped endpoints cosine:",
        result["correct_minus_swapped_endpoints_cosine"],
    )
    print(
        "within-pair predicate cosine:",
        result["within_pair_predicate_cosine"],
    )

    if cli.output:
        output_path = Path(cli.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print("saved:", output_path.resolve())

    assert result["live_to_cache_cosine"]["minimum"] > 0.999
    assert result["predicate_self_retrieval_top1_fraction"] > 0.99
    print("VG literal cache integrity: OK")


if __name__ == "__main__":
    main()
