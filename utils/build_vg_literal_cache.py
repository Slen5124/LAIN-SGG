"""Build a frozen VG150 literal S-P-O cache with the active CLIP code.

The flattened generation order is subject-major, then predicate, then object.
The saved tensor therefore has the explicit layout
``[subject, predicate, object, embedding]``.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CLIP.clip import build_model
from CLIP.customCLIP import tokenize
from utils.vg_list import VG150_PREDICATES, get_vg_object_names


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the frozen VG150 literal S-P-O CLIP cache",
    )
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--data-root", default="vg")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--template",
        default="a photo of {subject} {predicate} {object}",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def vocabulary_hash(objects, predicates, template):
    canonical = json.dumps(
        {
            "objects": objects,
            "predicates": predicates,
            "template": template,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def main():
    cli = parse_args()
    if cli.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    checkpoint_path = Path(cli.clip_checkpoint).resolve()
    output_path = Path(cli.output).resolve()
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.exists() and not cli.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Use a new path or pass --overwrite."
        )

    objects = list(get_vg_object_names(cli.data_root))
    predicates = list(VG150_PREDICATES)
    num_objects = len(objects)
    num_predicates = len(predicates)
    if num_objects != 150 or num_predicates != 50:
        raise ValueError(
            "Unexpected VG vocabulary sizes: "
            f"objects={num_objects}, predicates={num_predicates}"
        )

    print("Visible GPU count:", torch.cuda.device_count())
    if cli.device.startswith("cuda"):
        print("Visible GPU:", torch.cuda.get_device_name(0))
    print("Objects:", num_objects)
    print("Predicates:", num_predicates)
    print("Triplets:", num_objects * num_predicates * num_objects)
    print("Template:", cli.template)

    checkpoint_model = torch.jit.load(
        str(checkpoint_path),
        map_location="cpu",
    )
    state_dict = checkpoint_model.state_dict()
    del checkpoint_model
    clip_model = build_model(
        state_dict,
        use_adapter=False,
        adapter_pos="all",
        args=None,
    ).to(cli.device).eval()

    embedding_dim = int(clip_model.text_projection.shape[1])
    expected_embedding_dim = 512
    if embedding_dim != expected_embedding_dim:
        raise ValueError(
            "Unexpected CLIP text embedding dimension: "
            f"expected={expected_embedding_dim}, actual={embedding_dim}"
        )

    total = num_objects * num_predicates * num_objects
    flat_cache = torch.empty(
        (total, embedding_dim),
        dtype=torch.float16,
        device="cpu",
    )

    with torch.inference_mode():
        progress = tqdm(
            range(0, total, cli.batch_size),
            desc="Encoding literal S-P-O prompts",
        )
        for start in progress:
            end = min(start + cli.batch_size, total)
            prompts = []
            for flat_index in range(start, end):
                subject_index = flat_index // (
                    num_predicates * num_objects
                )
                remainder = flat_index % (
                    num_predicates * num_objects
                )
                predicate_index = remainder // num_objects
                object_index = remainder % num_objects
                prompts.append(
                    cli.template.format(
                        subject=objects[subject_index],
                        predicate=predicates[predicate_index],
                        object=objects[object_index],
                    )
                )

            tokens = torch.cat(
                [tokenize(prompt) for prompt in prompts],
                dim=0,
            ).to(cli.device)
            text_features = clip_model.encode_text(tokens).float()
            text_features = F.normalize(text_features, dim=-1)
            flat_cache[start:end].copy_(
                text_features.to(dtype=torch.float16, device="cpu")
            )

    cache = flat_cache.view(
        num_objects,
        num_predicates,
        num_objects,
        embedding_dim,
    )
    if not torch.isfinite(cache).all():
        raise ValueError("Generated cache contains NaN or infinity")

    norm_min = float("inf")
    norm_max = float("-inf")
    for start in range(0, num_objects, 10):
        norms = cache[start:start + 10].float().norm(dim=-1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))

    metadata = {
        "format_version": 2,
        "shape_order": [
            "subject",
            "predicate",
            "object",
            "embedding",
        ],
        "objects": objects,
        "predicates": predicates,
        "template": cli.template,
        "normalized": True,
        "dtype": "float16",
        "embedding_dim": embedding_dim,
        "clip_checkpoint": str(checkpoint_path),
        "causal_text_attention": True,
        "vocabulary_sha256": vocabulary_hash(
            objects,
            predicates,
            cli.template,
        ),
    }
    payload = {
        "text_features": cache,
        "metadata": metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temporary_path.exists():
        temporary_path.unlink()
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)

    print("Cache shape:", tuple(cache.shape))
    print("Cache dtype:", cache.dtype)
    print("Norm range:", norm_min, norm_max)
    print("Vocabulary hash:", metadata["vocabulary_sha256"])
    print("Saved:", output_path)
    print("VG causal-fixed literal S-P-O cache: OK")


if __name__ == "__main__":
    main()
