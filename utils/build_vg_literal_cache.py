"""Build a frozen CLIP cache for all VG150 literal S-P-O triplets.

The cache contains 150 subject classes x 50 predicates x 150 object classes.
Each literal prompt is encoded exactly once by the frozen CLIP text encoder,
L2-normalized, converted to FP16, and stored under /mnt/sdb by the caller.

This is preprocessing only. It does not alter LAIN, a training checkpoint, or
the VG annotations.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from CLIP.clip import build_model
from CLIP.customCLIP import tokenize
from utils.vg_list import VG150_PREDICATES, get_vg_object_names


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--template",
        default="a photo of {subject} {predicate} {object}",
        help=(
            "Literal prompt template. Required fields are {subject}, "
            "{predicate}, and {object}."
        ),
    )
    return parser.parse_args()


def validate_template(template):
    for field in ("{subject}", "{predicate}", "{object}"):
        if field not in template:
            raise ValueError(
                f"Literal template must contain {field}: {template}"
            )


def vocabulary_hash(object_names, predicate_names, template):
    payload = json.dumps(
        {
            "objects": list(object_names),
            "predicates": list(predicate_names),
            "template": template,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def flat_triplet_indices(flat_indices, num_predicates, num_objects):
    per_subject = num_predicates * num_objects
    subject_indices = torch.div(
        flat_indices,
        per_subject,
        rounding_mode="floor",
    )
    remainder = flat_indices.remainder(per_subject)
    predicate_indices = torch.div(
        remainder,
        num_objects,
        rounding_mode="floor",
    )
    object_indices = remainder.remainder(num_objects)
    return subject_indices, predicate_indices, object_indices


def main():
    args = parse_args()
    validate_template(args.template)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build the literal cache")

    object_names = list(get_vg_object_names(args.data_root))
    predicate_names = list(VG150_PREDICATES)
    num_objects = len(object_names)
    num_predicates = len(predicate_names)

    if num_objects != 150 or num_predicates != 50:
        raise ValueError(
            "Unexpected VG vocabulary sizes: "
            f"objects={num_objects}, predicates={num_predicates}"
        )

    checkpoint_path = Path(args.clip_checkpoint).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing cache: "
            f"{output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Visible GPU count:", torch.cuda.device_count())
    print("Visible GPU:", torch.cuda.get_device_name(0))
    print("Objects:", num_objects)
    print("Predicates:", num_predicates)
    print("Triplets:", num_objects * num_predicates * num_objects)
    print("Template:", args.template)

    clip_state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    ).state_dict()
    model = build_model(
        state_dict=clip_state,
        use_adapter=False,
        adapter_pos="all",
        args=None,
    ).cuda().eval()

    total_triplets = num_objects * num_predicates * num_objects
    output_dim = int(model.text_projection.shape[1])
    # [Frozen literal cache]
    # Allocate on CPU so only the active prompt batch occupies GPU memory.
    # FP16 requires about 1.07 GiB for VG150 with 512-dimensional CLIP text.
    flat_cache = torch.empty(
        total_triplets,
        output_dim,
        dtype=torch.float16,
        device="cpu",
    )

    started = time.time()
    with torch.inference_mode():
        for start in tqdm(
            range(0, total_triplets, args.batch_size),
            desc="Encoding literal S-P-O prompts",
        ):
            end = min(start + args.batch_size, total_triplets)
            flat_indices = torch.arange(start, end, dtype=torch.long)
            subject_indices, predicate_indices, object_indices = (
                flat_triplet_indices(
                    flat_indices,
                    num_predicates,
                    num_objects,
                )
            )

            prompts = [
                args.template.format(
                    subject=object_names[int(subject_index)],
                    predicate=predicate_names[int(predicate_index)],
                    object=object_names[int(object_index)],
                )
                for subject_index, predicate_index, object_index in zip(
                    subject_indices,
                    predicate_indices,
                    object_indices,
                )
            ]
            tokens = tokenize(prompts).cuda(non_blocking=False)
            embeddings = model.encode_text(tokens)
            embeddings = F.normalize(
                embeddings.float(),
                dim=-1,
                eps=1e-6,
            )
            flat_cache[start:end].copy_(embeddings.to("cpu", torch.float16))

    cache = flat_cache.view(
        num_objects,
        num_predicates,
        num_objects,
        output_dim,
    )
    metadata = {
        "format_version": 1,
        "shape_order": [
            "subject_class",
            "predicate_class",
            "object_class",
            "embedding",
        ],
        "shape": list(cache.shape),
        "dtype": "float16",
        "normalized": True,
        "template": args.template,
        "objects": object_names,
        "predicates": predicate_names,
        "vocabulary_sha256": vocabulary_hash(
            object_names,
            predicate_names,
            args.template,
        ),
        "clip_checkpoint": str(checkpoint_path),
    }

    torch.save(
        {
            "text_features": cache,
            "metadata": metadata,
        },
        output_path,
    )

    elapsed = time.time() - started
    print("Cache shape:", tuple(cache.shape))
    print("Cache dtype:", cache.dtype)
    print(
        "Norm range:",
        float(cache.float().norm(dim=-1).min()),
        float(cache.float().norm(dim=-1).max()),
    )
    print("Elapsed seconds:", round(elapsed, 1))
    print("Saved:", output_path)
    print("VG frozen literal S-P-O cache: OK")


if __name__ == "__main__":
    main()
