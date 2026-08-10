"""Official OvSGTR open-vocabulary relation split for VG150.

The original VG150 predicate vocabulary contains 50 foreground predicates.
OvSGTR divides them into 35 base and 15 novel predicates for OvR-SGG.

This module stores predicate names rather than hard-coded indices so that the
split is validated against the active VG dictionary before training.
"""

from typing import Dict, Sequence, Tuple


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


def resolve_ovsgtr_predicate_split(
    predicate_names: Sequence[str],
) -> Dict[str, Tuple[int, ...]]:
    """Resolve the official name-based split to active 0-based indices."""

    predicate_names = tuple(predicate_names)

    if len(predicate_names) != 50:
        raise ValueError(
            "OvSGTR relation split requires exactly 50 VG predicates, "
            f"got {len(predicate_names)}."
        )

    if len(set(predicate_names)) != len(predicate_names):
        raise ValueError(
            "The active VG predicate vocabulary contains duplicate names."
        )

    name_to_index = {
        name: index
        for index, name in enumerate(predicate_names)
    }

    expected_names = (
        set(OVSGTR_BASE_PREDICATES)
        | set(OVSGTR_NOVEL_PREDICATES)
    )
    active_names = set(predicate_names)

    missing_names = sorted(expected_names - active_names)
    unexpected_names = sorted(active_names - expected_names)

    if missing_names or unexpected_names:
        raise ValueError(
            "The active VG predicate vocabulary does not match OvSGTR. "
            f"Missing={missing_names}, "
            f"unexpected={unexpected_names}."
        )

    overlap = (
        set(OVSGTR_BASE_PREDICATES)
        & set(OVSGTR_NOVEL_PREDICATES)
    )
    if overlap:
        raise ValueError(
            "OvSGTR base and novel predicates overlap: "
            f"{sorted(overlap)}"
        )

    base_indices = tuple(
        name_to_index[name]
        for name in OVSGTR_BASE_PREDICATES
    )
    novel_indices = tuple(
        name_to_index[name]
        for name in OVSGTR_NOVEL_PREDICATES
    )

    if len(base_indices) != 35:
        raise ValueError(
            f"Expected 35 base predicates, got {len(base_indices)}."
        )

    if len(novel_indices) != 15:
        raise ValueError(
            f"Expected 15 novel predicates, got {len(novel_indices)}."
        )

    if set(base_indices) | set(novel_indices) != set(range(50)):
        raise ValueError(
            "OvSGTR base and novel indices do not partition VG50."
        )

    return {
        "base": base_indices,
        "novel": novel_indices,
    }