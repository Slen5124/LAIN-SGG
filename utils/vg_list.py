"""Visual Genome class-name and predicate-prompt utilities."""

import json
from pathlib import Path


VG150_PREDICATES = [
    'above', 'across', 'against', 'along', 'and',
    'at', 'attached to', 'behind', 'belonging to', 'between',
    'carrying', 'covered in', 'covering', 'eating', 'flying in',
    'for', 'from', 'growing on', 'hanging from', 'has',
    'holding', 'in', 'in front of', 'laying on', 'looking at',
    'lying on', 'made of', 'mounted on', 'near', 'of',
    'on', 'on back of', 'over', 'painted on', 'parked on',
    'part of', 'playing', 'riding', 'says', 'sitting on',
    'standing on', 'to', 'under', 'using', 'walking in',
    'walking on', 'watching', 'wearing', 'wears', 'with',
]

assert len(VG150_PREDICATES) == 50


vg_predicates_person = [
    f'a photo of a person is {predicate} the object'
    for predicate in VG150_PREDICATES
]

vg_predicates_something = [
    f'a photo of something {predicate} something'
    for predicate in VG150_PREDICATES
]

vg_predicates_bare = list(VG150_PREDICATES)


VG_PROMPT_FORMATS = {
    'person': vg_predicates_person,
    'something': vg_predicates_something,
    'bare': vg_predicates_bare,
}


def get_vg_predicates(prompt_format='something'):
    """Return the 50 VG predicates using the requested prompt format."""
    if prompt_format not in VG_PROMPT_FORMATS:
        raise ValueError(
            f"Unknown VG prompt format '{prompt_format}'. "
            f"Choose from {list(VG_PROMPT_FORMATS.keys())}"
        )

    return VG_PROMPT_FORMATS[prompt_format]


def get_vg_object_names(data_root):
    """Load the canonical 150 VG object names in zero-based label order."""
    dictionary_path = (
        Path(data_root)
        / 'vg_data'
        / 'stanford_filtered'
        / 'VG-SGG-dicts.json'
    )

    if not dictionary_path.is_file():
        raise FileNotFoundError(
            'VG dictionary was not found: '
            f'{dictionary_path}'
        )

    with dictionary_path.open('r', encoding='utf-8') as handle:
        dictionary = json.load(handle)

    idx_to_label = dictionary.get('idx_to_label')
    if not isinstance(idx_to_label, dict):
        raise ValueError(
            'VG dictionary does not contain a valid idx_to_label mapping'
        )

    object_names = []
    for one_based_index in range(1, 151):
        key = str(one_based_index)
        if key not in idx_to_label:
            raise ValueError(
                'VG idx_to_label is missing object index '
                f'{one_based_index}'
            )
        object_names.append(idx_to_label[key])

    if len(set(object_names)) != 150:
        raise ValueError(
            'VG object names must contain 150 unique classes'
        )

    return object_names
