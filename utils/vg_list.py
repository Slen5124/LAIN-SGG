"""
VG150 predicate prompt lists for LAIN (SGG task)
새롭게 추가한 코드

VG150 술어 50개를 CLIP classnames 형식으로.
인덱스 순서 = VG-SGG-dicts.json의 idx_to_predicate 1~50 (즉 0-based로 0~49).
→ vg.py에서 verb = predicate_idx - 1 로 변환한 것과 정확히 대응.

세 가지 프롬프트 형식 (프롬프트 ablation 축, 문서 v2 결정6):
    - vg_predicates_person:    HOI 원본 틀 유지 (baseline, subject=person 고정)
    - vg_predicates_something: SGG 태스크 정의 (subject 중립)
    - vg_predicates_bare:      술어 단독 (최소)

주의:
    - VG 술어 원본을 그대로 사용 (of/and/says 등 어색한 것도 손대지 않음).
      술어를 다듬는 것은 데이터 개입이 되므로 원본 유지.
    - person 형식은 전치사 술어(on, above 등)에서 "is on the object"처럼
      문법이 어색하지만, hico 원본 틀과의 일관성을 위해 그대로 둠.
"""

# VG150 술어 50개 (idx_to_predicate 1~50 순서 = verb 0~49)
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

assert len(VG150_PREDICATES) == 50, f'술어 개수 오류: {len(VG150_PREDICATES)}'


# --- 프롬프트 형식 3종 (ablation 축) ---

# P-person: HOI 원본 틀 유지 (baseline)
#   hico 형식 'a photo of a person is {verb} the object' 를 술어만 교체
vg_predicates_person = [
    f'a photo of a person is {p} the object' for p in VG150_PREDICATES
]

# P-something: SGG 태스크 정의 (subject 중립)
vg_predicates_something = [
    f'a photo of something {p} something' for p in VG150_PREDICATES
]

# P-bare: 술어 단독 (최소)
vg_predicates_bare = [p for p in VG150_PREDICATES]


# --- 형식 선택 헬퍼 (build_detector에서 사용) ---

VG_PROMPT_FORMATS = {
    'person': vg_predicates_person,
    'something': vg_predicates_something,
    'bare': vg_predicates_bare,
}


def get_vg_predicates(prompt_format='something'):
    """
    프롬프트 형식 이름으로 술어 리스트 반환.
    prompt_format: 'person' | 'something' | 'bare' (default 'something')
    """
    if prompt_format not in VG_PROMPT_FORMATS:
        raise ValueError(
            f"Unknown VG prompt format '{prompt_format}'. "
            f"Choose from {list(VG_PROMPT_FORMATS.keys())}"
        )
    return VG_PROMPT_FORMATS[prompt_format]