
## 2026-08-04 VG 로더 단독 검증 통과
- datasets/vg.py 작성, VGDataset 단독 테스트 성공
- train split: 62723 images (손상4개 제외, 관계있는 이미지만)
- 샘플0 (VG_100K_2/1.jpg): man-wears->sneaker, sidewalk-on->building, man-has->shirt
  → h5 원본과 정확히 일치
- 인덱스 변환 검증: verb 0~49, object 0~149 (1-based→0-based -1 정확)
- 박스 변환 검증: xcywh_1024 → xyxy, 이미지 안(0~800), sneaker가 man 발 위치
- target 형식: HICODet과 동일 키 (boxes_h/o, verb, object, hoi)
- 결론: 로더 레벨 완전 검증. 이후 에러는 모델/DataFactory 문제로 좁혀짐
