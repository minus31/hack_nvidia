# 08. NeMo Curator — Track C

> 대규모 데이터셋을 큐레이션하여 LLM 학습에 최적화된 고품질 데이터를 준비하는 NVIDIA 프레임워크

## Track C 파이프라인에서의 위치
```
NeMo Data Designer (합성 데이터 생성)
    ↓ LLM Judge 기반 1차 필터링
NeMo Curator (후처리)   ← 여기
├── 중복 제거 (정확/퍼지)
├── 품질 필터링
└── PII 삭제
    ↓
최종 학습용 데이터셋
```

## 주요 기능
| 기능 | 설명 |
|------|------|
| 정확 중복 제거 | 해시 기반으로 완전히 동일한 문서 제거 |
| 퍼지 중복 제거 | MinHash LSH로 의미적으로 유사한 문서 제거 |
| 언어 식별 | fastText 모델로 언어 자동 감지/필터링 |
| 품질 필터링 | 휴리스틱 + 분류기 기반 저품질 제거 |
| 개인정보 삭제 | PII 탐지 및 삭제/마스킹 |
| 의미 중복 제거 | 임베딩 기반 의미적 유사도 중복 탐지 |

## 설치
```bash
pip install nemo-curator                    # 기본
pip install "nemo-curator[cuda12x]"         # GPU 가속 (RAPIDS 포함)
```

## 핵심 코드 예시

### 정확 중복 제거
```python
import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import ExactDuplicates
import pandas as pd

df = pd.read_parquet("output/korean_math_cot_full.parquet")
dataset = DocumentDataset.from_pandas(df, text_field="problem")

exact_dedup = ExactDuplicates(id_field="id", text_field="problem")
deduplicated = exact_dedup(dataset)
```

### 퍼지 중복 제거 (MinHash LSH)
```python
from nemo_curator.modules import FuzzyDuplicates, FuzzyDuplicatesConfig

fuzzy_config = FuzzyDuplicatesConfig(
    char_ngrams=5,
    num_buckets=20,
    hashes_per_bucket=13,
    jaccard_threshold=0.8,    # 80% 이상 유사하면 중복으로 판정
)
fuzzy_dedup = FuzzyDuplicates(config=fuzzy_config, id_field="id", text_field="problem")
result = fuzzy_dedup(dataset)
```

### 커스텀 품질 필터
```python
from nemo_curator.filters import DocumentFilter

class KoreanMathQualityFilter(DocumentFilter):
    def __init__(self, min_problem_len=50):
        super().__init__()
        self.min_problem_len = min_problem_len

    def score_document(self, text: str) -> bool:
        if len(text) < self.min_problem_len:
            return False
        # 한국어 비율 30% 이상 요구
        korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
        return korean_chars / max(len(text), 1) >= 0.3

    def keep_document(self, score) -> bool:
        return score

quality_filter = nc.ScoreFilter(
    KoreanMathQualityFilter(min_problem_len=50),
    text_field="problem",
)
filtered_dataset = quality_filter(dataset)
```

## Data Designer와의 연계
- Data Designer 출력 (Parquet/JSONL) → Curator 입력
- `DocumentDataset.from_pandas()` 또는 `from_parquet()`로 로드
- 필터링 후 `dataset.to_pandas().to_parquet()` 또는 `to_json()` 내보내기

## 참고 링크
- 공식 문서: https://docs.nvidia.com/nemo/curator/latest/
- GitHub: https://github.com/NVIDIA/NeMo-Curator
