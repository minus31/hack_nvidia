# NeMo Data Designer

> LLM 기반 합성 데이터셋을 스키마 선언만으로 생성하는 NVIDIA 오케스트레이션 프레임워크

- **공식 문서**: https://nvidia-nemo.github.io/DataDesigner/latest/
- **GitHub**: https://github.com/NVIDIA-NeMo/DataDesigner

---

## 1. Data Designer란?

NeMo Data Designer는 **고품질 합성(Synthetic) 데이터셋**을 체계적으로 생성하기 위한 프레임워크이다.

직접 LLM API를 호출하는 것과 달리, Data Designer는 다음을 자동으로 관리한다:

| 직접 LLM 호출 | Data Designer 사용 |
|---|---|
| 수동 프롬프트 작성 | Jinja2 템플릿 기반 자동 프롬프트 |
| 데이터 다양성 보장 어려움 | 통계적 분포 제어(SamplerColumn) |
| 검증 로직 별도 구현 | 내장 Validation/Judge 컬럼 |
| 배치 처리 직접 구현 | 자동 배칭, 병렬 처리, 재시도 |
| 재현성 없음 | YAML 스키마로 완전한 재현성 |

---

## 2. 설치

```bash
pip install data-designer
```

설치 후 설정 확인:

```bash
data-designer config list
```

### 지원 LLM 프로바이더

- **NVIDIA NIM** (권장) — 로컬 vLLM 또는 build.nvidia.com
- **OpenAI**
- **OpenRouter**

환경변수로 API 키를 설정한다:

```bash
export NVIDIA_API_KEY="your-key"
# 또는 로컬 vLLM 사용 시
export NVIDIA_BASE_URL="http://localhost:5000/v1"
export NVIDIA_API_KEY="not-used"
```

---

## 3. 핵심 개념: 컬럼(Column) 시스템

Data Designer의 데이터셋은 **컬럼 단위**로 구성된다. 각 컬럼은 하나의 데이터 필드를 생성하며, 컬럼 간 의존성에 따라 실행 순서가 자동으로 결정된다.

### 3.1 SamplerColumn — 결정적 값 샘플링

LLM을 호출하지 않고, 미리 정의된 값 목록에서 **확률 분포에 따라 샘플링**한다.

```python
from data_designer.interface import SamplerColumnConfig, CategorySamplerParams

grade_column = SamplerColumnConfig(
    name="grade",
    description="학년",
    sampler_params=CategorySamplerParams(
        values=["초3", "초4", "초5", "초6", "중1", "중2", "중3"],
        weights=[0.1, 0.1, 0.15, 0.15, 0.2, 0.15, 0.15],
    ),
)

difficulty_column = SamplerColumnConfig(
    name="difficulty",
    description="난이도",
    sampler_params=CategorySamplerParams(
        values=["easy", "medium", "hard"],
        weights=[0.3, 0.5, 0.2],
    ),
)
```

**포인트**: `weights`를 통해 데이터셋 내 각 카테고리의 비율을 정밀하게 제어할 수 있다.

### 3.2 LLMTextColumn — 텍스트 생성

Jinja2 템플릿 프롬프트를 LLM에 보내 **자유 형식 텍스트**를 생성한다. 다른 컬럼의 값을 `{{ }}` 구문으로 참조할 수 있다.

```python
from data_designer.interface import LLMTextColumnConfig

problem_column = LLMTextColumnConfig(
    name="problem",
    description="수학 문제",
    prompt_template="""
    당신은 한국 {{grade}} 학생을 위한 수학 문제 출제 전문가입니다.

    다음 조건에 맞는 {{difficulty}} 난이도의 {{topic}} 문제를 한 문제 작성하세요.

    조건:
    - 학년 수준에 맞는 어휘 사용
    - 실생활 맥락 포함
    - 명확한 문제 진술
    """,
    model="problem-generator",  # temperature=0.9 (창의적 생성)
)

solution_column = LLMTextColumnConfig(
    name="solution",
    description="풀이 과정",
    prompt_template="""
    다음 수학 문제에 대한 단계별 풀이를 작성하세요.

    문제: {{problem}}

    풀이 형식:
    1단계: ...
    2단계: ...
    최종 답: ...
    """,
    model="solution-generator",  # temperature=0.3 (정확한 추론)
)
```

**포인트**: `model` 파라미터로 각 컬럼에 서로 다른 추론 설정(temperature 등)을 적용한다.

### 3.3 LLMStructuredColumn — 구조화된 출력 생성

**Pydantic 스키마**를 통해 LLM 출력을 검증 가능한 구조화된 데이터로 추출한다.

```python
from pydantic import BaseModel
from data_designer.interface import LLMStructuredColumnConfig

class MathMetadata(BaseModel):
    correct_answer: str       # 정답
    concepts: list[str]       # 사용된 수학 개념들

metadata_column = LLMStructuredColumnConfig(
    name="metadata",
    description="문제 메타데이터",
    prompt_template="""
    다음 수학 문제와 풀이에서 정답과 사용된 수학 개념을 추출하세요.

    문제: {{problem}}
    풀이: {{solution}}
    """,
    output_schema=MathMetadata,
    model="extractor",  # temperature=0.1 (정확한 추출)
)
```

**포인트**: Pydantic 스키마 덕분에 LLM 출력이 올바른 형식인지 자동 검증된다.

### 3.4 LLMJudgeColumn — 품질 평가

LLM을 **평가자(Judge)**로 활용하여 생성된 데이터의 품질을 점수화한다.

```python
from data_designer.interface import LLMJudgeColumnConfig

quality_column = LLMJudgeColumnConfig(
    name="quality_score",
    description="품질 점수",
    prompt_template="""
    다음 수학 문제와 풀이의 품질을 평가하세요.

    문제: {{problem}}
    풀이: {{solution}}

    평가 기준:
    1. 문제의 명확성
    2. 풀이의 정확성
    3. 교육적 가치
    """,
    model="quality-judge",  # temperature=0.1 (일관된 평가)
)
```

**포인트**: 생성 후 자동으로 품질 필터링을 수행하여 저품질 데이터를 걸러낼 수 있다.

### 3.5 ExpressionColumn — 후처리 변환

기존 컬럼 값을 Jinja2 템플릿으로 **변환/조합**한다. LLM을 호출하지 않는다.

```python
from data_designer.interface import ExpressionColumnConfig

# 학습용 chat 포맷으로 변환
chat_format_column = ExpressionColumnConfig(
    name="conversation",
    description="학습용 대화 형식",
    expression="""[
        {"role": "user", "content": "{{problem}}"},
        {"role": "assistant", "content": "{{solution}}"}
    ]""",
)
```

**포인트**: LLM fine-tuning에 필요한 chat 형식으로 데이터를 자동 변환하는 데 유용하다.

### 3.6 ValidationColumn — 데이터 검증

Python 함수를 사용해 생성된 데이터를 **프로그래밍적으로 검증**한다.

```python
from data_designer.interface import ValidationColumnConfig

validation_column = ValidationColumnConfig(
    name="is_valid",
    description="검증 결과",
    validation_fn=lambda row: len(row["solution"]) > 100,  # 풀이가 충분히 긴지 확인
)
```

---

## 4. 모델 프로파일 설정

하나의 LLM 엔드포인트에 여러 **추론 파라미터 프로파일**을 설정할 수 있다.

```python
from data_designer.interface import ChatCompletionInferenceParams

models = {
    "problem-generator": ChatCompletionInferenceParams(
        temperature=0.9,   # 높은 창의성
        top_p=0.95,
        max_tokens=1024,
    ),
    "solution-generator": ChatCompletionInferenceParams(
        temperature=0.3,   # 정확한 추론
        top_p=0.9,
        max_tokens=2048,
    ),
    "extractor": ChatCompletionInferenceParams(
        temperature=0.1,   # 정확한 추출
        max_tokens=512,
    ),
    "quality-judge": ChatCompletionInferenceParams(
        temperature=0.1,   # 일관된 평가
        max_tokens=256,
    ),
}
```

| 프로파일 | Temperature | 용도 |
|---|---|---|
| problem-generator | 0.9 | 다양하고 창의적인 문제 생성 |
| solution-generator | 0.3 | 정확하고 논리적인 풀이 작성 |
| extractor | 0.1 | 정답/개념 정확 추출 |
| quality-judge | 0.1 | 일관된 품질 평가 |

---

## 5. 파이프라인 실행 흐름

### 실행 순서

```
SamplerColumn (결정적)
    ↓ grade, topic, difficulty 값 생성
LLMTextColumn (problem)
    ↓ 문제 생성 (grade, topic, difficulty 참조)
LLMTextColumn (solution)
    ↓ 풀이 생성 (problem 참조)
LLMStructuredColumn (metadata)
    ↓ 정답/개념 추출 (problem, solution 참조)
LLMJudgeColumn (quality_score)
    ↓ 품질 평가 (problem, solution 참조)
ExpressionColumn (conversation)
    ↓ chat 포맷 변환 (problem, solution 참조)
ValidationColumn (is_valid)
    ↓ 최종 검증
```

Data Designer는 **컬럼 간 의존성을 자동 분석**하여 실행 순서를 결정한다. Jinja2 템플릿의 `{{ variable }}` 참조를 파싱하여 DAG(방향 비순환 그래프)를 구성한다.

### 실행 방법

```python
from data_designer.interface import DataDesigner

designer = DataDesigner(
    columns=[
        grade_column,
        topic_column,
        difficulty_column,
        problem_column,
        solution_column,
        metadata_column,
        quality_column,
        chat_format_column,
    ],
    models=models,
)

# 미리보기 (소량 샘플 생성하여 확인)
preview = designer.preview(n=5)
print(preview)

# 전체 데이터셋 생성
dataset = designer.generate(n=1000)
dataset.to_parquet("math_dataset.parquet")
```

---

## 6. 실전 예제: 한국어 수학 CoT 데이터셋

Nemotron Developer Days에서 사용된 실습 예제이다.

### 목표

한국 초등·중학교 수학 문제 + 단계별 풀이(Chain-of-Thought)를 자동 생성한다.

### 사전 준비

- Python 3.10+
- NVIDIA GPU 환경
- vLLM으로 Nemotron 3 배포

```bash
pip install ipykernel data-designer openai pandas pyarrow requests
```

### 지원 모델 및 GPU 요구사항

| 모델 | 정밀도 | GPU 요구사항 |
|---|---|---|
| Nemotron-3-Super (120B A12B) | BF16 | 4x H100 |
| Nemotron-3-Super (120B A12B) | FP8 | 2x H100 |
| Nemotron-3-Super (120B A12B) | NVFP4 | 1x B200 |
| Nemotron-3-Nano (30B A3B) | BF16 | 1x H100 또는 A100 |
| Nemotron-3-Nano (30B A3B) | FP8 | 1x H100 |
| Nemotron-3-Nano (30B A3B) | NVFP4 | 1x B200 |

### 데이터 분포 설계

```python
# 학년 분포
grades = ["초3", "초4", "초5", "초6", "중1", "중2", "중3"]
grade_weights = [0.1, 0.1, 0.15, 0.15, 0.2, 0.15, 0.15]

# 주제 분포
topics = ["사칙연산", "분수/소수", "도형", "측정", "방정식", "함수", "통계/확률"]
topic_weights = [0.15, 0.15, 0.15, 0.1, 0.2, 0.1, 0.15]

# 난이도 분포
difficulties = ["easy", "medium", "hard"]
difficulty_weights = [0.3, 0.5, 0.2]
```

---

## 7. 주요 기능 정리

| 기능 | 설명 |
|---|---|
| **다중 모달** | 텍스트 외에 이미지 컨텍스트, 이미지 생성/편집 튜토리얼 제공 |
| **MCP 통합** | MCP(Model Context Protocol) 도구 사용 지원 |
| **Trace 수집** | 생성 과정의 상세 트레이스 기록 |
| **Seed 데이터셋** | 외부 데이터셋을 시드로 활용하여 증강 |
| **레시피** | Python/SQL 코드 생성, QA/Chat 시스템, 플러그인 개발 등 사전 구성된 레시피 |

---

## 8. 참고 자료

- [공식 문서](https://nvidia-nemo.github.io/DataDesigner/latest/)
- [GitHub 리포지토리](https://github.com/NVIDIA-NeMo/DataDesigner)
- [vLLM 문서](https://docs.vllm.ai)
- [Nemotron Cookbooks (GitHub)](https://github.com/NVIDIA-NeMo/Nemotron)
