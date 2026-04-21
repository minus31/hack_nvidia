# 07. NeMo Data Designer — Track C (핵심)

> 스키마 기반으로 합성 데이터셋을 생성하는 NVIDIA 프레임워크. LLM을 활용한 고품질 학습 데이터 생성 파이프라인.

## 핵심 요약
- **목적:** vLLM으로 배포한 Nemotron 3를 백엔드로, YAML 스키마로 선언적 합성 데이터 파이프라인 구성
- **예제:** 한국어 수학 CoT(Chain-of-Thought) 데이터셋 생성
- **방식:** 컬럼 간 의존성을 Jinja2 템플릿 `{{ column_name }}`으로 정의 → 실행 순서 자동 결정

## SDK 핵심 컴포넌트
| 컴포넌트 | 설명 |
|---------|------|
| `SamplerColumn` | LLM 호출 없이 결정론적 샘플링 (카테고리, 분포) |
| `LLMTextColumn` | Jinja2 프롬프트 템플릿으로 LLM 텍스트 생성 |
| `LLMStructuredColumn` | Pydantic 스키마 기반 구조화된 LLM 출력 생성 |
| `LLMJudgeColumn` | LLM-as-a-Judge로 데이터 품질 평가/점수 |
| `ExpressionColumn` | Jinja2로 기존 컬럼 변환 (LLM 호출 없음) |
| `ValidationColumn` | Python 함수로 생성 데이터 검증 |

## 파이프라인 흐름
```
SamplerColumn (grade, topic, difficulty)
    ↓ {{ topic }}, {{ difficulty }}
LLMTextColumn → problem (한국어 수학 문제)
    ↓ {{ problem }}
LLMTextColumn → solution (단계별 CoT 풀이)
    ↓ {{ problem }}, {{ solution }}
LLMStructuredColumn → answer_metadata (정답 + 사용 개념)
LLMJudgeColumn → quality_score (품질 평가)
    ↓
ExpressionColumn → chat_formatted (학습용 대화 형식)
```

## 설치 및 환경
```bash
pip install data-designer openai pandas pyarrow requests
```

## GPU 요구사항
| 모델 | 변형 | 필요 GPU |
|------|------|---------|
| Nemotron 3 Super (120B) | BF16 | 4x H100 80GB |
| Nemotron 3 Super (120B) | FP8 | 2x H100 80GB |
| Nemotron 3 Super (120B) | NVFP4 | 1x B200 |
| Nemotron 3 Nano (30B) | BF16 | 1x H100/A100 80GB |
| Nemotron 3 Nano (30B) | FP8 | 1x H100 80GB |
| Nemotron 3 Nano (30B) | NVFP4 | 1x B200 |

## vLLM 배포 (별도 터미널)
```bash
# Nano BF16 예시 (1x H100)
bash scripts/launch_nemotron_nano.sh bf16
# → 준비되면 로그에 'Uvicorn running on http://0.0.0.0:5000' 표시

# Super BF16 예시 (4x H100)
bash scripts/launch_nemotron_super.sh bf16
```

주요 vLLM 옵션:
- `--served-model-name nemotron` (클라이언트 코드 공통화)
- `--reasoning-parser nano_v3` / `super_v3`
- `--tool-call-parser qwen3_coder`

## Data Designer 코드 구조
```python
from data_designer.interface import DataDesigner
from data_designer.config import (
    SamplerColumnConfig, LLMTextColumnConfig,
    LLMStructuredColumnConfig, LLMJudgeColumnConfig,
    ExpressionColumnConfig, ModelConfig, ModelProvider,
    CategorySamplerParams, ChatCompletionInferenceParams,
)

# 1. vLLM 프로바이더 정의
vllm_provider = ModelProvider(
    name="vllm-local",
    endpoint="http://localhost:5000/v1",
    provider_type="openai",
    api_key="not-used",
)

# 2. 모델 설정 (역할별 temperature 다르게)
problem_model = ModelConfig(
    alias="problem-generator",
    model="nemotron",
    provider="vllm-local",
    inference_parameters=ChatCompletionInferenceParams(temperature=0.9, max_tokens=4096),
)
solution_model = ModelConfig(alias="solution-generator", model="nemotron", provider="vllm-local",
    inference_parameters=ChatCompletionInferenceParams(temperature=0.3, max_tokens=2048))
judge_model = ModelConfig(alias="quality-judge", model="nemotron", provider="vllm-local",
    inference_parameters=ChatCompletionInferenceParams(temperature=0.1, max_tokens=1024))
```

## 스키마 정의 예시
```python
# 컬럼 정의 (예시)
grade_col = SamplerColumnConfig(
    name="grade",
    sampler_type=SamplerType.CATEGORY,
    sampler_params=CategorySamplerParams(
        categories=["초3", "초4", "초5", "초6", "중1", "중2", "중3"],
        weights=[1, 1, 1, 1, 1.5, 1.5, 1.5],
    ),
)

problem_col = LLMTextColumnConfig(
    name="problem",
    model_alias="problem-generator",
    prompt_template="""{{ grade }} 학생을 위한 {{ topic }} 관련 {{ difficulty }} 수학 문제를 만들어주세요.
문제만 작성하고, 풀이는 포함하지 마세요.""",
)

quality_col = LLMJudgeColumnConfig(
    name="quality_score",
    model_alias="quality-judge",
    scores=[
        Score(name="correctness", description="수학적 정확성", min=1, max=5),
        Score(name="clarity", description="문제 명확성", min=1, max=5),
    ],
    prompt_template="문제: {{ problem }}\n풀이: {{ solution }}\n\n위 수학 문제와 풀이를 평가해주세요.",
)
```

## 단계별 파이프라인
1. **Step 0:** 환경 설정 + vLLM 배포 (별도 터미널)
2. **Step 1:** 데이터 스키마 정의 (컬럼 구조)
3. **Step 2:** DataDesigner 인스턴스 생성 + 컬럼 등록
4. **Step 3:** 데이터 생성 실행 (`designer.generate(n=1000)`)
5. **Step 4:** 품질 필터링 (`quality_score >= 3.5`)
6. **Step 5:** 데이터셋 내보내기 (Parquet/JSONL)
7. **Step 6:** HuggingFace Hub 업로드 (GPU 서버로 전송)

## 출력 형식
```jsonl
{"messages": [
  {"role": "system", "content": "당신은 수학 선생님입니다."},
  {"role": "user", "content": "<문제 텍스트>"},
  {"role": "assistant", "content": "<CoT 풀이 + 정답>"}
]}
```

## 참고 링크
- 공식 문서: https://nvidia-nemo.github.io/DataDesigner/latest/
- GitHub: https://github.com/NVIDIA-NeMo/DataDesigner
