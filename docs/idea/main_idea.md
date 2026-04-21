# EvalOps + Self-Evolving SDG Pipeline

> **Track C — Nemotron for SDG**
> "합성 데이터 파이프라인을 스스로 개선하는 시스템"

---

## 한줄 컨셉

> **EvalOps 인프라** 위에서 **Self-Evolving 알고리즘**이 프롬프트를 자동 최적화하고, 그 결과를 실험으로 검증한다.

---

## 전체 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                     EvalOps Layer                        │
│                                                          │
│  [스키마 정의]   →   [Data Designer]   →   [LLMJudge]   │
│   (컬럼 설계)        (데이터 생성)        (품질 평가)     │
│                           ↑                    │        │
│                           │                    ▼        │
│                    [프롬프트 템플릿]    [품질 메트릭]     │
│                           ↑                    │        │
└───────────────────────────┼────────────────────┼────────┘
                            │                    │
                            │   Self-Evolving     │
                            │      Layer          │
                            │                    ▼
                     [개선된 프롬프트] ← [OPRO Optimizer]
                                          (Nemotron이
                                           옵티마이저)
                            │
                            ▼
                     [수렴 판단 → 종료]
                            │
                            ▼
               ┌────────────────────────┐
               │   Curator Filtering    │
               │  (중복 제거 + 품질)    │
               └────────────────────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │    Reporting Dashboard │
               │  • 라운드별 품질 그래프 │
               │  • 프롬프트 변화 추적  │
               │  • 최종 데이터셋 통계  │
               └────────────────────────┘
```

---

## Layer 1: EvalOps Infrastructure

### 역할
Self-Evolving 루프에 필요한 **평가 신호(fitness signal)**를 생성하는 인프라.

### 구성 요소

#### 1-A. 다기준 Judge System
단일 점수가 아닌 **여러 기준으로 분해된 품질 평가**를 수행한다.
Self-Evolving 알고리즘이 어느 기준이 약한지 파악하고 그 부분을 집중 개선할 수 있게 한다.

```python
# 예시: 수학 CoT 데이터셋의 다기준 평가
class QualityDimensions(BaseModel):
    clarity: int          # 1~5: 문제의 명확성
    correctness: int      # 1~5: 풀이의 정확성
    step_quality: int     # 1~5: 단계별 풀이 완성도
    grade_fit: int        # 1~5: 학년 수준 적합성
    overall: int          # 1~5: 종합 점수

judge_column = LLMJudgeColumnConfig(
    name="quality",
    output_schema=QualityDimensions,
    prompt_template="""
    다음 수학 문제와 풀이를 5가지 기준으로 각각 1~5점으로 평가하세요.
    학년: {{grade}} / 난이도: {{difficulty}}
    문제: {{problem}}
    풀이: {{solution}}
    """
)
```

#### 1-B. Prompt Registry
모든 프롬프트 버전을 버전 관리하고, 각 버전의 평가 결과를 기록한다.

```python
# prompt_registry.py
@dataclass
class PromptVersion:
    version: int
    template: str
    avg_score: float
    scores_by_dim: dict[str, float]
    sample_outputs: list[dict]    # 좋은/나쁜 예시
    created_at: str
```

#### 1-C. Mini-Batch Evaluator
전체 생성 없이 소량 샘플(n=20~50)로 빠르게 품질을 평가한다. Self-Evolving의 inner loop에서 호출된다.

---

## Layer 2: Self-Evolving Algorithm (OPRO 기반)

### 알고리즘 선택: OPRO

**OPRO** (Optimization by PROmpting, Google DeepMind, 2023)를 SDG에 적용한다.

> 핵심 아이디어: LLM 자체를 옵티마이저로 사용한다. 이전 프롬프트들과 그 점수를 컨텍스트로 주면, LLM이 더 좋은 프롬프트를 제안한다.

### OPRO Meta-Prompt 구조

```
당신은 합성 데이터 생성 프롬프트 최적화 전문가입니다.

## 목표
다음 기준에서 평균 4.0 이상을 달성하는 데이터 생성 프롬프트를 작성하세요:
- 명확성 (clarity)
- 정확성 (correctness)
- 단계 품질 (step_quality)
- 학년 적합성 (grade_fit)

## 이전 시도 결과 (최근 → 오래된 순)
[v5] 점수: 3.8 | 명확성:4.1, 정확성:3.9, 단계:3.4, 학년:3.8
프롬프트: "당신은 {{grade}} 학생을 위한 {{difficulty}} 수학 문제 전문가입니다..."
약점: 단계별 풀이가 너무 간략함

[v4] 점수: 3.5 | 명확성:3.8, 정확성:3.6, 단계:3.1, 학년:3.5
프롬프트: "{{grade}} 학생에게 {{topic}} 문제를 출제하세요..."
약점: 문제의 맥락 부족, 단계 구조 없음

## 고품질 예시 (score >= 4.5)
[예시 1] 문제: "..." / 풀이: "1단계:... 2단계:... 최종답:..."

## 저품질 예시 (score <= 2.5)
[예시 2] 문제: "..." / 풀이: "..." (단계 없음)

## 지시사항
위 정보를 바탕으로 개선된 프롬프트를 작성하세요.
특히 약점인 "단계별 풀이 품질"을 개선하는 데 집중하세요.
```

### 전체 루프

```python
def self_evolving_loop(
    schema: DataSchema,
    initial_prompt: str,
    target_score: float = 4.0,
    max_rounds: int = 10,
    samples_per_round: int = 30,
):
    registry = PromptRegistry()
    registry.add(version=0, template=initial_prompt)

    for round_idx in range(max_rounds):
        # 1. 현재 프롬프트로 소량 생성
        samples = designer.preview(n=samples_per_round)

        # 2. 다기준 평가
        scores = evaluator.score(samples)
        avg = scores["overall"].mean()

        print(f"[Round {round_idx}] avg_score={avg:.2f}")
        registry.update_scores(round_idx, scores, samples)

        # 3. 수렴 조건 확인
        if avg >= target_score:
            print(f"수렴! {round_idx}라운드에서 목표 달성")
            break

        # 4. OPRO: 약점 분석 + 새 프롬프트 생성
        weak_dims = scores.mean().nsmallest(2).index.tolist()
        new_prompt = opro_optimizer.suggest(
            history=registry.get_recent(n=5),
            weak_dims=weak_dims,
            good_examples=registry.get_top_samples(n=3),
            bad_examples=registry.get_bottom_samples(n=3),
        )

        # 5. 프롬프트 업데이트
        designer.update_prompt(new_prompt)
        registry.add(version=round_idx+1, template=new_prompt)

    # 6. 최적 프롬프트로 전체 데이터셋 생성
    best_version = registry.get_best()
    designer.update_prompt(best_version.template)
    final_dataset = designer.generate(n=1000)

    return final_dataset, registry
```

---

## Layer 3: Curator Filtering

Self-Evolving으로 생성된 데이터를 Curator로 최종 정제한다.

```python
# curator_pipeline.py
pipeline = Pipeline([
    # 1. 중복 제거 (Fuzzy)
    FuzzyDeduplicator(threshold=0.85),

    # 2. 품질 필터링 (Judge 점수 기반)
    ScoreFilter(min_score={"overall": 3.5, "correctness": 4.0}),

    # 3. 데이터 균형 (학년별, 난이도별 비율 조정)
    DistributionBalancer(
        target_dist={"grade": grade_weights, "difficulty": difficulty_weights}
    ),
])

curated = pipeline.run(final_dataset)
```

---

## 실험 설계

### 실험 도메인
**한국어 수학 CoT 데이터셋** (Nemotron Developer Days 참고 예제 기반)

### 측정 지표
| 지표 | 설명 | 측정 방법 |
|---|---|---|
| Quality Score | 4개 기준의 평균 점수 | LLMJudge (Nemotron) |
| Score Variance | 점수의 표준편차 | 데이터 일관성 지표 |
| Prompt Convergence | 목표 점수 도달 라운드 수 | 루프 로그 |
| Data Diversity | 생성된 데이터의 다양성 | Semantic 중복률 역수 |

### 비교군 (Baseline)
| 비교군 | 설명 |
|---|---|
| **Baseline A** | 초기 수동 작성 프롬프트 (단일 사용) |
| **Baseline B** | 수동 프롬프트 엔지니어링 3회 반복 |
| **Ours** | OPRO Self-Evolving (자동 최적화) |

### 기대 결과
```
품질 점수
5.0 │                              ••••••• (Ours - 수렴)
4.5 │                   •••••
4.0 │─────────────────── Target Line ────────────────
3.5 │              •
3.0 │     •    ────────────────────────── (Baseline B)
2.5 │ ─────────────────────────────────── (Baseline A)
    └────────────────────────────────────────────────
         R0   R1   R2   R3   R4   R5   R6   R7
```

---

## Reporting Dashboard

실험 결과를 시각화하는 Streamlit 대시보드.

### 탭 구성
1. **Overview**: 라운드별 품질 점수 추이 + 비교군 차트
2. **Prompt Evolution**: 프롬프트 버전 변화 (diff 뷰어)
3. **Sample Inspector**: 각 라운드의 고품질/저품질 샘플 비교
4. **Final Dataset**: Curator 이후 최종 데이터셋 통계 (학년/난이도 분포)

---

## 구현 로드맵 (해커톤 22시간)

| 시간 | 작업 | 산출물 |
|---|---|---|
| 0~3h | 환경 세팅 + Data Designer 기본 파이프라인 | 수학 데이터 생성 확인 |
| 3~6h | 다기준 LLMJudge + Prompt Registry 구현 | 평가 시스템 동작 확인 |
| 6~10h | OPRO Optimizer 구현 + Self-Evolving 루프 | 최소 3라운드 실험 |
| 10~14h | 실험 실행 (10라운드) + 데이터 수집 | 실험 결과 로그 |
| 14~16h | Curator 정제 파이프라인 | 최종 데이터셋 |
| 16~20h | Streamlit 대시보드 | 시각화 완성 |
| 20~22h | 발표 자료 + 정리 | 최종 제출 |

---

## 핵심 차별점

1. **평가와 최적화의 통합**: LLMJudge(EvalOps)가 생성하는 다기준 점수가 OPRO의 fitness signal로 직접 연결됨
2. **Nemotron 이중 활용**: 데이터 생성자 + 프롬프트 옵티마이저 둘 다 Nemotron 사용
3. **실험 결과**: "자동 최적화가 수동보다 X% 더 높은 품질을 달성"이라는 검증 가능한 주장
4. **범용성**: 도메인만 바꾸면 어느 SDG 파이프라인에나 적용 가능

---

## 리스크 및 대응

| 리스크 | 대응 |
|---|---|
| OPRO 수렴 실패 | 라운드 상한(10회) 설정 + 최고 점수 버전 사용 |
| API 비용/시간 부족 | samples_per_round=20으로 축소, preview 모드 활용 |
| 개선 효과 미미 | Baseline A와 비교 시 최소한 "자동화"의 가치로 포지셔닝 |
| 스코프 오버 | Curator 파이프라인은 기본 FuzzyDedup + ScoreFilter만 |
