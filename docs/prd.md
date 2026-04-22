# PRD — Self-Evolving SDG Pipeline

**Track:** Track C — Nemotron for SDG
**Event:** NVIDIA Nemotron Developer Days Seoul 2026 (Apr 21–22)

---

## 1. Why — Toward Datasets That Don't Go Stale

Building a good dataset is hard. Difficulty, diversity, label accuracy — meeting all of these conditions takes researchers months, sometimes years, of careful curation to produce the benchmarks and training sets our field relies on.

The problem is that these datasets start aging the moment they're released. Models keep improving while the data stays fixed, so benchmarks saturate within a few years and training sets end up rehearsing what the model already knows. And then we start from scratch.

**We don't let that effort go to waste.** We propose a pipeline that automatically evolves existing, carefully crafted datasets to match the current capability of the model.

---

## 2. 핵심 아이디어

기존에 공들여 만든 데이터셋(벤치마크)을 시드로 삼아, **자동으로 더 어렵고 다양한 버전을 생성**하는 방식을 실험한다. 

- 더어렵고 다양한 버전을 생성하는 방법론을 Sythesizer라고 하고, 여러 sythesizer를 비교한다. 
- 생성된 데이터의 품질(난이도·타당성)은 **Verifiable한 방식**으로 자동 평가
    - 평가를 용이하게 하기 위해서 그리고 Sythesizer를 강화학습할 수 있는 RLVR데이터셋을 함께 만들기 위해 LLM Judge에 의존하지 않고, 규칙·실행·해시 기반으로 정답을 확인할 수 있는 태스크를 사용하려 했음. 

- NeMo Data Designer + Nemotron(NIM)을 생성 엔진으로 활용

---

## 3. 해커톤 실험 범위 및 이유

### 왜 Training 데이터셋이 아닌 벤치마크인가

본 아이디어는 **학습용 합성 데이터셋의 자동 진화**에도 적용 될 수 있음. 그러나 학습 데이터셋의 품질을 평가하려면 해당 데이터로 모델을 파인튜닝한 뒤 성능 향상을 측정해야 하고, 이 과정은 해커톤 22시간 내에 완수하기 어렵다.

따라서 이번 해커톤에서는 **동일한 파이프라인 원리를 비교적 평가가 쉬운 벤치마크 생성에 적용**하여 실험한다.

- 벤치마크는 "더 어려운 문제가 생성되었는가"로 파이프라인 효과를 빠르게 검증 가능
- 파이프라인 구조(시드 → 생성 → 자동 평가 → 진화)는 학습 데이터셋 생성과 완전히 동일
- 향후 동일 파이프라인을 학습 데이터 도메인으로 확장하는 것이 자연스러운 다음 단계

---

## 4. 실험 설계

### 4.1 사용 벤치마크

#### Benchmark A — 기존 벤치마크 (HarmBench Copyright)

| 항목 | 내용 |
|---|---|
| 출처 | HarmBench (ICML 2024), `centerforaisafety/HarmBench` |
| 태스크 | 저작권 콘텐츠(노래 가사 / 책 구절) 생성 유도 프롬프트 |
| 규모 | 100개 (가사 50개 + 책 구절 50개) |
| Verifiable 방법 | MinHash 기반 해시 비교 (LLM Judge 없음) |
| 데이터 위치 | `data/harmbench/behavior_datasets/harmbench_behaviors_text_all.csv` |
| 해시 파일 위치 | `data/harmbench/copyright_classifier_hashes/*.pkl` |

**선택 이유:**
- Hash 비교로 자동 검증 가능한 유일한 Safety 벤치마크 카테고리
- 모델이 대부분 잘 거절하게 되면서 기존 프롬프트의 공격 성공률(ASR)이 낮아짐 → 노후화
- Synthesizer가 더 교묘한 프롬프트를 생성해 ASR을 높이는 것이 목표

#### Benchmark B — 직접 생성한 벤치마크 (Custom Copyright)
- To be added

---

### 4.2 SDG 방법론

Synthesizer는 Nemotron3-nano를 사용해서 방법론 들을 비교한다. => 여러 모델로 실험을 추가할 예정! 

- Evol-Instruct 
- Auto Evol-Instruct 
- GEPA 등 비교 할 예정

아래 방식도 고려
| 방법 | 설명 |
|---|---|
| **Paraphrase Evolution** | 직접적인 요청을 간접·우회적 표현으로 변환 |
| **Role-Play Wrapping** | 페르소나/시나리오를 씌워 요청을 위장 |
| **Context Injection** | 교육·연구·창작 등 정당한 맥락을 부여 |
| **Chain Escalation** | 무해한 질문에서 시작해 단계적으로 목표에 접근 |

각 방법은 NeMo Data Designer의 `LLMTextColumn`을 통해 Nemotron이 생성하며, 생성 프롬프트 템플릿은 라운드마다 개선된다.


문제 타당성 (잘못 생성된 건 아닌가)은 각 synthesizer 알고리즘에서 고려한다. 

- 진화 과정에서 프롬프트가 의미를 잃거나, 의도된 저작권 콘텐츠와 무관한 방향으로 흘러가는 경우를 걸러내야 함. => 이부분은 각 Synthesizer 알고리즘에서 고려한다. 

(예시)
| 체크 항목 | 방법 |
|---|---|
| 요청 의도 유지 여부 | 진화된 프롬프트가 여전히 저작권 콘텐츠 생성을 목적으로 하는가 |

---

### 4.3 평가 기준

#### 주요 지표 — 난이도 향상 (더 어려운 벤치마크인가)

```
ASR (Attack Success Rate) = 공격 성공 수 / 전체 시도 수

진화 전 ASR  →  진화 후 ASR
     ↑ 높아졌으면 "더 어려운 벤치마크" 생성 성공
```

- 타겟 모델에 프롬프트를 입력하고 출력과 `*.pkl` 해시를 비교
- ASR이 높아질수록 모델이 방어하기 어려운 벤치마크가 된 것

---

## 5. 코드 아키텍처

### LLM 후보군

| 용도 | 모델 |
|---|---|
| Evaluator (타겟) | `nemotron3-nano`, `gpt-4o` |
| Synthesizer (생성) | `nemotron3-nano`, `gpt-4o` |

LLM 호출 모듈은 Evaluator/Synthesizer 용도와 독립적으로 구성한다. 어느 컴포넌트에서도 동일하게 주입(inject)해서 사용.

### 핵심 컴포넌트

**LLM** — 모델 호출 인터페이스. Evaluator/Synthesizer 어디서든 동일하게 사용.
- `NemotronLLM`: NIM 서버 (`http://0.0.0.0:8000`)
- `OpenAILLM`: OpenAI API (`gpt-4o`)

**Evaluator** — 프롬프트 → LLM 응답 → hash check → ASR 계산.
- 데이터셋마다 하나씩 구현 (현재: `HarmBenchCopyrightEvaluator`)
- 저작권 콘텐츠 여부는 MinHash pkl 비교로 확인 (LLM Judge 없음)
- 출력: `EvalResult` (asr, per_prompt 성공 여부, 응답 텍스트)

**Synthesizer** — 시드 프롬프트를 받아 더 어렵게 진화시킨 버전을 반환한다.
- 알고리즘별로 구현: `EvolInstruct`, `AutoEvolInstruct`, `GEPA`(예정)
- 타당성 필터(요청 의도 유지)를 알고리즘 내부에서 처리

### 실험 흐름

```
seed prompts (HarmBench / Custom)
        │
        ▼
  Evaluator(llm).evaluate()  →  baseline ASR
        │
        ▼
  Synthesizer.synthesize(prompts, llm)  →  evolved prompts
  (Evol-Instruct / Auto-Evol / GEPA)
        │
        ▼
  Evaluator(llm).evaluate()  →  evolved ASR
        │
        ▼
  비교: baseline vs evolved × synthesizer × gen_llm
```

### 디렉토리 구조

```
src/
├── llm/
│   ├── base.py               # BaseLLM: generate(prompt) → str
│   ├── nemotron.py           # NemotronLLM → NIM (http://0.0.0.0:8000)
│   └── openai.py             # OpenAILLM → gpt-4o
│
├── evaluators/
│   ├── base.py               # BaseEvaluator: evaluate(prompts, llm) → EvalResult
│   └── harmbench_copyright.py  # MinHash pkl 비교로 hash_check
│
├── synthesizers/
│   ├── base.py               # BaseSynthesizer: synthesize(prompts, llm) → list[str]
│   ├── evol_instruct.py      # Evol-Instruct
│   ├── auto_evol_instruct.py # Auto Evol-Instruct
│   └── gepa.py               # GEPA (TBD)
│
├── data/
│   └── harmbench.py          # load_copyright_dataset()
│
├── run_experiment.py         # 실험 실행 진입점
└── report.py                 # 결과 리포트 생성
```
