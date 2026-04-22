# Synthetic Data Generation for Training — 2025 Research Trend Review

> 2025년 이후 합성 데이터 생성 연구 동향 정리

---

## 1. 전체 흐름

합성 데이터는 2025년 기준 AI 학습의 핵심 전략으로 자리잡았다.

- Gartner 예측: 2028년까지 AI 학습 데이터의 **80%가 합성 데이터** (2020년 기준 5% 미만)
- 핵심 동인: 레이블 데이터 부족, 비용, 개인정보 규제(GDPR/HIPAA), 희귀 케이스 확보

---

## 2. 주요 기법 분류

### 2.1 프롬프트 기반 생성 (Prompt-based Generation)

LLM에 직접 프롬프트를 주어 데이터를 생성하는 가장 기본적인 방식.

- **Self-Instruct** (Stanford): LLaMA 7B를 52,000개 합성 instruction 예제로 fine-tuning → ChatGPT 수준 달성
- **Alpaca 방식**: 강력한 teacher LLM(GPT-4 등) → 약한 student 모델 학습용 데이터 생성

### 2.2 Self-Play / 반복 정제

모델이 자기 자신과 경쟁하거나 스스로 생성한 데이터를 필터링하는 방식.

- **SPIN (Self-Play Fine-Tuning)**: GAN 방식 응용. 생성자는 인간 응답과 구별되지 않는 응답을 생성하고, 판별자는 이를 구분. ICLR 2025 채택
- **Active SDG** ([arxiv 2512.00884](https://arxiv.org/abs/2512.00884)): 학생 모델의 현재 상태 기반으로 합성 데이터를 반복 조정하는 closed-loop 방식

### 2.3 Reward Model 기반 파이프라인

LLM 생성 + Reward 모델 평가를 결합하는 2-스테이지 파이프라인. 2025년 표준으로 자리잡는 중.

```
Query → Generator LLM (복수 응답 생성) → Reward Model (품질 점수) → 상위 응답 선별 → 학습 데이터
```

---

## 3. 주요 연구 사례

### Case 1: NVIDIA Nemotron-4-340B SDG Pipeline

**출처**: [NVIDIA Blog](https://developer.nvidia.com/blog/leverage-our-latest-open-models-for-synthetic-data-generation-with-nvidia-nemotron-4-340b/)

| 항목 | 내용 |
|---|---|
| 구성 | Instruct 모델(생성) + Reward 모델(필터링) |
| Reward 모델 성능 | RewardBench 92.2점, Chat-Hard에서 2위 대비 +10점 |
| 검증 결과 | 10만 행 합성 데이터("Daring Anteater")로 Llama-3-70B 정렬 → 인간 주석 데이터의 **1%만으로** Llama-3-70B-Instruct 수준 달성 |
| 자체 학습 | Nemotron-4-340B-Instruct alignment 학습 데이터 중 **98%가 합성 데이터** |

Reward 모델은 응답을 Helpfulness, Correctness, Coherence, Complexity, Verbosity 5개 축(0~4 Likert)으로 평가한다.

---

### Case 2: Nemotron-CC — Common Crawl 기반 1조 토큰 데이터셋

**출처**: [NVIDIA Blog](https://developer.nvidia.com/blog/building-nemotron-cc-a-high-quality-trillion-token-dataset-for-llm-pretraining-from-common-crawl-using-nvidia-nemo-curator/)

NeMo Curator + SDG를 결합한 대규모 pretraining 데이터 구축 사례.

**파이프라인:**
```
Common Crawl 원시 데이터
  → HTML 파싱 + FastText 언어 감지
  → MinHash + LSH 중복 제거
  → 3개 분류기 앙상블로 품질 점수 산출 (0~19점, 5 티어)
  → 저품질 문서: LLM으로 Wikipedia 스타일 재작성
  → 고품질 문서: 4종 LLM으로 QA쌍 / 요약 / 지식 추출 / 리스트화
```

**규모**: 원본 6.3조 토큰 + 합성 **2조 토큰** 추가

**MMLU 벤치마크 결과** (Llama 3.1 8B 학습 기준):

| 학습 데이터 | MMLU | 비고 |
|---|---|---|
| DCLM | 53.4 | 기존 최고 |
| Nemotron-CC (1T 토큰) | **59.0** | +5.6점 |
| Nemotron-CC (15T 토큰) | **70.3** | Llama 3.1 baseline 대비 +5점 |

필터링으로 제거된 콘텐츠의 **90%를 합성 데이터로 복구** — 대규모 학습에 필요한 토큰 확보의 핵심 전략.

---

### Case 3: Microsoft Phi-4 / Phi-4-Reasoning

**출처**: [기술 보고서](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/phi_4_reasoning.pdf)

- 합성 데이터를 사전학습부터 fine-tuning까지 전 과정에 통합
- **Teacher 초월**: STEM 중심 QA에서 teacher 모델(GPT-4)보다 높은 성능 달성 → 단순 distillation이 아님을 증명
- **수학 추론 성능**: AIME 2025에서 DeepSeek-R1(전체) 수준, o1-mini와 DeepSeek-R1-Distill-Llama-70B 초과
- 학습 데이터: 수학 / 과학 / 코딩 Q&A + chat 형식 혼합

---

### Case 4: 데이터 효율적 Distillation

**출처**: [arxiv 2508.09883](https://www.arxiv.org/pdf/2508.09883)

- **단 800개** 정밀 큐레이션 예제로 AIME 2024/2025 및 LiveCodeBench SOTA 달성
- "대량 합성 < 소량 고품질"의 흐름을 실증

---

### Case 5: Google Gemini — Simula 시스템

- Gemini 2.5 Flash(teacher) → Gemma-3 4B(student) 합성 distillation
- 사이버보안, 법률 추론 등 전문 도메인 특화 합성 데이터 생성

---

## 4. 도메인별 응용

| 도메인 | 사례 | 핵심 성과 |
|---|---|---|
| 수학/추론 | SPIN, Phi-4-Reasoning | 합성 CoT 데이터로 reasoning 대폭 향상 |
| 코드 | Phi-4-Reasoning-Plus, CodeLLM | 실행 피드백 기반 RL로 검증 자동화 |
| 의료 | 희귀질환 합성 데이터 (7,133명 환자 기반) | GDPR/HIPAA 준수하면서 예측 모델 학습 |
| 자율주행 | Waymo | 하루 **200억 마일** 시뮬레이션 |
| 농업 | 작물 성장 시뮬레이션 | 수확량 예측 AI 학습 데이터 자동 생성 |

---

## 5. 과제 및 한계

| 과제 | 내용 |
|---|---|
| 사실 오류 | LLM 환각(hallucination)이 합성 데이터에도 전이 |
| 분포 편향 | 합성 데이터가 현실 분포와 다를 수 있음 |
| 편향 증폭 | 모델 편향이 데이터로 이어져 심화될 위험 |
| 모델 붕괴 | 합성 데이터만으로 반복 학습 시 다양성 저하 |

**완화 전략**: Reward 모델 필터링, 실행 피드백 RL(코드), 합성-실제 데이터 혼합

---

## 6. 2025년 핵심 트렌드 요약

1. **Reward 모델 필수화** — 생성 LLM + 평가 LLM의 2단계 파이프라인 표준화
2. **소량 고품질 > 대량 저품질** — 800개 정밀 예제가 수만 개 랜덤 예제를 압도
3. **Closed-loop 자동화** — 학생 모델 성능 피드백으로 합성 데이터를 동적 조정
4. **멀티모달 확장** — 텍스트를 넘어 이미지/비디오/오디오로 합성 데이터 영역 확대
5. **도메인 특화** — 의료, 법률, 코드 등 전문 도메인에서 합성 데이터가 실제 데이터 대체 시작

---

## 참고 문헌

- [Synthetic Data Generation Using LLMs: Survey (arxiv 2503.14023)](https://arxiv.org/abs/2503.14023)
- [NVIDIA Nemotron-4-340B SDG Pipeline](https://developer.nvidia.com/blog/leverage-our-latest-open-models-for-synthetic-data-generation-with-nvidia-nemotron-4-340b/)
- [Nemotron-CC: Trillion Token Dataset](https://developer.nvidia.com/blog/building-nemotron-cc-a-high-quality-trillion-token-dataset-for-llm-pretraining-from-common-crawl-using-nvidia-nemo-curator/)
- [Phi-4-Reasoning Technical Report](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/phi_4_reasoning.pdf)
- [Active Synthetic Data Generation (arxiv 2512.00884)](https://arxiv.org/abs/2512.00884)
- [Data-Efficient Distillation (arxiv 2508.09883)](https://www.arxiv.org/pdf/2508.09883)
- [LLM-Synthetic-Data Reading List (GitHub)](https://github.com/pengr/LLM-Synthetic-Data)
