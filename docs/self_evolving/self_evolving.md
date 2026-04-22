# Self-Evolving AI 알고리즘 조사 (2022~2026)

> 조사일: 2026-04-20

---

## 1. 합성 데이터 기반 자기 학습 (Synthetic Data Bootstrapping)

### SELF-INSTRUCT (ACL 2023)
- **논문:** [arxiv.org/abs/2212.10560](https://arxiv.org/abs/2212.10560)
- **핵심 아이디어:** 소수의 시드 예제만으로 모델이 스스로 instruction 데이터를 생성해 자기 학습
- **알고리즘:**
  1. 175개 수동 작성 (instruction, input, output) 시드 준비
  2. 시드를 few-shot 예제로 사용해 새 instruction 생성
  3. 분류형 / 개방형 여부 판단
  4. 각 instruction에 대응하는 input-output 생성
  5. ROUGE-L 유사도 기반 중복/저품질 필터링
  6. 통과한 예제를 시드 풀에 추가 → 반복
  7. 누적된 합성 데이터로 SFT

### Evol-Instruct / WizardLM (ICLR 2024)
- **논문:** [arxiv.org/abs/2304.12244](https://arxiv.org/abs/2304.12244)
- **핵심 아이디어:** LLM이 instruction을 자동으로 더 어렵게/다양하게 진화
- **알고리즘:**
  1. 단순 instruction 풀 준비 (SELF-INSTRUCT 결과물 등)
  2. **In-Depth Evolving:** 제약 추가, 심화, 구체화, 추론 단계 증가, 입력 복잡화 등 5가지 변이
  3. **In-Breadth Evolving:** 주제/기술 다양성 확장
  4. 실패한 진화 필터링
  5. N라운드 반복 → 점점 난이도 높은 데이터 축적
  6. 최종 데이터로 SFT
- **확장 — Auto Evol-Instruct:** optimizer LLM이 진화 전략 자체를 메타 최적화

---

## 2. Chain-of-Thought 자기 학습 (Bootstrapped Reasoning)

### STaR — Self-Taught Reasoner (NeurIPS 2022)
- **논문:** [arxiv.org/abs/2203.14465](https://arxiv.org/abs/2203.14465)
- **핵심 아이디어:** 정답으로 이어지는 추론 경로만 학습 + rationalization으로 어려운 문제 보완
- **알고리즘:**
  1. 소수의 rationale 예제로 프롬프트
  2. 전체 훈련 문제에 대해 rationale + 답 생성
  3. 정답인 (rationale, answer) 쌍만 보존
  4. **Rationalization:** 틀린 문제는 정답을 힌트로 주고 역으로 rationale 생성
  5. 정답 생성 + rationalization 데이터 합쳐서 SFT
  6. 수렴까지 반복

### Quiet-STaR (COLM 2024 Oral)
- **논문:** [arxiv.org/abs/2403.09629](https://arxiv.org/abs/2403.09629)
- **핵심 아이디어:** QA 경계가 아닌 모든 토큰 위치에서 내부 "thought" 생성 학습 (사전학습에 적용)
- **알고리즘:**
  1. 각 토큰 위치에서 M개의 후보 thought 샘플링
  2. thought 있을 때 / 없을 때의 다음 토큰 분포 예측
  3. mixing head로 두 분포 가중 결합
  4. REINFORCE로 이후 텍스트를 더 잘 예측하는 thought 강화
  5. 레이블 없이 raw 텍스트에서 추론 능력 학습

---

## 3. Self-Play Fine-Tuning

### SPIN — Self-Play Fine-Tuning (ICML 2024)
- **논문:** [arxiv.org/abs/2401.01335](https://arxiv.org/abs/2401.01335)
- **핵심 아이디어:** 이전 버전 모델(opponent) vs 현재 모델(player)의 2인 게임으로 SFT 데이터만으로 정렬
- **알고리즘:**
  1. M_0 = SFT 체크포인트로 초기화
  2. 반복 t:
     - 현재 모델 M_t (학습 대상)
     - 이전 모델 M_{t-1} (고정, opponent)
  3. M_{t-1}으로 응답 생성 (synthetic = rejected)
  4. 인간 응답 (chosen) vs M_{t-1} 응답 (rejected) 쌍 구성
  5. DPO loss로 M_t 학습
  6. t 증가 → 반복
- **이론적 보장:** M_t가 인간 데이터 분포와 일치할 때만 global optimum 도달 → 치팅 불가

---

## 4. 추론 시점 자기 개선 (Inference-Time Refinement)

### Self-Refine (NeurIPS 2023)
- **논문:** [arxiv.org/abs/2303.17651](https://arxiv.org/abs/2303.17651)
- **핵심 아이디어:** 가중치 업데이트 없이 inference 단계에서 출력 → 비판 → 수정 루프 반복
- **알고리즘:**
  1. 초기 응답 y_0 생성
  2. **Feedback:** 동일 모델이 자연어 비판 생성
  3. **Refine:** 비판을 반영해 y_1 생성
  4. 중지 조건(max iterations 또는 충분히 좋음)까지 반복
- **한계:** 기반 모델의 자기 비판 능력에 의존

### SelfEvolve for Code (2023)
- **논문:** [arxiv.org/abs/2306.02907](https://arxiv.org/abs/2306.02907)
- **알고리즘:**
  1. LLM으로 관련 도메인 지식 생성
  2. 생성된 지식을 컨텍스트로 초기 코드 작성
  3. 코드 실행 → 에러/테스트 피드백 수집
  4. 에러를 LLM에 피드백해 코드 수정
  5. 모든 테스트 통과 또는 반복 예산 소진까지 반복

---

## 5. Constitutional AI & RLAIF

### Constitutional AI — CAI (Anthropic, 2022)
- **논문:** [arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073)
- **핵심 아이디어:** "헌법(원칙 집합)"으로 AI 피드백을 유도해 인간 해악 레이블 대체
- **알고리즘 (2단계):**
  - **Phase 1 — SL-CAI:**
    1. 잠재적으로 해로운 응답 생성
    2. 헌법 원칙 중 하나로 비판
    3. 원칙에 맞게 수정
    4. N회 비판-수정 연쇄 반복
    5. 최종 수정 응답으로 SFT
  - **Phase 2 — RLHF + RLAIF:**
    1. 응답 쌍 생성
    2. AI가 헌법 기준으로 어느 쪽이 더 나은지 라벨링
    3. AI 라벨로 preference model 학습
    4. PPO로 RL

### RLAIF (Google, 2023)
- **논문:** [arxiv.org/abs/2309.00267](https://arxiv.org/abs/2309.00267)
- 별도 reward model 학습 없이 off-the-shelf LLM으로 RL 훈련 시 직접 스칼라 보상 쿼리

---

## 6. RLVR — 검증 가능한 보상으로 강화학습 (2025 주류)

### DeepSeek-R1 / GRPO (2025)
- **논문:** [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- **핵심 아이디어:** 인간 라벨 없이 규칙 기반 검증기(정답/오답)만으로 RL → 추론 능력 자발적 창발
- **알고리즘 — Group Relative Policy Optimization (GRPO):**
  1. 프롬프트 q에 대해 G개 응답 샘플링
  2. 규칙 기반 검증기로 각 응답 채점 (수학: 정/오답, 코드: 테스트 pass/fail)
  3. 그룹 상대적 advantage: `A_i = (reward_i - mean) / std`
  4. PPO-style clipped objective로 정책 업데이트 (별도 value 모델 불필요)
  5. KL penalty로 reference 모델 대비 드리프트 방지
- **창발 현상:** 명시적 훈련 없이 자기 검증, 되돌아가기, 장시간 반성 등 spontaneous 발생
- **전체 파이프라인:**
  1. Long CoT 예제로 Cold-start SFT
  2. GRPO 기반 RL (검증 가능한 보상)
  3. Rejection sampling → 고품질 롤아웃 수집
  4. 정확도 + 형식 + 유용성 통합 보상으로 최종 RL

### GRPO 개선 변형
| 논문 | 개선점 |
|---|---|
| **DAPO** | 높은 클리핑 threshold + dynamic sampling으로 양의 보상 집중 |
| **GSPO** | 토큰 레벨 → 시퀀스 레벨 importance sampling으로 편향 보정 |

---

## 7. 진화 탐색 (Evolutionary Search)

### AlphaEvolve (Google DeepMind, 2025)
- **논문:** [arxiv.org/abs/2506.13131](https://arxiv.org/abs/2506.13131)
- **핵심 아이디어:** LLM이 코드 돌연변이 제안 → 자동 평가기 점수 → 세대 간 진화로 알고리즘 최적화
- **알고리즘:**
  1. 프로그램 변형 집단(population) 유지
  2. 고품질 개체 편향 샘플링
  3. LLM 앙상블(Flash: 다량, Pro: 고품질)이 코드 돌연변이 제안
  4. 자동 평가기로 객관적 지표 측정
  5. 생존 개체 population에 추가, 열등 개체 제거
  6. 수천 세대 반복
- **결과:** Strassen 행렬 곱 알고리즘 56년 만에 개선 (4×4 복소수: 49 → 48 곱셈)

---

## 8. 커리큘럼 & 생애주기 자기 진화

| 논문 | 아이디어 |
|---|---|
| **SEC** (2025) | RL 학습과 동시에 커리큘럼 정책(어떤 문제를 풀지)도 함께 학습 |
| **EvolveR** (2024) | perceive → plan → act → reflect → update 생애주기로 경험 축적 |
| **SEAL** (MIT, 2025) | 모델이 런타임에 RL 기반 자기 편집으로 자신의 가중치 직접 업데이트 |

---

## 9. 알고리즘 비교표

| 알고리즘 | 가중치 업데이트 | 인간 라벨 필요 | 보상 출처 | Self-Play |
|---|---|---|---|---|
| SELF-INSTRUCT | SFT | 시드만 | 없음 (필터링) | X |
| Evol-Instruct | SFT | 시드만 | LLM 평가 | X |
| STaR | SFT | 정답 레이블 | Ground truth | X |
| Quiet-STaR | RL (REINFORCE) | 없음 | 자기 likelihood | X |
| SPIN | DPO | SFT 데이터만 | 인간 분포 vs 자기 분포 | O |
| Self-Refine | 없음 (추론만) | 없음 | LLM 자기 비판 | X |
| Constitutional AI | SFT + RL | 원칙만 | AI 피드백 | X |
| RLAIF | RL (PPO) | 없음 | LLM 선호도 | X |
| DeepSeek-R1 / GRPO | RL | 없음 | 규칙 기반 검증기 | X |
| AlphaEvolve | 코드 돌연변이 | 없음 | 자동 측정 지표 | X |
| SEAL | gradient | 없음 | RL 자기 편집 신호 | X |

---

## 10. 주요 트렌드 (2024-2026)

1. **RLVR 우세** — 인간 라벨 없이 검증 가능한 보상만으로 SOTA 추론 달성 (DeepSeek-R1)
2. **Self-play vs Self-critique** — SPIN(과거 자신과 게임) vs Constitutional AI(자기 비판)
3. **진화 탐색 부활** — AlphaEvolve: LLM 코드 돌연변이로 수십 년 된 수학 알고리즘 개선
4. **커리큘럼 자동화** — 무엇을 학습할지도 모델이 스스로 결정 (SEC)
5. **가중치 자기 수정** — SEAL: 런타임 파라미터 업데이트로 진정한 재귀적 자기 개선

---

## 참고 자료

- [SELF-INSTRUCT](https://arxiv.org/abs/2212.10560) | [Evol-Instruct](https://arxiv.org/abs/2304.12244)
- [STaR](https://arxiv.org/abs/2203.14465) | [Quiet-STaR](https://arxiv.org/abs/2403.09629)
- [SPIN](https://arxiv.org/abs/2401.01335) | [Self-Refine](https://arxiv.org/abs/2303.17651)
- [Constitutional AI](https://arxiv.org/abs/2212.08073) | [RLAIF](https://arxiv.org/abs/2309.00267)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [AlphaEvolve](https://arxiv.org/abs/2506.13131)
- [Survey of Self-Evolving Agents](https://arxiv.org/abs/2507.21046)
