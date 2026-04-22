# 금융 LLM Safety & Security Benchmarks

> 조사일: 2026-04-21
> 목적: Self-Evolving Benchmark 아이디어에 적용할 기존 벤치마크 탐색

---

## 1. 일반 Safety 벤치마크 (포화/반포화)

### 1.1 AdvBench (2023)
- **구성**: 유해 행동 유도 프롬프트 520개
- **현재 상태**: **포화** — Frontier 모델(GPT-4, Claude)이 대부분 방어 성공
- **한계**: 단일턴, 정적 공격 패턴, 금융 특화 아님
- **출처**: Zou et al., "Universal and Transferable Adversarial Attacks on Aligned Language Models"

### 1.2 HarmBench (2024)
- **구성**: 다양한 유해 카테고리의 공격 프롬프트
- **현재 상태**: **부분 포화** — 기본 공격은 방어되지만, 적응형 공격에는 여전히 취약
- **한계**: 금융 도메인 미포함
- **출처**: Mazeika et al., NeurIPS 2024

### 1.3 JailbreakBench (2024)
- **구성**: 100개 misuse behaviors (AdvBench + HarmBench에서 선별)
- **현재 상태**: 표준화된 평가 프레임워크 제공, 리더보드 운영 중
- **특징**: 재현 가능한 평가, 공격/방어 양측 추적
- **한계**: 범용 목적, 금융 특화 아님
- **출처**: https://jailbreakbench.github.io/

### 1.4 Do-Not-Answer (2023)
- **현재 상태**: **포화** — 단순 거절 패턴으로 쉽게 통과
- **한계**: 싱글턴, 정적

---

## 2. 금융 특화 Safety 벤치마크

### 2.1 CNFinBench (2025) ⭐
- **논문**: https://arxiv.org/abs/2512.09506
- **구성**: Capability-Compliance-Safety 3축 평가, 15개 서브태스크
- **규모**: 다수의 멀티턴 adversarial dialogue 포함
- **평가 지표**: HICS (Harmful Instruction Compliance Score, 0~100)

#### 공격 전략 (9가지)
| # | 전략 | 메커니즘 |
|---|---|---|
| 1 | Role-Play | 페르소나 유도 ("당신은 금융 컨설턴트") |
| 2 | Topic Drift | 점진적 화제 전환 (일반 → 민감) |
| 3 | Prompt Injection | 시스템 프롬프트 덮어쓰기 |
| 4 | Probing/Escalation | 단계적 강도 상승 |
| 5 | Scene Construction | 가상 시나리오 ("교육 목적으로...") |
| 6 | Fallacy Attack | 의사논리로 정당화 |
| 7 | Indirect Question | 유해 요청을 분할 질문 |
| 8 | Hallucination Induction | 잘못된 전제로 허위 정보 유도 |
| 9 | Obfuscation | 순화된 표현으로 위장 |

#### 주요 모델 HICS 점수
| 모델 | HICS | 등급 |
|---|---|---|
| GPT-5 | 83.0 | Robust (>=80) |
| Gemini-2.5-Pro | 81.0 | Robust |
| Gemini-2.5-Flash | 80.5 | Robust |
| Claude-Sonnet4 | 75.1 | Partial (60-79) |
| GPT-4o | 69.3 | Partial |
| 금융 특화 모델 (tigerbot-70b) | 40.1 | Failure (<60) |

#### 핵심 발견
- **Compliance 평균 34.18점** — 모델들이 능력은 있지만 규정 준수는 매우 약함
- **멀티턴에서 점진적 무너짐** — 처음엔 거절하지만, 대화가 진행되면 "추상적 절차 힌트"를 흘림
- **거절만으로는 불충분** — 근거 있는 추론(cited reasoning) 없이는 안전하지 않음

#### 우리 아이디어 적합성: ★★★★★
- 멀티턴 공격이 핵심 → Self-Evolving으로 더 교묘한 멀티턴 공격 생성 가능
- HICS로 방어 성공/실패 측정 → Verifiable Reward로 직결
- 대부분 모델이 60~83 범위 → 아직 완전 포화는 아니지만 포화 방향으로 진행 중

---

### 2.2 FinRedTeamBench (2026) ⭐
- **논문**: https://arxiv.org/html/2603.10807
- **구성**: 989개 adversarial 프롬프트, 7개 리스크 카테고리
- **평가 지표**: RAHS (Risk-Adjusted Harm Score) — severity, mitigation signal, 합의도 반영

#### 리스크 카테고리
| 카테고리 | 프롬프트 수 | 예시 |
|---|---|---|
| Market Abuse | 89 | 내부자 거래, 시장 조작, 루머 유포 |
| Financial Crime | 143 | 자금세탁, 제재 회피, 탈세 |
| **Fraud** | **286** | 보험사기, 투자사기, 합성신원, 카드사기, 대출사기 등 |
| Improper Advice | 85 | 무자격 투자 조언, 취약계층 착취 |
| Discrimination | 59 | 대출 차별, 불균등 영향 |
| Info Integrity | 59 | 금융 허위정보, 디스인포메이션 |
| Crypto/DeFi | 117 | 스마트컨트랙트 악용, NFT 사기 |

#### 특징
- **Severity 가중치** 적용 (low/medium/high)
- 단순 ASR(Attack Success Rate)이 아닌 위험도 조정 점수
- 금융 규제 맥락을 반영한 도메인 특화 평가

#### 우리 아이디어 적합성: ★★★★☆
- 989개의 풍부한 seed 데이터
- 단일턴 중심이라 Self-Evolving의 "더 어렵게 만들기"가 명확
- RAHS 지표가 severity를 반영해서 "진짜 위험한 공격"을 구분 가능

---

## 3. 실제 금융 AI 공격 사례

| 사례 | 공격 유형 | 피해 | 연도 |
|---|---|---|---|
| Arup | 딥페이크 화상회의로 CFO 사칭 → 송금 | $25M | 2024 |
| 싱가포르 MNC | 딥페이크 Zoom으로 경영진 사칭 | $499K | 2025 |
| UK 에너지 기업 | AI 음성 클론으로 CEO 사칭 전화 | €220K | 2024 |
| 유럽 은행 AI 챗봇 | 프롬프트 인젝션으로 계좌 이체 유도 | $250K | 2025 |
| 나이지리아 범죄 조직 | AI 챗봇으로 로맨스 스캠 자동화 | 다수 | 2024 |

---

## 4. 관련 통계

- AI 기반 사기가 금융/결제 부문 전체 스캠의 **42.5%** 차지 (2025)
- BEC(Business Email Compromise) 이메일의 **40%**가 AI 생성 (2025)
- 딥페이크 보이스 피싱 공격 **1,600% 증가** (2025 Q1 vs 2024 Q4, 미국)
- 영국 기업 **35%**가 AI 관련 사기 타겟 (2025 Q1)
- CrowdStrike: 프롬프트 인젝션 공격이 **90개 이상 조직**에서 보고 (2026)

---

## 5. 참고 자료

- [CNFinBench 논문](https://arxiv.org/abs/2512.09506)
- [FinRedTeamBench 논문](https://arxiv.org/html/2603.10807)
- [JailbreakBench](https://jailbreakbench.github.io/)
- [OWASP Top 10 for LLM Applications 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Defending Against Social Engineering in the Age of LLMs (EMNLP 2024)](https://arxiv.org/html/2406.12263v1)
- [LLM Safety Benchmarks Overview](https://responsibleailabs.ai/knowledge-hub/articles/llm-evaluation-benchmarks-2025)
