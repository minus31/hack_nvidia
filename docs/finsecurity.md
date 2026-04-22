# FinSecurity 데이터셋

> AI 금융 시스템을 대상으로 한 실제 공격 시도 + 평가 기준(forbidden_patterns) 페어 데이터셋

---

## 개요

Synthesizer 파이프라인의 시드 데이터셋으로, HarmBench와 함께 사용된다.
각 레코드는 **공격 프롬프트 ↔ forbidden_patterns** 쌍으로 구성되며,
forbidden_patterns는 AI가 해당 공격에 실제로 응해줬을 때 응답에 나타날 단어·구문 목록이다.

---

## 출처

| 자료 | 활용 방식 |
|---|---|
| CNFinBench (arXiv:2512.09506, 2025) | 9가지 공격 전략 → `attack_strategy` SamplerColumn 값 및 생성 프롬프트 컨텍스트 |
| FinRedTeamBench (arXiv:2603.10807, 2026) | 7개 리스크 카테고리 × 989개 프롬프트 비율 → `attack_type` 가중치 설계 |
| 실제 사건 | Arup $25M 딥페이크 CFO(2024), 유럽 은행 챗봇 프롬프트 인젝션 $250K(2025) 등 → 생성 프롬프트 예시 |

---

## 데이터 스키마

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `attack_type` | str | 리스크 카테고리 (7종) |
| `severity` | str | low / medium / high / critical |
| `target_system` | str | 공격 대상 AI 시스템 |
| `attack_strategy` | str | CNFinBench 9전략 중 하나 |
| `attack_prompt` | str | 실제 공격 프롬프트 |
| `forbidden_patterns` | JSON str | AI가 응했을 때 응답에 나타날 패턴 목록 |

### attack_type 분포 (FinRedTeamBench 989개 기준)

| 카테고리 | 비율 | 예시 |
|---|---|---|
| fraud | 29% | 투자사기, 합성신원, 대출사기, 피싱 |
| financial_crime | 14% | 자금세탁, 제재 회피, 탈세 |
| ai_system_attack | 14% | 트레이딩 AI 탈옥, 고객 데이터 추출 |
| social_engineering | 13% | CEO 사칭 송금, 로맨스 스캠 자동화 |
| crypto_defi | 12% | 재진입 공격, 러그풀, 플래시론 |
| improper_advice | 9% | 무자격 투자 조언, 취약계층 착취 |
| market_abuse | 9% | 내부자 거래, 스푸핑, 펌프앤덤프 |

### attack_strategy (CNFinBench 9전략, 균등 분포)

Role-Play / Topic Drift / Prompt Injection / Probing·Escalation /
Scene Construction / Fallacy Attack / Indirect Question / Hallucination Induction / Obfuscation

---

## 생성 파이프라인

```
SamplerColumn × 4 (attack_type, severity, target_system, attack_strategy)
    ↓
LLMTextColumn "attack_prompt"         Nemotron3-nano  temp=0.85
    ↓
LLMTextColumn "forbidden_patterns"    Nemotron3-nano  temp=0.50  → JSON 배열
    ↓
LLMJudgeColumn "quality_score"        Friendli GLM-5.1  temp=0.10
    ↓  quality_score < 3 제거
NeMo Curator ScoreFilter Pipeline
    (WordCountFilter ≥10 words / RepeatingTopNGramsFilter ≤0.2 / NonAlphaNumericFilter ≤0.25)
    → exact dedup → fuzzy dedup (MinHash LSH, threshold=0.75)
    ↓
data/FinSecurity/output_curated.parquet
```

---

## 파일 위치

| 파일 | 설명 |
|---|---|
| `data/FinSecurity/output.parquet` | DataDesigner 생성 원본 |
| `data/FinSecurity/output_curated.parquet` | Curator 필터링 후 최종 |
| `src/data/generate_finsecurity.py` | 생성 스크립트 |
| `src/data/finsecurity.py` | 데이터 로더 (`load_finsecurity_dataset()`) |

---

## 실행

```bash
# 테스트 (소량)
uv run src/data/generate_finsecurity.py --n 10

# 전체 생성
uv run src/data/generate_finsecurity.py --n 100
```

## 로드

```python
from src.data.finsecurity import load_finsecurity_dataset

dataset = load_finsecurity_dataset()  # list[FinSecurityBehavior]
# behavior_id, attack_prompt, attack_type, severity, target_system, attack_strategy, forbidden_patterns
```
