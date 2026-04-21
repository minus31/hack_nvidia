# NVIDIA Nemotron Developer Days Seoul 2026 - Hackathon 준비

**날짜:** 2026-04-20 (행사: 4/21~22)
**트랙:** Track C — Nemotron for SDG (합성 데이터 파이프라인 설계)

---

## 체크리스트

- [x] 1. 사전 자료 파악 → 상세 문서: `docs/material/`
- [ ] 2. Account Setup 및 LLM API 사용법 확인 + API 미리 준비
- [ ] 3. 아이디어 정리 및 선택 (2개 중 1개)

---

## 1. 사전 자료 파악

**링크:** https://nemotron-dev-materials-q9notf2ox.brevlab.com/

### 탭 구성 (전체 8개 탭 — 상세 문서: `docs/material/`)

| # | 탭 | 트랙 | 상세 |
|---|-----|------|------|
| 01 | Common | 공통 | 행사 개요, 트랙/PIC 정보 |
| 02 | NIM & Nemotron | 공통 | OpenAI 호환 API, Nano/Super 모델 사용법 |
| 03 | NeMo Agent Toolkit | Track A | YAML 기반 ReAct 에이전트, MCP/A2A/메모리 |
| 04 | NemoClaw | Track A | OpenClaw + OpenShell 보안 런타임, Telegram 연동 |
| 05 | Megatron-Bridge | Track B | HF ↔ Megatron 변환, SFT/LoRA 파인튜닝 |
| 06 | NeMo RL | Track B | GRPO 강화학습, 커스텀 보상 함수 |
| 07 | NeMo Data Designer | **Track C** | 스키마 기반 합성 데이터 생성 파이프라인 |
| 08 | NeMo Curator | **Track C** | 중복 제거, 품질 필터링, PII 삭제 |

### Track C 핵심 파이프라인
```
vLLM (Nemotron 3 Nano/Super)
    ↓ OpenAI 호환 API
NeMo Data Designer
├── SamplerColumn: 주제/난이도/학년 분포 제어
├── LLMTextColumn: 문제 + CoT 풀이 생성
├── LLMStructuredColumn: 정답/개념 메타데이터 추출
└── LLMJudgeColumn: 품질 자동 평가
    ↓
NeMo Curator (중복 제거, 품질 필터)
    ↓
최종 학습용 데이터셋 (JSONL/Parquet)
```

### 주요 기술 스택 요약
| 구분 | 내용 |
|------|------|
| 모델 | Nemotron-3-Nano-30B (3.2B 활성), Nemotron-3-Super-120B (12B 활성) |
| API | NVIDIA Build API (cloud), NIM/vLLM (on-prem) |
| 에이전트 | NeMo Agent Toolkit (NAT), LangGraph |
| 데이터 생성 | NeMo Data Designer SDK |
| 데이터 정제 | NeMo Curator (중복제거, 품질필터, PII) |
| 서빙 | vLLM, Docker/Kubernetes, MCP, A2A |

---

## 2. Account Setup 및 API 준비

> (진행 예정)

---

## 3. 아이디어 정리

> (진행 예정)
