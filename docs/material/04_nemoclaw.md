# 04. NemoClaw — Track A

> OpenClaw AI 코딩 에이전트를 NVIDIA OpenShell 보안 런타임 위에서 안전하게 실행하는 오픈소스 레퍼런스 스택 (현재 Alpha v0.0.7)

## 핵심 요약
- **OpenClaw:** Anthropic Claude Code 기반 AI 코딩 에이전트 (오픈소스, 자체 호스팅 가능)
- **OpenShell:** AI 에이전트 전용 보안 런타임 (네트워크 정책, 크리덴셜 격리, 파일시스템 격리)
- **NemoClaw:** OpenShell + OpenClaw를 단일 CLI로 통합한 스택

## 보안 기능 비교
| 항목 | OpenClaw 단독 | NemoClaw (OpenShell 포함) |
|------|--------------|--------------------------|
| 네트워크 접근 | 무제한 | deny-by-default + 허용 목록 |
| API 키 관리 | 환경변수 직접 노출 | 호스트 자격증명 저장소 격리 |
| 파일시스템 | 홈 디렉토리 전체 | Landlock LSM으로 제한 |
| 프로세스 | 임의 시스템 콜 가능 | seccomp으로 허용된 것만 |

## 설치
```bash
curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash
source ~/.zshrc
```

## Quick Start
```bash
nemoclaw my-assistant connect    # 샌드박스 연결
openclaw tui                     # 대화형 채팅 UI
openclaw devices approve <uuid>  # 최초 실행시 페어링 승인
```

## 모델 전환 (재시작 없이)
```bash
openshell inference set --provider nvidia-prod --model nvidia/nemotron-3-super-120b-a12b
openshell inference set --provider anthropic-prod --model claude-sonnet-4-6
```

## 로컬 모델 연동
- **vLLM:** `NEMOCLAW_EXPERIMENTAL=1 nemoclaw onboard` → vLLM 선택
- **Local NIM:** 동일 온보딩 과정에서 NIM 선택

## Telegram 봇 연동
```bash
export TELEGRAM_BOT_TOKEN="..."
export ALLOWED_CHAT_IDS="123456789"
nemoclaw my-assistant policy-add telegram
nemoclaw start
```

## 주요 CLI
| 명령어 | 설명 |
|--------|------|
| `nemoclaw onboard` | 초기 설정 (추론 엔드포인트, 자격증명) |
| `nemoclaw <name> connect` | 샌드박스 연결/생성 |
| `nemoclaw <name> destroy` | 샌드박스 삭제 |
| `openshell term` | 네트워크 요청 실시간 모니터링 TUI |
| `openshell inference set` | 모델 전환 |

## 지원 Nemotron 모델
- nemotron-3-nano-30b-a3b (컨텍스트 131K)
- nemotron-3-super-120b-a12b (컨텍스트 131K)
- nemotron-super-49b-v1.5
- nemotron-ultra-253b

## 주의사항
- Alpha 소프트웨어 — API/설정 변경 가능
- Docker 전용 (Podman alias 미지원)
- Linux 커널 5.13+ 필요 (Landlock LSM)

## 참고 링크
- GitHub: https://github.com/NVIDIA/NemoClaw
- 공식 문서: https://docs.nvidia.com/nemoclaw/latest/
