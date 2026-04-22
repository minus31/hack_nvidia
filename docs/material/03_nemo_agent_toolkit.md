# 03. NeMo Agent Toolkit (NAT) — Track A

> ReAct 에이전트를 YAML 설정 파일 하나로 빌드, 배포하는 NVIDIA 툴킷

## 핵심 요약
- 패키지: `nvidia-nat` (v1.5.0)
- YAML 설정 파일만으로 에이전트 정의 → CLI/REST API 서버로 즉시 배포
- MCP, A2A, 메모리(Mem0) 등 고급 기능 지원

## 설치
```bash
uv pip install "nvidia-nat[langchain]"        # 기본 + 내장 도구
uv pip install "nvidia-nat[mcp]"              # MCP Client/Server
uv pip install "nvidia-nat[a2a]"              # A2A 프로토콜
uv pip install "nvidia-nat[mem0ai]"           # Mem0 메모리
export NVIDIA_API_KEY=<nvapi-...>
```

## YAML 설정 구조
```yaml
llms:
  nim_llm:
    _type: nim
    model_name: nvidia/nemotron-3-nano-30b-a3b
    temperature: 0.0

functions:
  wikipedia_search:
    _type: wiki_search
    max_results: 3
  code_generator:
    _type: code_generation
    programming_language: "Python"
    llm_name: nim_llm

workflow:
  _type: react_agent
  tool_names: [wikipedia_search, code_generator]
  llm_name: nim_llm
  verbose: true
```

## CLI 명령어
| 명령어 | 설명 |
|--------|------|
| `nat run --config_file config.yml --input "질문"` | 단일 질의 실행 |
| `nat serve --config_file config.yml` | REST API 서버 시작 (포트 8000) |
| `nat eval --config_file config.yml` | RAGAS 메트릭으로 자동 평가 |
| `nat mcp` | MCP 서버로 노출 |

## REST API 엔드포인트
| 엔드포인트 | 설명 |
|-----------|------|
| `POST /generate` | 동기 응답 |
| `POST /generate/stream` | SSE 스트리밍 응답 |
| `GET /health` | 서버 상태 |

## 지원 기능
### Basic Example
- Wikipedia 검색 + 코드 생성 ReAct 에이전트

### Advanced Example #1
- **Custom Function:** Python으로 커스텀 도구 구현 및 등록
- **Memory (Mem0):** `auto_memory_agent`로 대화 기록 자동 저장/검색
- **MCP Client:** 외부 MCP 서버 도구를 에이전트에 연결
- **A2A + MCP Server:** 완성된 워크플로우를 A2A/MCP로 노출

### Advanced Example #2
- LangGraph 워크플로우를 NAT로 래핑
- Multi-agent supervisor 패턴

## 지원 LLM 공급자
OpenAI, Azure, Anthropic, Google, Ollama, AWS Bedrock, NVIDIA NIM

## 참고 링크
- 공식 문서: https://docs.nvidia.com/nemo/agent-toolkit/latest/
- GitHub: https://github.com/NVIDIA/NeMo-Agent-Toolkit
