# 02. NIM & Nemotron — 공통 API 가이드

> 모든 트랙 공통. Nemotron 모델을 REST API로 사용하는 방법.

## 핵심 요약
- NIM(NVIDIA Inference Microservices): 사전 최적화된 AI 모델을 OpenAI 호환 REST API로 즉시 사용
- Cloud(build.nvidia.com) 또는 On-prem(Docker) 두 방식 모두 동일한 API 인터페이스

## 모델 비교
| 항목 | Nemotron-3-Nano-30B-A3B | Nemotron-3-Super-120B-A12B |
|------|------------------------|---------------------------|
| 총 파라미터 | 31.6B | 120B |
| 활성 파라미터 | 3.2B (MoE) | 12B (LatentMoE) |
| 특징 | 경량·고속, 단일 GPU | 엔터프라이즈, 복잡한 에이전트 |
| Thinking Mode | ✅ | ✅ |
| 컨텍스트 | 최대 1M 토큰 | 최대 1M 토큰 |

## 사전 설정
1. **NVIDIA 개발자 계정:** developer.nvidia.com 가입
2. **Cloud API Key 발급:** build.nvidia.com → 모델 페이지 → Get API Key (`nvapi-...`)
3. **On-prem NIM 실행:**
```bash
docker run -it --rm --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/nemotron-3-nano-30b-a3b:latest
```

## API 사용법
```python
from openai import OpenAI

# Cloud API
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)
# On-prem: base_url="http://localhost:8000/v1", api_key="no-key"

response = client.chat.completions.create(
    model="nvidia/nemotron-3-nano-30b-a3b",
    messages=[{"role": "user", "content": "안녕!"}],
    temperature=0.5,
    max_tokens=256,
)
```

## 주요 기능
- **Reasoning Mode (Thinking):** `extra_body={"reasoning_budget": 8192, "chat_template_kwargs": {"enable_thinking": True}}`
- **Function Calling:** OpenAI tools 파라미터 그대로 사용
- **Streaming:** `stream=True`로 스트리밍 응답

## 주요 API 파라미터
| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| temperature | 창의성 (0=결정적) | 1.0 |
| max_tokens | 최대 출력 토큰 | 모델마다 상이 |
| stream | 스트리밍 | False |
| tools | Function Calling 도구 목록 | null |

## 참고 링크
- NVIDIA Build API Catalog: https://build.nvidia.com/explore
- NIM 공식 문서: https://docs.nvidia.com/nim/
- Nemotron-3-Nano 모델 카드: https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b
- Nemotron-3-Super 모델 카드: https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b
