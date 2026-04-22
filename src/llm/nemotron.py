import os
from openai import OpenAI
from .base import BaseLLM

CLOUD_BASE_URL = "https://integrate.api.nvidia.com/v1"
LOCAL_BASE_URL = "http://0.0.0.0:8000/v1"


class NemotronLLM(BaseLLM):
    """
    Nemotron — OpenAI 호환 클라이언트 사용.

    On-prem (NIM 서버 실행 중일 때):
        NemotronLLM()
    Cloud (build.nvidia.com, NVIDIA_API_KEY 필요):
        NemotronLLM(use_cloud=True)
    """

    def __init__(
        self,
        model: str = "nvidia/nemotron-3-nano",
        max_tokens: int = 8192,
        use_cloud: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._model = model
        self._max_tokens = max_tokens

        if base_url:
            _base_url = base_url
            _api_key = api_key or "no-key"
        elif use_cloud:
            _base_url = CLOUD_BASE_URL
            _api_key = api_key or os.environ["NVIDIA_API_KEY"]
        else:
            _base_url = LOCAL_BASE_URL
            _api_key = "no-key"

        self._client = OpenAI(base_url=_base_url, api_key=_api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content
