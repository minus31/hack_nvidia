import os
from openai import OpenAI
from .base import BaseLLM

FRIENDLI_BASE_URL = "https://api.friendli.ai/serverless/v1"


class FriendliLLM(BaseLLM):
    """Friendli AI serverless endpoint — OpenAI 호환 클라이언트 사용."""

    def __init__(
        self,
        model: str = "zai-org/GLM-5.1",
        max_tokens: int = 8192,
        api_key: str | None = None,
        enable_thinking: bool = False,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking
        self._client = OpenAI(
            base_url=FRIENDLI_BASE_URL,
            api_key=api_key or os.environ["API_KEY"],
        )

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = dict(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
        )
        if self._enable_thinking:
            kwargs["extra_body"] = {
                "parse_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": True},
            }

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        return msg.content or getattr(msg, "reasoning_content", None) or ""
