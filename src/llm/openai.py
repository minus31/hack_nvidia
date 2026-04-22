import os
from openai import OpenAI
from .base import BaseLLM


class OpenAILLM(BaseLLM):
    """OpenAI API — gpt-4o."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        api_key: str | None = None,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

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
