from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """LLM 호출 인터페이스. Evaluator/Synthesizer 어디서든 동일하게 주입해서 사용."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """프롬프트를 받아 응답 텍스트를 반환한다."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
