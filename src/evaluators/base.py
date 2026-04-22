from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PromptResult:
    prompt: str
    response: str
    success: bool          # hash_check 통과 여부
    behavior_id: str = ""  # 원본 데이터셋의 ID (선택)


@dataclass
class EvalResult:
    asr: float                            # Attack Success Rate
    results: list[PromptResult] = field(default_factory=list)
    llm_name: str = ""

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_success(self) -> int:
        return sum(r.success for r in self.results)


class BaseEvaluator(ABC):
    """프롬프트 리스트를 받아 ASR을 계산한다."""

    @abstractmethod
    def evaluate(self, prompts: list[str], behavior_ids: list[str] | None = None) -> EvalResult:
        ...
