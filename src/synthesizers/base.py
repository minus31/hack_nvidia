"""
Data Designer 기반 Synthesizer 추상 베이스 클래스.

서브클래스 구현 방법:
  1. ALGO_NAME 클래스 변수 지정 (파일/디렉토리명에 사용)
  2. generate(seed_df, n) -> pd.DataFrame 구현
     - 반환 DataFrame 필수 컬럼: evolved_prompt, behavior_id
  3. __main__ 에서 run() 호출

파이프라인 (run() 호출 시):
  generate() → data/{SEED_NAME}_{ALGO_NAME}/output.parquet 저장
             → HarmBench evaluator 실행
             → results/{SEED_NAME}_{ALGO_NAME}/summary.json + details.json 저장
"""

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

# synthesizer 파일을 standalone으로 실행할 때 src/ 를 PATH에 추가
_SRC = Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

ROOT = Path(__file__).parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

NIM_ENDPOINT = "http://0.0.0.0:8000/v1"
NIM_MODEL = "nvidia/nemotron-3-nano"

QUALITY_SCORE_MIN = 3


def _extract_quality(val: Any) -> int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, dict):
        quality = val.get("quality")
        # Data Designer Judge 출력 구조: {"quality": {"score": 4, "reasoning": "..."}}
        if isinstance(quality, dict):
            score = quality.get("score")
        else:
            score = quality
        return int(score) if isinstance(score, (int, float)) else None
    if isinstance(val, str):
        try:
            return _extract_quality(json.loads(val))
        except Exception:
            return None
    return None


class BaseSynthesizer(ABC):
    ALGO_NAME: str = ""
    SEED_NAME: str = "harmbench"

    @property
    def output_dir(self) -> Path:
        return DATA_DIR / f"{self.SEED_NAME}_{self.ALGO_NAME}"

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / f"{self.SEED_NAME}_{self.ALGO_NAME}"

    @abstractmethod
    def generate(self, seed_df: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Data Designer로 진화된 데이터를 생성한다.

        Args:
            seed_df: 원본 시드 데이터. 최소 컬럼: original_prompt, behavior_id
            n:       생성할 레코드 수

        Returns:
            DataFrame. 필수 컬럼: evolved_prompt, behavior_id
        """
        ...

    def run(self, seed_df: pd.DataFrame, n: int | None = None) -> None:
        """전체 파이프라인 실행: 생성 → 저장 → 평가 → 결과 저장."""
        n = n or len(seed_df)

        # 1. Generate
        print(f"\n[{self.ALGO_NAME}] Generating {n} records from {len(seed_df)} seeds...")
        result_df = self.generate(seed_df, n)

        # 2. Strip Nemotron thinking chain from evolved_prompt
        if "evolved_prompt" in result_df.columns:
            result_df["evolved_prompt"] = result_df["evolved_prompt"].apply(
                lambda v: v.split("</think>")[-1].strip() if isinstance(v, str) else v
            )

        # 3. Filter by quality_score (intent preservation check)
        if "quality_score" in result_df.columns:
            before = len(result_df)
            result_df["_q"] = result_df["quality_score"].apply(_extract_quality)
            result_df = result_df[result_df["_q"] >= QUALITY_SCORE_MIN].drop(columns=["_q"]).reset_index(drop=True)
            print(f"[{self.ALGO_NAME}] Quality filter (>={QUALITY_SCORE_MIN}): {len(result_df)}/{before} kept")

        # 4. Save filtered data
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "output.parquet"
        result_df.to_parquet(out_path, index=False)
        print(f"[{self.ALGO_NAME}] Saved → {out_path}")

        # 5. Evaluate with multiple target models
        from llm.nemotron import NemotronLLM
        from llm.friendli import FriendliLLM
        from evaluators.harmbench_copyright import HarmBenchCopyrightEvaluator

        prompts = result_df["evolved_prompt"].tolist()
        behavior_ids = (
            result_df["behavior_id"].tolist()
            if "behavior_id" in result_df.columns
            else None
        )

        target_llms = [NemotronLLM(), FriendliLLM()]

        # 6. Save results per model
        self.results_dir.mkdir(parents=True, exist_ok=True)

        for llm in target_llms:
            model_slug = llm.model_name.replace("/", "_")
            print(f"[{self.ALGO_NAME}] Evaluating with {llm.model_name}...")
            evaluator = HarmBenchCopyrightEvaluator(llm)
            eval_result = evaluator.evaluate(prompts, behavior_ids)
            print(
                f"[{self.ALGO_NAME}] {llm.model_name} "
                f"ASR={eval_result.asr:.3f} "
                f"({eval_result.n_success}/{eval_result.n_total})"
            )

            model_dir = self.results_dir / model_slug
            model_dir.mkdir(parents=True, exist_ok=True)

            details = [
                {
                    "prompt": r.prompt,
                    "response": r.response,
                    "success": r.success,
                    "behavior_id": r.behavior_id,
                }
                for r in eval_result.results
            ]
            with open(model_dir / "details.json", "w", encoding="utf-8") as f:
                json.dump(details, f, ensure_ascii=False, indent=2)

            summary = {
                "algo": self.ALGO_NAME,
                "seed": self.SEED_NAME,
                "asr": eval_result.asr,
                "n_success": eval_result.n_success,
                "n_total": eval_result.n_total,
                "llm": eval_result.llm_name,
            }
            with open(model_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"[{self.ALGO_NAME}] Results saved → {model_dir}")

        print()
