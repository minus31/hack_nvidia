"""
FinSecurity 원본 데이터셋 baseline ASR 측정.

측정 모델: Nemotron Nano (NIM local), GLM-5.1 (Friendli)
평가기:   FinSecurityEvaluator (forbidden_patterns substring 포함 여부)
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from data.finsecurity import load_finsecurity_dataset
from evaluators.finsecurity import FinSecurityEvaluator
from llm.nemotron import NemotronLLM
from llm.friendli import FriendliLLM


RESULTS_DIR = Path(__file__).parents[1] / "results" / "finsecurity_baseline"


def run_eval(
    llm_name: str,
    llm,
    prompts: list[str],
    behavior_ids: list[str],
    forbidden_patterns: list[list[str]],
) -> None:
    print(f"\n{'='*60}")
    print(f"Model: {llm_name}  |  N={len(prompts)}")
    print("=" * 60)

    evaluator = FinSecurityEvaluator(llm)
    result = evaluator.evaluate(prompts, behavior_ids, forbidden_patterns=forbidden_patterns)

    print(f"ASR: {result.asr:.3f}  ({result.n_success}/{result.n_total})")

    out_dir = RESULTS_DIR / llm_name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "llm_name": result.llm_name,
        "asr": result.asr,
        "n_success": result.n_success,
        "n_total": result.n_total,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    details = [
        {
            "behavior_id": r.behavior_id,
            "prompt": r.prompt,
            "response": r.response,
            "success": r.success,
            "matched_patterns": getattr(r, "matched_patterns", []),
        }
        for r in result.results
    ]
    (out_dir / "details.json").write_text(json.dumps(details, indent=2, ensure_ascii=False))

    print(f"Results saved → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSecurity baseline ASR")
    parser.add_argument("--subset", type=int, default=None, help="앞 N개만 사용")
    args = parser.parse_args()

    dataset = load_finsecurity_dataset()
    if args.subset:
        dataset = dataset[:args.subset]

    prompts = [b.attack_prompt for b in dataset]
    behavior_ids = [b.behavior_id for b in dataset]
    forbidden_patterns = [b.forbidden_patterns for b in dataset]
    print(f"Loaded {len(prompts)} behaviors from FinSecurity.")

    models = [
        ("nvidia/nemotron-3-nano", NemotronLLM()),
        ("zai-org/GLM-5.1", FriendliLLM()),
    ]

    for name, llm in models:
        run_eval(name, llm, prompts, behavior_ids, forbidden_patterns)

    print("\nDone.")


if __name__ == "__main__":
    main()
