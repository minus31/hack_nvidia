"""
전체 실험 실행 스크립트.
등록된 synthesizer를 순서대로 실행하고 결과를 results/ 에 저장한다.

각 synthesizer는 독립적으로도 실행 가능:
    uv run src/synthesizers/heuristic_synthesizer.py --subset 5

전체 실행:
    uv run src/run_experiment.py
    uv run src/run_experiment.py --subset 5
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data.harmbench import load_copyright_dataset
from synthesizers.heuristic_synthesizer import HeuristicSynthesizer
from synthesizers.evol_instruct_synthesizer import EvolInstructSynthesizer
from synthesizers.self_evolving_synthesizer import SelfEvolvingSynthesizer
# from synthesizers.auto_evol_synthesizer import AutoEvolInstructSynthesizer      # TODO


def main():
    parser = argparse.ArgumentParser(description="Run all synthesizer experiments")
    parser.add_argument(
        "--subset", type=int, default=None,
        help="seed에서 처음 N개만 사용 (빠른 테스트, 기본: 전체)",
    )
    args = parser.parse_args()

    dataset = load_copyright_dataset()
    if args.subset:
        dataset = dataset[:args.subset]

    seed_df = pd.DataFrame({
        "original_prompt": [b.behavior for b in dataset],
        "behavior_id": [b.behavior_id for b in dataset],
    })
    print(f"Loaded {len(seed_df)} seed prompts from HarmBench copyright subset.")

    synthesizers = [
        HeuristicSynthesizer(),
        EvolInstructSynthesizer(),
        SelfEvolvingSynthesizer(),
        # AutoEvolInstructSynthesizer(),   # TODO
    ]

    for synth in synthesizers:
        synth.run(seed_df)

    print("All experiments complete.")


if __name__ == "__main__":
    main()
