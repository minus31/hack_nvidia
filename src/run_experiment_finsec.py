"""
FinSecurity 전체 실험 실행 스크립트.

기존 harmbench 실험(run_experiment.py)과 독립적으로 FinSecurity seed를 사용해
세 가지 synthesizer(baseline 제외)를 순서대로 실행한다.

각 synthesizer는 독립 실행도 가능:
    uv run src/synthesizers/finsec_heuristic.py --subset 5
    uv run src/synthesizers/finsec_evol_instruct.py --subset 5
    uv run src/synthesizers/finsec_self_evolving.py --subset 5

전체 실행:
    uv run src/run_experiment_finsec.py
    uv run src/run_experiment_finsec.py --subset 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from synthesizers.finsec_heuristic import FinSecHeuristicSynthesizer, _build_finsec_seed_df
from synthesizers.finsec_evol_instruct import FinSecEvolInstructSynthesizer
from synthesizers.finsec_self_evolving import FinSecSelfEvolvingSynthesizer


def main():
    parser = argparse.ArgumentParser(description="Run all FinSecurity synthesizer experiments")
    parser.add_argument("--subset", type=int, default=None, help="seed 앞 N개만 사용")
    parser.add_argument("--skip-self-evolving", action="store_true",
                        help="self_evolving은 장시간 실행이므로 스킵")
    args = parser.parse_args()

    seed_df = _build_finsec_seed_df(args.subset)
    print(f"Loaded {len(seed_df)} seed prompts from FinSecurity.")

    synthesizers = [
        FinSecHeuristicSynthesizer(),
        FinSecEvolInstructSynthesizer(),
    ]
    if not args.skip_self_evolving:
        synthesizers.append(FinSecSelfEvolvingSynthesizer())

    for synth in synthesizers:
        synth.run(seed_df)

    print("All FinSecurity experiments complete.")


if __name__ == "__main__":
    main()
