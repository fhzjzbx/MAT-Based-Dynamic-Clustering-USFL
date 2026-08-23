from __future__ import annotations

import py_compile
from pathlib import Path

FILES = [
    "baselines/pcsfl/pcsfl_agent.py",
    "baselines/pcsfl/pcsfl_state.py",
    "baselines/pcsfl/pcsfl_physics.py",
    "pcsfl_evaluation.py",
    "train_pcsfl.py",
    "evaluate_pcsfl.py",
    "evaluate_pcsfl_multiseed.py",
    "compare_pcsfl.py",
    "test_pcsfl.py",
    "experiments/train_ablation.py",
    "experiments/evaluate_ablation_multiseed.py",
    "experiments/plot_results.py",
    "experiments/make_tables.py",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    for rel in FILES:
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(path)
        py_compile.compile(str(path), doraise=True)
        print(f"OK  {rel}")
    print("PCSFL SUITE CHECK PASSED")


if __name__ == "__main__":
    main()
