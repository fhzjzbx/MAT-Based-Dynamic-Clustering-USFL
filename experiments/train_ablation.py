from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_pcsfl import train

VARIANTS = {
    "no_phase_balance": dict(phase_balanced_replay=False),
    "legacy_reward": dict(reward_mode="legacy"),
    "random_channel_reward": dict(channel_mode="random"),
    "slack1": dict(capacity_slack=1),
    "separate_q": dict(use_joint_q=False),
}


def run_variant(name: str, episodes: int, rounds: int, seed: int, device: str) -> None:
    if name not in VARIANTS:
        raise ValueError(f"unknown variant: {name}")
    kwargs = dict(VARIANTS[name])
    train(
        episodes=episodes,
        rounds=rounds,
        seed=seed,
        checkpoint_path=f"checkpoints/ablation_{name}.pt",
        log_path=f"logs/pcsfl_ablation/{name}/training_history.json",
        device=device,
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PCSFL ablation variants")
    parser.add_argument("--variant", choices=[*VARIANTS.keys(), "all"], required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    names = list(VARIANTS) if args.variant == "all" else [args.variant]
    for name in names:
        print(f"\n=== training ablation: {name} ===")
        run_variant(name, args.episodes, args.rounds, args.seed, args.device)


if __name__ == "__main__":
    main()
