from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from baselines.pcsfl import PCSFLAgent, TensorProfileCache, build_agent_state, evaluate_actions
from data.cifar_10_provider import CIFAR10NonIIDProvider
from envs.liquid_airan_env import LiquidAIRANEnv
from models.usfl_networks import ResNet18_USFL


def train(
    *,
    episodes: int = 10,
    rounds: int = 150,
    seed: int = 42,
    checkpoint_path: str = "checkpoints/pcsfl_trained.pt",
    log_path: str = "logs/pcsfl/training_history.json",
    capacity_slack: int = 0,
    phase_balanced_replay: bool = True,
    use_joint_q: bool = True,
    reward_mode: str = "full",
    channel_mode: str = "state",
    device: str = "cpu",
) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    provider = CIFAR10NonIIDProvider(num_clients=25, alpha=0.7, data_dir="./data")
    model = ResNet18_USFL(num_classes=10).to(torch.device(device))
    model.eval()
    tensor_cache = TensorProfileCache()
    agent = PCSFLAgent(
        state_dim=15,
        max_clients=20,
        max_migs=7,
        num_layers=7,
        hidden_dim=256,
        gamma=0.0,
        learning_rate=1e-4,
        replay_capacity=20_000,
        batch_size=48,
        target_sync_interval=50,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997,
        capacity_slack=capacity_slack,
        phase_balanced_replay=phase_balanced_replay,
        use_joint_q=use_joint_q,
        device=device,
        seed=seed,
    )

    history = []
    global_step = 0
    random_channel_rng = np.random.default_rng(seed + 100_000)

    for episode in range(1, episodes + 1):
        episode_seed = seed + episode - 1
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        env = LiquidAIRANEnv(data_provider=provider, max_vehicles=25)

        for epoch in range(1, rounds + 1):
            global_step += 1
            raw, migs, bandwidth, _ = env.step()
            state = build_agent_state(raw, migs, bandwidth)
            clusters, l1, l2, _ = agent.step(state, migs)
            metrics = evaluate_actions(
                raw_client_states=raw,
                available_migs=migs,
                current_bandwidth=bandwidth,
                cluster_choices=clusters,
                l1_choices=l1,
                l2_choices=l2,
                resnet=model,
                tensor_cache=tensor_cache,
                reward_mode=reward_mode,
                channel_mode=channel_mode,
                rng=random_channel_rng,
            )
            # The project is a one-step exogenous resource-allocation problem; next state does not bootstrap.
            loss = agent.observe(
                reward=metrics["reward"],
                next_state=state,
                next_available_migs=migs,
                done=True,
            )
            row = {
                "global_step": global_step,
                "episode": episode,
                "epoch": epoch,
                "seed": episode_seed,
                "n_clients": int(raw.shape[0]),
                "n_migs": int(migs),
                "bandwidth_mhz": float(bandwidth / 1e6),
                **metrics,
                "loss": loss,
                "epsilon": float(agent.epsilon),
                "buffer_size": len(agent.replay_buffer),
                "gradient_steps": int(agent.gradient_steps),
            }
            history.append(row)
            if epoch % 25 == 0:
                loss_text = "buffering" if loss is None else f"{loss:.6f}"
                print(
                    f"episode={episode:02d} round={epoch:03d} "
                    f"full={metrics['full_total_delay_ms']:.3f} ms "
                    f"conflict={metrics['conflict_rate']:.1%} "
                    f"imbalance={metrics['load_imbalance']:.4f} "
                    f"loss={loss_text} epsilon={agent.epsilon:.4f}"
                )

    agent.save(checkpoint_path)
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"checkpoint: {checkpoint_path}")
    print(f"training log: {log_path}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the final PCSFL policy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(episodes=args.episodes, rounds=args.rounds, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
