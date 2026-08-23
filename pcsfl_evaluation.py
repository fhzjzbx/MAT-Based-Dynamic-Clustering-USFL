from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from baselines.pcsfl import PCSFLAgent, TensorProfileCache, build_agent_state, evaluate_actions
from data.cifar_10_provider import CIFAR10NonIIDProvider
from envs.liquid_airan_env import LiquidAIRANEnv
from models.usfl_networks import ResNet18_USFL


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_project_objects(seed: int, device: str | torch.device = "cpu"):
    set_seed(seed)
    provider = CIFAR10NonIIDProvider(num_clients=25, alpha=0.7, data_dir="./data")
    env = LiquidAIRANEnv(data_provider=provider, max_vehicles=25)
    model = ResNet18_USFL(num_classes=10).to(torch.device(device))
    model.eval()
    return provider, env, model


def generate_trace(seed: int, rounds: int = 150) -> list[dict]:
    _, env, _ = build_project_objects(seed, device="cpu")
    trace = []
    for epoch in range(1, rounds + 1):
        raw_state, available_migs, bandwidth, vehicle_ids = env.step()
        trace.append(
            {
                "epoch": epoch,
                "raw_state": np.asarray(raw_state, dtype=np.float32).copy(),
                "available_migs": int(available_migs),
                "bandwidth": float(bandwidth),
                "vehicle_ids": np.asarray(vehicle_ids).copy(),
            }
        )
    return trace


def make_agent(
    *,
    seed: int,
    device: str | torch.device = "cpu",
    checkpoint: str | Path | None = None,
    capacity_slack: int = 0,
    phase_balanced_replay: bool = True,
    use_joint_q: bool = True,
) -> PCSFLAgent:
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
    if checkpoint is not None:
        agent.load(checkpoint, load_optimizer=False)
    agent.set_training(False)
    return agent


def evaluate_trace(
    *,
    trace: Iterable[dict],
    model,
    agent: PCSFLAgent,
    reward_mode: str = "full",
    channel_mode: str = "state",
    random_channel_seed: int = 99173,
) -> list[dict]:
    cache = TensorProfileCache()
    rng = np.random.default_rng(random_channel_seed)
    detail = []
    for item in trace:
        raw = np.asarray(item["raw_state"], dtype=np.float32)
        migs = int(item["available_migs"])
        bandwidth = float(item["bandwidth"])
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
            tensor_cache=cache,
            reward_mode=reward_mode,
            channel_mode=channel_mode,
            rng=rng,
        )
        detail.append(
            {
                "epoch": int(item["epoch"]),
                "n_clients": int(raw.shape[0]),
                "n_migs": migs,
                "bandwidth_mhz": bandwidth / 1e6,
                **metrics,
            }
        )
    return detail


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def summarize_detail(detail: list[dict]) -> dict:
    def block(rows: list[dict]) -> dict:
        return {
            "mean_total_delay_ms": _mean(rows, "total_delay_ms"),
            "mean_full_total_delay_ms": _mean(rows, "full_total_delay_ms"),
            "mean_tx_delay_ms": _mean(rows, "tx_delay_ms"),
            "mean_uplink_delay_ms": _mean(rows, "uplink_delay_ms"),
            "mean_downlink_delay_ms": _mean(rows, "downlink_delay_ms"),
            "mean_comp_delay_ms": _mean(rows, "comp_delay_ms"),
            "mean_conflict_rate": _mean(rows, "conflict_rate"),
            "mean_load_imbalance": _mean(rows, "load_imbalance"),
            "mean_l1": _mean(rows, "mean_l1"),
            "mean_l2": _mean(rows, "mean_l2"),
            "mean_upload_bytes": _mean(rows, "mean_upload_bytes"),
            "mean_downlink_bytes": _mean(rows, "mean_downlink_bytes"),
        }

    phases = []
    for start, end in ((1, 50), (51, 100), (101, 150)):
        rows = [row for row in detail if start <= int(row["epoch"]) <= end]
        phases.append({"epochs": f"{start}-{end}", **block(rows)})
    return {"all_epochs": block(detail), "phases": phases}


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_detail_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = [k for k in rows[0].keys() if k not in {"cluster_loads", "mig_splits"}]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})
