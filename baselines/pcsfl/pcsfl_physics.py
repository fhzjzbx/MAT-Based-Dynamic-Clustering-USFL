from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class TensorProfile:
    upload_bytes: float
    downlink_bytes: float


class TensorProfileCache:
    """Caches tensor sizes for every legal U-shaped split pair."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int, str], TensorProfile] = {}

    def get(self, resnet, l1: int, l2: int) -> TensorProfile:
        device = next(resnet.parameters()).device
        key = (int(l1), int(l2), str(device))
        if key in self._cache:
            return self._cache[key]

        x = torch.zeros(1, 3, 32, 32, device=device)
        with torch.inference_mode():
            part_a = resnet.forward_partA(x, int(l1))
            part_b = resnet.forward_partB(part_a, int(l1), int(l2))
        profile = TensorProfile(
            upload_bytes=float(part_a.numel() * part_a.element_size()),
            downlink_bytes=float(part_b.numel() * part_b.element_size()),
        )
        self._cache[key] = profile
        return profile


def calculate_conflict_rate(cluster_choices, l1_choices, l2_choices, available_migs: int) -> float:
    cluster_choices = np.asarray(cluster_choices, dtype=np.int64)
    l1_choices = np.asarray(l1_choices, dtype=np.int64)
    l2_choices = np.asarray(l2_choices, dtype=np.int64)
    non_empty = 0
    conflict = 0
    for mig in range(int(available_migs)):
        mask = cluster_choices == mig
        if not np.any(mask):
            continue
        non_empty += 1
        if len(np.unique(l1_choices[mask])) != 1 or len(np.unique(l2_choices[mask])) != 1:
            conflict += 1
    return 0.0 if non_empty == 0 else float(conflict / non_empty)


def calculate_load_imbalance(cluster_choices, available_migs: int) -> float:
    choices = np.asarray(cluster_choices, dtype=np.int64)
    n = len(choices)
    m = int(available_migs)
    if n == 0 or m <= 1:
        return 0.0
    counts = np.bincount(choices, minlength=m).astype(np.float64)
    distribution = counts / n
    uniform = np.full(m, 1.0 / m, dtype=np.float64)
    tv = 0.5 * np.abs(distribution - uniform).sum()
    max_tv = 1.0 - 1.0 / m
    return float(np.clip(tv / max(max_tv, 1e-12), 0.0, 1.0))


def _cluster_wireless_delay(
    channel_gains: np.ndarray,
    tensor_bytes: float,
    current_bandwidth: float,
    available_migs: int,
) -> float:
    """Same-round Shannon delay with equal bandwidth weights inside one MIG."""
    n = len(channel_gains)
    if n == 0:
        return 0.0
    cluster_bandwidth = float(current_bandwidth) / max(int(available_migs), 1)
    allocated_bw = cluster_bandwidth / n
    snr = 10.0 * np.clip(np.asarray(channel_gains, dtype=np.float64), 0.0, None)
    rates = allocated_bw * np.log2(1.0 + snr)
    delays = float(tensor_bytes) * 8.0 / np.maximum(rates, 1e-9)
    return float(np.max(delays))


def evaluate_actions(
    *,
    raw_client_states: np.ndarray,
    available_migs: int,
    current_bandwidth: float,
    cluster_choices: np.ndarray,
    l1_choices: np.ndarray,
    l2_choices: np.ndarray,
    resnet,
    tensor_cache: TensorProfileCache | None = None,
    reward_mode: str = "full",
    channel_mode: str = "state",
    rng: np.random.Generator | None = None,
    reward_scale_seconds: float = 0.1,
) -> dict:
    """Evaluate one PCSFL joint action using the project's U-shaped physical model.

    ``total_delay_ms`` retains the legacy-comparable uplink+logical-compute metric.
    ``full_total_delay_ms`` adds the U-shaped downlink and is the final training objective.
    """
    raw = np.asarray(raw_client_states, dtype=np.float64)
    clusters = np.asarray(cluster_choices, dtype=np.int64)
    l1 = np.asarray(l1_choices, dtype=np.int64)
    l2 = np.asarray(l2_choices, dtype=np.int64)
    n_clients = len(clusters)
    if raw.shape[0] != n_clients:
        raise ValueError("state/action client count mismatch")
    if not (len(l1) == len(l2) == n_clients):
        raise ValueError("split action length mismatch")

    if channel_mode == "state":
        channels = raw[:, 0].copy()
    elif channel_mode == "random":
        rng = rng or np.random.default_rng()
        channels = rng.rayleigh(scale=1.0, size=n_clients)
    else:
        raise ValueError("channel_mode must be 'state' or 'random'")

    cache = tensor_cache or TensorProfileCache()
    legacy_system_delay = 0.0
    full_system_delay = 0.0
    legacy_uplink_at_bottleneck = 0.0
    full_uplink_at_bottleneck = 0.0
    full_downlink_at_bottleneck = 0.0
    full_compute_at_bottleneck = 0.0

    cluster_loads = np.bincount(clusters, minlength=int(available_migs)).astype(int)
    mig_splits = []
    upload_per_client = np.zeros(n_clients, dtype=np.float64)
    downlink_per_client = np.zeros(n_clients, dtype=np.float64)

    for mig in range(int(available_migs)):
        idx = np.where(clusters == mig)[0]
        if idx.size == 0:
            mig_splits.append(None)
            continue
        unique_pairs = np.unique(np.stack([l1[idx], l2[idx]], axis=1), axis=0)
        if len(unique_pairs) != 1:
            # Retain the legacy serial-compute penalty if an invalid conflicting action is supplied.
            split_pair = tuple(map(int, unique_pairs[0]))
            comp_delay = 0.06 * idx.size
        else:
            split_pair = tuple(map(int, unique_pairs[0]))
            comp_delay = 0.05 + 0.005 * idx.size
        mig_splits.append(list(split_pair))

        profile = cache.get(resnet, *split_pair)
        upload_per_client[idx] = profile.upload_bytes
        downlink_per_client[idx] = profile.downlink_bytes

        uplink = _cluster_wireless_delay(
            channels[idx], profile.upload_bytes, current_bandwidth, available_migs
        )
        downlink = _cluster_wireless_delay(
            channels[idx], profile.downlink_bytes, current_bandwidth, available_migs
        )
        legacy = comp_delay + uplink
        full = comp_delay + uplink + downlink

        if legacy > legacy_system_delay:
            legacy_system_delay = legacy
            legacy_uplink_at_bottleneck = uplink
        if full > full_system_delay:
            full_system_delay = full
            full_uplink_at_bottleneck = uplink
            full_downlink_at_bottleneck = downlink
            full_compute_at_bottleneck = comp_delay

    conflict_rate = calculate_conflict_rate(clusters, l1, l2, available_migs)
    load_imbalance = calculate_load_imbalance(clusters, available_migs)
    objective_delay = full_system_delay if reward_mode == "full" else legacy_system_delay
    if reward_mode not in {"full", "legacy"}:
        raise ValueError("reward_mode must be 'full' or 'legacy'")
    reward = -float(np.clip(objective_delay / reward_scale_seconds, 0.0, 2.0))

    return {
        "total_delay_ms": legacy_system_delay * 1000.0,
        "full_total_delay_ms": full_system_delay * 1000.0,
        "tx_delay_ms": legacy_uplink_at_bottleneck * 1000.0,
        "uplink_delay_ms": full_uplink_at_bottleneck * 1000.0,
        "downlink_delay_ms": full_downlink_at_bottleneck * 1000.0,
        "comp_delay_ms": full_compute_at_bottleneck * 1000.0,
        "conflict_rate": conflict_rate,
        "load_imbalance": load_imbalance,
        "cluster_loads": cluster_loads.tolist(),
        "mig_splits": mig_splits,
        "mean_l1": float(np.mean(l1)) if n_clients else 0.0,
        "mean_l2": float(np.mean(l2)) if n_clients else 0.0,
        "mean_upload_bytes": float(np.mean(upload_per_client)) if n_clients else 0.0,
        "mean_downlink_bytes": float(np.mean(downlink_per_client)) if n_clients else 0.0,
        "reward": reward,
    }
