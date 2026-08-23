from __future__ import annotations

import numpy as np


def build_agent_state(
    raw_client_states: np.ndarray,
    available_migs: int,
    current_bandwidth: float,
    *,
    max_clients: int = 20,
    max_migs: int = 7,
) -> np.ndarray:
    """Build the 15-dimensional PCSFL state used by the final policy.

    Raw state layout is expected to be:
        [channel_gain, compute_power, 10-dim label_distribution]

    Three global features are appended to every active client:
        normalized bandwidth, available MIG ratio, active-client ratio.
    """
    raw = np.asarray(raw_client_states, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError("raw_client_states must have shape (N, state_dim)")
    if raw.shape[1] < 2:
        raise ValueError("raw_client_states must contain channel and compute features")
    n_clients = raw.shape[0]
    if n_clients > max_clients:
        raise ValueError(f"active clients N={n_clients} exceeds max_clients={max_clients}")

    local = raw.copy()

    # Preserve absolute channel quality instead of per-round z-score normalization.
    h = np.clip(local[:, 0], 0.0, None)
    local[:, 0] = h / (1.0 + h)

    # The provided environment registers compute power in [1.0, 2.5] GHz.
    compute = local[:, 1]
    if np.nanmax(np.abs(compute), initial=0.0) > 1e6:
        compute = compute / 1e9
    local[:, 1] = np.clip((compute - 1.0) / 1.5, 0.0, 1.0)

    # Label distributions are already bounded distributions in the project.
    if local.shape[1] > 2:
        local[:, 2:] = np.clip(local[:, 2:], 0.0, 1.0)

    bandwidth_norm = float(np.clip(current_bandwidth / 1e8, 0.0, 1.0))
    mig_norm = float(np.clip(available_migs / max_migs, 0.0, 1.0))
    clients_norm = float(np.clip(n_clients / max_clients, 0.0, 1.0))
    global_context = np.tile(
        np.asarray([bandwidth_norm, mig_norm, clients_norm], dtype=np.float32),
        (n_clients, 1),
    )
    return np.concatenate([local, global_context], axis=1).astype(np.float32)


def infer_phase_id(available_migs: int, current_bandwidth: float) -> int:
    """Map the three project tide regimes to replay phases 0, 1, 2."""
    if current_bandwidth < 50e6:
        return 2
    if available_migs >= 5:
        return 1
    return 0
