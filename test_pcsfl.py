from __future__ import annotations

import numpy as np
import torch

from baselines.pcsfl import PCSFLAgent, TensorProfileCache, build_agent_state, evaluate_actions
from models.usfl_networks import ResNet18_USFL


def synthetic_raw_state(n: int, rng: np.random.Generator) -> np.ndarray:
    h = rng.rayleigh(1.0, size=(n, 1))
    compute = rng.uniform(1.0, 2.5, size=(n, 1))
    labels = rng.dirichlet(np.ones(10), size=n)
    return np.concatenate([h, compute, labels], axis=1).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(1234)
    model = ResNet18_USFL(num_classes=10).to(torch.device("cpu"))
    model.eval()
    cache = TensorProfileCache()
    agent = PCSFLAgent(seed=1234, batch_size=48, device="cpu")

    cases = [(8, 2, 100e6), (11, 2, 100e6), (15, 5, 100e6), (19, 5, 100e6), (20, 7, 20e6)]
    for n, migs, bw in cases:
        raw = synthetic_raw_state(n, rng)
        state = build_agent_state(raw, migs, bw)
        clusters, l1, l2, weights = agent.step(state, migs)
        counts = np.bincount(clusters, minlength=migs)
        assert counts.max() - counts.min() <= 1
        assert np.all(l1 <= l2)
        assert np.allclose(weights, 1.0)
        for mig in range(migs):
            mask = clusters == mig
            if np.any(mask):
                assert len(np.unique(l1[mask])) == 1
                assert len(np.unique(l2[mask])) == 1
        metrics = evaluate_actions(
            raw_client_states=raw,
            available_migs=migs,
            current_bandwidth=bw,
            cluster_choices=clusters,
            l1_choices=l1,
            l2_choices=l2,
            resnet=model,
            tensor_cache=cache,
        )
        assert metrics["conflict_rate"] == 0.0
        agent.observe(metrics["reward"], state, migs, done=True)
        print(f"N={n}, MIG={migs}, loads={counts.tolist()}, full={metrics['full_total_delay_ms']:.3f} ms -> PASS")

    # Exercise the replay/Joint-Q update path.
    for _ in range(55):
        n, migs, bw = cases[int(rng.integers(0, len(cases)))]
        raw = synthetic_raw_state(n, rng)
        state = build_agent_state(raw, migs, bw)
        clusters, l1, l2, _ = agent.step(state, migs)
        metrics = evaluate_actions(
            raw_client_states=raw,
            available_migs=migs,
            current_bandwidth=bw,
            cluster_choices=clusters,
            l1_choices=l1,
            l2_choices=l2,
            resnet=model,
            tensor_cache=cache,
        )
        loss = agent.observe(metrics["reward"], state, migs, done=True)
    assert agent.gradient_steps > 0 and loss is not None
    print("PCSFL smoke test PASSED")


if __name__ == "__main__":
    main()
