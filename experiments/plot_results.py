from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    multi = load("logs/pcsfl/multiseed/multiseed_summary.json")
    ablation = load("logs/pcsfl_ablation/multiseed/ablation_summary.json")
    history_path = Path("logs/pcsfl/training_history.json")
    history = load(str(history_path)) if history_path.exists() else []
    out = Path("figures/pcsfl")
    out.mkdir(parents=True, exist_ok=True)

    rows = multi["per_seed"]
    seeds = [r["eval_seed"] for r in rows]
    x = np.arange(len(seeds))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w/2, [r["untrained_full_ms"] for r in rows], w, label="Untrained")
    ax.bar(x + w/2, [r["trained_full_ms"] for r in rows], w, label="Trained")
    ax.set_xticks(x, seeds); ax.set_ylabel("Full U-shaped delay (ms)"); ax.legend()
    save(fig, out, "01_multiseed_full_delay.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar([str(s) for s in seeds], [r["full_gain_pct"] for r in rows])
    ax.axhline(0, linewidth=1); ax.set_ylabel("Training gain (%)")
    save(fig, out, "02_multiseed_training_gain.png")

    agg = multi["aggregate"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(["Phase 1", "Phase 2", "Phase 3"], [agg["phase1_gain_mean_pct"], agg["phase2_gain_mean_pct"], agg["phase3_gain_mean_pct"]])
    ax.axhline(0, linewidth=1); ax.set_ylabel("Mean gain (%)")
    save(fig, out, "03_phase_training_gain.png")

    # Reconstruct mean phase delays from phase_summaries.
    phase_store = multi.get("phase_summaries", {})
    u_phase, t_phase = [], []
    for p in range(3):
        u_phase.append(np.mean([phase_store[str(s)]["untrained"]["phases"][p]["mean_full_total_delay_ms"] for s in seeds]))
        t_phase.append(np.mean([phase_store[str(s)]["trained"]["phases"][p]["mean_full_total_delay_ms"] for s in seeds]))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xp = np.arange(3)
    ax.bar(xp - w/2, u_phase, w, label="Untrained")
    ax.bar(xp + w/2, t_phase, w, label="Trained")
    ax.set_xticks(xp, ["Phase 1", "Phase 2", "Phase 3"]); ax.set_ylabel("Full delay (ms)"); ax.legend()
    save(fig, out, "04_phase_full_delay.png")

    a = ablation["aggregate"]
    names = ["full", "no_phase_balance", "legacy_reward", "random_channel_reward", "slack1", "separate_q"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, [a[n]["full_delay_mean_ms"] for n in names])
    ax.set_ylabel("Full delay (ms)"); ax.tick_params(axis="x", rotation=25)
    save(fig, out, "05_ablation_full_delay.png")

    variants = [n for n in names if n != "full"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(variants, [a[n]["delta_vs_full_mean_ms"] for n in variants])
    ax.axhline(0, linewidth=1); ax.set_ylabel("Delta vs full PCSFL (ms)"); ax.tick_params(axis="x", rotation=25)
    save(fig, out, "06_ablation_delta_vs_full.png")

    if history:
        steps = [r["global_step"] for r in history]
        delay = np.asarray([r["full_total_delay_ms"] for r in history], dtype=float)
        window = 50
        kernel = np.ones(window) / window
        ma = np.convolve(delay, kernel, mode="valid")
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(steps[window-1:], ma); ax.set_xlabel("Training step"); ax.set_ylabel("Moving-average full delay (ms)")
        save(fig, out, "07_training_full_delay_ma.png")

    print(f"figures: {out}")


if __name__ == "__main__":
    main()
