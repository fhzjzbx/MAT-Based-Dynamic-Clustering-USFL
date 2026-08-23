from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcsfl_evaluation import build_project_objects, evaluate_trace, generate_trace, make_agent, summarize_detail, write_json

SEEDS = [2026, 2027, 2028, 2029, 2030]
VARIANTS = {
    "untrained": dict(checkpoint=None, capacity_slack=0, phase_balanced_replay=True, use_joint_q=True),
    "full": dict(checkpoint="checkpoints/pcsfl_trained.pt", capacity_slack=0, phase_balanced_replay=True, use_joint_q=True),
    "no_phase_balance": dict(checkpoint="checkpoints/ablation_no_phase_balance.pt", capacity_slack=0, phase_balanced_replay=False, use_joint_q=True),
    "legacy_reward": dict(checkpoint="checkpoints/ablation_legacy_reward.pt", capacity_slack=0, phase_balanced_replay=True, use_joint_q=True),
    "random_channel_reward": dict(checkpoint="checkpoints/ablation_random_channel_reward.pt", capacity_slack=0, phase_balanced_replay=True, use_joint_q=True),
    "slack1": dict(checkpoint="checkpoints/ablation_slack1.pt", capacity_slack=1, phase_balanced_replay=True, use_joint_q=True),
    "separate_q": dict(checkpoint="checkpoints/ablation_separate_q.pt", capacity_slack=0, phase_balanced_replay=True, use_joint_q=False),
}


def main() -> None:
    out = Path("logs/pcsfl_ablation/multiseed")
    out.mkdir(parents=True, exist_ok=True)
    per_seed = []

    for seed in SEEDS:
        print(f"seed={seed}")
        trace = generate_trace(seed, 150)
        _, _, model = build_project_objects(seed, device="cpu")
        summaries = {}
        for name, cfg in VARIANTS.items():
            ckpt = cfg["checkpoint"]
            if ckpt is not None and not Path(ckpt).exists():
                raise FileNotFoundError(ckpt)
            agent = make_agent(
                seed=seed,
                checkpoint=ckpt,
                capacity_slack=cfg["capacity_slack"],
                phase_balanced_replay=cfg["phase_balanced_replay"],
                use_joint_q=cfg["use_joint_q"],
            )
            detail = evaluate_trace(trace=trace, model=model, agent=agent)
            summaries[name] = summarize_detail(detail)

        untrained_full = summaries["untrained"]["all_epochs"]["mean_full_total_delay_ms"]
        full_ref = summaries["full"]["all_epochs"]["mean_full_total_delay_ms"]
        for name, summary in summaries.items():
            allm = summary["all_epochs"]
            full = allm["mean_full_total_delay_ms"]
            per_seed.append({
                "eval_seed": seed,
                "variant": name,
                "full_delay_ms": full,
                "legacy_delay_ms": allm["mean_total_delay_ms"],
                "gain_vs_untrained_pct": (untrained_full - full) / untrained_full * 100.0,
                "delta_vs_full_ms": full - full_ref,
                "phase1_full_ms": summary["phases"][0]["mean_full_total_delay_ms"],
                "phase2_full_ms": summary["phases"][1]["mean_full_total_delay_ms"],
                "phase3_full_ms": summary["phases"][2]["mean_full_total_delay_ms"],
                "conflict_rate": allm["mean_conflict_rate"],
                "load_imbalance": allm["mean_load_imbalance"],
                "mean_l1": allm["mean_l1"],
                "mean_l2": allm["mean_l2"],
                "mean_upload_bytes": allm["mean_upload_bytes"],
                "mean_downlink_bytes": allm["mean_downlink_bytes"],
            })

    aggregate = {}
    for name in VARIANTS:
        rows = [r for r in per_seed if r["variant"] == name]
        aggregate[name] = {
            "full_delay_mean_ms": float(np.mean([r["full_delay_ms"] for r in rows])),
            "full_delay_std_ms": float(np.std([r["full_delay_ms"] for r in rows])),
            "legacy_delay_mean_ms": float(np.mean([r["legacy_delay_ms"] for r in rows])),
            "legacy_delay_std_ms": float(np.std([r["legacy_delay_ms"] for r in rows])),
            "gain_vs_untrained_mean_pct": float(np.mean([r["gain_vs_untrained_pct"] for r in rows])),
            "gain_vs_untrained_std_pct": float(np.std([r["gain_vs_untrained_pct"] for r in rows])),
            "delta_vs_full_mean_ms": float(np.mean([r["delta_vs_full_ms"] for r in rows])),
            "phase1_full_mean_ms": float(np.mean([r["phase1_full_ms"] for r in rows])),
            "phase2_full_mean_ms": float(np.mean([r["phase2_full_ms"] for r in rows])),
            "phase3_full_mean_ms": float(np.mean([r["phase3_full_ms"] for r in rows])),
            "conflict_rate_mean": float(np.mean([r["conflict_rate"] for r in rows])),
            "load_imbalance_mean": float(np.mean([r["load_imbalance"] for r in rows])),
            "mean_l1": float(np.mean([r["mean_l1"] for r in rows])),
            "mean_l2": float(np.mean([r["mean_l2"] for r in rows])),
            "mean_upload_bytes": float(np.mean([r["mean_upload_bytes"] for r in rows])),
            "mean_downlink_bytes": float(np.mean([r["mean_downlink_bytes"] for r in rows])),
        }

    result = {
        "eval_seeds": SEEDS,
        "fairness_note": "All variants use the same pre-generated 150-round trace and the same final PCSFL physical evaluator for each seed.",
        "variants": VARIANTS,
        "aggregate": aggregate,
        "per_seed": per_seed,
    }
    write_json(out / "ablation_summary.json", result)
    with (out / "ablation_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        writer.writeheader(); writer.writerows(per_seed)
    print(f"results: {out}")


if __name__ == "__main__":
    main()
