from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pcsfl_evaluation import build_project_objects, evaluate_trace, generate_trace, make_agent, summarize_detail, write_json

DEFAULT_SEEDS = [2026, 2027, 2028, 2029, 2030]


def phase_gain(untrained: dict, trained: dict, phase_index: int) -> float:
    u = untrained["phases"][phase_index]["mean_full_total_delay_ms"]
    t = trained["phases"][phase_index]["mean_full_total_delay_ms"]
    return (u - t) / u * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair multi-seed PCSFL evaluation")
    parser.add_argument("--checkpoint", default="checkpoints/pcsfl_trained.pt")
    parser.add_argument("--output-dir", default="logs/pcsfl/multiseed")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    per_seed = []
    detailed = {}
    for seed in DEFAULT_SEEDS:
        print(f"evaluation seed={seed}")
        trace = generate_trace(seed, 150)
        _, _, model = build_project_objects(seed, device=args.device)
        untrained_agent = make_agent(seed=seed, device=args.device, checkpoint=None)
        trained_agent = make_agent(seed=seed, device=args.device, checkpoint=args.checkpoint)
        u_detail = evaluate_trace(trace=trace, model=model, agent=untrained_agent)
        t_detail = evaluate_trace(trace=trace, model=model, agent=trained_agent)
        u = summarize_detail(u_detail)
        t = summarize_detail(t_detail)
        ufull = u["all_epochs"]["mean_full_total_delay_ms"]
        tfull = t["all_epochs"]["mean_full_total_delay_ms"]
        ulegacy = u["all_epochs"]["mean_total_delay_ms"]
        tlegacy = t["all_epochs"]["mean_total_delay_ms"]
        row = {
            "eval_seed": seed,
            "untrained_legacy_ms": ulegacy,
            "trained_legacy_ms": tlegacy,
            "legacy_gain_ms": ulegacy - tlegacy,
            "legacy_gain_pct": (ulegacy - tlegacy) / ulegacy * 100.0,
            "untrained_full_ms": ufull,
            "trained_full_ms": tfull,
            "full_gain_ms": ufull - tfull,
            "full_gain_pct": (ufull - tfull) / ufull * 100.0,
            "phase1_full_gain_pct": phase_gain(u, t, 0),
            "phase2_full_gain_pct": phase_gain(u, t, 1),
            "phase3_full_gain_pct": phase_gain(u, t, 2),
            "trained_conflict_rate": t["all_epochs"]["mean_conflict_rate"],
            "trained_load_imbalance": t["all_epochs"]["mean_load_imbalance"],
            "trained_mean_l1": t["all_epochs"]["mean_l1"],
            "trained_mean_l2": t["all_epochs"]["mean_l2"],
            "trained_upload_bytes": t["all_epochs"]["mean_upload_bytes"],
            "trained_downlink_bytes": t["all_epochs"]["mean_downlink_bytes"],
        }
        per_seed.append(row)
        detailed[str(seed)] = {"untrained": u, "trained": t}
        print(f"full: {ufull:.3f} -> {tfull:.3f} ms | gain={row['full_gain_pct']:.2f}%")

    def mean(key): return float(np.mean([x[key] for x in per_seed]))
    def std(key): return float(np.std([x[key] for x in per_seed]))
    summary = {
        "algorithm": "PCSFL multi-seed fair evaluation",
        "checkpoint": args.checkpoint,
        "eval_seeds": DEFAULT_SEEDS,
        "num_seeds": len(DEFAULT_SEEDS),
        "fairness_note": "For every seed, one 150-round trace is generated first; untrained and trained policies use that exact trace.",
        "per_seed": per_seed,
        "aggregate": {
            "untrained_full_mean_ms": mean("untrained_full_ms"),
            "untrained_full_std_ms": std("untrained_full_ms"),
            "trained_full_mean_ms": mean("trained_full_ms"),
            "trained_full_std_ms": std("trained_full_ms"),
            "gain_mean_pct": mean("full_gain_pct"),
            "gain_std_pct": std("full_gain_pct"),
            "wins": int(sum(x["trained_full_ms"] < x["untrained_full_ms"] for x in per_seed)),
            "phase1_gain_mean_pct": mean("phase1_full_gain_pct"),
            "phase2_gain_mean_pct": mean("phase2_full_gain_pct"),
            "phase3_gain_mean_pct": mean("phase3_full_gain_pct"),
        },
        "phase_summaries": detailed,
    }
    summary["aggregate"]["pass"] = summary["aggregate"]["gain_mean_pct"] >= 5.0 and summary["aggregate"]["wins"] >= 4
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "multiseed_summary.json", summary)
    with (out / "multiseed_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        writer.writeheader(); writer.writerows(per_seed)
    a = summary["aggregate"]
    print(f"untrained full: {a['untrained_full_mean_ms']:.3f} +/- {a['untrained_full_std_ms']:.3f} ms")
    print(f"trained full:   {a['trained_full_mean_ms']:.3f} +/- {a['trained_full_std_ms']:.3f} ms")
    print(f"gain: {a['gain_mean_pct']:.2f} +/- {a['gain_std_pct']:.2f}% | wins={a['wins']}/5 | PASS={a['pass']}")


if __name__ == "__main__":
    main()
