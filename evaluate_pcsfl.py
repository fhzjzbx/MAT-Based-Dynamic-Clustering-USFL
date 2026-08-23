from __future__ import annotations

import argparse
from pathlib import Path

from pcsfl_evaluation import (
    build_project_objects,
    evaluate_trace,
    generate_trace,
    make_agent,
    summarize_detail,
    write_detail_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PCSFL on one fixed 150-round trace")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint", default="checkpoints/pcsfl_trained.pt")
    parser.add_argument("--untrained", action="store_true", help="evaluate the untrained structured policy")
    parser.add_argument("--output-dir", default="logs/pcsfl/evaluation")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    trace = generate_trace(args.seed, 150)
    _, _, model = build_project_objects(args.seed, device=args.device)
    checkpoint = None if args.untrained else args.checkpoint
    if checkpoint is not None and not Path(checkpoint).exists():
        raise FileNotFoundError(checkpoint)
    agent = make_agent(seed=args.seed, device=args.device, checkpoint=checkpoint)
    detail = evaluate_trace(trace=trace, model=model, agent=agent)
    summary = {
        "algorithm": "PCSFL-Untrained" if args.untrained else "PCSFL",
        "checkpoint": None if args.untrained else str(args.checkpoint),
        "exploration_disabled": True,
        "eval_seed": args.seed,
        "state_dim": 15,
        "capacity_slack": 0,
        "headline_metric_note": (
            "mean_total_delay_ms is legacy-comparable uplink+logical-compute; "
            "mean_full_total_delay_ms adds U-shaped downlink and is the final PCSFL objective."
        ),
        **summarize_detail(detail),
    }
    out = Path(args.output_dir)
    stem = "untrained" if args.untrained else "trained"
    write_json(out / f"{stem}_summary.json", summary)
    write_json(out / f"{stem}_detail.json", detail)
    write_detail_csv(out / f"{stem}_detail.csv", detail)
    print(f"full delay: {summary['all_epochs']['mean_full_total_delay_ms']:.3f} ms")
    print(f"results: {out}")


if __name__ == "__main__":
    main()
