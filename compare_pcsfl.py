from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained and untrained PCSFL summaries")
    parser.add_argument("--untrained", default="logs/pcsfl/evaluation/untrained_summary.json")
    parser.add_argument("--trained", default="logs/pcsfl/evaluation/trained_summary.json")
    args = parser.parse_args()
    u = load(args.untrained)["all_epochs"]
    t = load(args.trained)["all_epochs"]
    legacy_gain = (u["mean_total_delay_ms"] - t["mean_total_delay_ms"]) / u["mean_total_delay_ms"] * 100.0
    full_gain = (u["mean_full_total_delay_ms"] - t["mean_full_total_delay_ms"]) / u["mean_full_total_delay_ms"] * 100.0
    print("PCSFL: trained vs untrained")
    print(f"legacy: {u['mean_total_delay_ms']:.3f} -> {t['mean_total_delay_ms']:.3f} ms | gain={legacy_gain:.2f}%")
    print(f"full:   {u['mean_full_total_delay_ms']:.3f} -> {t['mean_full_total_delay_ms']:.3f} ms | gain={full_gain:.2f}%")
    print(f"conflict rate: {t['mean_conflict_rate']:.3%}")
    print(f"load imbalance: {t['mean_load_imbalance']:.4f}")


if __name__ == "__main__":
    main()
