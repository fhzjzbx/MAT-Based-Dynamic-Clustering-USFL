from __future__ import annotations

import csv
import json
from pathlib import Path


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    multi = load("logs/pcsfl/multiseed/multiseed_summary.json")
    ablation = load("logs/pcsfl_ablation/multiseed/ablation_summary.json")
    out = Path("tables/pcsfl")
    out.mkdir(parents=True, exist_ok=True)

    rows = multi["per_seed"]
    with (out / "multiseed.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["eval_seed", "untrained_full_ms", "trained_full_ms", "full_gain_pct", "phase1_full_gain_pct", "phase2_full_gain_pct", "phase3_full_gain_pct"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{k:r[k] for k in fields} for r in rows])

    with (out / "ablation.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["variant", "full_delay_mean_ms", "full_delay_std_ms", "gain_vs_untrained_mean_pct", "delta_vs_full_mean_ms"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name, m in ablation["aggregate"].items():
            w.writerow({"variant": name, **{k:m[k] for k in fields[1:]}})

    key = {
        "multiseed": multi["aggregate"],
        "full_ablation": ablation["aggregate"]["full"],
    }
    (out / "key_metrics.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tables: {out}")


if __name__ == "__main__":
    main()
