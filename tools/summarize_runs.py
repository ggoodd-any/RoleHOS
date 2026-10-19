import argparse, json, csv
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    a = p.parse_args()
    groups = {}
    for p in a.root.glob("*/seed*/summary.json"):
        x = json.loads(p.read_text())
        if x.get("partial") or "Avg" not in x:
            continue
        groups.setdefault(x["variant"], {})[x["seed"]] = x["Avg"]
    rows = []
    for variant, seeds in groups.items():
        vals = [seeds[k] for k in sorted(seeds)]
        rows.append(
            dict(
                variant=variant,
                n=len(vals),
                complete_three_seeds=set(seeds) == {0, 1, 2},
                mean_Avg=float(np.mean(vals)),
                sample_SD=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            )
        )
    (a.root / "aggregate.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
