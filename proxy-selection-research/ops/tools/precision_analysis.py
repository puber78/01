"""How precise is the pilot, really?  Data-grounded precision analysis.

Uses the EXAM01 technical scores (6 frozen detectors x 80 images) to estimate
the sampling variability of

  * a single AUC, and
  * the PAIRED AUC difference between two detectors scored on the same images,

then extrapolates to larger capture counts assuming variance ~ 1/n.

Why this matters: the pilot's T_pilot is 160 images (80 real + 80 AI).  The
question "is 240 captures enough?" is a question about the half-width of the
paired-difference interval relative to delta = 0.01 AUC, not about training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from proxy_selection.metrics.auc import roc_auc
from proxy_selection.runlog import RunLogger

rng = np.random.default_rng(20261026)
run_dir = sorted((ROOT / "scores" / "EXAM01").glob("run_*"), key=lambda p: p.stat().st_mtime)[-1]
src = pd.read_parquet(ROOT / "manifests" / "EXAM01" / run_dir.name / "sources.parquet")
lab = dict(zip(src["source_id"], src["label"]))
sc = pd.read_parquet(run_dir / "technical_scores.parquet")
sc = sc[sc["status"] == "ok"].copy()
sc["label"] = sc["source_id"].map(lab)
pivot = sc.pivot_table(index="sample_id", columns="model_id", values="raw_score")
y = sc.drop_duplicates("sample_id").set_index("sample_id").loc[pivot.index, "label"].to_numpy()
models = list(pivot.columns)
X = pivot.to_numpy()

aucs = {m: roc_auc(y, X[:, i]) for i, m in enumerate(models)}
order = sorted(aucs, key=lambda m: -aucs[m])
print("Technical-sample AUCs:")
for m in order:
    print(f"  {m}: {aucs[m]:.4f}")

# closest pair = hardest discrimination problem
pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1 :]]
closest = min(pairs, key=lambda ab: abs(aucs[ab[0]] - aucs[ab[1]]))
wide = max(pairs, key=lambda ab: abs(aucs[ab[0]] - aucs[ab[1]]))
print(f"closest pair: {closest} delta={aucs[closest[0]]-aucs[closest[1]]:+.4f}")
print(f"widest pair : {wide} delta={aucs[wide[0]]-aucs[wide[1]]:+.4f}")


def paired_bootstrap_se(ia, ib, n_boot=2000, n_per_class=None):
    idxs = np.arange(len(y))
    se = []
    for _ in range(n_boot):
        if n_per_class is None:
            pick = rng.choice(idxs, size=len(idxs), replace=True)
        else:
            pos = idxs[y == 1]
            neg = idxs[y == 0]
            pick = np.concatenate(
                [rng.choice(pos, size=n_per_class, replace=True),
                 rng.choice(neg, size=n_per_class, replace=True)]
            )
        yy = y[pick]
        d = roc_auc(yy, X[pick, ia]) - roc_auc(yy, X[pick, ib])
        se.append(d)
    return float(np.std(se, ddof=1))


def single_auc_se(i, n_boot=2000):
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(y), size=len(y), replace=True)
        yy = y[pick]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc(yy, X[pick, i]))
    return float(np.std(vals, ddof=1))


se80 = {m: single_auc_se(i) for i, m in enumerate(models)}
pair_se80 = paired_bootstrap_se(models.index(closest[0]), models.index(closest[1]))
print(f"\nAt 80 per class: single AUC SE ~ {np.mean(list(se80.values())):.4f}; "
      f"paired-delta SE (closest pair) = {pair_se80:.4f}")

rows = []
for n_per_class in [40, 80, 160, 320, 500, 1000]:
    scale = np.sqrt(80.0 / n_per_class)
    se_single = float(np.mean(list(se80.values()))) * scale
    se_pair = pair_se80 * scale
    rows.append(
        {
            "n_per_class": n_per_class,
            "n_total_images": 2 * n_per_class,
            "single_auc_se": round(se_single, 4),
            "single_auc_halfwidth_95": round(1.96 * se_single, 4),
            "paired_delta_se_closest_pair": round(se_pair, 4),
            "paired_delta_halfwidth_95": round(1.96 * se_pair, 4),
            "delta_0.01_resolvable": bool(1.96 * se_pair <= 0.01),
        }
    )
table = pd.DataFrame(rows)
print("\n" + table.to_string(index=False))

need = (pair_se80 / (0.01 / 1.96)) ** 2 * 80
print(f"\nper-class n needed for paired half-width <= 0.01: ~{need:.0f} "
      f"(~{2*need:.0f} images total)")

out = ROOT / "metrics" / "EXAM02_precision"
out.mkdir(parents=True, exist_ok=True)
log = RunLogger.start(ROOT, note="EXAM02 precision analysis")
table.to_csv(out / f"precision_table_{log.run_id}.csv", index=False)
summary = {
    "run_id": log.run_id,
    "source_run": run_dir.name,
    "aucs": {m: round(aucs[m], 4) for m in order},
    "closest_pair": list(closest),
    "paired_delta_se_at_80_per_class": round(pair_se80, 4),
    "mean_single_auc_se_at_80_per_class": round(float(np.mean(list(se80.values()))), 4),
    "per_class_needed_for_halfwidth_0.01": int(round(need)),
    "table": rows,
    "caveat": "Extrapolation assumes variance ~ 1/n and that the technical-sample "
              "score correlation is representative of the recapture setting.",
}
(out / f"precision_summary_{log.run_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
log.finish("PASS", need=int(round(need)))
print(json.dumps(summary, indent=2))
