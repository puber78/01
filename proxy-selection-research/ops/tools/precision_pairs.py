"""Refined precision analysis: paired AUC differences for realistic model pairs.

The earlier run used the *closest* pair, which happened to be two near-chance
detectors with weakly correlated scores -- the least favourable case.  Here we
report every pair, the score correlation, and the per-class n needed for a
paired 95% half-width of 0.01 / 0.02 / 0.03.
"""
from __future__ import annotations

import itertools
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
sc = sc[sc["status"] == "ok"]
sc["label"] = sc["source_id"].map(lab)
pivot = sc.pivot_table(index="sample_id", columns="model_id", values="raw_score")
y = sc.drop_duplicates("sample_id").set_index("sample_id").loc[pivot.index, "label"].to_numpy()
models = list(pivot.columns)
X = pivot.to_numpy()
aucs = {m: roc_auc(y, X[:, i]) for i, m in enumerate(models)}

# map model_id -> model_key via the registry fields recorded in the audit
from proxy_selection.contracts.ids import mint_model_id

mdl_to_key = {}
for f in (run_dir / "model_audits").glob("*.json"):
    a = json.loads(f.read_text("utf-8"))
    if a.get("status") != "OK":
        continue
    mid = mint_model_id(
        repo=a["registry"]["repo"],
        revision=a["registry"]["revision"] or "",
        weight_sha256=a["config_fingerprint"]["checkpoint_hash"],
    )
    mdl_to_key[mid] = a["model_key"]


def paired_se(i, j, n_boot=1500):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = []
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yy = y[pick]
        out.append(roc_auc(yy, X[pick, i]) - roc_auc(yy, X[pick, j]))
    return float(np.std(out, ddof=1))


rows = []
for i, j in itertools.combinations(range(len(models)), 2):
    a, b = models[i], models[j]
    delta = aucs[a] - aucs[b]
    se = paired_se(i, j)
    r = float(np.corrcoef(X[:, i], X[:, j])[0, 1])
    rows.append(
        {
            "model_a": mdl_to_key.get(a, a[-6:]),
            "model_b": mdl_to_key.get(b, b[-6:]),
            "auc_a": round(aucs[a], 3),
            "auc_b": round(aucs[b], 3),
            "delta": round(delta, 4),
            "score_corr": round(r, 2),
            "paired_se_at_80": round(se, 4),
            "halfwidth_95_at_80": round(1.96 * se, 4),
            "n_per_class_for_hw_0.01": int(round(80 * (se / (0.01 / 1.96)) ** 2)),
            "n_per_class_for_hw_0.02": int(round(80 * (se / (0.02 / 1.96)) ** 2)),
            "n_per_class_for_hw_0.03": int(round(80 * (se / (0.03 / 1.96)) ** 2)),
        }
    )
df = pd.DataFrame(rows).sort_values("delta", key=lambda s: s.abs())
pd.set_option("display.width", 200)
print(df.to_string(index=False))

strong = df[(df.auc_a > 0.8) & (df.auc_b > 0.8)]
print("\nTwo strong models (correlated):")
print(strong.to_string(index=False))
log = RunLogger.start(ROOT, note="EXAM02 precision refined")
out = ROOT / "metrics" / "EXAM02_precision"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / f"pairwise_precision_{log.run_id}.csv", index=False)
log.finish("PASS", n_pairs=len(df))
