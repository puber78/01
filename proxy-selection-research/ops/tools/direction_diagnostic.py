"""Engineering diagnostic: per-model score separation on the TECHNICAL sample.

Purpose: flag checkpoints whose declared label orientation contradicts the
observed direction on clean technical images.  The output is a QC flag only --
the project does NOT flip orientation based on this diagnostic.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from proxy_selection.io import Ledger, write_json  # noqa: E402
from proxy_selection.metrics.auc import roc_auc  # noqa: E402
from proxy_selection.runlog import RunLogger  # noqa: E402


def main() -> int:
    run_dir = sorted((ROOT / "scores" / "EXAM01").glob("run_*"), key=lambda p: p.stat().st_mtime)[-1]
    log = RunLogger.start(ROOT, note="direction diagnostic (technical sample)")
    ledger = Ledger(ROOT / "ops")

    src = pd.read_parquet(ROOT / "manifests" / "EXAM01" / run_dir.name / "sources.parquet")
    lab = dict(zip(src["source_id"], src["label"]))
    scores = pd.read_parquet(run_dir / "technical_scores.parquet")
    scores["label"] = scores["source_id"].map(lab)
    if scores["label"].isna().any():
        raise SystemExit("some scored sources have no label")

    out = []
    for model_id, g in scores.groupby("model_id"):
        g = g[g["status"] == "ok"]
        y = g["label"].astype(int).to_numpy()
        s = g["raw_score"].astype(float).to_numpy()
        auc = roc_auc(y, s)
        out.append(
            {
                "model_id": model_id,
                "n": int(len(g)),
                "auc_declared_orientation": auc,
                "mean_score_real": float(s[y == 0].mean()),
                "mean_score_ai": float(s[y == 1].mean()),
                "flag": "OK" if auc >= 0.5 else "BELOW_CHANCE_DIRECTION_SUSPECT",
            }
        )
    out.sort(key=lambda r: -r["auc_declared_orientation"])
    payload = {"run_id": run_dir.name, "diagnostic_only": True, "models": out}
    write_json(
        ROOT / "ops" / "logs" / f"25_direction_diagnostic_{run_dir.name}.json",
        payload,
        ledger=ledger,
        run_id=log.run_id,
        kind="diagnostic",
    )
    log.finish("PASS", n_models=len(out))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
