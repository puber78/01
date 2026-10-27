"""EXAM01 minimal real chain: read -> official preprocess -> score -> statistics.

Runs every registered candidate on the TECHNICAL sample (role TECH, never
research data), writes schema-valid score rows, measures FP32 repeat-inference
tolerance, and recomputes AUC from the persisted parquet (independent scoring
interface).  All outputs are new files; the run is logged end to end.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "models" / "hf_home" / "hub"))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from proxy_selection.adapters.hf_classifier import HFClassifierAdapter  # noqa: E402
from proxy_selection.config import (  # noqa: E402
    MODELS,
    PROTOCOLS,
    SCORES,
    load_proposed_parameters,
    load_yaml,
)
from proxy_selection.contracts.ids import (  # noqa: E402
    mint_display_asset_id,
    mint_sample_id,
    mint_source_id,
    mint_split_group_id,
)
from proxy_selection.hashing import sha256_file, sha256_json  # noqa: E402
from proxy_selection.io import Ledger, write_json, write_new, write_parquet  # noqa: E402
from proxy_selection.runlog import RunLogger, timed  # noqa: E402
from proxy_selection.scoring import compute_auc_table, coverage_report, score_rows  # noqa: E402


def load_registry() -> list[dict]:
    return json.loads((MODELS / "registry.json").read_text("utf-8"))["candidates"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="model_key subset (default: all)")
    ap.add_argument("--repeat-subset", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    import torch

    torch.manual_seed(20261026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20261026)

    log = RunLogger.start(ROOT, note="EXAM01 technical chain")
    run_dir = SCORES / "EXAM01" / log.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(ROOT / "ops")
    params = load_proposed_parameters()
    tol = load_yaml(PROTOCOLS / "inference_tolerances.yaml")["fp32_repeat_inference"]
    atol, rtol = float(tol["atol"]), float(tol["rtol"])
    log.event(
        "environment",
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        python=platform.python_version(),
        parameters_approval_status=params.approval_status,
        parameters_approved=params.approved,
    )

    tech_csv = ROOT / "data" / "raw" / "TECH" / "technical_sources.csv"
    tech = pd.read_csv(tech_csv, dtype=str, keep_default_na=False)
    tech["label"] = tech["label"].astype(int)
    tech["width"] = tech["width"].astype(int)
    tech["height"] = tech["height"].astype(int)
    if args.max_samples:
        tech = tech.groupby("label", group_keys=False).head(args.max_samples // 2 + 1)
    log.input_hashes({str(tech_csv): sha256_file(tech_csv)})

    # ---- build TECH samples: source -> display asset (no display) -> clean sample ----
    import datetime as _dt

    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    samples = []
    sources_rows = []
    splits_rows = []
    display_rows = []
    spec_hash = sha256_json({"display_spec": "no_display_technical_source", "render": "identity"})
    for rec in tech.itertuples(index=False):
        lineage_key = f"TECH:{rec.source_dataset}:{rec.record_id}"
        src_id = mint_source_id(
            lineage_key=lineage_key,
            label=int(rec.label),
            provenance_uri=f"{rec.source_dataset}#{rec.record_id}",
        )
        grp_id = mint_split_group_id(
            lineage_key=lineage_key,
            generation_family=(rec.generator_model or ("camera_original" if int(rec.label) == 0 else "unknown")),
        )
        dsp_id = mint_display_asset_id(source_id=src_id, display_spec_hash=spec_hash)
        smp_id = mint_sample_id(branch="digital", transform_id="clean_tech", parent_key=dsp_id)
        sources_rows.append(
            {
                "source_id": src_id,
                "split_group_id": grp_id,
                "lineage_root_id": lineage_key,
                "label": int(rec.label),
                "content_category": "photograph" if int(rec.label) == 0 else "diffusion_generated",
                "generator_id": rec.generator_model or "camera_original",
                "provenance_uri": f"{rec.source_dataset}#{rec.record_id}",
                "provenance_sha256": sha256_json(
                    {"dataset": rec.source_dataset, "record_id": rec.record_id, "split": rec.source_split}
                ),
                "license_id": rec.license_id,
                "license_evidence_sha256": sha256_json(
                    {"license_id": rec.license_id, "license_url": rec.license_url}
                ),
                "label_evidence_sha256": sha256_json({"tech_id": rec.tech_id, "label": int(rec.label)}),
                "original_file_sha256": rec.sha256,
                "ingested_at": now,
                "ingest_run_id": log.run_id,
                "notes": "TECH role: engineering verification only",
            }
        )
        display_rows.append(
            {
                "display_asset_id": dsp_id,
                "source_id": src_id,
                "display_file_sha256": rec.sha256,
                "width": int(rec.width),
                "height": int(rec.height),
                "color_space": "sRGB_unmanaged",
                "display_spec_hash": spec_hash,
                "created_at": now,
                "run_id": log.run_id,
            }
        )
        samples.append(
            {
                "sample_id": smp_id,
                "source_id": src_id,
                "split_group_id": grp_id,
                "display_asset_id": dsp_id,
                "split_role": "TECH",
                "environment_id": "TECH_no_display",
                "image_branch": "digital",
                "file_path": str(ROOT / rec.file_relpath),
                "label": int(rec.label),
                "tech_id": rec.tech_id,
            }
        )
    assignment_hash = sha256_json(sorted(r["source_id"] for r in sources_rows))
    for r in sources_rows:
        splits_rows.append(
            {
                "split_group_id": r["split_group_id"],
                "source_id": r["source_id"],
                "split_role": "TECH",
                "assigned_at": now,
                "assignment_hash": assignment_hash,
                "assignment_run_id": log.run_id,
            }
        )
    manifest_dir = ROOT / "manifests" / "EXAM01" / log.run_id
    write_parquet(manifest_dir / "sources.parquet", sources_rows, "sources", ledger=ledger, run_id=log.run_id,
                  note="TECH sources (technical sample)")
    write_parquet(manifest_dir / "splits.parquet", splits_rows, "splits", ledger=ledger, run_id=log.run_id,
                  note="TECH splits")
    write_parquet(manifest_dir / "display_assets.parquet", display_rows, "display_assets", ledger=ledger,
                  run_id=log.run_id, note="identity display spec for technical samples")
    log.event("tech_samples_prepared", n=len(samples), n_ai=sum(s["label"] for s in samples),
              assignment_hash=assignment_hash)

    registry = load_registry()
    wanted = set(args.models) if args.models else None
    score_rows_all: list[dict] = []
    audits: list[dict] = []
    repeat_report: dict[str, dict] = {}

    for cand in registry:
        if wanted and cand["model_key"] not in wanted:
            continue
        print(f"=== {cand['model_key']} ({cand['repo']}) ===", flush=True)
        adapter = HFClassifierAdapter(
            repo=cand["repo"],
            revision=cand["revision"],
            cache_dir=str(ROOT / "models" / "hf_cache"),
            ai_label_override=cand.get("ai_label"),
            batch_size=int(cand.get("batch_size", 16)),
        )
        try:
            with timed() as t_load:
                adapter.load()
        except Exception as exc:  # noqa: BLE001
            log.failure("adapter_load", repr(exc), model_key=cand["model_key"], repo=cand["repo"])
            audits.append(
                {
                    "model_key": cand["model_key"],
                    "repo": cand["repo"],
                    "status": "LOAD_FAILED",
                    "error": repr(exc),
                }
            )
            continue
        print(f"  loaded in {t_load.elapsed:.1f}s arch={adapter.metadata['architectures']} "
              f"ai_label={adapter.metadata['ai_label']} params={adapter.metadata['num_parameters']}", flush=True)

        with timed() as t_score:
            rows = score_rows(
                adapter,
                samples,
                run_id=log.run_id,
                transform_id="clean_tech",
                is_fixture=False,
                progress=True,
            )
        score_rows_all.extend(rows)
        label_map_local = {s["sample_id"]: s["label"] for s in samples}
        df_rows = pd.DataFrame(rows)
        df_rows["label"] = df_rows["sample_id"].map(label_map_local)
        aucs = compute_auc_table(df_rows)
        audit = {
            "model_key": cand["model_key"],
            "status": "OK",
            "registry": cand,
            "metadata": adapter.metadata,
            "config_fingerprint": adapter.config_fingerprint(),
            "load_seconds": round(t_load.elapsed, 2),
            "score_seconds": round(t_score.elapsed, 2),
            "n_scored": len(rows),
            "n_failed": sum(1 for r in rows if r["status"] != "ok"),
            "tech_auc": None if aucs.empty else float(aucs["auc"].iloc[0]),
            "tech_n": None if aucs.empty else int(aucs["n"].iloc[0]),
        }

        # ---- FP32 repeat-inference tolerance on a subset ----
        # The subset is a whole number of inference batches so the batching (and
        # therefore the kernel selection) is identical to the first batches of the
        # main pass; otherwise a batch-shape change would be mistaken for
        # nondeterminism.
        bs = int(adapter.batch_size)
        n_rep = (max(bs, args.repeat_subset) // bs) * bs
        subset = samples[:n_rep]
        with timed() as t_rep:
            rows2 = score_rows(
                adapter, subset, run_id=log.run_id, transform_id="clean_tech_repeat", is_fixture=False
            )
        by_sample_first = {r["sample_id"]: r["raw_score"] for r in rows}
        diffs = []
        for r in rows2:
            s1 = by_sample_first.get(r["sample_id"])
            s2 = r["raw_score"]
            if s1 is not None and s2 is not None:
                diffs.append((abs(s1 - s2), atol + rtol * abs(s2), s1, s2))
        max_diff = max((d[0] for d in diffs), default=None)
        worst_allowed = max((d[1] for d in diffs), default=None)
        repeats_ok = all(d[0] <= d[1] for d in diffs) if diffs else None
        audit["repeat_inference"] = {
            "n": len(diffs),
            "batch_size": bs,
            "max_abs_diff": max_diff,
            "max_allowed": worst_allowed,
            "atol": atol,
            "rtol": rtol,
            "passes": repeats_ok,
            "seconds": round(t_rep.elapsed, 2),
        }
        if repeats_ok is False:
            log.failure("fp32_repeat_tolerance", "repeat difference exceeds tolerance", model_key=cand["model_key"])
        audits.append(audit)
        print(f"  scored={len(rows)} failed={audit['n_failed']} auc={audit['tech_auc']} "
              f"repeat_max={max_diff} pass={repeats_ok}", flush=True)

        write_json(
            run_dir / "model_audits" / f"{cand['model_key']}.json",
            audit,
            ledger=ledger,
            run_id=log.run_id,
            kind="model_audit",
        )
        del adapter
        import torch as _torch
        _torch.cuda.empty_cache()

    if not score_rows_all:
        log.finish("FAIL", reason="no model produced scores")
        print("NO SCORES")
        return 1

    # ---- persist scores (schema-validated) ----
    score_path = SCORES / "EXAM01" / log.run_id / "technical_scores.parquet"
    digest = write_parquet(
        score_path, score_rows_all, "scores", ledger=ledger, run_id=log.run_id,
        note="TECH role technical sample; is_fixture=False; not research data",
    )
    log.artifact(score_path, digest)

    # ---- independent scoring: re-read parquet, join labels, recompute ----
    df = pd.read_parquet(score_path)
    label_map = {s["sample_id"]: s["label"] for s in samples}
    df["label"] = df["sample_id"].map(label_map)
    assert df["label"].notna().all(), "every scored sample must have a TECH label"
    table = compute_auc_table(df)
    metrics = {
        "run_id": log.run_id,
        "role": "TECH",
        "n_samples": len(samples),
        "n_rows": len(df),
        "coverage": coverage_report(df, expected_units=len(samples) * int(df["model_id"].nunique())),
        "auc_table": table.to_dict(orient="records"),
        "models": audits,
        "repeat_tolerance": {"atol": atol, "rtol": rtol},
        "parameters_approval_status": params.approval_status,
        "note": "Technical-only metrics: no scientific claim; never used for candidate selection.",
    }
    write_json(
        SCORES / "EXAM01" / log.run_id / "technical_metrics.json",
        metrics,
        ledger=ledger,
        run_id=log.run_id,
        kind="tech_metrics",
    )
    write_new(
        SCORES / "EXAM01" / log.run_id / "technical_auc_table.csv",
        table.to_csv(index=False),
        ledger=ledger,
        run_id=log.run_id,
        kind="tech_auc_table",
    )
    log.finish(
        "PASS" if all(a.get("status") == "OK" for a in audits) else "PARTIAL",
        models=[a["model_key"] for a in audits],
        failed_models=[a["model_key"] for a in audits if a.get("status") != "OK"],
        score_parquet=str(score_path),
    )
    print(json.dumps({"run_id": log.run_id, "auc": table.to_dict(orient="records")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
