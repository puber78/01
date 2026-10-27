"""Verify that the execution package is byte-identical to the recorded manifest."""
import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
PKG = PROJECT.parent
MAN = PKG / "migration" / "迁移清单.json"
CSV = PROJECT / "ops" / "manifests" / "inputs_sha256.csv"

with CSV.open(encoding="utf-8-sig") as fh:
    rows = {r["rel"]: r["sha256"] for r in csv.DictReader(fh)}
man = json.loads(MAN.read_text(encoding="utf-8"))

ok = True
for c in man["changed_files"]:
    key = c["path"].replace("/", "\\")
    got = rows.get(key)
    match = got == c["new_sha256"]
    ok &= match
    print(("OK   " if match else "MISMATCH "), c["path"], (got or "MISSING")[:12])
print("params:", rows.get("验收参数_待批准.yaml", "MISSING")[:12])
print("files:", len(rows), "ALL_MATCH:", ok)
