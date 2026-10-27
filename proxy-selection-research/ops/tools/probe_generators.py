"""Probe generator-model distribution across ELSA1M shards (1 row per shard)."""
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from proxy_selection.hfstream import UA, iter_rows, resolve_url  # noqa: E402

u = "https://hf-mirror.com/api/datasets/elsaEU/ELSA1M_track1/tree/main/data"
files = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read())
paths = sorted(f["path"] for f in files if f["path"].endswith(".parquet"))
print("total shards:", len(paths))

idxs = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 890]
for i in idxs:
    if i >= len(paths):
        continue
    try:
        rows = list(iter_rows(resolve_url("elsaEU/ELSA1M_track1", paths[i]), limit=1,
                             columns=["id", "model", "positive_prompt"]))
        row = rows[0]
        print(f"shard {i:3d} | {row.get('model')} | {str(row.get('positive_prompt'))[:60]}")
    except Exception as exc:  # noqa: BLE001
        print(f"shard {i:3d} | FAIL {exc}")
    time.sleep(2.5)

