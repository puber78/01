import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from proxy_selection.hfstream import UA, iter_rows, resolve_url

u = "https://hf-mirror.com/api/datasets/elsaEU/ELSA1M_track1/tree/main/data"
files = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read())
name = files[0]["path"]
url = resolve_url("elsaEU/ELSA1M_track1", name)
rows = list(iter_rows(url, limit=2))
print("file:", name)
print("keys:", list(rows[0].keys()))
for r in rows:
    print({k: (v if k != "image" else type(v).__name__) for k, v in r.items()})
