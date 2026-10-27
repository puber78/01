import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from proxy_selection.hfstream import iter_rows, resolve_url  # noqa: E402

t0 = time.time()
url = resolve_url("OwensLab/CommunityForensics-Small", "data/HFCF_small_0.parquet")
rows = list(iter_rows(url, limit=3))
print("elapsed", round(time.time() - t0, 1), "s")
print("keys:", list(rows[0].keys()))
for r in rows:
    print({k: (type(v).__name__ if k == "image" else v) for k, v in r.items()})
