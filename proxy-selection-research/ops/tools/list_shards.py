import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from proxy_selection.hfstream import UA  # noqa: E402


def tree(repo, sub=""):
    u = f"https://hf-mirror.com/api/datasets/{repo}/tree/main"
    if sub:
        u += f"/{sub}"
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read())


coco = [x["path"] for x in tree("detection-datasets/coco", "data") if x["path"].endswith(".parquet")]
elsa = sorted(x["path"] for x in tree("elsaEU/ELSA1M_track1", "data") if x["path"].endswith(".parquet"))
print("COCO")
for p in coco[:4]:
    print(p)
print("ELSA")
for p in elsa[:2]:
    print(p)
