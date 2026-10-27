"""Search HF mirror for multi-generator AI-image datasets and show their file trees."""
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from proxy_selection.hfstream import UA  # noqa: E402


def api(u):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read())


for q in ["GenImage", "genimage", "community-forensics", "b-free", "fake image dataset"]:
    try:
        rows = api(f"https://hf-mirror.com/api/datasets?search={urllib.parse.quote(q)}&limit=6")
        print("QUERY", q)
        for d in rows:
            print("   ", d["id"], d.get("downloads"), d.get("gated"))
    except Exception as exc:  # noqa: BLE001
        print(q, "FAIL", exc)
    time.sleep(1.5)

for ds in ["yuweiyang/GenImage", "elsaEU/ELSA1M_track2", "InfImagine/FakeImageDataset"]:
    try:
        t = api(f"https://hf-mirror.com/api/datasets/{ds}/tree/main")
        print(ds, "->", [(x["path"], x.get("size")) for x in t[:6]])
    except Exception as exc:  # noqa: BLE001
        print(ds, "FAIL", str(exc)[:80])
