import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from proxy_selection.hfstream import UA  # noqa: E402


def api(u):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read())


for ds in ["jzousz/GenImage", "OwensLab/CommunityForensics-Small"]:
    try:
        t = api(f"https://hf-mirror.com/api/datasets/{ds}/tree/main")
        print(ds, "->", [(x["path"], x.get("size"), x.get("type")) for x in t[:10]])
        for sub in [x["path"] for x in t if x.get("type") == "directory"][:6]:
            tt = api(f"https://hf-mirror.com/api/datasets/{ds}/tree/main/{sub}")
            print("   ", sub, "->", [(x["path"], x.get("size"), x.get("type")) for x in tt[:6]])
    except Exception as exc:  # noqa: BLE001
        print(ds, "FAIL", str(exc)[:120])
