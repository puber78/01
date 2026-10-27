import json
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 research-agent/1.0"}


def api(u):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40).read())


def get(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read()


queries = [
    ("ai-generated-image-detector", 25),
    ("ai-image-detector", 25),
    ("deepfake-detector", 15),
]
seen = set()
for q, lim in queries:
    try:
        rows = api(f"https://hf-mirror.com/api/models?search={q}&sort=downloads&direction=-1&limit={lim}")
    except Exception as exc:  # noqa: BLE001
        print("QUERY FAIL", q, exc)
        continue
    for m in rows:
        cid = m["id"]
        if cid in seen:
            continue
        seen.add(cid)
        try:
            cfg = json.loads(get(f"https://hf-mirror.com/{cid}/resolve/main/config.json"))
            print(f"{cid:65s} {str(cfg.get('architectures')):42s} {cfg.get('id2label')} dl={m.get('downloads')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{cid:65s} FAIL {str(exc)[:50]}")
