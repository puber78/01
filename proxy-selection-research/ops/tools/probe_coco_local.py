import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from proxy_selection.localdata import read_local_rows  # noqa: E402
from proxy_selection.hfstream import decode_image_cell  # noqa: E402

rows = read_local_rows(ROOT / "models" / "datasets" / "coco_shard0.parquet",
                       columns=["image_id", "image", "width", "height"], limit=420)
ids = [int(r["image_id"]) for r in rows]
print("rows", len(rows), "unique", len(set(ids)), "first", ids[:5], "last", ids[-3:])
ok = small = bad = 0
sizes = []
for r in rows:
    try:
        img = decode_image_cell(r["image"])
    except Exception:
        bad += 1
        continue
    if min(img.size) < 480:
        small += 1
    else:
        ok += 1
        sizes.append(img.size)
print("ok", ok, "small", small, "bad", bad)
print("sample sizes", sizes[:5])
