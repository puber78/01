import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from proxy_selection.hfstream import decode_image_cell, iter_rows, resolve_url

url = resolve_url(
    "detection-datasets/coco",
    "data/train-00000-of-00040-67e35002d152155c.parquet",
)
rows = list(iter_rows(url, limit=3, columns=["image_id", "image", "width", "height"]))
print("rows:", len(rows), "keys:", list(rows[0].keys()))
img = decode_image_cell(rows[0]["image"])
print("image:", img.size, img.mode, "id:", rows[0]["image_id"])
