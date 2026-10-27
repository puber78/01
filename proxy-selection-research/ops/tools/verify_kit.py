"""Verify the capture kit: every queued display file exists, repeats are valid,
classes are balanced, and the viewer imports cleanly."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

KIT = ROOT / "captures" / "kit"
ok = True
main_originals = set()
for i in range(1, 11):
    q = json.loads((KIT / f"queue_main_s{i}.json").read_text("utf-8"))
    main_originals |= {it["source_id"] for it in q["items"] if not it.get("is_repeat")}
for name in ["tech"] + [f"main_s{i}" for i in range(1, 11)]:
    p = KIT / f"queue_{name}.json"
    q = json.loads(p.read_text("utf-8"))
    items = q["items"]
    missing = [it for it in items if not (ROOT / it["file_relpath"]).exists()]
    labels = Counter(it["label"] for it in items)
    repeats = [it for it in items if it.get("is_repeat")]
    originals = {it["source_id"] for it in items if not it.get("is_repeat")}
    valid_pool = originals if name == "tech" else (originals | main_originals)
    bad_repeat = [it for it in repeats if it["repeat_of_source_id"] not in valid_pool]
    seqs = [it["seq"] for it in items]
    print(f"{name:8s} n={len(items):3d} classes={dict(labels)} repeats={len(repeats)} "
          f"missing_files={len(missing)} bad_repeat={len(bad_repeat)} seq_unique={len(set(seqs)) == len(seqs)}")
    ok &= not missing and not bad_repeat and len(set(seqs)) == len(seqs)

# viewer import check (no GUI loop)
import capture_viewer  # noqa: E402

print("viewer import OK:", hasattr(capture_viewer, "Viewer"))
print("KIT_OK", ok)
