"""Build the physical-capture kit: display assets, randomised queues, protocol.

Outputs (all under data/ and captures/kit/):
  data/display/TECH/*.png           technical-session display assets
  captures/kit/queue_tech.json      technical session queue (16 + 4 repeats)
  captures/kit/queue_main_s1..s4.json  main session queues (264 total)
  captures/kit/checklist_*.csv      human-readable backup list
  captures/kit/capture_protocol.md  the protocol the photographer follows
  captures/kit/run_*.bat            one-click viewers
  captures/kit/kit_manifest.json    hashes of everything in the kit
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from proxy_selection.config import MANIFESTS  # noqa: E402
from proxy_selection.contracts.ids import mint_display_asset_id, mint_source_id  # noqa: E402
from proxy_selection.hashing import sha256_bytes, sha256_file, sha256_json  # noqa: E402
from proxy_selection.io import Ledger, replace_with_archive, write_new  # noqa: E402
from proxy_selection.runlog import RunLogger  # noqa: E402

CANVAS_W, CANVAS_H = 2560, 1440


def kit_write(path: Path, data, *, kind: str) -> str:
    """Write a kit file; if it exists, archive the old bytes first."""
    if path.exists():
        return replace_with_archive(path, data, archive_root=ROOT / "ops" / "archive", kind=kind)
    return write_new(path, data, kind=kind)
SEED = 20261026


def latest_exam02() -> Path:
    dirs = sorted((MANIFESTS / "EXAM02").glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not dirs:
        raise SystemExit("no EXAM02 manifest run found")
    return dirs[-1]


def make_display(src: Path, out: Path, ledger: Ledger, run_id: str) -> tuple[str, dict]:
    img = Image.open(src).convert("RGB")
    scale = min(CANVAS_W / img.width, CANVAS_H / img.height)
    w, h = max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale)))
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (0, 0, 0))
    x, y = (CANVAS_W - w) // 2, (CANVAS_H - h) // 2
    canvas.paste(img.resize((w, h), Image.LANCZOS), (x, y))
    if out.exists():
        digest = sha256_file(out)
        return digest, {"content_x": x, "content_y": y, "content_w": w, "content_h": h,
                        "canvas_w": CANVAS_W, "canvas_h": CANVAS_H}
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=False)
    data = buf.getvalue()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    digest = sha256_bytes(data)
    ledger.record_artifact(out, kind="display_asset_tech", run_id=run_id, sha256=digest)
    return digest, {"content_x": x, "content_y": y, "content_w": w, "content_h": h,
                    "canvas_w": CANVAS_W, "canvas_h": CANVAS_H}


def main() -> int:
    log = RunLogger.start(ROOT, note="build capture kit")
    ledger = Ledger(ROOT / "ops")
    run = latest_exam02()
    rng = np.random.default_rng(SEED)

    sources = pd.read_parquet(run / "sources.parquet")
    splits = pd.read_parquet(run / "splits.parquet")
    assets = pd.read_parquet(run / "display_assets.parquet")
    samples = pd.DataFrame([json.loads(x) for x in (run / "samples.jsonl").read_text("utf-8").splitlines()])
    meta = pd.read_csv(ROOT / "data" / "raw" / "TECH" / "technical_sources.csv", dtype=str, keep_default_na=False)

    merged = splits.merge(sources, on=["source_id", "split_group_id"], how="left")
    merged = merged.merge(samples[["sample_id", "source_id", "display_asset_id", "file_relpath"]], on="source_id", how="left")
    merged = merged[merged["split_role"].isin(["C_pool", "T_pilot"])].sort_values("source_id").reset_index(drop=True)

    # ---------------- technical session display assets ----------------
    tech_photo = meta[meta.label == "0"].sort_values("tech_id").iloc[::5].head(8)
    tech_ai = meta[meta.label == "1"].sort_values("tech_id").iloc[::5].head(8)
    tech_rows = []
    for r in pd.concat([tech_photo, tech_ai]).itertuples(index=False):
        src = ROOT / r.file_relpath
        spec = sha256_json({"canvas": [CANVAS_W, CANVAS_H], "fit": "lanczos_letterbox_black", "role": "TECH"})
        # technical assets use their own namespace: id from source + spec
        src_id = mint_source_id(lineage_key=f"TECH:{r.source_dataset}:{r.record_id}", label=int(r.label),
                                provenance_uri=f"{r.source_dataset}#{r.record_id}")
        dsp_id = mint_display_asset_id(source_id=src_id, display_spec_hash=spec)
        out = ROOT / "data" / "display" / "TECH" / f"{dsp_id}.png"
        digest, geom = make_display(src, out, ledger, log.run_id)
        tech_rows.append({
            "role": "TECH",
            "label": int(r.label),
            "source_id": src_id,
            "display_asset_id": dsp_id,
            "file_relpath": str(out.relative_to(ROOT)).replace("\\", "/"),
            "display_sha256": digest,
            "geometry": geom,
            "repeat_of": None,
            "is_repeat": False,
        })

    def build_queue(rows: list[dict], repeats: int, rng: np.random.Generator, session: str) -> list[dict]:
        photo = [r for r in rows if r["label"] == 0]
        ai = [r for r in rows if r["label"] == 1]
        n_pairs = min(len(photo), len(ai))
        pairs = []
        for i in range(n_pairs):
            pair = [photo[i], ai[i]]
            if rng.random() < 0.5:
                pair.reverse()
            pairs.append(pair)
        leftovers = photo[n_pairs:] + ai[n_pairs:]
        rng.shuffle(pairs)
        ordered = [x for pair in pairs for x in pair] + leftovers
        # Insert planned repeats 20-70 positions after their original so that a
        # repeat never lands next to itself and is spread across sessions.
        gap_lo = min(20, max(2, len(ordered) // 8))
        gap_hi = min(70, max(gap_lo + 1, len(ordered) // 2))
        n_avail = max(0, len(ordered) - gap_lo - 1)
        n_repeats = min(repeats, n_avail)
        chosen = rng.choice(len(ordered), size=n_repeats, replace=False) if n_repeats else []
        insertions = []
        for idx in sorted(int(x) for x in chosen):
            target = min(idx + int(rng.integers(gap_lo, gap_hi + 1)), len(ordered) - 1)
            clone = dict(ordered[idx])
            clone["is_repeat"] = True
            clone["repeat_of_source_id"] = ordered[idx]["source_id"]
            insertions.append((target, clone))
        for target, clone in sorted(insertions, key=lambda x: x[0], reverse=True):
            ordered.insert(target + 1, clone)
        for item in ordered:
            item.setdefault("repeat_of_source_id", None)
            item.setdefault("is_repeat", False)
        out = ordered
        for i, item in enumerate(out):
            item["seq"] = i + 1
            item["session"] = session
        return out

    tech_queue = build_queue(tech_rows, 4, np.random.default_rng(SEED + 10), "tech")
    geom_all = json.loads((run / "display_geometry.json").read_text("utf-8"))
    main_rows = [
        {
            "role": r.split_role,
            "label": int(r.label),
            "source_id": r.source_id,
            "display_asset_id": r.display_asset_id,
            "file_relpath": r.file_relpath,
            "geometry": geom_all.get(r.display_asset_id, {}),
            "is_repeat": False,
            "repeat_of_source_id": None,
        }
        for r in merged.itertuples(index=False)
    ]
    main_queue = build_queue(main_rows, 100, np.random.default_rng(SEED + 20), "main")

    kit_dir = ROOT / "captures" / "kit"
    kit_dir.mkdir(parents=True, exist_ok=True)

    def write_queue(path: Path, items: list[dict], session: str, extra: dict | None = None) -> None:
        payload = {
            "session": session,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "items": items,
            **(extra or {}),
        }
        kit_write(path, json.dumps(payload, ensure_ascii=False, indent=2), kind="queue")

    write_queue(kit_dir / "queue_tech.json", tech_queue, "tech")
    n_sessions = 10
    per = (len(main_queue) + n_sessions - 1) // n_sessions
    for s in range(n_sessions):
        chunk = main_queue[s * per : (s + 1) * per]
        if not chunk:
            continue
        write_queue(kit_dir / f"queue_main_s{s + 1}.json", chunk, f"main_s{s + 1}")

    for name, items in (("tech", tech_queue), ("main", main_queue)):
        csv = "seq,role,label,source_id,is_repeat,repeat_of_source_id\n" + "".join(
            f"{it['seq']},{it['role']},{it['label']},{it['source_id']},{int(it['is_repeat'])},"
            f"{it.get('repeat_of_source_id') or ''}\n"
            for it in items
        )
        kit_write(kit_dir / f"checklist_{name}.csv", csv, kind="checklist")

    protocol = f"""# 屏幕翻拍采集协议 v1（EXAM02）

> 生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}。
> 显示画布：2560×1440 sRGB；每个显示文件已按源图等比缩放并居中，四周黑边。
> 采集期间**不得**改变窗口大小、缩放、屏幕分辨率、相机焦距或拍摄距离。

## 0. 设备与环境（本协议固定）

| 项 | 设定 |
|---|---|
| 主环境 e1 | 显示器 R27qa（2560×1440）+ 手机 Redmi K80 |
| 备选环境 e2 | 笔记本 华硕天选四 / NE156FHM-NX6（1920×1080）+ Redmi K90（仅 EXAM08 外部面板使用，主批次不混用） |
| 屏幕设置 | 分辨率原生；缩放 100%；亮度 70%；关闭夜间模式/护眼/自动亮度/HDR；关闭屏保与通知 |
| 相机 | 主摄 1×；照片比例 4:3；JPEG 最高质量；HDR 关闭；闪光灯关闭；对焦锁定屏幕中心；ISO 自动 |
| 几何 | 镜头光轴垂直于屏幕中心，距离约 **45 cm**（画面中屏幕宽度占约 85–90%），手机固定不晃动 |
| 光照 | 室内顶灯开启；避免窗户/台灯在屏幕上形成高光反射；不要改变灯光 |
| 输出 | 原生 JPEG，不裁剪、不旋转、不美颜、不加滤镜 |

## 1. 摆位（只做一次）

1. 显示器放在桌面，屏幕正对拍摄者。
2. 手机用支架/书堆固定，镜头对准屏幕中心，**距离 45 cm**，屏幕四边都应落在画面内并留约 5–8% 边距。
3. 拍一张测试图，检查：屏幕四角完整、无透视倾斜、无明显反光、画面清晰。
4. 固定后在整个批次中**不再移动**手机和屏幕。

## 2. 运行显示程序

双击 `run_tech.bat`（技术批）或 `run_main_s1.bat`…`run_main_s4.bat`（主批次）。

- 全屏黑底显示一张图；按 **→ 或空格** 表示"这张已拍"，程序记录并切到下一张；
- 每张图出现前会短暂显示 `序号 / 总数`（只在这 0.7 秒内出现，不会被拍到）；
- 按 **←** 可回退一张；按 **Q** 退出（进度自动保存）；
- 程序退出后进度保存在 `queue_*_progress.json`，下次从断点继续。

## 3. 拍摄流程

1. 看到图片后，用手机拍照（默认相机，点击屏幕中心对焦，等 0.5 秒）。
2. 确认照片已保存，再按 →。
3. **不要删除任何照片**；如果拍糊/反光/没对上，立即原地重拍一张（相机里会多一张），继续按 →
   —— 多出的照片在导入时按"额外尝试"处理，不会打乱队列。
4. 如果整张图完全失败（例如屏幕黑屏），按 **S** 跳过并在纸上记下序号。
5. 每 30–40 张休息一次；重新开始前确认手机/屏幕没有移动。

## 4. 批次

| 批次 | 张数 | 用途 |
|---|---:|---|
| tech（技术批） | {len(tech_queue)} | 先验证整条链：拍摄→导入→配准→QC→评分；**不进入研究数据** |
| main s1–s{n_sessions}（主批） | {len(main_queue)} | C_pool 200 + T_pilot 700 + 预定重复 100 |

**先只拍技术批**，把照片交回并确认通过后，再拍主批。

## 5. 照片交接

把照片原样复制到：

```
captures/incoming/tech/        ← 技术批（全部照片，按文件名排序即拍摄顺序）
captures/incoming/main_s1/ ...  ← 主批各段
```

不要改名、不要压缩、不要用微信/QQ 传输（会重编码）。用数据线复制原文件。

## 6. 失败与重拍规则

- 单张失败（模糊、反光、曝光过度、屏幕未显示完整）：原地重拍，原照片保留；
- 整段失败：停止，记录原因，整段重拍前先确认设备未移动；
- 不允许为了"好看"筛掉困难样本；所有照片都进入导入与 QC 记录。
"""
    kit_write(kit_dir / "capture_protocol.md", protocol, kind="protocol")

    for name in ["tech"] + [f"main_s{i}" for i in range(1, n_sessions + 1)]:
        bat = f'@echo off\r\ncd /d "{ROOT}"\r\n".venv\\Scripts\\python.exe" tools\\capture_viewer.py "captures\\kit\\queue_{name}.json" "{ROOT}"\r\npause\r\n'
        kit_write(kit_dir / f"run_{name}.bat", bat.encode("gbk"), kind="launcher")

    manifest = {
        "run_id": log.run_id,
        "exam02_run": run.name,
        "n_tech": len(tech_queue),
        "n_main": len(main_queue),
        "n_main_unique": sum(1 for it in main_queue if not it["is_repeat"]),
        "n_main_repeats": sum(1 for it in main_queue if it["is_repeat"]),
        "sessions": {"tech": len(tech_queue), **{f"main_s{i}": len(main_queue[i * per:(i + 1) * per]) for i in range(4)}},
        "files": {},
    }
    for p in sorted(kit_dir.rglob("*")):
        if p.is_file() and p.name != "kit_manifest.json":
            manifest["files"][str(p.relative_to(ROOT)).replace("\\", "/")] = sha256_file(p)
    kit_write(kit_dir / "kit_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2), kind="kit_manifest")
    log.finish("PASS", n_tech=len(tech_queue), n_main=len(main_queue))
    print(json.dumps(manifest["sessions"], ensure_ascii=False))
    print("tech queue:", len(tech_queue), "main queue:", len(main_queue),
          "unique:", manifest["n_main_unique"], "repeats:", manifest["n_main_repeats"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
