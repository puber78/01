"""Full-screen capture viewer for the screen-recapture sessions.

Displays one prepared display asset at a time on a black background and records
exactly which asset was shown at which time, so that captured photos can be
mapped back to sources without relying on memory.

Controls
--------
Right arrow / Space : "shot taken" -> record and show the next image
Left arrow          : step back (does not delete a recorded shot)
S                   : mark the current item as skipped (recorded as skipped)
Q / Esc             : save and quit

The viewer never draws anything on top of the image while it is displayed.
Progress is written after every step to ``capture_progress.json`` next to the
queue file, so a crash or a break loses nothing.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tkinter import Tk, Canvas, TclError

from PIL import Image, ImageTk

BLACK = "#000000"


class Viewer:
    def __init__(self, root: Tk, queue_path: Path, project_root: Path):
        self.root = root
        self.queue_path = queue_path
        self.project_root = project_root
        self.queue = json.loads(queue_path.read_text(encoding="utf-8"))
        self.items = self.queue["items"]
        self.progress_path = queue_path.with_name(queue_path.stem + "_progress.json")
        self.state = {"items": {}, "started_at": self._now(), "session": self.queue.get("session", "")}
        if self.progress_path.exists():
            try:
                saved = json.loads(self.progress_path.read_text(encoding="utf-8"))
                if len(saved.get("items", {})) and saved.get("session") == self.state["session"]:
                    self.state = saved
            except json.JSONDecodeError:
                pass
        self.index = self._first_pending()
        self.photo = None
        self.canvas = Canvas(root, bg=BLACK, highlightthickness=0, cursor="none")
        self.canvas.pack(fill="both", expand=True)
        self.image_id = None
        self.label_id = None
        root.bind("<Right>", lambda e: self.advance("shot"))
        root.bind("<space>", lambda e: self.advance("shot"))
        root.bind("<Left>", lambda e: self.back())
        root.bind("<s>", lambda e: self.advance("skipped"))
        root.bind("<q>", lambda e: self.quit())
        root.bind("<Escape>", lambda e: self.quit())
        root.bind("<Configure>", lambda e: self.render())
        self.show_current()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _first_pending(self) -> int:
        for i, it in enumerate(self.items):
            st = self.state["items"].get(str(it["seq"]), {}).get("status")
            if st not in ("shot", "skipped"):
                return i
        return len(self.items) - 1

    def save(self) -> None:
        self.progress_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def render(self) -> None:
        if self.index >= len(self.items):
            self.show_message("队列完成。按 Q 退出。", "全部完成")
            return
        item = self.items[self.index]
        path = self.project_root / item["file_relpath"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            self.show_message(f"无法打开图像：{exc}", item.get("label", ""))
            return
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        scale = min(sw / img.width, sh / img.height)
        size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img = img.resize(size, Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(sw // 2, sh // 2, image=self.photo, anchor="center")
        self.image_id = None

    def show_message(self, text: str, subtitle: str = "") -> None:
        self.canvas.delete("all")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.canvas.create_text(sw // 2, sh // 2 - 40, text=text, fill="#888888", font=("Segoe UI", 40))
        if subtitle:
            self.canvas.create_text(sw // 2, sh // 2 + 40, text=subtitle, fill="#444444", font=("Segoe UI", 28))

    def show_current(self) -> None:
        if self.index >= len(self.items):
            self.show_message("队列完成。按 Q 退出。", "全部完成")
            return
        item = self.items[self.index]
        st = self.state["items"].get(str(item["seq"]), {}).get("status")
        mode = "已拍，可重拍" if st == "shot" else ""
        # Short interstitial: the number is visible only between images.
        self.show_message(f"{self.index + 1} / {len(self.items)}", mode)
        self.root.after(700, self.render)

    def advance(self, status: str) -> None:
        if self.index >= len(self.items):
            return
        item = self.items[self.index]
        self.state["items"][str(item["seq"])] = {
            "status": status,
            "display_asset_id": item["display_asset_id"],
            "recorded_at": self._now(),
        }
        self.save()
        if self.index < len(self.items) - 1:
            self.index += 1
            self.show_current()
        else:
            self.index += 1
            self.show_message("队列完成。按 Q 退出。", "全部完成")

    def back(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.show_current()

    def quit(self) -> None:
        self.save()
        self.root.destroy()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: capture_viewer.py <queue.json> [project_root]")
        return 2
    queue_path = Path(sys.argv[1]).resolve()
    project_root = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else queue_path.parents[2]
    root = Tk()
    root.title("Capture Viewer")
    root.configure(bg=BLACK)
    root.attributes("-fullscreen", True)
    try:
        root.attributes("-topmost", True)
    except TclError:
        pass
    Viewer(root, queue_path, project_root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
