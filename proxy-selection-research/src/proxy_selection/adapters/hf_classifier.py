"""HuggingFace ``transformers`` image-classification adapter for AI-image
detectors.

Design points that matter for auditability:

* the snapshot is resolved once with a pinned *revision* (commit sha when the
  caller supplies one) and the weight file bytes are hashed;
* preprocessing comes from the model's own ``AutoImageProcessor`` -- no
  hand-rolled resize/normalise;
* the AI-positive class is derived from ``config.id2label`` by explicit keyword
  matching, and an ambiguous label map is a hard engineering failure
  (``ORIENTATION_UNDETERMINED``) rather than a guess;
* inference is ``model.eval()`` + ``torch.inference_mode()`` with a fixed batch
  size, and ``predict_batch`` returns explicit failure codes instead of imputed
  scores.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from ..adapters.base import DetectorAdapter, Prediction
from ..contracts.ids import mint_model_id
from ..hashing import sha256_file, sha256_json

AI_TOKENS = {
    "ai",
    "aigenerated",
    "artificial",
    "fake",
    "generated",
    "synthetic",
    "deepfake",
    "sdxl",
    "midjourney",
    "dalle",
    "diffusion",
    "gan",
    "machine",
}
REAL_TOKENS = {"real", "human", "authentic", "natural", "photo", "photographic", "camera", "original"}


class OrientationUndetermined(RuntimeError):
    pass


def _tokens(label: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(label).strip().lower()) if t}


def resolve_ai_index(id2label: dict[int, str], override_label: str | None = None) -> tuple[int, str]:
    """Return (index, label_name) of the AI-generated class.

    ``override_label`` is an exact label string documented in the model
    registry together with evidence; it is required when the id2label map is
    generic (``LABEL_0``/``LABEL_1``).  Anything ambiguous raises instead of
    guessing, because a flipped orientation would silently mirror every AUC.
    """
    if override_label is not None:
        for idx, raw in id2label.items():
            if str(raw) == override_label:
                return int(idx), str(raw)
        raise OrientationUndetermined(
            f"override label {override_label!r} not present in id2label={id2label}"
        )

    ai_hits = [int(i) for i, raw in id2label.items() if _tokens(raw) & AI_TOKENS]
    real_hits = [int(i) for i, raw in id2label.items() if _tokens(raw) & REAL_TOKENS]

    if len(ai_hits) == 1 and len(real_hits) == 1 and ai_hits[0] != real_hits[0]:
        return ai_hits[0], id2label[ai_hits[0]]
    if len(id2label) == 2 and len(ai_hits) == 1 and len(real_hits) == 0:
        other = [i for i in id2label if i != ai_hits[0]]
        return ai_hits[0], id2label[ai_hits[0]] if other else id2label[ai_hits[0]]
    if len(id2label) == 2 and len(real_hits) == 1 and len(ai_hits) == 0:
        other = [i for i in id2label if i != real_hits[0]]
        if len(other) == 1:
            return other[0], id2label[other[0]]
    raise OrientationUndetermined(
        f"cannot determine AI-positive class from id2label={id2label}; "
        "supply a documented exact ai_label in the model registry"
    )


def weight_files(snapshot_dir: Path) -> list[Path]:
    names = ("model.safetensors", "pytorch_model.bin", "model.pt", "weights.pth")
    found = [snapshot_dir / n for n in names if (snapshot_dir / n).exists()]
    if not found:
        found = sorted(
            p for p in snapshot_dir.iterdir() if p.suffix in {".safetensors", ".bin", ".pt", ".pth"}
        )
    return found


@dataclass
class HFClassifierAdapter(DetectorAdapter):
    repo: str
    revision: str | None = None
    cache_dir: str | Path | None = None
    local_files_only: bool = False
    device: str = "cuda"
    batch_size: int = 16
    name: str = "hf-classifier"
    ai_label_override: str | None = None
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _checkpoint_hash: str = field(default="", init=False)
    _preprocess_hash: str = field(default="", init=False)
    _model_id: str = field(default="", init=False)
    _orientation: str = field(default="", init=False)
    _ai_index: int = field(default=-1, init=False)
    _snapshot_dir: Path | None = field(default=None, init=False, repr=False)
    metadata: dict = field(default_factory=dict, init=False)

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        kwargs: dict[str, Any] = {}
        if self.revision:
            kwargs["revision"] = self.revision
        if self.cache_dir:
            kwargs["cache_dir"] = str(self.cache_dir)
        snapshot = snapshot_download(
            repo_id=self.repo,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.bin",
                "*.txt",
                "*.md",
            ],
            local_files_only=self.local_files_only,
            **kwargs,
        )
        snap = Path(snapshot)
        self._snapshot_dir = snap

        self._processor = AutoImageProcessor.from_pretrained(snapshot)
        self._model = AutoModelForImageClassification.from_pretrained(snapshot)
        self._model.eval()
        self._model.to(self.device if torch.cuda.is_available() else "cpu")
        self.device = str(next(self._model.parameters()).device)

        files = weight_files(snap)
        if not files:
            raise FileNotFoundError(f"no weight file found in snapshot {snap}")
        self._checkpoint_hash = sha256_json(
            {str(f.name): sha256_file(f) for f in sorted(files)}
        )
        self._preprocess_hash = sha256_json(
            {
                "preprocessor_config": json.loads((snap / "preprocessor_config.json").read_text("utf-8"))
                if (snap / "preprocessor_config.json").exists()
                else {},
                "image_processor_type": self._processor.__class__.__name__,
            }
        )
        cfg = json.loads((snap / "config.json").read_text("utf-8"))
        id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()} or {0: "LABEL_0", 1: "LABEL_1"}
        self._ai_index, ai_label = resolve_ai_index(id2label, self.ai_label_override)
        self._orientation = "higher_is_ai"
        self._model_id = mint_model_id(
            repo=self.repo, revision=self.revision or "", weight_sha256=self._checkpoint_hash
        )
        self.metadata = {
            "repo": self.repo,
            "requested_revision": self.revision,
            "snapshot_dir": str(snap),
            "snapshot_files": {p.name: sha256_file(p) for p in sorted(snap.iterdir()) if p.is_file()},
            "id2label": id2label,
            "ai_index": self._ai_index,
            "ai_label": ai_label,
            "architectures": cfg.get("architectures"),
            "num_parameters": int(sum(p.numel() for p in self._model.parameters())),
            "device": self.device,
            "batch_size": self.batch_size,
        }

    # -- interface ---------------------------------------------------------
    def preprocess(self, images: Sequence[Image.Image]):
        import torch

        if self._processor is None:
            raise RuntimeError("load() must be called first")
        inputs = self._processor(images=list(images), return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict_batch(self, images: Sequence[Image.Image]) -> list[Prediction]:
        import torch
        import torch.nn.functional as F

        if self._model is None:
            raise RuntimeError("load() must be called first")
        out: list[Prediction] = []
        for start in range(0, len(images), self.batch_size):
            chunk = list(images[start : start + self.batch_size])
            try:
                inputs = self.preprocess(chunk)
                with torch.inference_mode():
                    logits = self._model(**inputs).logits.float()
                if logits.shape[1] == 2:
                    probs = F.softmax(logits, dim=-1)
                elif logits.shape[1] == 1:
                    probs = torch.sigmoid(logits)
                else:
                    raise ValueError(f"unexpected logits shape {tuple(logits.shape)}")
                p_ai = probs[:, self._ai_index].detach().cpu().numpy()
                for v in p_ai:
                    if not torch.isfinite(torch.tensor(float(v))):
                        out.append(Prediction(None, "failed", "NON_FINITE_SCORE"))
                    else:
                        out.append(Prediction(float(v), "ok", None))
            except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - environment specific
                torch.cuda.empty_cache()
                out.extend(Prediction(None, "failed", "OOM", str(exc)) for _ in chunk)
            except Exception as exc:  # noqa: BLE001 - explicit failure path
                out.extend(Prediction(None, "failed", "MODEL_ERROR", str(exc)) for _ in chunk)
        return out

    @property
    def score_direction(self) -> str:
        return self._orientation or "higher_is_ai"

    @property
    def checkpoint_hash(self) -> str:
        if not self._checkpoint_hash:
            raise RuntimeError("load() must be called first")
        return self._checkpoint_hash

    @property
    def preprocess_hash(self) -> str:
        if not self._preprocess_hash:
            raise RuntimeError("load() must be called first")
        return self._preprocess_hash

    @property
    def model_id(self) -> str:
        if not self._model_id:
            raise RuntimeError("load() must be called first")
        return self._model_id

    def config_fingerprint(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "revision": self.revision,
            "checkpoint_hash": self.checkpoint_hash,
            "preprocess_hash": self.preprocess_hash,
            "orientation": self.score_direction,
            "ai_index": self._ai_index,
            "batch_size": self.batch_size,
            "device": self.device,
        }
