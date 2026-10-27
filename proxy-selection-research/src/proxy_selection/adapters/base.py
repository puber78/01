"""Detector adapter interface (unlabeled, single-image, RGB only).

A detector adapter must never accept labels, device ids, source ids or data
roles: it maps a PIL RGB image to one scalar score.  Everything that identifies
where an image came from is joined later, in the scoring layer, from manifests.

Contract
--------
* ``load()``             deterministic weights + official preprocessing
* ``preprocess()``       returns the official tensor batch (auditable spec)
* ``predict_batch()``    list[Prediction]; failures are explicit, never imputed
* ``score_direction``    'higher_is_ai' or 'higher_is_real'
* ``checkpoint_hash``    sha256 of the weight file(s)
* ``preprocess_hash``    sha256 of the preprocessing configuration
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from PIL import Image


@dataclass(frozen=True)
class Prediction:
    raw_score: float | None
    status: str
    failure_code: str | None = None
    detail: str = ""


class DetectorAdapter(ABC):
    """Abstract frozen single-image RGB detector."""

    name: str = "abstract"

    @abstractmethod
    def load(self) -> None:
        """Load weights and preprocessing; must be idempotent."""

    @abstractmethod
    def preprocess(self, images: Sequence[Image.Image]) -> Any:
        """Apply the official preprocessing to a batch of RGB images."""

    @abstractmethod
    def predict_batch(self, images: Sequence[Image.Image]) -> list[Prediction]:
        """Score a batch of RGB images without any label/identity input."""

    @property
    @abstractmethod
    def score_direction(self) -> str:
        """'higher_is_ai' or 'higher_is_real'."""

    @property
    @abstractmethod
    def checkpoint_hash(self) -> str:
        """SHA-256 over the resolved weight file bytes."""

    @property
    @abstractmethod
    def preprocess_hash(self) -> str:
        """SHA-256 over the preprocessing configuration used above."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable model identifier minted from repo+revision+weight hash."""

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "checkpoint_hash": self.checkpoint_hash,
            "preprocess_hash": self.preprocess_hash,
            "score_direction": self.score_direction,
        }
