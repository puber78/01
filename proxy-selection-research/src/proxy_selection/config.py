"""Project configuration namespace.

Everything the research code reads lives under this package's project root
(``proxy-selection-research/``).  Paths are resolved from the module location so
scripts behave the same regardless of the shell's working directory.

The parameter registry deliberately refuses to treat proposed values as
approved: ``proposed_parameters.yaml`` carries ``approval_status:
DRAFT_UNAPPROVED`` and the loader records that status with every run.  Scientific
thresholds must not be silently promoted to "approved" by code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPS = PROJECT_ROOT / "ops"
LOGS = OPS / "logs"
MANIFESTS = PROJECT_ROOT / "manifests"
PREDICTIONS = PROJECT_ROOT / "predictions"
SCORES = PROJECT_ROOT / "scores"
SELECTIONS = PROJECT_ROOT / "selections"
METRICS = PROJECT_ROOT / "metrics"
RESULTS = PROJECT_ROOT / "results"
REPORTS = PROJECT_ROOT / "reports"
PROTOCOLS = PROJECT_ROOT / "protocols"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
ARCHIVE = OPS / "archive"
TMP = PROJECT_ROOT / "tmp"

DATA_ROLE_DIRS = {
    "TECH": "EXAM01",
    "C_pool": "C_pool",
    "D_sel": "D_sel",
    "C_confirm": "C_confirm",
    "D_sel_confirm": "D_sel_confirm",
    "T_pilot": "T_pilot",
    "T_confirm": "T_confirm",
    "EXTERNAL": "EXTERNAL",
}


def ensure_dirs() -> None:
    for p in (
        LOGS,
        LOGS / "runs",
        MANIFESTS,
        SCORES,
        SELECTIONS,
        METRICS,
        RESULTS,
        REPORTS,
        PROTOCOLS,
        DATA_RAW,
        DATA_PROCESSED,
        MODELS,
        ARCHIVE,
        TMP,
    ):
        p.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return data


@dataclass(frozen=True)
class ProposedParameters:
    """The still-unapproved numerical initial values from the execution outline.

    ``approved`` is False unless a human edits the registry to APPROVED with an
    approver name; research code must check ``require_approved`` before using a
    threshold in a scientific decision.
    """

    path: Path
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def approval_status(self) -> str:
        return str(self.data.get("approval_status", "MISSING"))

    @property
    def approved(self) -> bool:
        return self.approval_status == "APPROVED" and bool(self.data.get("approved_by"))

    def values(self) -> dict[str, Any]:
        return dict(self.data.get("proposed_parameters", {}))

    def require_approved(self) -> None:
        if not self.approved:
            raise PermissionError(
                f"parameters at {self.path} are {self.approval_status}; "
                "a human approval is required before scientific use"
            )


def load_proposed_parameters() -> ProposedParameters:
    path = PROTOCOLS / "proposed_parameters.yaml"
    return ProposedParameters(path=path, data=load_yaml(path))


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    scores: Path = SCORES
    selections: Path = SELECTIONS
    metrics: Path = METRICS
    results: Path = RESULTS
    reports: Path = REPORTS
    manifests: Path = MANIFESTS
    models: Path = MODELS
    data_raw: Path = DATA_RAW
    tmp: Path = TMP

    def score_dir(self, exam: str) -> Path:
        p = self.scores / exam
        p.mkdir(parents=True, exist_ok=True)
        return p


PATHS = Paths()


def hf_endpoint() -> str | None:
    return os.environ.get("HF_ENDPOINT")
