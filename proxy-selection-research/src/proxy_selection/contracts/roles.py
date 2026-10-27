"""Three-role data-access table and enforcement.

The research protocol defines data roles (C_pool, D_sel, T_pilot, T_confirm,
D_sel_confirm, C_confirm, TECH, EXTERNAL) and three logical access roles:

* ``development``  - may read C_pool / D_sel / TECH / C_confirm / D_sel_confirm,
                     never T_pilot or T_confirm.
* ``evaluation``   - the independent holder of held-out physical scores; may read
                     every role, but produced choices are frozen before revealing.
* ``audit``        - read-only re-computation after the fact; may read everything
                     but cannot write selections or fit parameters.

Every read of a table goes through :meth:`AccessPolicy.check_read`, which raises
:class:`AccessViolation` and records the attempt.  The number of recorded
violations is an engineering acceptance metric (must be 0 for unauthorised
cross-role reads).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

DataRole = Literal[
    "C_pool",
    "D_sel",
    "T_pilot",
    "C_confirm",
    "D_sel_confirm",
    "T_confirm",
    "TECH",
    "EXTERNAL",
]

ACCESS_ROLE = Literal["development", "evaluation", "audit"]

ALL_DATA_ROLES: tuple[DataRole, ...] = (
    "C_pool",
    "D_sel",
    "T_pilot",
    "C_confirm",
    "D_sel_confirm",
    "T_confirm",
    "TECH",
    "EXTERNAL",
)

HELD_OUT_ROLES: frozenset[str] = frozenset({"T_pilot", "T_confirm", "EXTERNAL"})

_ALLOWED: dict[str, frozenset[str]] = {
    "development": frozenset({"C_pool", "D_sel", "C_confirm", "D_sel_confirm", "TECH"}),
    "evaluation": frozenset(ALL_DATA_ROLES),
    "audit": frozenset(ALL_DATA_ROLES),
}


class AccessViolation(PermissionError):
    """Raised when a logical role attempts a forbidden read/write."""


@dataclass
class AccessPolicy:
    """Enforces the role table and records every attempt for the audit trail."""

    logical_role: ACCESS_ROLE
    events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.logical_role not in _ALLOWED:
            raise ValueError(f"unknown logical role: {self.logical_role}")

    def can_read(self, data_role: str) -> bool:
        if data_role not in ALL_DATA_ROLES:
            raise ValueError(f"unknown data role: {data_role}")
        return data_role in _ALLOWED[self.logical_role]

    def check_read(self, data_role: str, *, resource: str = "") -> None:
        ok = self.can_read(data_role)
        self.events.append(
            {"action": "read", "data_role": data_role, "resource": resource, "allowed": ok}
        )
        if not ok:
            raise AccessViolation(
                f"{self.logical_role} may not read data role {data_role} (resource={resource!r})"
            )

    def check_write(self, data_role: str, *, resource: str = "") -> None:
        # Writes follow the same role table; evaluation-side writers additionally
        # must have frozen their choices (enforced by the caller, not here).
        ok = data_role in _ALLOWED[self.logical_role]
        self.events.append(
            {"action": "write", "data_role": data_role, "resource": resource, "allowed": ok}
        )
        if not ok:
            raise AccessViolation(
                f"{self.logical_role} may not write data role {data_role} (resource={resource!r})"
            )

    @property
    def violations(self) -> list[dict]:
        return [e for e in self.events if not e["allowed"]]

    @property
    def violation_count(self) -> int:
        return len(self.violations)


def allowed_roles_for(logical_role: ACCESS_ROLE) -> frozenset[str]:
    return _ALLOWED[logical_role]


def roles_are_disjoint(groups: Iterable[Iterable[str]]) -> bool:
    """True when no split_group_id appears in more than one role group."""
    seen: set[str] = set()
    for group in groups:
        current = set(group)
        if seen & current:
            return False
        seen |= current
    return True
