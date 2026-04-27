from .base import Target, TargetState
from .strategies import (
    FixedTarget,
    CenterlineRandomTarget,
    BranchEndTarget,
    BranchIndexTarget,
    ManualRandomTarget,
)

_REGISTRY = {
    "fixed":             FixedTarget,
    "centerline_random": CenterlineRandomTarget,
    "branch_end":        BranchEndTarget,
    "branch_index":      BranchIndexTarget,
    "manual_random":     ManualRandomTarget,
}


def make_target(kind: str, **kwargs) -> Target:
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown target '{kind}'. Options: {list(_REGISTRY)}")
    return _REGISTRY[kind](**kwargs)


def register(name: str, cls):
    _REGISTRY[name] = cls
