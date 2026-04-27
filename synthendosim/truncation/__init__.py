from .base import Truncation, AnyTruncation, NeverTruncate
from .conditions import (
    MaxStepsTruncation, SimErrorTruncation,
    VesselEndTruncation, NoProgressTruncation,
)

_REGISTRY = {
    "max_steps":    MaxStepsTruncation,
    "sim_error":    SimErrorTruncation,
    "vessel_end":   VesselEndTruncation,
    "no_progress":  NoProgressTruncation,
    "never":        NeverTruncate,
}


def make_truncation(kind: str = "max_steps", **kwargs) -> Truncation:
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown truncation: {kind!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[kind](**kwargs)


def build_default_truncation(max_steps: int = 300) -> Truncation:
    """Default: timeout OR physics error."""
    return MaxStepsTruncation(max_steps) | SimErrorTruncation()


def register_truncation(name: str, cls):
    _REGISTRY[name] = cls
