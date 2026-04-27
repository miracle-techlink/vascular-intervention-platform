from .base import InterimTarget, NoInterimTarget
from .waypoints import (
    FixedWaypointTarget,
    CenterlineWaypointTarget,
    ProximityWindowTarget,
    StageTarget,
)

_REGISTRY = {
    "none":             NoInterimTarget,
    "fixed":            FixedWaypointTarget,
    "centerline":       CenterlineWaypointTarget,
    "proximity_window": ProximityWindowTarget,
    "stage":            StageTarget,
}


def make_interimtarget(kind: str = "none", **kwargs) -> InterimTarget:
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown interimtarget kind '{kind}'. Choose from: {list(_REGISTRY)}")
    return _REGISTRY[kind](**kwargs)


def register(name: str, cls):
    """Register a custom InterimTarget class."""
    _REGISTRY[name] = cls
