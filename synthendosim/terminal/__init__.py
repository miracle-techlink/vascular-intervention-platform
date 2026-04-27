from .base import Terminal, AnyTerminal, AllTerminal, NeverTerminal
from .conditions import TargetReachedTerminal, AllTargetsReachedTerminal, PathRatioTerminal

_REGISTRY = {
    "target_reached":      TargetReachedTerminal,
    "all_targets_reached": AllTargetsReachedTerminal,
    "path_ratio":          PathRatioTerminal,
    "never":               NeverTerminal,
}


def make_terminal(kind: str = "target_reached", **kwargs) -> Terminal:
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown terminal: {kind!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[kind](**kwargs)


def register_terminal(name: str, cls):
    _REGISTRY[name] = cls
