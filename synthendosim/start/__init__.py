from .base import Start
from .strategies import (
    InsertionPointStart,
    RandomAdvanceStart,
    CurriculumStart,
    MultiAnatomyStart,
)

_REGISTRY = {
    "insertion_point":  InsertionPointStart,
    "random_advance":   RandomAdvanceStart,
    "curriculum":       CurriculumStart,
    "multi_anatomy":    MultiAnatomyStart,
}


def make_start(kind: str = "insertion_point", **kwargs) -> Start:
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown start strategy: {kind!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[kind](**kwargs)


def register_start(name: str, cls):
    _REGISTRY[name] = cls
