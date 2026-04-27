from .dsa import BiplaneDSA, MonoplaneDSA, NullImager
from .pillow import PillowImager


class Imager:
    """Abstract base for all imagers."""
    def set_anatomy_bounds(self, bbox_min, bbox_max): ...
    def render(self, wire_nodes, rng=None): ...
    def render_lao(self, wire_nodes, rng=None): ...


def make_imager(kind: str, **kwargs):
    if kind == "biplane_dsa":
        return BiplaneDSA(**kwargs)
    elif kind == "monoplane":
        return MonoplaneDSA(**kwargs)
    elif kind == "pillow":
        return PillowImager(**kwargs)
    elif kind == "none":
        return NullImager()
    raise ValueError(f"Unknown imager kind: {kind!r}. Options: biplane_dsa, monoplane, pillow, none")
