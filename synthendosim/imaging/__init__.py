from .dsa import BiplaneDSA, MonoplaneDSA, NullImager

def make_imager(kind: str, **kwargs):
    if kind == "biplane_dsa":
        return BiplaneDSA(**kwargs)
    elif kind == "monoplane":
        return MonoplaneDSA(**kwargs)
    elif kind == "none":
        return NullImager()
    raise ValueError(f"Unknown imager kind: {kind!r}")
