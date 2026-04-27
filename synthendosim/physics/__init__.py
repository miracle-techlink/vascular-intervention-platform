from .backend import PhysicsBackend
from .mock_backend import MockBackend

def make_backend(kind: str, **kwargs) -> PhysicsBackend:
    if kind == "sofa":
        from .sofa_backend import SofaBackend
        return SofaBackend(**kwargs)
    elif kind == "mock":
        return MockBackend(**kwargs)
    raise ValueError(f"Unknown physics backend: {kind!r}")
