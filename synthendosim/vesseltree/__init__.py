from .base import VesselTree, Insertion
from .aorticarch import AorticArch, ARCH_TYPES
from .aorticarchrandom import AorticArchRandom
from .frommesh import VesselTreeFromMesh
from .util.branch import Branch, BranchWithRadii, BranchingPoint

_REGISTRY = {
    "aortic_arch":        AorticArch,
    "aortic_arch_random": AorticArchRandom,
    "from_mesh":          VesselTreeFromMesh,
}


def make_vessel_tree(kind: str, **kwargs) -> VesselTree:
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown vessel tree '{kind}'. Options: {list(_REGISTRY)}")
    return _REGISTRY[kind](**kwargs)


def register(name: str, cls):
    _REGISTRY[name] = cls
