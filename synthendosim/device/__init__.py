from .base import Device, SofaDeviceParams
from .guidewire import JShapedGuidewire, StraightGuidewire, HydrophilicGuidewire
from .catheter import SimmonsCatheter, PigtailCatheter, SheathCatheter

_DEVICE_REGISTRY = {
    "j_guidewire":          JShapedGuidewire,
    "straight_guidewire":   StraightGuidewire,
    "hydrophilic_guidewire": HydrophilicGuidewire,
    "simmons_catheter":     SimmonsCatheter,
    "pigtail_catheter":     PigtailCatheter,
    "sheath":               SheathCatheter,
}


def make_device(kind: str, **kwargs) -> Device:
    """Factory: make_device("j_guidewire", tip_angle=0.3)"""
    if kind not in _DEVICE_REGISTRY:
        raise ValueError(
            f"Unknown device: {kind!r}. Available: {list(_DEVICE_REGISTRY)}"
        )
    return _DEVICE_REGISTRY[kind](**kwargs)


def register_device(name: str, cls):
    """Register a custom device class."""
    _DEVICE_REGISTRY[name] = cls
