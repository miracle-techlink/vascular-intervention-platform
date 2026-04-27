"""
SynthEndoSim — Environment registry.
Preset configs keyed by name, extensible at runtime.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
from ..config.schema import EnvConfig, config_from_dict, load_config

_REGISTRY: Dict[str, EnvConfig] = {}

_PRESETS_DIR = Path(__file__).parent.parent / "config" / "defaults"


def _lazy_load_presets():
    if _REGISTRY:
        return
    for yaml_path in _PRESETS_DIR.glob("*.yaml"):
        if yaml_path.stem == "base":
            continue
        try:
            cfg = load_config(yaml_path)
            _REGISTRY[cfg.name] = cfg
        except Exception:
            pass


def register(name: str, cfg: EnvConfig):
    """Register a custom environment config under a name."""
    _REGISTRY[name] = cfg


def get_config(name: str) -> EnvConfig:
    _lazy_load_presets()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown environment: {name!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_envs():
    _lazy_load_presets()
    return list(_REGISTRY.keys())
