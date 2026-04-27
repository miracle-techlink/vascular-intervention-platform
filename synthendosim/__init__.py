"""
SynthEndoSim — Synthetic Endovascular Simulation Engine
========================================================

A clean, config-driven vascular intervention RL environment built on
SOFA + BeamAdapter FEM, with native biplane DSA rendering and
curvature-weighted manifold reward.

Quick start:
    import synthendosim as ses

    # Use preset
    env = ses.make("SynthEndoSim-AorticArch-CAS-v1",
                   anatomy_mesh="path/to/aorta.obj")

    # Build from mesh directly
    env = ses.from_mesh(
        mesh="path/to/aorta.obj",
        insertion_point=(-23, -180, -15),
        target_point=(-31.7, 70.3, 11.6),
    )

    # Fully custom
    env = ses.make_env(config_dict={
        "anatomy": {"mesh_path": "...", "target_point": [...]},
        "reward": {"manifold_distance_weight": 2.0},
        "physics": {"backend": "mock"},   # for testing without SOFA
    })

    obs, info = env.reset(seed=0)
    obs, reward, done, trunc, info = env.step(env.action_space.sample())
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .core.env import SynthEndoEnv
from .core.registry import get_config, register, list_envs
from .config.schema import EnvConfig, config_from_dict, load_config
from .core.types import (
    SimState, Observation, DeviceState,
    AnatomySpec, DeviceSpec, ImagingSpec,
)
from .reward.components import (
    ManifoldDistanceDelta, TargetReached,
    StepPenalty, WallCollisionPenalty,
    EuclideanDistanceDelta, CompositeReward,
)
from .imaging.dsa import BiplaneDSA, MonoplaneDSA
from .anatomy.loader import load_anatomy
from .anatomy.centerline import ManifoldPathfinder


__version__ = "0.1.0"


# Plugin kwarg names accepted by SynthEndoEnv
_PLUGIN_KEYS = {
    "terminal", "truncation", "start", "interimtarget",
    "pathfinder", "reward", "info_builder", "renderer", "devices",
}


def make(name: str, **overrides) -> SynthEndoEnv:
    """
    Create environment from a registered preset name.

    Anatomy overrides:
        anatomy_mesh  : str  — override mesh path

    Plugin overrides (injected directly into SynthEndoEnv):
        terminal, truncation, start, interimtarget,
        pathfinder, reward, info_builder, renderer

    Example:
        env = ses.make("SynthEndoSim-AorticArch-CAS-v1",
                       anatomy_mesh="data/KiTS_K1.obj",
                       terminal=TargetReachedTerminal(threshold_mm=5.0))
    """
    cfg = get_config(name)
    plugins = {k: overrides.pop(k) for k in list(overrides) if k in _PLUGIN_KEYS}
    if "anatomy_mesh" in overrides:
        cfg.anatomy.mesh_path = overrides.pop("anatomy_mesh")
    return SynthEndoEnv(cfg, **plugins)


def make_env(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    **plugins,
) -> SynthEndoEnv:
    """
    Create environment from a YAML file or a plain dict.

    Merges with base.yaml defaults, so you only need to specify overrides.
    Plugin objects can be injected as kwargs.

    Example:
        env = ses.make_env(
            config_dict={"physics": {"backend": "mock"}, ...},
            terminal=TargetReachedTerminal(threshold_mm=5.0),
            interimtarget=CenterlineWaypointTarget(n_waypoints=5),
        )
    """
    if config_path is not None:
        cfg = load_config(config_path)
    elif config_dict is not None:
        cfg = config_from_dict(config_dict)
    else:
        cfg = EnvConfig()
    valid = {k: v for k, v in plugins.items() if k in _PLUGIN_KEYS}
    return SynthEndoEnv(cfg, **valid)


def from_mesh(
    mesh: str,
    insertion_point: Tuple[float, float, float],
    target_point: Tuple[float, float, float],
    insertion_direction: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    centerline: Optional[str] = None,
    device: str = "j_guidewire",
    imaging: str = "biplane_dsa",
    backend: str = "sofa",
    max_steps: int = 300,
    **reward_kwargs,
) -> SynthEndoEnv:
    """
    Shortest path to create an env from a mesh file.

    Example:
        env = ses.from_mesh(
            mesh="data/aorta_K1.obj",
            insertion_point=(-23, -180, -15),
            target_point=(-31.7, 70.3, 11.6),
        )
    """
    cfg_dict: Dict[str, Any] = {
        "anatomy": {
            "mesh_path": mesh,
            "insertion_point": list(insertion_point),
            "insertion_direction": list(insertion_direction),
            "target_point": list(target_point),
        },
        "device": {"kind": device},
        "imaging": {"kind": imaging},
        "physics": {"backend": backend},
        "episode": {"max_steps": max_steps},
    }
    if centerline:
        cfg_dict["anatomy"]["centerline_path"] = centerline
    if reward_kwargs:
        cfg_dict["reward"] = reward_kwargs
    return make_env(config_dict=cfg_dict)


def make_vec_env(
    n_envs: int,
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    mesh_list: Optional[list] = None,
):
    """
    Create a vectorised environment (multiprocessing-based).
    Each worker gets its own SOFA instance.

    If mesh_list is provided (len >= n_envs), each worker loads a different anatomy.
    """
    from .vec_env import SynthEndoVecEnv
    return SynthEndoVecEnv(
        n_envs=n_envs,
        config_path=config_path,
        config_dict=config_dict,
        mesh_list=mesh_list,
    )
