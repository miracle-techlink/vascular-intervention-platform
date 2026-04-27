"""
SynthEndoSim — YAML config schema + loader.
All environment parameters live here; no hardcoded values in env code.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


@dataclass
class PhysicsConfig:
    backend: str = "sofa"          # "sofa" | "mock" (for testing)
    dt_simulation: float = 0.006   # s per substep
    friction: float = 0.1


@dataclass
class DeviceConfig:
    kind: str = "j_guidewire"
    total_length: float = 450.0
    tip_length: float = 40.0
    tip_angle: float = 0.21
    diameter: float = 0.89
    stiffness: float = 1.71e4
    young_modulus: float = 1.7e5
    poisson_ratio: float = 0.49


@dataclass
class AnatomyConfig:
    mesh_path: str = ""
    insertion_point: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    insertion_direction: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    target_point: List[float] = field(default_factory=lambda: [0.0, 100.0, 0.0])
    centerline_path: Optional[str] = None
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    rotation_yzx_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    visu_mesh_path: Optional[str] = None
    name: str = "custom"


@dataclass
class ImagingConfig:
    kind: str = "biplane_dsa"
    image_size: List[int] = field(default_factory=lambda: [128, 128])
    lao_deg: float = 30.0
    lat_deg: float = 120.0
    n0_photons: int = 15_000
    mu_blood: float = 0.048
    mu_tissue: float = 0.020


@dataclass
class RewardConfig:
    manifold_distance_weight: float = 1.0
    target_reached_bonus: float = 10.0
    step_penalty: float = -0.01
    wall_collision_penalty: float = -1.0
    target_threshold_mm: float = 10.0
    use_manifold: bool = True         # False → euclidean fallback


@dataclass
class ObsConfig:
    use_tip_3d: bool = True
    use_wire_mask: bool = False       # set True to include DSA image obs
    use_path_remaining: bool = True
    use_insertion_length: bool = True
    use_rotation: bool = True
    normalise: bool = True


@dataclass
class EpisodeConfig:
    max_steps: int = 300
    seed_train: Optional[int] = None
    seed_eval: Optional[int] = None
    held_out_seeds: List[int] = field(default_factory=list)


@dataclass
class EnvConfig:
    name: str = "SynthEndoSim-v1"
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    anatomy: AnatomyConfig = field(default_factory=AnatomyConfig)
    imaging: ImagingConfig = field(default_factory=ImagingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def _dict_to_config(d: dict) -> EnvConfig:
    cfg = EnvConfig()
    if "name" in d:
        cfg.name = d["name"]
    if "physics" in d:
        cfg.physics = PhysicsConfig(**d["physics"])
    if "device" in d:
        cfg.device = DeviceConfig(**d["device"])
    if "anatomy" in d:
        cfg.anatomy = AnatomyConfig(**d["anatomy"])
    if "imaging" in d:
        cfg.imaging = ImagingConfig(**d["imaging"])
    if "reward" in d:
        cfg.reward = RewardConfig(**d["reward"])
    if "obs" in d:
        cfg.obs = ObsConfig(**d["obs"])
    if "episode" in d:
        cfg.episode = EpisodeConfig(**d["episode"])
    return cfg


def load_config(path: str | Path) -> EnvConfig:
    defaults_path = Path(__file__).parent / "defaults" / "base.yaml"
    base: dict = {}
    if defaults_path.exists():
        with open(defaults_path) as f:
            base = yaml.safe_load(f) or {}
    with open(path) as f:
        override = yaml.safe_load(f) or {}
    merged = _merge(base, override)
    return _dict_to_config(merged)


def config_from_dict(d: dict) -> EnvConfig:
    return _dict_to_config(d)
