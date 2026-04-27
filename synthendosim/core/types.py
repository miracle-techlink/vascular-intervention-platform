"""
SynthEndoSim — Core type definitions.
All data flowing through the simulation is typed here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DeviceState:
    """Physical state of one interventional device (guidewire / catheter)."""
    dof_positions: np.ndarray          # (N, 3) world-space node positions
    inserted_length: float             # mm inserted past entry point
    rotation: float                    # rad, cumulative rotation at handle
    tip_position: np.ndarray           # (3,) world-space tip coords
    tip_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class SimState:
    """Complete simulation state after each step."""
    devices: List[DeviceState]
    step: int
    episode: int
    target_position: np.ndarray        # (3,) world-space target coords
    interim_target: Optional[np.ndarray] = None   # current interim waypoint
    path_remaining: float = 0.0        # geodesic distance to target (mm)
    simulation_error: bool = False


@dataclass
class Observation:
    """Structured observation returned to the RL agent."""
    tip_3d: np.ndarray                 # (3,) normalised tip position
    wire_mask: Optional[np.ndarray]    # (H, W) or (2, H, W) for biplane
    insertion_length: float
    rotation: float
    target_3d: np.ndarray              # (3,) normalised target position
    path_remaining: float
    extra: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        """Flatten to 1D array for simple MLP policies."""
        parts = [
            self.tip_3d,
            self.target_3d,
            np.array([self.insertion_length, self.rotation, self.path_remaining]),
        ]
        return np.concatenate(parts).astype(np.float32)

    def to_dict(self) -> Dict[str, np.ndarray]:
        d: Dict[str, np.ndarray] = {
            "tip_3d": self.tip_3d.astype(np.float32),
            "target_3d": self.target_3d.astype(np.float32),
            "insertion_length": np.array([self.insertion_length], dtype=np.float32),
            "rotation": np.array([self.rotation], dtype=np.float32),
            "path_remaining": np.array([self.path_remaining], dtype=np.float32),
        }
        if self.wire_mask is not None:
            d["wire_mask"] = self.wire_mask.astype(np.float32)
        d.update(self.extra)
        return d


@dataclass
class StepResult:
    obs: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict


@dataclass
class AnatomySpec:
    """Describes a vascular anatomy for the simulation."""
    mesh_path: str                                      # OBJ/STL collision mesh
    insertion_point: Tuple[float, float, float]         # world-space entry (mm)
    insertion_direction: Tuple[float, float, float]     # unit vector
    target_point: Tuple[float, float, float]            # navigation target (mm)
    centerline: Optional[np.ndarray] = None             # (N, 3) waypoints
    name: str = "unnamed"
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation_yzx_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    visu_mesh_path: Optional[str] = None


@dataclass
class DeviceSpec:
    """Parameterises a simulated device."""
    kind: str                   # "j_guidewire" | "straight" | "simmons"
    total_length: float = 450.0 # mm
    tip_length: float = 40.0    # mm (shaped section)
    tip_angle: float = 0.21     # rad
    diameter: float = 0.89      # mm (0.035")
    stiffness: float = 1.71e4   # BeamAdapter EI
    young_modulus: float = 1.7e5
    poisson_ratio: float = 0.49


@dataclass
class ImagingSpec:
    kind: str = "biplane_dsa"   # "biplane_dsa" | "monoplane" | "none"
    image_size: Tuple[int, int] = (128, 128)
    lao_deg: float = 30.0
    lat_deg: float = 120.0
    n0_photons: int = 15_000    # Beer-Lambert Poisson noise
    mu_blood: float = 0.048     # cm^-1
    mu_tissue: float = 0.020
