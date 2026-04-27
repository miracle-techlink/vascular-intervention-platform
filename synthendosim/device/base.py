"""
SynthEndoSim — Abstract Device interface.

Any interventional device (guidewire, catheter, sheath) must implement this.
The physics backend receives a list of Device objects and builds SOFA nodes from them.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SofaDeviceParams:
    """Low-level parameters passed directly to SOFA WireRestShape."""
    is_a_procedural_shape: bool = True
    straight_length: float = 400.0
    length: float = 450.0
    spire_diameter: float = 4.0
    spire_height: float = 0.0
    radius_extremity: float = 0.45
    young_modulus_extremity: float = 85_000.0
    mass_density_extremity: float = 0.000155
    radius: float = 0.45
    young_modulus: float = 170_000.0
    mass_density: float = 0.000155
    poisson_ratio: float = 0.49
    key_points: List[float] = field(default_factory=lambda: [0.0, 400.0, 450.0])
    density_of_beams: List[int] = field(default_factory=lambda: [40, 10])
    num_edges_collis: List[int] = field(default_factory=lambda: [20, 10])
    num_edges: int = 50
    mesh_path: str = None      # None = procedural shape


class Device(ABC):
    """Abstract interventional device."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name (used as SOFA node identifier)."""

    @property
    @abstractmethod
    def sofa_params(self) -> SofaDeviceParams:
        """Parameters for SOFA WireRestShape construction."""

    @property
    def color(self) -> Tuple[float, float, float, float]:
        """RGBA color for 3D visualisation."""
        return (0.1, 0.1, 0.8, 1.0)
