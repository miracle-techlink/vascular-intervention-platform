"""
SynthEndoSim — Abstract physics backend.
Concrete implementations: SofaBackend (production), MockBackend (testing).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np

from ..core.types import DeviceState, DeviceSpec, AnatomySpec


class PhysicsBackend(ABC):
    """
    Contract for any physics engine powering SynthEndoSim.
    Swap SofaBackend → DiffSimBackend without touching env code.
    """

    @abstractmethod
    def reset(
        self,
        anatomy: AnatomySpec,
        devices: List[DeviceSpec],
        seed: Optional[int] = None,
    ) -> List[DeviceState]:
        """Load anatomy mesh, initialise devices at insertion point, return initial state."""

    @abstractmethod
    def step(
        self,
        actions: np.ndarray,        # (n_devices, 2) — [translation_mm/s, rotation_rad/s]
        dt: float,                  # seconds to advance
    ) -> List[DeviceState]:
        """Advance simulation by dt seconds, return updated device states."""

    @abstractmethod
    def get_dof_positions(self) -> np.ndarray:
        """Return (N, 3) array of all wire node world positions (for imaging)."""

    @property
    @abstractmethod
    def simulation_error(self) -> bool:
        """True if SOFA (or other backend) reported a numerical instability."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources."""
