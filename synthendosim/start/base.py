"""
SynthEndoSim — Abstract Start strategy.

Determines where and how the device is initialised at episode start.
The Start object receives the anatomy and returns the initial insertion state.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from ..core.types import AnatomySpec


@dataclass
class StartState:
    """Returned by Start.sample() — initial device state for this episode."""
    position: np.ndarray          # world-space insertion entry point (3,)
    direction: np.ndarray         # unit insertion direction (3,)
    inserted_length_mm: float = 0.0


class Start(ABC):
    """
    Determines the device starting configuration.
    Implement get_start() — sample() wraps it into StartState.
    """

    @abstractmethod
    def get_start(
        self,
        anatomy: AnatomySpec,
        episode: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position_3d, direction_unit_3d)."""

    def sample(
        self,
        anatomy: AnatomySpec,
        rng: Optional[np.random.Generator] = None,
        episode: int = 0,
    ) -> StartState:
        pos, direction = self.get_start(anatomy, episode=episode, rng=rng)
        return StartState(position=pos, direction=direction)

    def reset(self, anatomy: AnatomySpec = None, episode: int = 0) -> None: ...
