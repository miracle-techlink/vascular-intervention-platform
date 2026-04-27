"""
SynthEndoSim — Abstract Target.

The Target subsystem selects the navigation goal for each episode.
Decoupled from SimState — targets can change between episodes but not mid-episode.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..core.types import AnatomySpec


@dataclass
class TargetState:
    position: np.ndarray    # (3,) world-space goal position
    threshold_mm: float     # success radius
    label: str = "target"   # human-readable label (e.g. "LCCA")


class Target(ABC):
    """
    Selects and tracks the navigation target for each episode.

    reset() is called at the start of each episode and sets self.state.
    """
    state: TargetState

    @abstractmethod
    def reset(
        self,
        anatomy: AnatomySpec,
        episode: int = 0,
        seed: Optional[int] = None,
    ) -> TargetState:
        """Sample a target for this episode. Returns TargetState."""

    def is_reached(self, tip: np.ndarray) -> bool:
        dist = float(np.linalg.norm(tip - self.state.position))
        return dist <= self.state.threshold_mm

    @property
    def position(self) -> np.ndarray:
        return self.state.position

    @property
    def threshold_mm(self) -> float:
        return self.state.threshold_mm
