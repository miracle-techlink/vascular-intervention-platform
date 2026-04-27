"""
SynthEndoSim — Abstract InterimTarget.

Manages intermediate waypoints along the navigation path.
Instead of directly targeting the final vessel, the agent
follows a sequence of interim waypoints — useful for:
  - Dense reward shaping (each waypoint = bonus)
  - Multi-stage procedure modelling (CAS: aorta → arch → LCCA)
  - Curriculum: agent only sees the next local goal
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from ..core.types import SimState, AnatomySpec


class InterimTarget(ABC):
    """
    Provides the current sub-goal position.
    Returns None when no interim target is active (use final target).
    """

    @abstractmethod
    def current(self, state: SimState) -> Optional[np.ndarray]:
        """Return current interim waypoint (3,) or None."""

    @abstractmethod
    def advance(self, state: SimState) -> bool:
        """
        Check if current interim target is reached; if so, advance to next.
        Returns True if advanced (reward can be given).
        """

    def reset(self, anatomy: AnatomySpec, episode: int = 0) -> None: ...

    @property
    def remaining_count(self) -> int:
        """Number of remaining waypoints (excluding final target)."""
        return 0


class NoInterimTarget(InterimTarget):
    """No waypoints — agent goes directly for the final target."""
    def current(self, state: SimState) -> Optional[np.ndarray]:
        return None

    def advance(self, state: SimState) -> bool:
        return False
