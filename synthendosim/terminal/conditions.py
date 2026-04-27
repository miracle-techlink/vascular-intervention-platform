"""
SynthEndoSim — Terminal condition implementations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .base import Terminal
from ..core.types import SimState


@dataclass
class TargetReachedTerminal(Terminal):
    """
    Terminal when the leading device tip is within `threshold_mm` of the target.
    Default 10mm matches clinical "within target vessel" criterion.
    """
    threshold_mm: float = 10.0
    device_index: int = 0

    def __call__(self, state: SimState, prev_state: Optional[SimState] = None) -> bool:
        if not state.devices or self.device_index >= len(state.devices):
            return False
        tip = state.devices[self.device_index].tip_position
        dist = float(np.linalg.norm(tip - state.target_position))
        return dist <= self.threshold_mm


@dataclass
class AllTargetsReachedTerminal(Terminal):
    """
    Terminal when ALL interim targets have been visited and final target reached.
    Useful for multi-stage procedures (e.g. CAS: AAo → arch → LCCA → ICA).
    """
    threshold_mm: float = 10.0

    def __call__(self, state: SimState, prev_state: Optional[SimState] = None) -> bool:
        # No active interim target = all waypoints consumed
        return state.interim_target is None and self._main_reached(state)

    def _main_reached(self, state: SimState) -> bool:
        if not state.devices:
            return False
        tip = state.devices[0].tip_position
        return float(np.linalg.norm(tip - state.target_position)) <= self.threshold_mm


class PathRatioTerminal(Terminal):
    """Terminal when path_remaining / initial_path < ratio_threshold."""

    def __init__(self, ratio_threshold: float = 0.05):
        self.ratio_threshold = ratio_threshold
        self._initial_path: Optional[float] = None

    def reset(self, anatomy=None, episode: int = 0) -> None:
        self._initial_path = None

    def __call__(self, state: SimState, prev_state: Optional[SimState] = None) -> bool:
        if self._initial_path is None:
            self._initial_path = state.path_remaining + 1e-6
            return False
        ratio = state.path_remaining / self._initial_path
        return ratio < self.ratio_threshold
