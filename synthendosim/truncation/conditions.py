"""
SynthEndoSim — Truncation condition implementations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .base import Truncation
from ..core.types import SimState


@dataclass
class MaxStepsTruncation(Truncation):
    """Truncate after max_steps regardless of progress."""
    max_steps: int = 300

    def __call__(self, state: SimState, prev_state: Optional[SimState] = None) -> bool:
        return state.step >= self.max_steps


@dataclass
class SimErrorTruncation(Truncation):
    """Truncate when the physics backend reports numerical instability."""
    def __call__(self, state: SimState, prev_state: Optional[SimState] = None) -> bool:
        return state.simulation_error


class VesselEndTruncation(Truncation):
    """
    Truncate when the device tip exits the vessel bounding box.
    Prevents the wire from escaping into free space.
    """

    def __init__(self, bbox_min=None, bbox_max=None, margin_mm: float = 5.0):
        self.bbox_min = np.array(bbox_min, dtype=np.float32) if bbox_min is not None else None
        self.bbox_max = np.array(bbox_max, dtype=np.float32) if bbox_max is not None else None
        self.margin_mm = margin_mm

    def set_bounds(self, bbox_min, bbox_max):
        self.bbox_min = np.array(bbox_min, dtype=np.float32)
        self.bbox_max = np.array(bbox_max, dtype=np.float32)

    def reset(self, anatomy=None, episode: int = 0) -> None:
        if anatomy is not None and hasattr(anatomy, "mesh_path"):
            pass  # optionally reload bounds from anatomy

    def __call__(self, state: SimState, prev_state: Optional[SimState] = None) -> bool:
        if self.bbox_min is None or not state.devices:
            return False
        tip = state.devices[0].tip_position
        m = self.margin_mm
        return bool(
            np.any(tip < self.bbox_min - m) or np.any(tip > self.bbox_max + m)
        )


class NoProgressTruncation(Truncation):
    """
    Truncate if no meaningful progress toward target in `patience` steps.
    Prevents the agent from spinning in place indefinitely.
    """

    def __init__(self, patience: int = 100, min_delta_mm: float = 1.0):
        self.patience = patience
        self.min_delta_mm = min_delta_mm
        self._best_dist: Optional[float] = None
        self._steps_no_progress: int = 0

    def reset(self, anatomy=None, episode: int = 0) -> None:
        self._best_dist = None
        self._steps_no_progress = 0

    def __call__(self, state: SimState, prev_state: Optional[SimState] = None) -> bool:
        dist = state.path_remaining
        if self._best_dist is None or dist < self._best_dist - self.min_delta_mm:
            self._best_dist = dist
            self._steps_no_progress = 0
        else:
            self._steps_no_progress += 1
        return self._steps_no_progress >= self.patience
