"""
SynthEndoSim — Additional Start strategies.

VesselEndStart    : reset device when it reaches a vessel endpoint
MaxLengthStart    : reset device when it exceeds max insertion length
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .base import Start, StartState
from ..core.types import AnatomySpec


@dataclass
class VesselEndStart(Start):
    """
    Resets the device to the insertion point if the tip is at a vessel endpoint.
    Prevents the agent from camping at a dead-end between episodes.

    Use in combination with another start strategy via fallback.
    """
    threshold_mm: float = 5.0
    fallback: Start = None   # delegate to this if not at vessel end

    def get_start(self, anatomy: AnatomySpec, episode: int = 0, rng=None):
        # VesselTree-aware: check if tip is at a tree end
        vessel_tree = getattr(anatomy, "vessel_tree", None)
        if vessel_tree is not None:
            # We don't have current tip here — return insertion point unconditionally
            # (Actual check happens in env with state.devices[0].tip_position)
            pass
        base = np.array(anatomy.insertion_point, dtype=np.float32)
        dir_ = np.array(anatomy.insertion_direction, dtype=np.float32)
        return base, dir_

    def should_reset(self, tip: np.ndarray, anatomy: AnatomySpec) -> bool:
        """Call from env.step() to check if device should be reset."""
        vessel_tree = getattr(anatomy, "vessel_tree", None)
        if vessel_tree is None:
            return False
        return vessel_tree.at_tree_end(tip)


@dataclass
class MaxLengthStart(Start):
    """
    Resets device to insertion point when any device insertion length exceeds max_length_mm.
    Prevents infinite insertion without reaching target.
    """
    max_length_mm: float = 400.0
    device_index: int = 0

    def get_start(self, anatomy: AnatomySpec, episode: int = 0, rng=None):
        base = np.array(anatomy.insertion_point, dtype=np.float32)
        dir_ = np.array(anatomy.insertion_direction, dtype=np.float32)
        return base, dir_

    def should_reset(self, state) -> bool:
        """Call from env to check if device should be reset mid-episode."""
        if not state.devices or self.device_index >= len(state.devices):
            return False
        return state.devices[self.device_index].inserted_length > self.max_length_mm
