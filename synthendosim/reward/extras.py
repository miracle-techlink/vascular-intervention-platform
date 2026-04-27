"""
SynthEndoSim — Additional reward components.

LastActionPenalty         : penalise large actions (smooth control)
InsertionLengthDeltaReward: reward progress in insertion length
CoaxialClearanceReward    : keep guidewire within safe window ahead of catheter
FailurePenalty            : large one-time penalty on truncation (sim error / vessel end)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .components import RewardComponent
from ..core.types import SimState


@dataclass
class LastActionPenalty(RewardComponent):
    """
    Penalises large magnitude actions. Encourages smooth, precise control.

    penalty_factor : negative weight applied to L2 norm of action
    action_idx     : None = all dims, int = specific dim only
    """
    penalty_factor: float = -0.01
    action_idx: Optional[int] = None

    def __call__(self, state: SimState, prev_state: Optional[SimState] = None) -> float:
        action = getattr(state, "last_action", None)
        if action is None:
            return 0.0
        if self.action_idx is not None:
            a = float(action[self.action_idx])
        else:
            a = float(np.linalg.norm(action))
        return self.penalty_factor * abs(a)


@dataclass
class InsertionLengthDeltaReward(RewardComponent):
    """
    Rewards positive insertion progress (mm advanced per step).
    Penalises retraction (negative delta).

    forward_factor  : reward per mm of forward insertion
    backward_factor : penalty per mm of backward retraction (should be ≤ 0)
    device_index    : which device to track (default 0 = primary guidewire)
    """
    forward_factor: float = 0.01
    backward_factor: float = -0.005
    device_index: int = 0

    def __call__(self, state: SimState, prev_state: Optional[SimState] = None) -> float:
        if prev_state is None or not state.devices or not prev_state.devices:
            return 0.0
        if self.device_index >= len(state.devices):
            return 0.0
        delta = (
            state.devices[self.device_index].inserted_length
            - prev_state.devices[self.device_index].inserted_length
        )
        if delta >= 0:
            return self.forward_factor * delta
        return self.backward_factor * abs(delta)


@dataclass
class CoaxialClearanceReward(RewardComponent):
    """
    For coaxial systems (guidewire inside catheter):
    reward 0 when guidewire is within [lower, upper] mm ahead of catheter tip.
    Penalise proportionally otherwise.

    Prevents the guidewire from retracting into the catheter or advancing too far.
    """
    guidewire_device: int = 0
    catheter_device: int = 1
    lower_clearance_mm: float = 5.0
    upper_clearance_mm: float = 40.0
    penalty_factor: float = -0.02

    def __call__(self, state: SimState, prev_state: Optional[SimState] = None) -> float:
        n = len(state.devices)
        if n <= max(self.guidewire_device, self.catheter_device):
            return 0.0
        gw = state.devices[self.guidewire_device].inserted_length
        ca = state.devices[self.catheter_device].inserted_length
        relative = gw - ca
        if self.lower_clearance_mm < relative < self.upper_clearance_mm:
            return 0.0
        overshoot = max(0, relative - self.upper_clearance_mm)
        undershoot = max(0, self.lower_clearance_mm - relative)
        return self.penalty_factor * (overshoot + undershoot)


@dataclass
class FailurePenalty(RewardComponent):
    """
    One-time penalty applied when the episode ends by truncation (failure).
    Note: this requires the caller to pass `truncated=True` via state metadata.
    Attach via a wrapper, not directly — or use in terminal/truncation callbacks.
    """
    penalty: float = -10.0

    def __call__(self, state: SimState, prev_state: Optional[SimState] = None) -> float:
        # Applied externally by the env when truncated=True
        return 0.0

    def on_truncation(self) -> float:
        return self.penalty
