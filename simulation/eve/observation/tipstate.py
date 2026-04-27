"""Sim4EndoR-aligned observation: s_t = [p_t, v_t] (6D).

p_t: 3D tip position
v_t: 3D tip velocity (finite difference of position)
"""
import numpy as np
import gymnasium as gym

from .observation import Observation
from ..intervention import Intervention


class TipState(Observation):
    """6D observation: guidewire tip 3D position + 3D velocity."""

    def __init__(self, intervention: Intervention) -> None:
        self.intervention = intervention
        self._last_tip = None
        self.obs = np.zeros(6, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        pos_low = self.intervention.fluoroscopy.tracking3d_space.low
        pos_high = self.intervention.fluoroscopy.tracking3d_space.high
        vel_bound = pos_high - pos_low
        low = np.concatenate([pos_low, -vel_bound]).astype(np.float32)
        high = np.concatenate([pos_high, vel_bound]).astype(np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        tip = self.intervention.fluoroscopy.tracking3d[0].astype(np.float32)
        velocity = tip - self._last_tip if self._last_tip is not None else np.zeros(3, dtype=np.float32)
        self._last_tip = tip.copy()
        self.obs = np.concatenate([tip, velocity]).astype(np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        tip = self.intervention.fluoroscopy.tracking3d[0].astype(np.float32)
        self._last_tip = tip.copy()
        self.obs = np.concatenate([tip, np.zeros(3, dtype=np.float32)])
