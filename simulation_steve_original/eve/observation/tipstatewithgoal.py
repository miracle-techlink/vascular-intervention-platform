"""
TipStateWithGoal: 9D observation = tip position + tip velocity + target direction.

Extends TipState with the relative direction vector to the target:
    obs = [p_x, p_y, p_z,        (tip position, 3D)
           v_x, v_y, v_z,        (tip velocity, 3D)
           gx-px, gy-py, gz-pz]  (target direction, 3D)

This gives the agent spatial awareness of WHERE the target is,
which is the key missing piece in the vanilla TipState observation.
"""

import numpy as np
import gymnasium as gym

from .observation import Observation
from ..intervention import Intervention


class TipStateWithGoal(Observation):
    """9D observation: tip position + velocity + relative target direction."""

    def __init__(self, intervention: Intervention) -> None:
        self.intervention = intervention
        self._last_tip = None
        self.obs = np.zeros(9, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        pos_low  = self.intervention.fluoroscopy.tracking3d_space.low
        pos_high = self.intervention.fluoroscopy.tracking3d_space.high
        vel_bound = pos_high - pos_low
        dir_bound = pos_high - pos_low   # direction vector same scale as position
        low  = np.concatenate([pos_low,  -vel_bound, -dir_bound]).astype(np.float32)
        high = np.concatenate([pos_high,  vel_bound,  dir_bound]).astype(np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self) -> None:
        tip    = self.intervention.fluoroscopy.tracking3d[0].astype(np.float32)
        vel    = (tip - self._last_tip) if self._last_tip is not None \
                 else np.zeros(3, dtype=np.float32)
        self._last_tip = tip.copy()
        goal   = self.intervention.target.coordinates3d.astype(np.float32)
        direction = goal - tip
        self.obs = np.concatenate([tip, vel, direction]).astype(np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        tip  = self.intervention.fluoroscopy.tracking3d[0].astype(np.float32)
        self._last_tip = tip.copy()
        goal = self.intervention.target.coordinates3d.astype(np.float32)
        self.obs = np.concatenate([tip, np.zeros(3, dtype=np.float32),
                                   goal - tip]).astype(np.float32)
