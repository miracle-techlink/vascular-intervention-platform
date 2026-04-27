"""
SynthEndoSim — AorticArchRandom: randomises arch type + geometry each episode.
"""
from __future__ import annotations
import random
from typing import List, Optional
import numpy as np

from .base import VesselTree, Insertion
from .aorticarch import AorticArch, ARCH_TYPES


class AorticArchRandom(VesselTree):
    """
    Samples a random arch type and geometry variation every `episodes_between_change`
    episodes. Useful for multi-anatomy generalisation training.
    """

    def __init__(
        self,
        arch_types: Optional[List[str]] = None,
        scale_xy: Optional[List[float]] = None,
        scale_z: Optional[List[float]] = None,
        scale_d: Optional[List[float]] = None,
        rotate_y_deg: Optional[List[float]] = None,
        rotate_z_deg: Optional[List[float]] = None,
        rotate_x_deg: Optional[List[float]] = None,
        episodes_between_change: int = 1,
    ):
        self._arch_types = list(arch_types or ARCH_TYPES)
        self._scale_xy  = list(scale_xy  or np.linspace(0.8, 1.2, 5))
        self._scale_z   = list(scale_z   or np.linspace(0.8, 1.2, 5))
        self._scale_d   = list(scale_d   or np.linspace(0.8, 1.2, 5))
        self._rot_y     = list(rotate_y_deg or [0.0])
        self._rot_z     = list(rotate_z_deg or [0.0])
        self._rot_x     = list(rotate_x_deg or [0.0])
        self._n_change  = episodes_between_change
        self._rng       = random.Random()
        self._current   = self._sample_arch()
        self.mesh_path  = ""
        self.visu_mesh_path = None

    # delegate to current AorticArch ─────────────────────────────────────
    @property
    def branches(self):           return self._current.branches
    @property
    def branching_points(self):   return self._current.branching_points
    @property
    def centerline_coordinates(self): return self._current.centerline_coordinates
    @property
    def insertion(self):          return self._current.insertion
    @property
    def bbox_low(self):           return self._current.bbox_low
    @property
    def bbox_high(self):          return self._current.bbox_high
    @property
    def mesh_path(self):          return self._current.mesh_path
    @mesh_path.setter
    def mesh_path(self, v): pass   # read-only proxy

    def reset(self, episode: int = 0, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)
        if episode % self._n_change == 0:
            self._current = self._sample_arch()
        self._current.reset(episode, seed)

    def _sample_arch(self) -> AorticArch:
        return AorticArch(
            arch_type=self._rng.choice(self._arch_types),
            seed=self._rng.randint(0, 2**31 - 1),
            rotation_yzx_deg=(
                self._rng.choice(self._rot_y),
                self._rng.choice(self._rot_z),
                self._rng.choice(self._rot_x),
            ),
            scaling_xyzd=(
                self._rng.choice(self._scale_xy),
                self._rng.choice(self._scale_xy),
                self._rng.choice(self._scale_z),
                self._rng.choice(self._scale_d),
            ),
        )
