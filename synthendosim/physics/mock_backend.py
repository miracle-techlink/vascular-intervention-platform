"""
SynthEndoSim — Mock physics backend for unit testing / CI without SOFA.
Simulates a straight tube: tip advances linearly with translation action.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np

from .backend import PhysicsBackend
from ..core.types import DeviceState, DeviceSpec, AnatomySpec


class MockBackend(PhysicsBackend):
    def __init__(self, dt_simulation: float = 0.006, friction: float = 0.1):
        self.dt_simulation = dt_simulation
        self._inserted = 0.0
        self._rotation = 0.0
        self._target = np.zeros(3)
        self._tip = np.zeros(3)
        self._entry = np.zeros(3)
        self._direction = np.array([0.0, 1.0, 0.0])
        self._sim_error = False
        self._n = 1

    def reset(
        self,
        anatomy: AnatomySpec,
        devices: List[DeviceSpec],
        seed: Optional[int] = None,
    ) -> List[DeviceState]:
        self._n = len(devices)
        self._entry = np.array(anatomy.insertion_point, dtype=np.float32)
        self._direction = np.array(anatomy.insertion_direction, dtype=np.float32)
        self._direction /= np.linalg.norm(self._direction)
        self._target = np.array(anatomy.target_point, dtype=np.float32)
        self._inserted = 0.0
        self._rotation = 0.0
        self._tip = self._entry.copy()
        self._sim_error = False
        return self._states()

    def step(self, actions: np.ndarray, dt: float) -> List[DeviceState]:
        self._inserted += float(actions[0, 0]) * dt
        self._inserted = max(0.0, self._inserted)
        self._rotation += float(actions[0, 1]) * dt
        self._tip = self._entry + self._direction * self._inserted
        return self._states()

    def get_dof_positions(self) -> np.ndarray:
        n_nodes = max(2, int(self._inserted / 5) + 1)
        t = np.linspace(0, 1, n_nodes)[:, None]
        return (self._entry + t * self._direction * self._inserted).astype(np.float32)

    @property
    def simulation_error(self) -> bool:
        return self._sim_error

    def close(self) -> None:
        pass

    def _states(self) -> List[DeviceState]:
        dofs = self.get_dof_positions()
        return [
            DeviceState(
                dof_positions=dofs,
                inserted_length=self._inserted,
                rotation=self._rotation,
                tip_position=self._tip.copy(),
            )
            for _ in range(self._n)
        ]
