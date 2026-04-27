"""
SynthEndoSim — InterimTarget implementations.

Provides various waypoint strategies for staged navigation:
  NoInterimTarget        — direct-to-goal (baseline)
  FixedWaypointTarget    — user-specified list of 3D points
  CenterlineWaypointTarget — auto-spaced waypoints along anatomy centerline
  ProximityWindowTarget  — sliding window: advance when within threshold
  StageTarget            — named procedure stages (CAS: AAo → Arch → LCCA → ICA)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from .base import InterimTarget, NoInterimTarget
from ..core.types import SimState, AnatomySpec


@dataclass
class FixedWaypointTarget(InterimTarget):
    """
    User-specified ordered list of 3D waypoints.
    The agent must reach each within `threshold_mm` before the next is unlocked.
    """
    waypoints: List[np.ndarray]
    threshold_mm: float = 10.0
    _idx: int = field(default=0, init=False, repr=False)

    def reset(self, anatomy: AnatomySpec, episode: int = 0) -> None:
        self._idx = 0

    def current(self, state: SimState) -> Optional[np.ndarray]:
        if self._idx >= len(self.waypoints):
            return None
        return self.waypoints[self._idx]

    def advance(self, state: SimState) -> bool:
        if self._idx >= len(self.waypoints):
            return False
        wp = self.waypoints[self._idx]
        tip = state.devices[0].tip_position if state.devices else np.zeros(3)
        dist = float(np.linalg.norm(tip - wp))
        if dist <= self.threshold_mm:
            self._idx += 1
            return True
        return False

    @property
    def remaining_count(self) -> int:
        return max(0, len(self.waypoints) - self._idx)


@dataclass
class CenterlineWaypointTarget(InterimTarget):
    """
    Automatically places N evenly-spaced waypoints along the anatomy centerline.
    On reset the waypoints are re-derived from the provided centerline array.
    """
    n_waypoints: int = 5
    threshold_mm: float = 12.0
    skip_first_frac: float = 0.05   # skip first 5% of centerline (entry region)
    _waypoints: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _idx: int = field(default=0, init=False, repr=False)

    def reset(self, anatomy: AnatomySpec, episode: int = 0) -> None:
        self._idx = 0
        self._waypoints = []
        if anatomy.centerline is None or len(anatomy.centerline) < 2:
            return
        cl = anatomy.centerline
        skip = int(len(cl) * self.skip_first_frac)
        cl = cl[skip:] if skip < len(cl) - 1 else cl
        # evenly spaced indices along arc-length
        diffs = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        arc = np.concatenate([[0], np.cumsum(diffs)])
        total = arc[-1]
        targets = np.linspace(0, total, self.n_waypoints + 2)[1:-1]
        for t in targets:
            idx = int(np.searchsorted(arc, t))
            idx = min(idx, len(cl) - 1)
            self._waypoints.append(cl[idx].astype(np.float32))

    def current(self, state: SimState) -> Optional[np.ndarray]:
        if self._idx >= len(self._waypoints):
            return None
        return self._waypoints[self._idx]

    def advance(self, state: SimState) -> bool:
        if self._idx >= len(self._waypoints):
            return False
        wp = self._waypoints[self._idx]
        tip = state.devices[0].tip_position if state.devices else np.zeros(3)
        if float(np.linalg.norm(tip - wp)) <= self.threshold_mm:
            self._idx += 1
            return True
        return False

    @property
    def remaining_count(self) -> int:
        return max(0, len(self._waypoints) - self._idx)


@dataclass
class ProximityWindowTarget(InterimTarget):
    """
    Sliding-window waypoints: always expose the nearest un-reached point on
    the centerline ahead of the tip, within a look-ahead distance.
    More forgiving than strict sequential gating.
    """
    lookahead_mm: float = 30.0
    threshold_mm: float = 8.0
    n_waypoints: int = 8
    _waypoints: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _idx: int = field(default=0, init=False, repr=False)

    def reset(self, anatomy: AnatomySpec, episode: int = 0) -> None:
        self._idx = 0
        self._waypoints = []
        if anatomy.centerline is None or len(anatomy.centerline) < 2:
            return
        cl = anatomy.centerline
        diffs = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        arc = np.concatenate([[0], np.cumsum(diffs)])
        total = arc[-1]
        idxs = np.linspace(0, total, self.n_waypoints + 2)[1:-1]
        for t in idxs:
            i = min(int(np.searchsorted(arc, t)), len(cl) - 1)
            self._waypoints.append(cl[i].astype(np.float32))

    def current(self, state: SimState) -> Optional[np.ndarray]:
        if not self._waypoints or self._idx >= len(self._waypoints):
            return None
        tip = state.devices[0].tip_position if state.devices else np.zeros(3)
        # advance window to nearest waypoint within lookahead
        for j in range(self._idx, len(self._waypoints)):
            d = float(np.linalg.norm(tip - self._waypoints[j]))
            if d <= self.lookahead_mm:
                return self._waypoints[j]
        return None

    def advance(self, state: SimState) -> bool:
        wp = self.current(state)
        if wp is None:
            return False
        tip = state.devices[0].tip_position if state.devices else np.zeros(3)
        if float(np.linalg.norm(tip - wp)) <= self.threshold_mm:
            self._idx += 1
            return True
        return False

    @property
    def remaining_count(self) -> int:
        return max(0, len(self._waypoints) - self._idx)


@dataclass
class StageTarget(InterimTarget):
    """
    Named procedure stages with explicit 3D positions.
    Useful for CAS (aorta → arch → LCCA → ICA) or other multi-stage workflows.
    Stage positions can be set at construction or derived from anatomy on reset.
    """
    stages: List[Tuple[str, np.ndarray]]   # [(name, position_mm), ...]
    threshold_mm: float = 10.0
    _idx: int = field(default=0, init=False, repr=False)

    def reset(self, anatomy: AnatomySpec, episode: int = 0) -> None:
        self._idx = 0

    @property
    def current_stage_name(self) -> Optional[str]:
        if self._idx >= len(self.stages):
            return None
        return self.stages[self._idx][0]

    def current(self, state: SimState) -> Optional[np.ndarray]:
        if self._idx >= len(self.stages):
            return None
        return self.stages[self._idx][1]

    def advance(self, state: SimState) -> bool:
        if self._idx >= len(self.stages):
            return False
        wp = self.stages[self._idx][1]
        tip = state.devices[0].tip_position if state.devices else np.zeros(3)
        if float(np.linalg.norm(tip - wp)) <= self.threshold_mm:
            self._idx += 1
            return True
        return False

    @property
    def remaining_count(self) -> int:
        return max(0, len(self.stages) - self._idx)
