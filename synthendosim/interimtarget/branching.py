"""
SynthEndoSim — BranchingPointTarget.

Uses vessel tree branching points as interim waypoints.
Unlike CenterlineWaypointTarget (evenly spaced), this places waypoints
at anatomically meaningful locations: branch junctions.

For CAS: aortic root → BCT junction → LCCA origin → target vessel.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .base import InterimTarget, NoInterimTarget
from ..core.types import SimState, AnatomySpec


@dataclass
class BranchingPointTarget(InterimTarget):
    """
    Waypoints placed at branch junction points of the vessel tree.
    Requires anatomy.vessel_tree to be set (from VesselTree module).

    threshold_mm : radius to consider a branching point "reached"
    branches     : optional list of branch names — only junctions between
                   listed branches are used (None = all junctions)
    """
    threshold_mm: float = 12.0
    branches: Optional[List[str]] = None
    _waypoints: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _idx: int = field(default=0, init=False, repr=False)

    def reset(self, anatomy: AnatomySpec, episode: int = 0) -> None:
        self._idx = 0
        self._waypoints = []
        vessel_tree = getattr(anatomy, "vessel_tree", None)
        if vessel_tree is None or not vessel_tree.branching_points:
            return
        bps = vessel_tree.branching_points
        # filter by branch membership if requested
        if self.branches is not None:
            branch_set = set(self.branches)
            bps = [
                bp for bp in bps
                if any(b.name in branch_set for b in bp.connections)
            ]
        # sort by distance from insertion point (proximal → distal order)
        entry = np.array(anatomy.insertion_point, dtype=np.float32)
        bps_sorted = sorted(bps, key=lambda bp: float(np.linalg.norm(bp.coordinates - entry)))
        self._waypoints = [bp.coordinates.astype(np.float32) for bp in bps_sorted]

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
