"""
SynthEndoSim — VesselTree abstract base.

A VesselTree describes the complete vascular anatomy available for navigation:
  - Branching topology (Branch objects with centerline + radii)
  - Insertion point / direction for device entry
  - Coordinate bounding box (used for obs normalisation)
  - Mesh path for SOFA collision geometry

Implementations:
  AorticArch     — procedural 7-type aortic arch generator
  VesselTreeFromMesh — load from OBJ/STL + optional branch list
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

from .util.branch import Branch, BranchWithRadii, BranchingPoint


@dataclass
class Insertion:
    position: np.ndarray   # (3,) world-space entry point (mm)
    direction: np.ndarray  # (3,) unit insertion direction


class VesselTree(ABC):
    """
    Abstract vessel tree.

    Subclasses MUST populate in __init__ or reset():
        branches              : Tuple[Branch | BranchWithRadii]
        branching_points      : List[BranchingPoint]
        centerline_coordinates: np.ndarray  (N, 3)
        insertion             : Insertion
        mesh_path             : str
        bbox_low, bbox_high   : np.ndarray (3,)
    """
    branches: Tuple[Branch, ...]
    branching_points: List[BranchingPoint]
    centerline_coordinates: np.ndarray
    insertion: Insertion
    bbox_low: np.ndarray
    bbox_high: np.ndarray
    mesh_path: str
    visu_mesh_path: Optional[str] = None

    @abstractmethod
    def reset(self, episode: int = 0, seed: Optional[int] = None) -> None:
        """Called at episode start. May randomise anatomy."""

    def step(self) -> None:
        """Called each sim step (most VesselTrees are static)."""

    # ── dict-like branch access ─────────────────────────────────────────
    def __getitem__(self, item: Union[int, str]) -> Branch:
        if isinstance(item, int):
            return self.branches[item]
        names = tuple(b.name for b in self.branches)
        return self.branches[names.index(item)]

    def keys(self) -> Tuple[str, ...]:
        return tuple(b.name for b in self.branches)

    def values(self) -> Tuple[Branch, ...]:
        return self.branches

    def items(self):
        return zip(self.keys(), self.values())

    # ── geometry helpers ────────────────────────────────────────────────
    def nearest_branch(self, point: np.ndarray) -> Branch:
        """Return the branch whose centerline is closest to `point`."""
        best, best_d = None, float("inf")
        for b in self.branches:
            d = float(np.linalg.norm(b.coordinates - point, axis=1).min())
            if d < best_d:
                best_d, best = d, b
        return best

    def at_tree_end(self, point: np.ndarray) -> bool:
        """True if `point` is beyond any open branch endpoint."""
        branch = self.nearest_branch(point)
        coords = branch.coordinates
        dists = np.linalg.norm(coords - point, axis=1)
        idx = int(np.argmin(dists))
        if idx not in (0, len(coords) - 1):
            return False
        # check the second nearest point to determine direction
        second = np.argpartition(dists, 1)[1]
        toward_branch = coords[second] - coords[idx]
        toward_point  = point - coords[idx]
        if np.dot(toward_branch, toward_point) > 0:
            return False  # point is between branch nodes
        # confirm endpoint is not a branching junction
        bp_coords = np.array([bp.coordinates for bp in (self.branching_points or [])])
        if len(bp_coords) > 0:
            near_bp = np.linalg.norm(bp_coords - coords[idx], axis=1).min()
            if near_bp < 5.0:
                return False
        return True
