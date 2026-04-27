"""
SynthEndoSim — Target selection strategies.

FixedTarget           — always the same point (from config or AnatomySpec)
CenterlineRandomTarget — random point on centerline of specified branches
BranchEndTarget       — targets branch endpoints (orifice / distal end)
BranchIndexTarget     — specific branch + index (reproducible)
ManualRandomTarget    — user-specified list, sampled randomly
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from .base import Target, TargetState
from ..core.types import AnatomySpec


class FixedTarget(Target):
    """
    Always uses the target_point from AnatomySpec (or a user-specified override).
    """
    def __init__(self, threshold_mm: float = 10.0, position: Optional[Tuple] = None):
        self.threshold_mm_ = threshold_mm
        self._override = np.array(position, dtype=np.float32) if position else None
        self.state = TargetState(np.zeros(3), threshold_mm, "fixed")

    def reset(self, anatomy: AnatomySpec, episode: int = 0, seed=None) -> TargetState:
        pos = self._override if self._override is not None else np.array(anatomy.target_point, np.float32)
        self.state = TargetState(pos, self.threshold_mm_, "fixed")
        return self.state


@dataclass
class CenterlineRandomTarget(Target):
    """
    Samples a random point on the anatomy centerline, optionally restricted
    to specific named branches (when a VesselTree centerline is available).
    """
    threshold_mm: float = 10.0
    branches: Optional[List[str]] = None    # None = all branches
    min_separation_mm: float = 0.0          # minimum distance between possible targets
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)

    def __post_init__(self):
        self.state = TargetState(np.zeros(3), self.threshold_mm)

    def reset(self, anatomy: AnatomySpec, episode: int = 0, seed=None) -> TargetState:
        if seed is not None:
            self._rng = random.Random(seed)
        candidates = self._get_candidates(anatomy)
        if len(candidates) == 0:
            pos = np.array(anatomy.target_point, np.float32)
        else:
            idx = self._rng.randrange(len(candidates))
            pos = candidates[idx].astype(np.float32)
        self.state = TargetState(pos, self.threshold_mm, "centerline_random")
        return self.state

    def _get_candidates(self, anatomy: AnatomySpec) -> np.ndarray:
        if anatomy.centerline is None or len(anatomy.centerline) == 0:
            return np.empty((0, 3))
        cl = anatomy.centerline
        # filter by branch name if vessel_tree info is attached
        vessel_tree = getattr(anatomy, "vessel_tree", None)
        if vessel_tree is not None and self.branches is not None:
            candidates = []
            for name in self.branches:
                try:
                    b = vessel_tree[name]
                    candidates.append(b.coordinates)
                except (KeyError, ValueError):
                    pass
            return np.concatenate(candidates) if candidates else cl
        return cl


@dataclass
class BranchEndTarget(Target):
    """
    Targets the distal endpoint of each branch.
    Randomly selects one branch endpoint per episode.
    Requires anatomy.vessel_tree to be set.
    """
    threshold_mm: float = 10.0
    branches: Optional[List[str]] = None   # None = all branches
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)

    def __post_init__(self):
        self.state = TargetState(np.zeros(3), self.threshold_mm)

    def reset(self, anatomy: AnatomySpec, episode: int = 0, seed=None) -> TargetState:
        if seed is not None:
            self._rng = random.Random(seed)
        vessel_tree = getattr(anatomy, "vessel_tree", None)
        endpoints = []
        if vessel_tree is not None and vessel_tree.branches:
            for b in vessel_tree.branches:
                if self.branches is None or b.name in self.branches:
                    endpoints.append((b.name, b.coordinates[-1]))
        if endpoints:
            name, pos = self._rng.choice(endpoints)
            self.state = TargetState(pos.astype(np.float32), self.threshold_mm, f"branch_end:{name}")
        else:
            pos = np.array(anatomy.target_point, np.float32)
            self.state = TargetState(pos, self.threshold_mm, "branch_end:fallback")
        return self.state


@dataclass
class BranchIndexTarget(Target):
    """
    Fixed target at a specific branch + index.
    Fully reproducible — useful for evaluation protocols.
    """
    branch: str
    idx: int
    threshold_mm: float = 10.0

    def __post_init__(self):
        self.state = TargetState(np.zeros(3), self.threshold_mm)

    def reset(self, anatomy: AnatomySpec, episode: int = 0, seed=None) -> TargetState:
        vessel_tree = getattr(anatomy, "vessel_tree", None)
        if vessel_tree is not None:
            try:
                b = vessel_tree[self.branch]
                pos = b.coordinates[self.idx].astype(np.float32)
                self.state = TargetState(pos, self.threshold_mm, f"{self.branch}[{self.idx}]")
                return self.state
            except Exception:
                pass
        pos = np.array(anatomy.target_point, np.float32)
        self.state = TargetState(pos, self.threshold_mm, "branch_index:fallback")
        return self.state


@dataclass
class ManualRandomTarget(Target):
    """
    Randomly samples from a user-specified list of 3-D positions.
    """
    positions: List[Tuple[float, float, float]]
    threshold_mm: float = 10.0
    labels: Optional[List[str]] = None
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)

    def __post_init__(self):
        self.state = TargetState(np.zeros(3), self.threshold_mm)

    def reset(self, anatomy: AnatomySpec, episode: int = 0, seed=None) -> TargetState:
        if seed is not None:
            self._rng = random.Random(seed)
        idx = self._rng.randrange(len(self.positions))
        pos = np.array(self.positions[idx], dtype=np.float32)
        label = self.labels[idx] if self.labels and idx < len(self.labels) else f"manual_{idx}"
        self.state = TargetState(pos, self.threshold_mm, label)
        return self.state
