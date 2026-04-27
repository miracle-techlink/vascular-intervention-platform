"""
SynthEndoSim — Branch data structures.
Ported and cleaned from stEVE (MIT licence).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass(frozen=True, eq=True)
class Branch:
    """Immutable ordered sequence of 3-D centerline coordinates."""
    name: str
    coordinates: np.ndarray = field(init=True, compare=False, repr=True)
    _coordinates: tuple = field(init=False, default=None, compare=True, repr=False)

    def __post_init__(self):
        coords = np.asarray(self.coordinates, dtype=np.float32)
        object.__setattr__(self, "coordinates", coords)
        coords.flags.writeable = False
        object.__setattr__(self, "_coordinates", tuple(map(tuple, coords)))

    def __repr__(self) -> str:
        return f"Branch({self.name}, n={len(self.coordinates)})"

    @property
    def low(self) -> np.ndarray:
        return self.coordinates.min(axis=0)

    @property
    def high(self) -> np.ndarray:
        return self.coordinates.max(axis=0)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.coordinates, axis=0), axis=1).sum())

    def in_branch(self, points: np.ndarray, radius: float) -> np.ndarray:
        pts = np.atleast_2d(points)
        dists = np.linalg.norm(self.coordinates[:, None] - pts[None], axis=-1)
        return np.any(dists < radius, axis=0)

    def get_path_along_branch(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        si = int(np.argmin(np.linalg.norm(self.coordinates - start, axis=1)))
        ei = int(np.argmin(np.linalg.norm(self.coordinates - end, axis=1)))
        d = ei - si
        if d == 0:
            return np.stack([start, end])
        step = int(d / abs(d))
        si += step; ei -= step
        segment = self.coordinates[si:ei + step:step]
        return np.concatenate([start.reshape(1, 3), segment, end.reshape(1, 3)])


@dataclass(frozen=True, eq=True)
class BranchWithRadii(Branch):
    """Branch with per-point vessel radii (mm)."""
    radii: np.ndarray = field(init=True, compare=False, repr=False)
    _radii: tuple = field(init=False, default=None, compare=True, repr=False)

    def __post_init__(self):
        super().__post_init__()
        r = np.asarray(self.radii, dtype=np.float32)
        object.__setattr__(self, "radii", r)
        r.flags.writeable = False
        object.__setattr__(self, "_radii", tuple(r.tolist()))

    @property
    def low(self) -> np.ndarray:
        r = self.radii.reshape(-1, 1)
        return (self.coordinates - np.broadcast_to(r, self.coordinates.shape)).min(axis=0)

    @property
    def high(self) -> np.ndarray:
        r = self.radii.reshape(-1, 1)
        return (self.coordinates + np.broadcast_to(r, self.coordinates.shape)).max(axis=0)

    def in_branch(self, points: np.ndarray, radius=None) -> np.ndarray:
        pts = np.atleast_2d(points)
        dists = np.linalg.norm(self.coordinates[:, None] - pts[None], axis=-1)
        return np.any(dists < self.radii[:, None], axis=0)


@dataclass(frozen=True, eq=True)
class BranchingPoint:
    """Point where two or more branches meet."""
    coordinates: np.ndarray = field(compare=False)
    radius: float
    connections: tuple  # Tuple[Branch, ...]
    _coordinates: tuple = field(init=False, default=None, compare=True, repr=False)

    def __post_init__(self):
        c = np.asarray(self.coordinates, dtype=np.float32)
        object.__setattr__(self, "coordinates", c)
        c.flags.writeable = False
        object.__setattr__(self, "_coordinates", tuple(c.tolist()))

    def __repr__(self) -> str:
        names = [b.name for b in self.connections]
        return f"BranchingPoint({names})"


# ── geometry helpers ───────────────────────────────────────────────────────────

def rotate_array(array: np.ndarray, y_deg: float, z_deg: float, x_deg: float) -> np.ndarray:
    to_rad = np.pi / 180
    Ry = _rot_y(y_deg * to_rad)
    Rz = _rot_z(z_deg * to_rad)
    Rx = _rot_x(x_deg * to_rad)
    R = Rx @ Rz @ Ry
    return (R @ np.atleast_2d(array).T).T


def _rot_y(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], np.float32)

def _rot_z(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32)

def _rot_x(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], np.float32)


def rotate_branches(branches, rotate_yzx_deg):
    result = []
    for b in branches:
        c = rotate_array(b.coordinates, *rotate_yzx_deg)
        result.append(BranchWithRadii(b.name, c, b.radii) if isinstance(b, BranchWithRadii) else Branch(b.name, c))
    return tuple(result)


def scale_branches_xyz(branches, xyz):
    result = []
    s = np.array(xyz, dtype=np.float32)
    for b in branches:
        c = b.coordinates * s
        result.append(BranchWithRadii(b.name, c, b.radii) if isinstance(b, BranchWithRadii) else Branch(b.name, c))
    return tuple(result)


def scale_branches_xyzd(branches, xyzd):
    branches = scale_branches_xyz(branches, xyzd[:3])
    result = []
    for b in branches:
        r = b.radii * xyzd[3] if isinstance(b, BranchWithRadii) else None
        result.append(BranchWithRadii(b.name, b.coordinates, r) if r is not None else b)
    return tuple(result)


# ── branching point computation ────────────────────────────────────────────────

def calc_branching_with_radii(branches: List[BranchWithRadii]) -> List[BranchingPoint]:
    raw = []
    for main in branches:
        for other in branches:
            if other is main:
                continue
            mask = main.in_branch(other.coordinates)
            if not np.any(mask):
                continue
            for idx in np.where(mask)[0]:
                raw.append(BranchingPoint(
                    other.coordinates[idx],
                    float(other.radii[idx]),
                    (main, other),
                ))
    return _consolidate(raw)


def calc_branching(branches: List[Branch], radii: Union[float, List[float]]) -> List[BranchingPoint]:
    if isinstance(radii, (int, float)):
        radii = [radii] * len(branches)
    raw = []
    for main, mr in zip(branches, radii):
        for other, _or in zip(branches, radii):
            if other is main:
                continue
            mask = main.in_branch(other.coordinates, mr)
            if not np.any(mask):
                continue
            for idx in np.where(mask)[0]:
                raw.append(BranchingPoint(other.coordinates[idx], _or, (main, other)))
    return _consolidate(raw)


def _consolidate(raw: List[BranchingPoint]) -> List[BranchingPoint]:
    # merge duplicates with same connection set
    merged = []
    remaining = list(raw)
    while remaining:
        bp = remaining.pop()
        group = [bp]
        keep = []
        for other in remaining:
            if set(other.connections) == set(bp.connections):
                group.append(other)
            else:
                keep.append(other)
        remaining = keep
        coords = np.mean([g.coordinates for g in group], axis=0)
        r = min(g.radius for g in group)
        merged.append(BranchingPoint(coords, r, tuple(bp.connections)))

    # merge overlapping points
    final = []
    while merged:
        bp = merged.pop()
        absorbed = False
        for i, other in enumerate(merged):
            if np.linalg.norm(bp.coordinates - other.coordinates) < bp.radius + other.radius:
                coord = (bp.coordinates + other.coordinates) / 2
                r = max(bp.radius, other.radius)
                conns = tuple(set(bp.connections) | set(other.connections))
                merged[i] = BranchingPoint(coord, r, conns)
                absorbed = True
                break
        if not absorbed:
            final.append(bp)
    return final


def calc_insertion_from_branch_start(branch: Branch):
    p0 = branch.coordinates[1]
    p1 = branch.coordinates[2]
    d = p1 - p0
    return p0, d / (np.linalg.norm(d) + 1e-9)
