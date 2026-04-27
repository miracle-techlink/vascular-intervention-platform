"""
SynthEndoSim — Pathfinder module.

Computes path-remaining distance from tip to target.
Pluggable: swap Euclidean for geodesic / curvature-weighted without changing env.

Implementations:
  EuclideanPathfinder   — straight-line distance (fast, no centerline needed)
  ManifoldPathfinder    — curvature-weighted geodesic along centerline graph
  DijkstraPathfinder    — full Dijkstra on centerline graph (accurate, slower)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Pathfinder(ABC):
    """Abstract base — given tip and target positions, return remaining distance."""

    @abstractmethod
    def path_remaining(self, tip: np.ndarray, target: np.ndarray) -> float:
        """Return estimated path length (mm) from tip to target."""

    def reset(self, centerline: Optional[np.ndarray] = None) -> None:
        """Called on episode reset with updated centerline (optional)."""


class EuclideanPathfinder(Pathfinder):
    """Straight-line distance. O(1), no centerline required."""

    def path_remaining(self, tip: np.ndarray, target: np.ndarray) -> float:
        return float(np.linalg.norm(tip - target))


class ManifoldPathfinder(Pathfinder):
    """
    Curvature-weighted geodesic along the anatomy centerline.

    Edge weight = arc_length * (1 + curvature_weight * kappa)
    where kappa = local angular change per mm.

    The tip and target are projected onto the nearest centerline node,
    then the weighted path length is accumulated.
    """

    def __init__(
        self,
        centerline: np.ndarray,
        curvature_weight: float = 2.0,
    ):
        self.curvature_weight = curvature_weight
        self._cl: Optional[np.ndarray] = None
        self._arc: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        if centerline is not None and len(centerline) > 1:
            self._build(centerline)

    def reset(self, centerline: Optional[np.ndarray] = None) -> None:
        if centerline is not None and len(centerline) > 1:
            self._build(centerline)

    def _build(self, cl: np.ndarray) -> None:
        self._cl = cl.astype(np.float32)
        segs = np.diff(cl, axis=0)
        seg_len = np.linalg.norm(segs, axis=1) + 1e-9

        # curvature: angle between consecutive segments
        kappa = np.zeros(len(seg_len))
        if len(segs) > 1:
            dots = np.einsum("ij,ij->i", segs[:-1], segs[1:])
            dots /= (seg_len[:-1] * seg_len[1:])
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)            # rad per segment junction
            # assign half curvature to each neighbouring segment
            kappa[:-1] += angles / seg_len[:-1] * 0.5
            kappa[1:]  += angles / seg_len[1:] * 0.5

        w = seg_len * (1.0 + self.curvature_weight * kappa)
        self._weights = w
        self._arc = np.concatenate([[0.0], np.cumsum(w)])

    def _project(self, point: np.ndarray) -> int:
        if self._cl is None:
            return 0
        dists = np.linalg.norm(self._cl - point, axis=1)
        return int(np.argmin(dists))

    def path_remaining(self, tip: np.ndarray, target: np.ndarray) -> float:
        if self._cl is None or self._arc is None:
            return float(np.linalg.norm(tip - target))
        i_tip    = self._project(tip)
        i_target = self._project(target)
        if i_tip == i_target:
            return float(np.linalg.norm(tip - target))
        lo, hi = min(i_tip, i_target), max(i_tip, i_target)
        return float(self._arc[hi] - self._arc[lo])


class DijkstraPathfinder(Pathfinder):
    """
    Full Dijkstra on the centerline graph with curvature-weighted edges.
    More accurate than ManifoldPathfinder for branching centerlines.
    Slightly slower — use for evaluation, not training.
    """

    def __init__(
        self,
        centerline: np.ndarray,
        curvature_weight: float = 2.0,
        branch_radius_mm: float = 5.0,
    ):
        self.curvature_weight = curvature_weight
        self.branch_radius_mm = branch_radius_mm
        self._cl: Optional[np.ndarray] = None
        self._adj = {}
        if centerline is not None and len(centerline) > 1:
            self._build(centerline)

    def reset(self, centerline: Optional[np.ndarray] = None) -> None:
        if centerline is not None and len(centerline) > 1:
            self._build(centerline)

    def _build(self, cl: np.ndarray) -> None:
        self._cl = cl.astype(np.float32)
        n = len(cl)
        self._adj = {i: [] for i in range(n)}
        for i in range(n - 1):
            d = float(np.linalg.norm(cl[i + 1] - cl[i])) + 1e-9
            self._adj[i].append((i + 1, d))
            self._adj[i + 1].append((i, d))
        # add cross-edges for branching geometries
        for i in range(n):
            dists = np.linalg.norm(cl - cl[i], axis=1)
            nearby = np.where((dists < self.branch_radius_mm) & (dists > 0))[0]
            for j in nearby:
                if abs(i - j) > 1:
                    d = float(dists[j])
                    self._adj[i].append((j, d))

    def _project(self, point: np.ndarray) -> int:
        if self._cl is None:
            return 0
        return int(np.argmin(np.linalg.norm(self._cl - point, axis=1)))

    def path_remaining(self, tip: np.ndarray, target: np.ndarray) -> float:
        if self._cl is None:
            return float(np.linalg.norm(tip - target))
        import heapq
        src = self._project(tip)
        dst = self._project(target)
        if src == dst:
            return float(np.linalg.norm(tip - target))
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                return d
            if d > dist.get(u, float("inf")):
                continue
            for v, w in self._adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return float(np.linalg.norm(tip - target))  # fallback


_REGISTRY = {
    "euclidean": EuclideanPathfinder,
    "manifold":  ManifoldPathfinder,
    "dijkstra":  DijkstraPathfinder,
}


def make_pathfinder(
    kind: str = "manifold",
    centerline: Optional[np.ndarray] = None,
    **kwargs,
) -> Pathfinder:
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown pathfinder '{kind}'. Choose from: {list(_REGISTRY)}")
    cls = _REGISTRY[kind]
    if kind == "euclidean":
        return cls()
    if centerline is not None:
        return cls(centerline, **kwargs)
    return cls(np.zeros((2, 3)), **kwargs)


def register(name: str, cls):
    _REGISTRY[name] = cls
