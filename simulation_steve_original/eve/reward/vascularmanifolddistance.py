"""
VascularManifoldDistanceDelta: potential-based reward from curvature-weighted
geodesic distance on the vascular manifold.

r_t = (D_{t-1} - D_t) * factor

D_t is computed by CurvatureWeightedDijkstra, so it integrates
(1 + alpha * kappa_hat(s)) ds along the centerline — segments with higher
curvature cost more, giving a richer reward gradient than plain arc-length.

Use with CurvatureWeightedDijkstra as the pathfinder.
Compatible with any Pathfinder (falls back to plain arc-length if BruteForceBFS).
"""
from .reward import Reward
from ..pathfinder import Pathfinder


class VascularManifoldDistanceDelta(Reward):

    def __init__(self, pathfinder: Pathfinder, factor: float = 0.01) -> None:
        self.pathfinder = pathfinder
        self.factor     = factor
        self._last_dist = None

    def step(self) -> None:
        d = self.pathfinder.path_length
        self.reward = (self._last_dist - d) * self.factor
        self._last_dist = d

    def reset(self, episode_nr: int = 0) -> None:
        self.reward     = 0.0
        self._last_dist = self.pathfinder.path_length
