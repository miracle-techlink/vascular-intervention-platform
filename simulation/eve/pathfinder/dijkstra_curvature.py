"""
CurvatureWeightedDijkstra: geodesic distance on the vascular manifold
weighted by local centerline curvature.

Edge weight: w_i = ||Δp_i|| * (1 + alpha * kappa_hat_i)
where kappa_hat = kappa / kappa_ref  (75th-percentile normalisation)

Physical motivation:
  A guidewire navigating a high-curvature segment (aortic arch, bifurcations)
  experiences larger frictional forces and deflection.  Weighting those
  segments more in the distance metric gives the RL agent a steeper reward
  gradient near curves and better reflects the real navigation difficulty.

alpha = 0  →  identical to BruteForceBFS (plain arc-length)
alpha > 0  →  high-curvature sections cost more
"""

import numpy as np
from math import inf

from .bruteforcebfs import BruteForceBFS, BPConnection
from ..intervention import Intervention


# ── curvature helpers ────────────────────────────────────────────────────────

def _local_curvature(points: np.ndarray) -> np.ndarray:
    """
    Discrete Frenet curvature kappa_i = |p'_i × p''_i| / |p'_i|^3.
    Uses np.gradient (central differences, 2nd-order accurate at boundaries).
    Returns shape (N,), values >= 0.
    """
    if len(points) < 3:
        return np.zeros(len(points), dtype=np.float64)
    dp  = np.gradient(points, axis=0)          # first derivative
    ddp = np.gradient(dp,     axis=0)          # second derivative
    cross_n = np.linalg.norm(np.cross(dp, ddp), axis=1)
    dp_n    = np.linalg.norm(dp,                axis=1)
    return cross_n / (dp_n ** 3 + 1e-12)


def curvature_weighted_length(points: np.ndarray,
                               alpha: float,
                               kappa_ref: float) -> float:
    """
    L_cw = sum_i  ||p_{i+1}-p_i||  *  (1 + alpha * kappa_hat_{i+0.5})
    kappa_hat_{i+0.5} = 0.5*(kappa_i + kappa_{i+1}) / kappa_ref
    """
    if len(points) < 2:
        return 0.0
    kappa     = _local_curvature(points)
    kappa_hat = kappa / (kappa_ref + 1e-12)
    seg_len   = np.linalg.norm(points[1:] - points[:-1], axis=1)
    seg_khat  = 0.5 * (kappa_hat[:-1] + kappa_hat[1:])
    return float(np.sum(seg_len * (1.0 + alpha * seg_khat)))


# ── pathfinder ───────────────────────────────────────────────────────────────

class CurvatureWeightedDijkstra(BruteForceBFS):
    """
    Drop-in replacement for BruteForceBFS that uses curvature-weighted
    arc-length as the edge cost.  path_length is therefore the geodesic
    distance on the vascular manifold rather than plain arc-length.

    Parameters
    ----------
    intervention : Intervention
    alpha        : float
        Curvature weighting coefficient.  0 = plain arc-length.  1.0 = full
        curvature weighting (recommended default).
    """

    def __init__(self, intervention: Intervention, alpha: float = 1.0):
        super().__init__(intervention)
        self.alpha      = alpha
        self._kappa_ref = 1.0       # set in _init_vessel_tree

    # ------------------------------------------------------------------
    def _init_vessel_tree(self) -> None:
        self._kappa_ref = self._compute_global_kappa_ref()
        super()._init_vessel_tree()   # calls _initialize_node_connections

    def _compute_global_kappa_ref(self) -> float:
        """75th-percentile curvature across all branch centerlines."""
        all_k = []
        for branch in self.intervention.vessel_tree.branches:
            all_k.append(_local_curvature(branch.coordinates))
        if not all_k:
            return 1.0
        merged = np.concatenate(all_k)
        ref = float(np.percentile(merged, 75))
        return ref if ref > 1e-12 else 1.0

    # ------------------------------------------------------------------
    def _initialize_node_connections(self, branching_points):
        """Same as parent but edge weights are curvature-weighted lengths."""
        node_connections = {}
        for bp in branching_points:
            node_connections[bp] = {}
            for connection in bp.connections:
                for target_bp in branching_points:
                    if bp == target_bp:
                        continue
                    if connection in target_bp.connections:
                        points = connection.get_path_along_branch(
                            bp.coordinates, target_bp.coordinates)
                        length = curvature_weighted_length(
                            points, self.alpha, self._kappa_ref)
                        node_connections[bp][target_bp] = BPConnection(length, points)
        return node_connections

    # ------------------------------------------------------------------
    def _get_shortest_path(self, start_branch, target_branch, start, target):
        """Same topology search as BruteForceBFS; curvature-weighted lengths."""
        search_graph = self._create_search_graph(start_branch, target_branch)
        bfs_paths    = self._get_bfs_paths_generator(search_graph)

        shortest_path_length = inf
        shortest_path        = None

        path = next(bfs_paths, None)

        if path is None:
            return None, 0.0, np.empty((1, 3))

        if len(path) == 2:
            pts    = start_branch.get_path_along_branch(start, target)
            length = curvature_weighted_length(pts, self.alpha, self._kappa_ref)
            return None, length, pts

        # multi-branch path
        pts    = start_branch.get_path_along_branch(start, path[1].coordinates)
        length = curvature_weighted_length(pts, self.alpha, self._kappa_ref)

        for node, next_node in zip(path[1:-2], path[2:-1]):
            conn    = self._node_connections[node][next_node]
            length += conn.length
            pts     = np.vstack((pts, conn.points[1:]))

        end_pts    = target_branch.get_path_along_branch(path[-2].coordinates, target)
        end_length = curvature_weighted_length(end_pts, self.alpha, self._kappa_ref)
        pts        = np.vstack((pts, end_pts[1:]))
        length    += end_length
        shortest_path = path[1:-1]

        return shortest_path, length, pts
