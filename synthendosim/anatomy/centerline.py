"""
SynthEndoSim — Centerline utilities.

Provides:
  - Manifold distance: curvature-weighted geodesic along centerline
  - Nearest point projection onto centerline
  - Path remaining from tip to target
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def local_curvature(points: np.ndarray) -> np.ndarray:
    """Discrete Frenet curvature κ_i = |p'×p''| / |p'|³ (shape: N,)."""
    if len(points) < 3:
        return np.zeros(len(points), dtype=np.float64)
    dp = np.gradient(points, axis=0)
    ddp = np.gradient(dp, axis=0)
    cross_n = np.linalg.norm(np.cross(dp, ddp), axis=1)
    dp_n = np.linalg.norm(dp, axis=1)
    return cross_n / (dp_n ** 3 + 1e-12)


def curvature_reference(points: np.ndarray, percentile: float = 75.0) -> float:
    k = local_curvature(points)
    ref = float(np.percentile(k, percentile))
    return ref if ref > 1e-12 else 1.0


def curvature_weighted_length(
    points: np.ndarray,
    alpha: float = 1.0,
    kappa_ref: float = 1.0,
) -> float:
    """
    L_cw = Σ‖Δpᵢ‖ · (1 + α · κ̂ᵢ)
    κ̂ = κ / kappa_ref   (75th-percentile normalised)
    alpha=0 → plain arc-length.
    """
    if len(points) < 2:
        return 0.0
    kappa = local_curvature(points)
    kappa_hat = kappa / (kappa_ref + 1e-12)
    seg_len = np.linalg.norm(points[1:] - points[:-1], axis=1)
    seg_khat = 0.5 * (kappa_hat[:-1] + kappa_hat[1:])
    return float(np.sum(seg_len * (1.0 + alpha * seg_khat)))


class ManifoldPathfinder:
    """
    Precomputed curvature-weighted distance lookup along a centerline.

    Usage:
        pf = ManifoldPathfinder(centerline_pts, alpha=1.0)
        remaining = pf.distance_to_target(tip_position, target_position)
    """

    def __init__(
        self,
        centerline: np.ndarray,
        alpha: float = 1.0,
    ):
        self.centerline = centerline.astype(np.float64)
        self.alpha = alpha
        self._kappa_ref = curvature_reference(self.centerline)
        self._cumulative = self._build_cumulative()

    def _build_cumulative(self) -> np.ndarray:
        """Curvature-weighted cumulative arc-length for each node (shape: N,)."""
        kappa = local_curvature(self.centerline)
        kappa_hat = kappa / (self._kappa_ref + 1e-12)
        seg_len = np.linalg.norm(
            self.centerline[1:] - self.centerline[:-1], axis=1
        )
        seg_khat = 0.5 * (kappa_hat[:-1] + kappa_hat[1:])
        weighted = seg_len * (1.0 + self.alpha * seg_khat)
        cumul = np.zeros(len(self.centerline))
        cumul[1:] = np.cumsum(weighted)
        return cumul

    def _nearest_index(self, point: np.ndarray) -> int:
        dists = np.linalg.norm(self.centerline - point, axis=1)
        return int(np.argmin(dists))

    def distance_to_target(
        self,
        tip: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Curvature-weighted geodesic distance from tip to target along centerline."""
        i_tip = self._nearest_index(tip)
        i_tgt = self._nearest_index(target)
        if i_tip == i_tgt:
            return float(np.linalg.norm(tip - target))
        i_lo, i_hi = min(i_tip, i_tgt), max(i_tip, i_tgt)
        return float(abs(self._cumulative[i_hi] - self._cumulative[i_lo]))

    def path_remaining(
        self,
        tip: np.ndarray,
        target: np.ndarray,
    ) -> float:
        return self.distance_to_target(tip, target)

    def nearest_point(self, tip: np.ndarray) -> np.ndarray:
        idx = self._nearest_index(tip)
        return self.centerline[idx].astype(np.float32)


def euclidean_path_remaining(tip: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(tip - target))
