"""
SynthEndoSim — Cubic Hermite Spline utilities.
Used to generate smooth procedural vessel branches.
Ported from stEVE (MIT licence).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.interpolate import CubicHermiteSpline


@dataclass
class CHSPoint:
    """Control point for a Cubic Hermite Spline (coords + radius + derivatives)."""
    y: Tuple[float, float, float, float]      # (x, y, z, radius)
    dydx: Tuple[float, float, float, float]   # derivatives

    @property
    def coords(self): return np.array(self.y[:3], dtype=np.float32)
    @property
    def r(self): return float(self.y[3])
    @property
    def d_coords(self): return np.array(self.dydx[:3], dtype=np.float32)
    @property
    def d_r(self): return float(self.dydx[3])


def chs_point_normal(
    coords_mean, coords_sigma,
    direction_mean, direction_sigma,
    direction_magnitude_mean_and_sigma,
    radius_mean_and_sigma,
    d_radius_mean_and_sigma,
    coord_offset=None,
    rng: np.random.Generator = None,
) -> CHSPoint:
    """Sample a CHSPoint from Gaussian distributions (for procedural generation)."""
    if coord_offset is None:
        coord_offset = (0., 0., 0.)
    rng = rng or np.random.default_rng()
    n = rng.normal

    y = [
        n(coords_mean[i], coords_sigma[i]) + coord_offset[i] for i in range(3)
    ] + [n(radius_mean_and_sigma[0], radius_mean_and_sigma[1])]

    d_dir = np.array([n(direction_mean[i], direction_sigma[i]) for i in range(3)], np.float32)
    mag = n(direction_magnitude_mean_and_sigma[0], direction_magnitude_mean_and_sigma[1])
    d_coords = d_dir / (np.linalg.norm(d_dir) + 1e-9) * mag
    dydx = list(d_coords) + [n(d_radius_mean_and_sigma[0], d_radius_mean_and_sigma[1])]

    return CHSPoint(tuple(y), tuple(dydx))


def chs_to_branch_points(
    points: List[CHSPoint],
    resolution: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate a list of CHSPoints into evenly-spaced (coordinates, radii).
    Returns:
        coordinates : (N, 3) float32
        radii       : (N,)   float32
    """
    y = np.array([p.y for p in points])
    dydx = np.array([p.dydx for p in points])
    # arc-length parameterisation
    x = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        x.append(x[-1] + np.linalg.norm(np.array(a.coords) - np.array(b.coords)))

    spline = CubicHermiteSpline(x, y, dydx)
    xs = np.arange(x[0], x[-1], 0.1)
    pts = spline(xs)  # (M, 4)

    # resample at `resolution` spacing
    sampled = [pts[0]]
    acc = 0.0
    for cur, nxt in zip(pts[:-1], pts[1:]):
        seg = np.linalg.norm(nxt[:3] - cur[:3])
        acc += seg
        if acc >= resolution:
            overshoot = acc - resolution
            t = 1.0 - overshoot / (seg + 1e-9)
            new_pt = cur + t * (nxt - cur)
            sampled.append(new_pt)
            acc = overshoot

    sampled = np.array(sampled, dtype=np.float32)
    return sampled[:, :3], sampled[:, 3] / 2   # radii = diameter/2
