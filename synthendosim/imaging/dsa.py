"""
SynthEndoSim — Biplane DSA renderer (Beer-Lambert + Poisson noise).

Renders two X-ray projections (LAO and LAT planes) from wire node positions.
Far faster than stEVE's fluoroscopy because we skip SOFA's GL rendering
and do everything in NumPy.

Physics model:
  I = I₀ · exp(-μ_blood · L_blood - μ_tissue · L_tissue)  (Beer-Lambert)
  Observed = Poisson(I)   (quantum noise, N₀ photons)
  DSA = -log(I_contrast / I_mask)                          (subtraction)
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def _rotation_matrix(azimuth_deg: float, elevation_deg: float = 0.0) -> np.ndarray:
    """
    Build 3×3 rotation: azimuth around Y, then elevation around X.
    Matches clinical C-arm convention (LAO/RAO = azimuth, CRA/CAU = elevation).
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    Ry = np.array([
        [ np.cos(az), 0, np.sin(az)],
        [          0, 1,          0],
        [-np.sin(az), 0, np.cos(az)],
    ])
    Rx = np.array([
        [1,          0,           0],
        [0, np.cos(el), -np.sin(el)],
        [0, np.sin(el),  np.cos(el)],
    ])
    return Rx @ Ry


def _project_points(
    points: np.ndarray,         # (N, 3) world mm
    R: np.ndarray,              # 3×3 rotation
    bbox_min: np.ndarray,       # (3,) world mm
    bbox_max: np.ndarray,       # (3,) world mm
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Orthographic projection: rotate world points → take XZ plane → pixel coords.
    Returns (N, 2) integer pixel indices.
    """
    H, W = image_size
    rotated = (R @ points.T).T                          # (N, 3)
    bbox_r_min = (R @ bbox_min).min(axis=0) if bbox_min.ndim > 1 else (R @ bbox_min)
    bbox_r_max = (R @ bbox_max).max(axis=0) if bbox_max.ndim > 1 else (R @ bbox_max)

    # project onto XZ
    px = rotated[:, 0]
    pz = rotated[:, 2]
    span_x = bbox_r_max[0] - bbox_r_min[0] + 1e-6
    span_z = bbox_r_max[2] - bbox_r_min[2] + 1e-6
    u = np.clip(((px - bbox_r_min[0]) / span_x * (W - 1)).astype(int), 0, W - 1)
    v = np.clip(((pz - bbox_r_min[2]) / span_z * (H - 1)).astype(int), 0, H - 1)
    return np.stack([v, u], axis=1)


class BiplaneDSA:
    """
    Biplane DSA renderer.

    Parameters
    ----------
    lao_deg     : azimuth of LAO plane (default 30°)
    lat_deg     : azimuth of LAT plane (default 120°)
    image_size  : (H, W) pixels
    n0_photons  : Poisson intensity baseline
    mu_blood    : linear attenuation coefficient blood (cm⁻¹)
    mu_tissue   : linear attenuation coefficient soft tissue (cm⁻¹)
    """

    def __init__(
        self,
        lao_deg: float = 30.0,
        lat_deg: float = 120.0,
        image_size: Tuple[int, int] = (128, 128),
        n0_photons: int = 15_000,
        mu_blood: float = 0.048,
        mu_tissue: float = 0.020,
    ):
        self.lao_deg = lao_deg
        self.lat_deg = lat_deg
        self.image_size = image_size
        self.n0 = n0_photons
        self.mu_blood = mu_blood
        self.mu_tissue = mu_tissue

        self._R_lao = _rotation_matrix(lao_deg)
        self._R_lat = _rotation_matrix(lat_deg)

        self._bbox_min: Optional[np.ndarray] = None
        self._bbox_max: Optional[np.ndarray] = None

    def set_anatomy_bounds(self, bbox_min: np.ndarray, bbox_max: np.ndarray):
        """Call once after anatomy is loaded so projections are normalised correctly."""
        self._bbox_min = bbox_min.astype(np.float64)
        self._bbox_max = bbox_max.astype(np.float64)

    def render(
        self,
        wire_nodes: np.ndarray,         # (N, 3) world mm
        vessel_nodes: Optional[np.ndarray] = None,  # (M, 3) vessel surface pts
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Returns (2, H, W) float32 DSA images: [LAO, LAT].
        Values in [0, 1] (DSA-subtracted, normalised).
        """
        if rng is None:
            rng = np.random.default_rng()

        bmin = self._bbox_min if self._bbox_min is not None else wire_nodes.min(0) - 10
        bmax = self._bbox_max if self._bbox_max is not None else wire_nodes.max(0) + 10

        lao = self._render_plane(wire_nodes, vessel_nodes, self._R_lao, bmin, bmax, rng)
        lat = self._render_plane(wire_nodes, vessel_nodes, self._R_lat, bmin, bmax, rng)
        return np.stack([lao, lat], axis=0).astype(np.float32)

    def render_lao(
        self,
        wire_nodes: np.ndarray,
        vessel_nodes: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        bmin = self._bbox_min if self._bbox_min is not None else wire_nodes.min(0) - 10
        bmax = self._bbox_max if self._bbox_max is not None else wire_nodes.max(0) + 10
        return self._render_plane(wire_nodes, vessel_nodes, self._R_lao, bmin, bmax, rng)

    def _render_plane(
        self,
        wire_nodes: np.ndarray,
        vessel_nodes: Optional[np.ndarray],
        R: np.ndarray,
        bmin: np.ndarray,
        bmax: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        H, W = self.image_size

        # --- vessel background image (mask) ---
        bg = np.full((H, W), float(self.n0), dtype=np.float64)
        if vessel_nodes is not None and len(vessel_nodes) > 0:
            vpx = _project_points(vessel_nodes, R, bmin, bmax, self.image_size)
            depth = ((R @ vessel_nodes.T)[1])          # depth along projection axis
            span = depth.max() - depth.min() + 1e-6
            for (v, u), d in zip(vpx, depth):
                thickness_mm = (d - depth.min()) / span * 30.0   # ~30mm vessel wall
                atten = np.exp(-self.mu_tissue * thickness_mm / 10)
                bg[v, u] *= atten

        bg_noisy = rng.poisson(bg).astype(np.float64)

        # --- contrast image (vessel + wire) ---
        fg = bg.copy()
        if vessel_nodes is not None and len(vessel_nodes) > 0:
            vpx = _project_points(vessel_nodes, R, bmin, bmax, self.image_size)
            depth = ((R @ vessel_nodes.T)[1])
            span = depth.max() - depth.min() + 1e-6
            for (v, u), d in zip(vpx, depth):
                lumen_mm = 8.0   # ~8mm vessel lumen
                atten = np.exp(-self.mu_blood * lumen_mm / 10)
                fg[v, u] *= atten

        # wire attenuation (thin, high-density)
        if len(wire_nodes) > 1:
            wpx = _project_points(wire_nodes, R, bmin, bmax, self.image_size)
            wire_atten = np.exp(-self.mu_blood * 4.0 * 5)  # 4× blood density, 5mm
            for v, u in wpx:
                fg[v, u] *= wire_atten

        fg_noisy = rng.poisson(fg).astype(np.float64)

        # --- DSA subtraction: -log(I_fg / I_bg) ---
        dsa = -np.log((fg_noisy + 1.0) / (bg_noisy + 1.0))
        dsa = np.clip(dsa, 0.0, None)

        # normalise to [0, 1]
        dmax = dsa.max()
        if dmax > 0:
            dsa /= dmax
        return dsa.astype(np.float32)


class MonoplaneDSA(BiplaneDSA):
    """Single-plane DSA — returns (H, W) instead of (2, H, W)."""

    def render(
        self,
        wire_nodes: np.ndarray,
        vessel_nodes: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        bmin = self._bbox_min if self._bbox_min is not None else wire_nodes.min(0) - 10
        bmax = self._bbox_max if self._bbox_max is not None else wire_nodes.max(0) + 10
        return self._render_plane(wire_nodes, vessel_nodes, self._R_lao, bmin, bmax, rng)


class NullImager:
    """No-op imager when imaging obs is disabled."""
    def set_anatomy_bounds(self, *_): pass
    def render(self, *_) -> None:
        return None
