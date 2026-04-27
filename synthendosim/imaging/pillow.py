"""
SynthEndoSim — Pillow (2-D line drawing) fluoroscopy.

Lightweight alternative to DRR: draws the guidewire as a line on a blank canvas.
~100× faster than DRR. Useful for rapid iteration and debug.

Output: (H, W) float32 in [0, 1]
  0.0 = background (white in DSA convention)
  1.0 = wire (black in DSA convention)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np



class PillowImager:
    """
    Pure-NumPy line rasteriser. No X-ray physics — just paints wire segments.

    Parameters
    ----------
    image_size : (H, W)
    projection_axis : int
        Which world axis to project out: 0=X, 1=Y (anterior-posterior), 2=Z
    line_radius_px : int
        Half-width of drawn wire in pixels
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        projection_axis: int = 1,
        line_radius_px: int = 1,
        **kwargs,
    ):
        self.image_size  = image_size
        self.proj_axis   = projection_axis
        self.radius      = max(0, line_radius_px)
        self._bbox_min: Optional[np.ndarray] = None
        self._bbox_max: Optional[np.ndarray] = None

    def set_anatomy_bounds(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> None:
        self._bbox_min = bbox_min.astype(np.float32)
        self._bbox_max = bbox_max.astype(np.float32)

    def render(
        self,
        wire_nodes: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        H, W = self.image_size
        canvas = np.zeros((H, W), dtype=np.float32)
        if wire_nodes is None or len(wire_nodes) < 2:
            return canvas

        # project to 2D (drop projection axis)
        axes = [i for i in range(3) if i != self.proj_axis]
        pts2d = wire_nodes[:, axes]  # (N, 2)

        # normalise to pixel coords
        if self._bbox_min is not None:
            lo = self._bbox_min[axes]
            hi = self._bbox_max[axes]
            denom = np.where(hi - lo > 0, hi - lo, 1.0)
            pts2d = (pts2d - lo) / denom
        pts2d = np.clip(pts2d, 0.0, 1.0)
        px = (pts2d[:, 0] * (W - 1)).astype(int)
        py = (pts2d[:, 1] * (H - 1)).astype(int)

        # rasterise each segment with Bresenham's line
        for i in range(len(px) - 1):
            _draw_line(canvas, px[i], py[i], px[i+1], py[i+1], self.radius)

        return canvas

    def render_lao(self, wire_nodes, rng=None) -> np.ndarray:
        return self.render(wire_nodes, rng)


def _draw_line(canvas: np.ndarray, x0, y0, x1, y1, radius: int) -> None:
    H, W = canvas.shape
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = y0 + dr, x0 + dc
                if 0 <= r < H and 0 <= c < W:
                    canvas[r, c] = 1.0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 < dx:
            err += dx; y0 += sy
