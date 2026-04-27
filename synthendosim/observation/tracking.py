"""
SynthEndoSim — Tracking observation types.

Tracking2D  : N evenly-spaced 2-D wire positions (image coords, [0,1]²)
Tracking3D  : N evenly-spaced 3-D wire positions (world-space, normalised)
LastAction  : the previous step's action vector
InsertionLengths : absolute insertion length(s) per device
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import gymnasium as gym

from ..core.types import SimState, DeviceState


# ── Tracking2D ─────────────────────────────────────────────────────────────────

@dataclass
class Tracking2D:
    """
    Evenly samples N points along the guidewire trajectory in 2-D image space.
    Image coords are (u, v) ∈ [0,1]², derived from the DSA image projection.

    Used by image-based policies that don't directly observe 3-D pose.
    """
    n_points: int = 5
    resolution_mm: float = 1.0   # spacing between sampled points
    name: str = "tracking2d"

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(0.0, 1.0, (self.n_points, 2), dtype=np.float32)

    def build(self, state: SimState, imager=None, bbox_min=None, bbox_max=None) -> np.ndarray:
        """Return (n_points, 2) normalised 2-D tracking."""
        if not state.devices:
            return np.zeros((self.n_points, 2), dtype=np.float32)
        dof = state.devices[0].dof_positions  # (K, 3) world-space nodes
        return self._sample(dof, bbox_min, bbox_max)

    def _sample(self, dof: np.ndarray, bbox_min, bbox_max) -> np.ndarray:
        if len(dof) == 0:
            return np.zeros((self.n_points, 2), dtype=np.float32)
        # project 3D → 2D by dropping z, normalise via bbox
        pts2d = dof[:, :2]  # (K, 2)
        if bbox_min is not None and bbox_max is not None:
            lo = bbox_min[:2]; hi = bbox_max[:2]
            denom = np.where(hi - lo > 0, hi - lo, 1.0)
            pts2d = (pts2d - lo) / denom
        pts2d = np.clip(pts2d, 0.0, 1.0).astype(np.float32)
        return self._evenly_spaced(pts2d)

    def _evenly_spaced(self, pts: np.ndarray) -> np.ndarray:
        result = [pts[0]]
        acc = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            if len(result) >= self.n_points:
                break
            seg = np.linalg.norm(b - a)
            acc += seg
            while acc >= self.resolution_mm and len(result) < self.n_points:
                t = (acc - self.resolution_mm) / (seg + 1e-9)
                pt = b - t * (b - a)
                result.append(pt.astype(np.float32))
                acc -= self.resolution_mm
        while len(result) < self.n_points:
            result.append(result[-1])
        return np.array(result[:self.n_points], dtype=np.float32)


# ── Tracking3D ─────────────────────────────────────────────────────────────────

@dataclass
class Tracking3D:
    """
    N evenly-spaced 3-D wire positions, normalised to [0,1]³ via bounding box.
    Gives the policy a sense of the full wire configuration in 3-D space.
    """
    n_points: int = 5
    resolution_mm: float = 1.0
    name: str = "tracking3d"

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(0.0, 1.0, (self.n_points, 3), dtype=np.float32)

    def build(self, state: SimState, bbox_min=None, bbox_max=None) -> np.ndarray:
        if not state.devices:
            return np.zeros((self.n_points, 3), dtype=np.float32)
        dof = state.devices[0].dof_positions  # (K, 3)
        pts = dof.copy().astype(np.float32)
        if bbox_min is not None and bbox_max is not None:
            denom = np.where(bbox_max - bbox_min > 0, bbox_max - bbox_min, 1.0)
            pts = (pts - bbox_min) / denom
        pts = np.clip(pts, 0.0, 1.0)
        return self._evenly_spaced(pts)

    def _evenly_spaced(self, pts: np.ndarray) -> np.ndarray:
        result = [pts[0]]
        acc = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            if len(result) >= self.n_points:
                break
            seg = float(np.linalg.norm(b - a))
            acc += seg
            while acc >= self.resolution_mm and len(result) < self.n_points:
                t = (acc - self.resolution_mm) / (seg + 1e-9)
                result.append((b - t * (b - a)).astype(np.float32))
                acc -= self.resolution_mm
        while len(result) < self.n_points:
            result.append(result[-1])
        return np.array(result[:self.n_points], dtype=np.float32)


# ── LastAction ─────────────────────────────────────────────────────────────────

@dataclass
class LastAction:
    """
    Includes the previous action in the observation.
    Useful for RNN/LSTM policies and action-smoothing regularisation.
    """
    n_devices: int = 1
    name: str = "last_action"

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-1.0, 1.0, (self.n_devices * 2,), dtype=np.float32)

    def build(self, last_action: Optional[np.ndarray]) -> np.ndarray:
        if last_action is None:
            return np.zeros(self.n_devices * 2, dtype=np.float32)
        return np.asarray(last_action, dtype=np.float32).flatten()


# ── InsertionLengths ───────────────────────────────────────────────────────────

@dataclass
class InsertionLengths:
    """
    Per-device absolute insertion lengths (mm), normalised by device total_length.
    For coaxial systems: [guidewire_insertion, catheter_insertion].
    """
    n_devices: int = 1
    max_length_mm: float = 450.0
    name: str = "insertion_lengths"

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(0.0, 1.0, (self.n_devices,), dtype=np.float32)

    def build(self, state: SimState) -> np.ndarray:
        lengths = []
        for i in range(self.n_devices):
            if i < len(state.devices):
                l = state.devices[i].inserted_length / (self.max_length_mm + 1e-9)
            else:
                l = 0.0
            lengths.append(float(np.clip(l, 0., 1.)))
        return np.array(lengths, dtype=np.float32)


# ── InsertionLengthRelative ────────────────────────────────────────────────────

@dataclass
class InsertionLengthRelative:
    """
    Relative insertion: device_i - device_j, normalised.
    Models the coaxial relationship (guidewire protrusion beyond catheter tip).
    """
    device_id: int = 0
    relative_to: int = 1
    max_relative_mm: float = 100.0
    name: str = "insertion_length_relative"

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def build(self, state: SimState) -> np.ndarray:
        if len(state.devices) <= max(self.device_id, self.relative_to):
            return np.zeros(1, dtype=np.float32)
        a = state.devices[self.device_id].inserted_length
        b = state.devices[self.relative_to].inserted_length
        rel = np.clip((a - b) / (self.max_relative_mm + 1e-9), -1., 1.)
        return np.array([rel], dtype=np.float32)
