"""
SynthEndoSim — Observation builders.

Converts SimState + imaging output → typed Observation.
Normalisation: all 3D coords mapped to [0,1] using anatomy bounding box.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from ..core.types import SimState, Observation
from ..config.schema import ObsConfig
import gymnasium as gym


class ObservationBuilder:
    """
    Builds Observation from SimState and optional DSA image.

    Parameters
    ----------
    cfg        : ObsConfig
    bbox_min   : anatomy bounding box minimum (3,)
    bbox_max   : anatomy bounding box maximum (3,)
    path_scale : normalisation scale for path_remaining (mm)
    """

    def __init__(
        self,
        cfg: ObsConfig,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        path_scale: float = 300.0,
    ):
        self.cfg = cfg
        self.bbox_min = bbox_min.astype(np.float32)
        self.bbox_max = bbox_max.astype(np.float32)
        self.bbox_span = np.maximum(self.bbox_max - self.bbox_min, 1e-6)
        self.path_scale = max(path_scale, 1.0)

    def build(
        self,
        state: SimState,
        wire_image: Optional[np.ndarray] = None,
    ) -> Observation:
        if not state.devices:
            tip = np.zeros(3, dtype=np.float32)
            ins_len = 0.0
            rot = 0.0
        else:
            d = state.devices[0]
            tip = d.tip_position.astype(np.float32)
            ins_len = float(d.inserted_length)
            rot = float(d.rotation)

        target = state.target_position.astype(np.float32)

        if self.cfg.normalise:
            tip_n = np.clip((tip - self.bbox_min) / self.bbox_span, 0.0, 1.0)
            tgt_n = np.clip((target - self.bbox_min) / self.bbox_span, 0.0, 1.0)
            ins_n = ins_len / self.path_scale
            rot_n = (rot % (2 * np.pi)) / (2 * np.pi)
            path_n = state.path_remaining / self.path_scale
        else:
            tip_n, tgt_n = tip, target
            ins_n, rot_n, path_n = ins_len, rot, state.path_remaining

        return Observation(
            tip_3d=tip_n,
            wire_mask=wire_image,
            insertion_length=float(ins_n),
            rotation=float(rot_n),
            target_3d=tgt_n,
            path_remaining=float(path_n),
        )

    def observation_space(self) -> gym.Space:
        """Return gymnasium observation space (dict or box)."""
        spaces: dict = {}

        if self.cfg.use_tip_3d:
            spaces["tip_3d"] = gym.spaces.Box(0.0, 1.0, (3,), np.float32)
        if self.cfg.use_path_remaining:
            spaces["path_remaining"] = gym.spaces.Box(0.0, 1.0, (1,), np.float32)
        if self.cfg.use_insertion_length:
            spaces["insertion_length"] = gym.spaces.Box(0.0, 1.0, (1,), np.float32)
        if self.cfg.use_rotation:
            spaces["rotation"] = gym.spaces.Box(0.0, 1.0, (1,), np.float32)

        spaces["target_3d"] = gym.spaces.Box(0.0, 1.0, (3,), np.float32)

        if self.cfg.use_wire_mask:
            # biplane: (2, H, W), monoplane: (H, W) — use generic shape
            spaces["wire_mask"] = gym.spaces.Box(0.0, 1.0, (2, 128, 128), np.float32)

        if len(spaces) == 1 and "tip_3d" in spaces:
            return spaces["tip_3d"]

        return gym.spaces.Dict(spaces)

    def filter(self, obs: Observation) -> dict | np.ndarray:
        """Return only the enabled fields per ObsConfig."""
        if not self.cfg.use_wire_mask and obs.wire_mask is None:
            parts = []
            if self.cfg.use_tip_3d:
                parts.append(obs.tip_3d)
            parts.append(obs.target_3d)
            if self.cfg.use_insertion_length:
                parts.append(np.array([obs.insertion_length], dtype=np.float32))
            if self.cfg.use_rotation:
                parts.append(np.array([obs.rotation], dtype=np.float32))
            if self.cfg.use_path_remaining:
                parts.append(np.array([obs.path_remaining], dtype=np.float32))
            return np.concatenate(parts).astype(np.float32)

        d = obs.to_dict()
        keep = {}
        if self.cfg.use_tip_3d:
            keep["tip_3d"] = d["tip_3d"]
        keep["target_3d"] = d["target_3d"]
        if self.cfg.use_insertion_length:
            keep["insertion_length"] = d["insertion_length"]
        if self.cfg.use_rotation:
            keep["rotation"] = d["rotation"]
        if self.cfg.use_path_remaining:
            keep["path_remaining"] = d["path_remaining"]
        if self.cfg.use_wire_mask and "wire_mask" in d:
            keep["wire_mask"] = d["wire_mask"]
        return keep
