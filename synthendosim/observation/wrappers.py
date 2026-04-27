"""
SynthEndoSim — Observation wrappers.

These wrap any obs builder component to add post-processing:

  Memory               : stack N consecutive observations (temporal history)
  RelativeToLastState  : delta obs — (current - previous)
  RelativeToFirstRow   : delta obs — (current - episode_start)
  Normalize            : scale obs to [lo, hi] range
  SelectiveMemory      : temporal memory for only selected dict keys
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Union
import numpy as np
import gymnasium as gym


class MemoryResetMode(IntEnum):
    FILL = 0   # fill all frames with the first observation
    ZERO = 1   # zero all frames except the first


# ── Memory ──────────────────────────────────────────────────────────────────────

class Memory:
    """
    Wraps a (N,) or (N, M) obs array and stacks it over `n_steps` time steps.
    Output shape: (n_steps, *obs_shape).

    Example:
        raw_obs : (3,)       → memory obs : (n_steps, 3)
        raw_obs : (5, 2)     → memory obs : (n_steps, 5, 2)
    """

    def __init__(
        self,
        inner_space: gym.spaces.Box,
        n_steps: int,
        reset_mode: MemoryResetMode = MemoryResetMode.FILL,
        name: str = "memory",
    ):
        self.inner_space = inner_space
        self.n_steps = n_steps
        self.reset_mode = reset_mode
        self.name = name
        self._buffer: Optional[np.ndarray] = None

    @property
    def space(self) -> gym.spaces.Box:
        lo = np.repeat(self.inner_space.low[np.newaxis], self.n_steps, axis=0)
        hi = np.repeat(self.inner_space.high[np.newaxis], self.n_steps, axis=0)
        return gym.spaces.Box(lo, hi, dtype=np.float32)

    def reset(self, first_obs: np.ndarray) -> np.ndarray:
        if self.reset_mode == MemoryResetMode.FILL:
            self._buffer = np.repeat(first_obs[np.newaxis], self.n_steps, axis=0)
        else:
            self._buffer = np.zeros((self.n_steps, *first_obs.shape), dtype=np.float32)
            self._buffer[0] = first_obs
        return self._buffer.copy()

    def update(self, new_obs: np.ndarray) -> np.ndarray:
        self._buffer = np.roll(self._buffer, 1, axis=0)
        self._buffer[0] = new_obs
        return self._buffer.copy()


# ── RelativeToLastState ─────────────────────────────────────────────────────────

class RelativeToLastState:
    """
    Computes delta obs: current - previous.
    Captures instantaneous velocity/change information.
    """

    def __init__(self, inner_space: gym.spaces.Box, name: str = "rel_last"):
        self.inner_space = inner_space
        self.name = name
        self._last: Optional[np.ndarray] = None

    @property
    def space(self) -> gym.spaces.Box:
        lo = self.inner_space.low  - self.inner_space.high
        hi = self.inner_space.high - self.inner_space.low
        return gym.spaces.Box(lo, hi, dtype=np.float32)

    def reset(self, first_obs: np.ndarray) -> np.ndarray:
        self._last = first_obs.copy()
        return np.zeros_like(first_obs)

    def update(self, new_obs: np.ndarray) -> np.ndarray:
        delta = new_obs - self._last
        self._last = new_obs.copy()
        return delta.astype(np.float32)


# ── RelativeToFirstRow ──────────────────────────────────────────────────────────

class RelativeToFirstRow:
    """
    Computes delta obs: current - episode_start.
    Captures cumulative displacement from initial state.
    """

    def __init__(self, inner_space: gym.spaces.Box, name: str = "rel_first"):
        self.inner_space = inner_space
        self.name = name
        self._first: Optional[np.ndarray] = None

    @property
    def space(self) -> gym.spaces.Box:
        lo = self.inner_space.low  - self.inner_space.high
        hi = self.inner_space.high - self.inner_space.low
        return gym.spaces.Box(lo, hi, dtype=np.float32)

    def reset(self, first_obs: np.ndarray) -> np.ndarray:
        self._first = first_obs.copy()
        return np.zeros_like(first_obs)

    def update(self, new_obs: np.ndarray) -> np.ndarray:
        return (new_obs - self._first).astype(np.float32)


# ── Normalize ───────────────────────────────────────────────────────────────────

class Normalize:
    """
    Normalises obs to [out_low, out_high] using known in_low / in_high bounds.
    """

    def __init__(
        self,
        inner_space: gym.spaces.Box,
        out_low: float = -1.0,
        out_high: float = 1.0,
        name: str = "normalized",
    ):
        self.inner_space = inner_space
        self.out_low  = out_low
        self.out_high = out_high
        self.name = name
        self._in_range = inner_space.high - inner_space.low
        self._in_range = np.where(self._in_range > 0, self._in_range, 1.0)

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            self.out_low, self.out_high, self.inner_space.shape, dtype=np.float32
        )

    def apply(self, obs: np.ndarray) -> np.ndarray:
        t = (obs - self.inner_space.low) / self._in_range
        return (t * (self.out_high - self.out_low) + self.out_low).astype(np.float32)


# ── SelectiveMemory ─────────────────────────────────────────────────────────────

class SelectiveMemory:
    """
    Applies temporal Memory wrapper to only selected keys in a Dict obs.
    Other keys pass through unchanged.

    Usage:
        sel_mem = SelectiveMemory(
            key_specs={"tracking3d": (tracking3d_space, 4, MemoryResetMode.FILL)},
            passthrough_keys=["tip_3d", "target_3d"],
        )
    """

    def __init__(
        self,
        key_specs: Dict[str, tuple],   # key → (space, n_steps, reset_mode)
        passthrough_keys: Optional[List[str]] = None,
    ):
        self._memories: Dict[str, Memory] = {}
        for key, (space, n_steps, mode) in key_specs.items():
            self._memories[key] = Memory(space, n_steps, mode, name=key)
        self._passthrough = set(passthrough_keys or [])

    def reset(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {}
        for k, mem in self._memories.items():
            if k in obs_dict:
                out[k] = mem.reset(obs_dict[k])
        for k in self._passthrough:
            if k in obs_dict:
                out[k] = obs_dict[k]
        return out

    def update(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {}
        for k, mem in self._memories.items():
            if k in obs_dict:
                out[k] = mem.update(obs_dict[k])
        for k in self._passthrough:
            if k in obs_dict:
                out[k] = obs_dict[k]
        return out
