"""
SynthEndoSim — Start strategy implementations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from .base import Start
from ..core.types import AnatomySpec


class InsertionPointStart(Start):
    """
    Fixed insertion: always start at anatomy.insertion_point.
    Standard for single-anatomy training.
    """
    def get_start(self, anatomy: AnatomySpec, episode: int = 0,
                  rng=None) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.array(anatomy.insertion_point, dtype=np.float32),
            np.array(anatomy.insertion_direction, dtype=np.float32),
        )


@dataclass
class RandomAdvanceStart(Start):
    """
    Start with a random initial advance along the centerline.
    Curriculum: device begins partially inserted so agent doesn't always start from scratch.

    advance_range_mm : (min_mm, max_mm) — random advance within this range
    """
    advance_range_mm: Tuple[float, float] = (0.0, 50.0)

    def get_start(self, anatomy: AnatomySpec, episode: int = 0,
                  rng=None) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()
        base = np.array(anatomy.insertion_point, dtype=np.float32)
        direction = np.array(anatomy.insertion_direction, dtype=np.float32)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        advance = float(rng.uniform(*self.advance_range_mm))
        return base + direction * advance, direction


@dataclass
class CurriculumStart(Start):
    """
    Progressive curriculum: early episodes start close to insertion,
    later episodes start closer to target (or vice versa).

    Modes
    -----
    "easy_first"  : start near target in early episodes (short navigation)
    "hard_first"  : start at insertion always (full navigation from scratch)
    "progressive" : linearly interpolate from near-target → insertion as episodes increase
    """
    mode: str = "progressive"
    curriculum_episodes: int = 10_000

    def get_start(self, anatomy: AnatomySpec, episode: int = 0,
                  rng=None) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        entry = np.array(anatomy.insertion_point, dtype=np.float32)
        target = np.array(anatomy.target_point, dtype=np.float32)
        direction = np.array(anatomy.insertion_direction, dtype=np.float32)
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        if self.mode == "hard_first":
            return entry, direction

        if self.mode == "easy_first":
            t = min(1.0, episode / max(self.curriculum_episodes, 1))
            # t=0 → start near target; t=1 → start at entry
            pos = target * (1 - t) + entry * t
            return pos, direction

        if self.mode == "progressive":
            t = min(1.0, episode / max(self.curriculum_episodes, 1))
            # t=0 → start near target; t=1 → full episode from entry
            pos = target * (1 - t) + entry * t
            noise = rng.normal(0, 2.0, 3).astype(np.float32)
            return pos + noise, direction

        return entry, direction


@dataclass
class MultiAnatomyStart(Start):
    """
    Cycles through a list of anatomies, selecting one per episode.
    Enables multi-anatomy generalisation training without vec_env.
    Each anatomy can have its own insertion point.
    """
    anatomy_list: List[AnatomySpec] = field(default_factory=list)
    shuffle: bool = True

    def __post_init__(self):
        self._order = list(range(len(self.anatomy_list)))

    def reset(self, episode: int = 0):
        if self.shuffle and episode == 0:
            import random
            random.shuffle(self._order)

    def get_start(self, anatomy: AnatomySpec, episode: int = 0,
                  rng=None) -> Tuple[np.ndarray, np.ndarray]:
        if not self.anatomy_list:
            return (np.array(anatomy.insertion_point, dtype=np.float32),
                    np.array(anatomy.insertion_direction, dtype=np.float32))
        idx = self._order[episode % len(self._order)]
        selected = self.anatomy_list[idx]
        return (np.array(selected.insertion_point, dtype=np.float32),
                np.array(selected.insertion_direction, dtype=np.float32))
