"""
SynthEndoSim — Info aggregation wrappers.

AverageEpisodesLastStep : rolling mean of last-step metrics over N episodes
AverageSteps            : rolling mean of per-step metrics within an episode
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np


class AverageEpisodesLastStep:
    """
    Wraps a scalar metric and computes its rolling average over the last N episodes.
    Metric is sampled at the LAST step of each episode.

    Usage:
        wrapper = AverageEpisodesLastStep("target_distance_mm", n_episodes=100)
        # in env.step():
        info = wrapper.update(info, terminated or truncated)
        # info will contain "avg100/target_distance_mm" key after N episodes
    """

    def __init__(
        self,
        metric_key: str,
        n_episodes: int = 100,
        output_prefix: str = None,
    ):
        self.key = metric_key
        self.n = n_episodes
        self.prefix = output_prefix or f"avg{n_episodes}"
        self._buffer: List[float] = []
        self._last_value: Optional[float] = None

    def update(self, info: Dict[str, Any], episode_done: bool) -> Dict[str, Any]:
        if self.key in info:
            self._last_value = float(info[self.key])
        if episode_done and self._last_value is not None:
            self._buffer.append(self._last_value)
            if len(self._buffer) > self.n:
                self._buffer.pop(0)
            info[f"{self.prefix}/{self.key}"] = float(np.mean(self._buffer))
        return info


class AverageSteps:
    """
    Tracks a scalar metric and computes its mean over the current episode.
    Resets at the start of each episode.

    Usage:
        wrapper = AverageSteps("reward")
        info = wrapper.update(info, reward, episode_done)
    """

    def __init__(self, metric_key: str, output_key: Optional[str] = None):
        self.key = metric_key
        self.out_key = output_key or f"mean_step/{metric_key}"
        self._values: List[float] = []

    def reset(self) -> None:
        self._values.clear()

    def update(self, info: Dict[str, Any], episode_done: bool) -> Dict[str, Any]:
        if self.key in info:
            self._values.append(float(info[self.key]))
        if episode_done and self._values:
            info[self.out_key] = float(np.mean(self._values))
            self._values.clear()
        return info


class InfoCompose:
    """
    Chains multiple info wrappers — apply all in sequence.

    Usage:
        composer = InfoCompose([
            AverageEpisodesLastStep("target_distance_mm", 100),
            AverageEpisodesLastStep("episode_stats/success", 100),
            AverageSteps("reward"),
        ])
        info = composer.update(info, reward, done)
    """

    def __init__(self, wrappers: list):
        self._wrappers = wrappers

    def reset(self) -> None:
        for w in self._wrappers:
            if hasattr(w, "reset"):
                w.reset()

    def update(
        self, info: Dict[str, Any], episode_done: bool
    ) -> Dict[str, Any]:
        for w in self._wrappers:
            if isinstance(w, AverageEpisodesLastStep):
                info = w.update(info, episode_done)
            elif isinstance(w, AverageSteps):
                info = w.update(info, episode_done)
        return info
