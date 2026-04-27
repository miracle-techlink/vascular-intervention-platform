"""
SynthEndoSim — Info builders.

Collects per-step and per-episode metrics. The `info` dict returned
by env.step() / env.reset() is populated by InfoBuilder.

Design: InfoBuilder is a plugin. Custom metrics (e.g. fluoroscopy dose,
        contrast volume) can be added by subclassing or composing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from ..core.types import SimState, AnatomySpec


@dataclass
class StepInfo:
    """Flat dict-compatible structure for per-step info."""
    step: int = 0
    episode: int = 0
    tip_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_distance_mm: float = 0.0
    path_remaining_mm: float = 0.0
    target_reached: bool = False
    sim_error: bool = False
    interim_target_position: Optional[np.ndarray] = None
    interim_target_distance_mm: Optional[float] = None
    interim_advanced: bool = False
    waypoints_remaining: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "step": self.step,
            "episode": self.episode,
            "tip_position": self.tip_position.copy(),
            "target_distance_mm": self.target_distance_mm,
            "path_remaining_mm": self.path_remaining_mm,
            "target_reached": self.target_reached,
            "sim_error": self.sim_error,
            "waypoints_remaining": self.waypoints_remaining,
            "interim_advanced": self.interim_advanced,
        }
        if self.interim_target_position is not None:
            d["interim_target_position"] = self.interim_target_position.copy()
            d["interim_target_distance_mm"] = self.interim_target_distance_mm
        return d


@dataclass
class EpisodeInfo:
    """Accumulated per-episode statistics. Appended to info on episode end."""
    episode: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    success: bool = False
    final_distance_mm: float = float("inf")
    min_distance_mm: float = float("inf")
    waypoints_reached: int = 0
    total_waypoints: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_stats/episode": self.episode,
            "episode_stats/steps": self.total_steps,
            "episode_stats/total_reward": self.total_reward,
            "episode_stats/success": self.success,
            "episode_stats/final_distance_mm": self.final_distance_mm,
            "episode_stats/min_distance_mm": self.min_distance_mm,
            "episode_stats/waypoints_reached": self.waypoints_reached,
            "episode_stats/waypoints_total": self.total_waypoints,
            "episode_stats/success_rate_waypoints": (
                self.waypoints_reached / self.total_waypoints
                if self.total_waypoints > 0 else 0.0
            ),
        }


# Type alias for extra metric providers
MetricFn = Callable[[SimState, Optional[SimState]], Dict[str, Any]]


class InfoBuilder:
    """
    Builds the info dict for each step/reset.

    Usage:
        builder = InfoBuilder(target_threshold_mm=5.0)
        builder.register_metric("dose_mGy", my_dose_fn)

        info = builder.build_step(state, prev_state, terminated, truncated)
    """

    def __init__(
        self,
        target_threshold_mm: float = 5.0,
        extra_metrics: Optional[List[MetricFn]] = None,
    ):
        self.target_threshold_mm = target_threshold_mm
        self._extra: List[MetricFn] = list(extra_metrics or [])
        self._ep_info = EpisodeInfo()
        self._cum_reward: float = 0.0
        self._min_dist: float = float("inf")

    def register_metric(self, name: str, fn: Callable) -> None:
        """Add a custom metric function: fn(state, prev_state) -> dict."""
        self._extra.append(fn)

    def reset_episode(self, episode: int, anatomy: AnatomySpec) -> Dict[str, Any]:
        self._ep_info = EpisodeInfo(episode=episode)
        self._cum_reward = 0.0
        self._min_dist = float("inf")
        return {"episode": episode, "anatomy": anatomy.name}

    def build_step(
        self,
        state: SimState,
        prev_state: Optional[SimState],
        reward: float,
        terminated: bool,
        truncated: bool,
        interim_advanced: bool = False,
        interimtarget=None,
    ) -> Dict[str, Any]:
        tip = state.devices[0].tip_position if state.devices else np.zeros(3)
        dist = float(np.linalg.norm(tip - state.target_position))
        self._min_dist = min(self._min_dist, dist)
        self._cum_reward += reward
        self._ep_info.total_steps = state.step
        self._ep_info.total_reward = self._cum_reward

        step_info = StepInfo(
            step=state.step,
            episode=state.episode,
            tip_position=tip,
            target_distance_mm=dist,
            path_remaining_mm=state.path_remaining,
            target_reached=dist <= self.target_threshold_mm,
            sim_error=state.simulation_error,
            interim_advanced=interim_advanced,
        )

        if interimtarget is not None:
            it_pos = interimtarget.current(state)
            step_info.interim_target_position = it_pos
            step_info.waypoints_remaining = interimtarget.remaining_count
            if it_pos is not None:
                step_info.interim_target_distance_mm = float(
                    np.linalg.norm(tip - it_pos)
                )

        info = step_info.to_dict()

        # extra user metrics
        for fn in self._extra:
            try:
                info.update(fn(state, prev_state))
            except Exception:
                pass

        # flush episode stats on done
        if terminated or truncated:
            self._ep_info.success = terminated
            self._ep_info.final_distance_mm = dist
            self._ep_info.min_distance_mm = self._min_dist
            if interimtarget is not None:
                from ..interimtarget import NoInterimTarget
                if not isinstance(interimtarget, NoInterimTarget):
                    total = getattr(interimtarget, "_waypoints", None)
                    total = len(total) if total is not None else len(
                        getattr(interimtarget, "stages", [])
                    )
                    self._ep_info.total_waypoints = total
                    self._ep_info.waypoints_reached = (
                        total - interimtarget.remaining_count
                    )
            info.update(self._ep_info.to_dict())

        return info


def build_info(
    target_threshold_mm: float = 5.0,
    extra_metrics: Optional[List[MetricFn]] = None,
) -> InfoBuilder:
    return InfoBuilder(
        target_threshold_mm=target_threshold_mm,
        extra_metrics=extra_metrics,
    )
