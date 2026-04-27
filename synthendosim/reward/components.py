"""
SynthEndoSim — Reward components.

Each component is a pure function: (state, prev_state, cfg) → float.
Compose with CompositeReward or the + operator.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import numpy as np

from ..core.types import SimState
from ..config.schema import RewardConfig


# ── base class ───────────────────────────────────────────────────────────────

@dataclass
class RewardComponent:
    """A single reward term. Subclass or use function_reward() to create."""
    weight: float = 1.0

    def __call__(self, state: SimState, prev_state: Optional[SimState]) -> float:
        raise NotImplementedError

    def __add__(self, other: "RewardComponent") -> "CompositeReward":
        return CompositeReward(components=[self, other])

    def __mul__(self, scalar: float) -> "ScaledReward":
        return ScaledReward(self, scalar)

    def __rmul__(self, scalar: float) -> "ScaledReward":
        return ScaledReward(self, scalar)


@dataclass
class ScaledReward(RewardComponent):
    component: RewardComponent = field(default_factory=RewardComponent)
    scale: float = 1.0

    def __call__(self, state: SimState, prev_state: Optional[SimState]) -> float:
        return self.scale * self.component(state, prev_state)


@dataclass
class CompositeReward(RewardComponent):
    components: List[RewardComponent] = field(default_factory=list)

    def __call__(self, state: SimState, prev_state: Optional[SimState]) -> float:
        return sum(c(state, prev_state) for c in self.components)

    def __add__(self, other: RewardComponent) -> "CompositeReward":
        return CompositeReward(components=self.components + [other])


# ── concrete reward terms ─────────────────────────────────────────────────────

@dataclass
class ManifoldDistanceDelta(RewardComponent):
    """
    Potential-based reward: r = (D_{t-1} - D_t) * weight
    D_t = curvature-weighted geodesic distance to target.
    Falls back to euclidean if no centerline available.
    """
    weight: float = 1.0

    def __call__(self, state: SimState, prev_state: Optional[SimState]) -> float:
        if prev_state is None:
            return 0.0
        delta = prev_state.path_remaining - state.path_remaining
        return delta * self.weight


@dataclass
class TargetReached(RewardComponent):
    """Sparse terminal bonus when tip is within threshold of target."""
    bonus: float = 10.0
    threshold_mm: float = 10.0

    def __call__(self, state: SimState, prev_state: Optional[SimState]) -> float:
        if not state.devices:
            return 0.0
        tip = state.devices[0].tip_position
        dist = float(np.linalg.norm(tip - state.target_position))
        return self.bonus if dist <= self.threshold_mm else 0.0


@dataclass
class StepPenalty(RewardComponent):
    """Constant per-step penalty to encourage efficiency."""
    penalty: float = -0.01

    def __call__(self, state: SimState, prev_state: Optional[SimState]) -> float:
        return self.penalty


@dataclass
class WallCollisionPenalty(RewardComponent):
    """Penalty triggered on SOFA simulation error (wall penetration)."""
    penalty: float = -1.0

    def __call__(self, state: SimState, prev_state: Optional[SimState]) -> float:
        return self.penalty if state.simulation_error else 0.0


@dataclass
class EuclideanDistanceDelta(RewardComponent):
    """Simple euclidean shaping — fallback when no centerline."""
    weight: float = 1.0

    def __call__(self, state: SimState, prev_state: Optional[SimState]) -> float:
        if prev_state is None or not state.devices or not prev_state.devices:
            return 0.0
        tip_now = state.devices[0].tip_position
        tip_prev = prev_state.devices[0].tip_position
        d_now = float(np.linalg.norm(tip_now - state.target_position))
        d_prev = float(np.linalg.norm(tip_prev - prev_state.target_position))
        return (d_prev - d_now) * self.weight


# ── factory ──────────────────────────────────────────────────────────────────

def build_reward(cfg: RewardConfig) -> CompositeReward:
    """Build composite reward from config."""
    components: List[RewardComponent] = []

    if cfg.use_manifold:
        components.append(
            ManifoldDistanceDelta(weight=cfg.manifold_distance_weight)
        )
    else:
        components.append(
            EuclideanDistanceDelta(weight=cfg.manifold_distance_weight)
        )

    components.append(TargetReached(
        bonus=cfg.target_reached_bonus,
        threshold_mm=cfg.target_threshold_mm,
    ))
    components.append(StepPenalty(penalty=cfg.step_penalty))
    components.append(WallCollisionPenalty(penalty=cfg.wall_collision_penalty))

    return CompositeReward(components=components)
