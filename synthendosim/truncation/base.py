"""
SynthEndoSim — Abstract Truncation condition.

Truncation = episode FAILURE or TIMEOUT (not success).
Triggers done=True with truncated=True in Gymnasium interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
from ..core.types import SimState


class Truncation(ABC):
    @abstractmethod
    def __call__(self, state: SimState, prev_state: Optional[SimState] = None) -> bool: ...

    def reset(self, anatomy: Any = None, episode: int = 0) -> None: ...

    def __or__(self, other: "Truncation") -> "AnyTruncation":
        return AnyTruncation([self, other])


class AnyTruncation(Truncation):
    def __init__(self, conditions):
        self._conditions = conditions

    def __call__(self, state: SimState, prev_state=None) -> bool:
        return any(c(state, prev_state) for c in self._conditions)

    def reset(self, anatomy=None, episode: int = 0):
        for c in self._conditions:
            c.reset(anatomy, episode)


class NeverTruncate(Truncation):
    def __call__(self, state: SimState, prev_state=None) -> bool:
        return False
