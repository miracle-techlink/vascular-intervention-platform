"""
SynthEndoSim — Abstract Terminal condition.

Terminal = episode SUCCESS (target reached, procedure complete).
Distinct from Truncation (timeout, failure, error).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
from ..core.types import SimState


class Terminal(ABC):
    """Returns True when the episode has successfully completed."""

    @abstractmethod
    def __call__(self, state: SimState, prev_state: Optional["SimState"] = None) -> bool: ...

    def reset(self, anatomy: Any = None, episode: int = 0) -> None:
        """Called at episode start. Override if stateful."""

    def __or__(self, other: "Terminal") -> "AnyTerminal":
        return AnyTerminal([self, other])

    def __and__(self, other: "Terminal") -> "AllTerminal":
        return AllTerminal([self, other])


class AnyTerminal(Terminal):
    """Terminal if ANY condition fires."""
    def __init__(self, conditions):
        self._conditions = conditions

    def __call__(self, state: SimState, prev_state=None) -> bool:
        return any(c(state, prev_state) for c in self._conditions)

    def reset(self, anatomy=None, episode: int = 0):
        for c in self._conditions:
            c.reset(anatomy, episode)


class AllTerminal(Terminal):
    """Terminal only if ALL conditions fire."""
    def __init__(self, conditions):
        self._conditions = conditions

    def __call__(self, state: SimState, prev_state=None) -> bool:
        return all(c(state, prev_state) for c in self._conditions)

    def reset(self, anatomy=None, episode: int = 0):
        for c in self._conditions:
            c.reset(anatomy, episode)


class NeverTerminal(Terminal):
    """Never signals terminal (use only truncation for episode end)."""
    def __call__(self, state: SimState, prev_state=None) -> bool:
        return False
