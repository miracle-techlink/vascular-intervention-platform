"""
SynthEndoSim — Episode State Recorder & Replay.

InterventionStateRecorder records full episode trajectories to disk.
replay_episode() replays a saved trajectory and returns SAR tuples.

Useful for:
  - Offline analysis of failure cases
  - Behaviour cloning dataset generation
  - Video rendering from saved episodes

Usage:
    recorder = InterventionStateRecorder()

    obs, info = env.reset()
    recorder.on_reset(obs, info)
    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, term, trunc, info = env.step(action)
        recorder.on_step(action, obs, reward, term, trunc, info)
        if term or trunc:
            break
    recorder.save("episode_0042.pkl")

    # Later: replay
    data = InterventionStateRecorder.load("episode_0042.pkl")
    for step in data.steps:
        print(step.action, step.reward)
"""
from __future__ import annotations
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class StepRecord:
    action: np.ndarray
    obs: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


@dataclass
class EpisodeRecord:
    episode: int
    seed: Optional[int]
    initial_obs: Any
    initial_info: Dict[str, Any]
    steps: List[StepRecord] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def success(self) -> bool:
        return bool(self.steps and self.steps[-1].terminated)

    def to_sar(self):
        """Return (observations, actions, rewards) arrays for BC training."""
        obs_list = [self.initial_obs] + [s.obs for s in self.steps[:-1]]
        actions  = np.stack([s.action  for s in self.steps])
        rewards  = np.array([s.reward  for s in self.steps])
        return obs_list, actions, rewards


class InterventionStateRecorder:
    """Records a single episode trajectory."""

    def __init__(self):
        self._record: Optional[EpisodeRecord] = None

    def on_reset(
        self,
        obs: Any,
        info: Dict,
        episode: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        self._record = EpisodeRecord(
            episode=episode,
            seed=seed,
            initial_obs=obs,
            initial_info=dict(info),
        )

    def on_step(
        self,
        action: np.ndarray,
        obs: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict,
    ) -> None:
        if self._record is None:
            raise RuntimeError("Call on_reset() before on_step()")
        self._record.steps.append(StepRecord(
            action=np.asarray(action, dtype=np.float32),
            obs=obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        ))

    @property
    def current(self) -> Optional[EpisodeRecord]:
        return self._record

    def save(self, path: str) -> None:
        """Pickle the current episode record to disk."""
        if self._record is None:
            raise RuntimeError("No episode recorded yet.")
        with open(path, "wb") as f:
            pickle.dump(self._record, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> EpisodeRecord:
        """Load a previously saved EpisodeRecord."""
        with open(path, "rb") as f:
            return pickle.load(f)


class MultiEpisodeRecorder:
    """
    Records multiple episodes and saves them as a dataset.
    Useful for collecting demonstration data for imitation learning.
    """

    def __init__(self, save_dir: str, prefix: str = "episode"):
        self._dir = Path(save_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._prefix = prefix
        self._ep_recorder = InterventionStateRecorder()
        self._episode_count = 0

    def on_reset(self, obs, info, seed=None):
        self._episode_count += 1
        self._ep_recorder.on_reset(obs, info, self._episode_count, seed)

    def on_step(self, action, obs, reward, terminated, truncated, info):
        self._ep_recorder.on_step(action, obs, reward, terminated, truncated, info)
        if terminated or truncated:
            path = self._dir / f"{self._prefix}_{self._episode_count:05d}.pkl"
            self._ep_recorder.save(str(path))

    def load_dataset(self) -> List[EpisodeRecord]:
        """Load all saved episodes from the directory."""
        records = []
        for p in sorted(self._dir.glob(f"{self._prefix}_*.pkl")):
            records.append(InterventionStateRecorder.load(str(p)))
        return records

    def to_bc_dataset(self):
        """Return all episodes as stacked arrays for behaviour cloning."""
        records = self.load_dataset()
        all_obs, all_actions, all_rewards = [], [], []
        for rec in records:
            obs_list, actions, rewards = rec.to_sar()
            all_obs.extend(obs_list)
            all_actions.append(actions)
            all_rewards.append(rewards)
        return (
            all_obs,
            np.concatenate(all_actions),
            np.concatenate(all_rewards),
        )
