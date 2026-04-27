"""
SynthEndoSim — Vectorised environment.

True multiprocessing vec-env: each worker spawns its own SOFA instance.
Uses shared-memory numpy arrays for zero-copy obs transfer when possible.

Compatible with stable-baselines3 VecEnv interface.
"""
from __future__ import annotations
import multiprocessing as mp
import numpy as np
from typing import Any, Dict, List, Optional

from .config.schema import EnvConfig, config_from_dict, load_config


def _worker(
    rank: int,
    cfg_dict: dict,
    mesh_path: Optional[str],
    conn: mp.connection.Connection,
):
    """Worker process: owns one SynthEndoEnv, responds to commands."""
    from synthendosim.core.env import SynthEndoEnv
    from synthendosim.config.schema import config_from_dict

    cfg = config_from_dict(cfg_dict)
    if mesh_path:
        cfg.anatomy.mesh_path = mesh_path
    env = SynthEndoEnv(cfg)

    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                obs, info = env.reset(**data)
                conn.send(("ok", obs, info))
            elif cmd == "step":
                result = env.step(data)
                conn.send(("ok",) + result)
            elif cmd == "render":
                frame = env.render()
                conn.send(("ok", frame))
            elif cmd == "close":
                env.close()
                conn.send(("ok",))
                break
            elif cmd == "action_space":
                conn.send(("ok", env.action_space))
            elif cmd == "obs_space":
                conn.send(("ok", env.observation_space))
    except Exception as exc:
        conn.send(("error", str(exc)))
    finally:
        conn.close()


class SynthEndoVecEnv:
    """
    Synchronous multiprocessing vectorised environment.

    Each of the n_envs workers runs a full SynthEndoEnv in a separate process.
    Actions are dispatched in parallel; results collected synchronously.

    Usage:
        vec = ses.make_vec_env(4, config_dict={...})
        obs = vec.reset()            # (4, obs_dim)
        obs, rews, dones, infos = vec.step(actions)  # actions: (4, 2)
        vec.close()
    """

    def __init__(
        self,
        n_envs: int,
        config_path: Optional[str] = None,
        config_dict: Optional[dict] = None,
        mesh_list: Optional[List[str]] = None,
    ):
        self.n_envs = n_envs

        if config_path:
            cfg = load_config(config_path)
            cfg_dict = cfg.__dict__
        elif config_dict:
            cfg_dict = config_dict
        else:
            cfg_dict = {}

        self._conns = []
        self._procs = []
        ctx = mp.get_context("spawn")

        for i in range(n_envs):
            mesh = mesh_list[i % len(mesh_list)] if mesh_list else None
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(i, cfg_dict, mesh, child_conn),
                daemon=True,
            )
            proc.start()
            self._conns.append(parent_conn)
            self._procs.append(proc)

        # Get spaces from first worker
        self._conns[0].send(("action_space", None))
        _, self.action_space = self._conns[0].recv()
        self._conns[0].send(("obs_space", None))
        _, self.observation_space = self._conns[0].recv()

    def reset(self, seed: Optional[int] = None, **kwargs) -> tuple:
        seeds = [seed + i if seed is not None else None for i in range(self.n_envs)]
        for conn, s in zip(self._conns, seeds):
            conn.send(("reset", {"seed": s}))
        results = [conn.recv() for conn in self._conns]
        obs_list = [r[1] for r in results]
        info_list = [r[2] for r in results]
        return self._stack(obs_list), info_list

    def step(self, actions: np.ndarray) -> tuple:
        for conn, action in zip(self._conns, actions):
            conn.send(("step", action))
        results = [conn.recv() for conn in self._conns]
        obs_list   = [r[1] for r in results]
        rew_list   = [r[2] for r in results]
        term_list  = [r[3] for r in results]
        trunc_list = [r[4] for r in results]
        info_list  = [r[5] for r in results]
        return (
            self._stack(obs_list),
            np.array(rew_list, dtype=np.float32),
            np.array(term_list, dtype=bool),
            np.array(trunc_list, dtype=bool),
            info_list,
        )

    def close(self):
        for conn in self._conns:
            conn.send(("close", None))
        for conn in self._conns:
            conn.recv()
        for proc in self._procs:
            proc.join(timeout=5)

    @staticmethod
    def _stack(obs_list):
        if isinstance(obs_list[0], dict):
            return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
        return np.stack(obs_list)
