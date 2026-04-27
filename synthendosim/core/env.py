"""
SynthEndoSim — Main Gymnasium environment.

Clean, config-driven, single-entry-point design.
No class inheritance chains, no hidden globals.

Usage:
    import synthendosim as ses

    # From preset
    env = ses.make("SynthEndoSim-AorticArch-CAS-v1", anatomy_mesh="path/to/mesh.obj")

    # Fully custom
    env = ses.make_env(config_dict={...})

    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(action)
"""
from __future__ import annotations
import copy
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
import gymnasium as gym

from ..core.types import SimState, AnatomySpec, DeviceSpec
from ..config.schema import EnvConfig
from ..physics import make_backend, PhysicsBackend
from ..anatomy.loader import load_anatomy
from ..anatomy.centerline import ManifoldPathfinder, euclidean_path_remaining
from ..imaging import make_imager
from ..reward.components import build_reward, CompositeReward
from ..observation.builders import ObservationBuilder

logger = logging.getLogger(__name__)


class SynthEndoEnv(gym.Env):
    """
    SynthEndoSim Gymnasium environment.

    Action space : Box(-1, 1, (2,)) per device
                   [translation_scale, rotation_scale]
                   Actual velocity = action * [max_trans_mm_s, max_rot_rad_s]
    Obs space    : Dict or Box depending on ObsConfig.use_wire_mask
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    # Action scaling defaults (overridable via config extras)
    MAX_TRANSLATION = 10.0   # mm/s
    MAX_ROTATION    = np.pi  # rad/s

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg

        # Physics backend
        self._backend: PhysicsBackend = make_backend(
            cfg.physics.backend,
            dt_simulation=cfg.physics.dt_simulation,
            friction=cfg.physics.friction,
        )

        # Anatomy
        self._anatomy_spec: AnatomySpec = self._build_anatomy_spec(cfg)

        # Device specs
        self._device_specs = [self._build_device_spec(cfg.device)]

        # Pathfinder (curvature-weighted or euclidean)
        self._pathfinder: Optional[ManifoldPathfinder] = None

        # Imager
        self._imager = make_imager(
            cfg.imaging.kind,
            lao_deg=cfg.imaging.lao_deg,
            lat_deg=cfg.imaging.lat_deg,
            image_size=tuple(cfg.imaging.image_size),
            n0_photons=cfg.imaging.n0_photons,
            mu_blood=cfg.imaging.mu_blood,
            mu_tissue=cfg.imaging.mu_tissue,
        )

        # Reward
        self._reward_fn: CompositeReward = build_reward(cfg.reward)

        # State
        self._state: Optional[SimState] = None
        self._prev_state: Optional[SimState] = None
        self._step_count: int = 0
        self._episode: int = 0
        self._rng = np.random.default_rng()

        # Bounding box (set on first reset after anatomy is loaded)
        self._bbox_min = np.zeros(3, dtype=np.float32)
        self._bbox_max = np.ones(3, dtype=np.float32) * 200.0

        # Observation builder (placeholder; updated after first reset)
        self._obs_builder = ObservationBuilder(
            cfg.obs, self._bbox_min, self._bbox_max
        )

        # Gym spaces (updated after first reset when bbox is known)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 * len(self._device_specs),),
            dtype=np.float32,
        )
        self.observation_space = self._obs_builder.observation_space()

    # ------------------------------------------------------------------ #
    # Core Gym interface                                                   #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._episode += 1
        self._step_count = 0

        # Apply any per-reset anatomy overrides (e.g. multi-anatomy training)
        anatomy = self._resolve_anatomy(options)

        device_states = self._backend.reset(
            anatomy, self._device_specs, seed=seed
        )

        # Update bbox from anatomy vertices (needed for normalisation)
        self._update_bbox(anatomy)
        self._imager.set_anatomy_bounds(self._bbox_min, self._bbox_max)

        # Build pathfinder
        if anatomy.centerline is not None and self.cfg.reward.use_manifold:
            self._pathfinder = ManifoldPathfinder(anatomy.centerline)
        else:
            self._pathfinder = None

        target = np.array(anatomy.target_point, dtype=np.float32)
        path_rem = self._compute_path_remaining(
            device_states[0].tip_position if device_states else np.zeros(3),
            target,
        )

        self._state = SimState(
            devices=device_states,
            step=0,
            episode=self._episode,
            target_position=target,
            path_remaining=path_rem,
        )
        self._prev_state = None

        wire_img = self._render_image()
        obs_struct = self._obs_builder.build(self._state, wire_img)
        obs = self._obs_builder.filter(obs_struct)
        info = self._make_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Any, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32)
        n_dev = len(self._device_specs)
        actions_scaled = action.reshape(n_dev, 2).copy()
        actions_scaled[:, 0] *= self.MAX_TRANSLATION
        actions_scaled[:, 1] *= self.MAX_ROTATION

        device_states = self._backend.step(
            actions_scaled, dt=self.cfg.physics.dt_simulation * 10
        )
        self._step_count += 1

        target = self._state.target_position
        path_rem = self._compute_path_remaining(
            device_states[0].tip_position if device_states else np.zeros(3),
            target,
        )

        self._prev_state = self._state
        self._state = SimState(
            devices=device_states,
            step=self._step_count,
            episode=self._episode,
            target_position=target,
            path_remaining=path_rem,
            simulation_error=self._backend.simulation_error,
        )

        reward = float(self._reward_fn(self._state, self._prev_state))
        terminated = self._check_terminated()
        truncated = self._check_truncated()

        wire_img = self._render_image()
        obs_struct = self._obs_builder.build(self._state, wire_img)
        obs = self._obs_builder.filter(obs_struct)
        info = self._make_info()
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Return (H, W, 3) uint8 LAO DSA image for visualisation."""
        if self._state is None:
            return None
        wire_nodes = self._backend.get_dof_positions()
        if hasattr(self._imager, "render_lao"):
            lao = self._imager.render_lao(wire_nodes, rng=self._rng)
            img = (lao * 255).astype(np.uint8)
            return np.stack([img, img, img], axis=-1)
        return None

    def close(self):
        self._backend.close()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _resolve_anatomy(self, options: Optional[dict]) -> AnatomySpec:
        if options and "anatomy_mesh" in options:
            spec = copy.copy(self._anatomy_spec)
            spec.mesh_path = options["anatomy_mesh"]
            return spec
        return self._anatomy_spec

    def _update_bbox(self, anatomy: AnatomySpec):
        try:
            from ..anatomy.loader import _load_vertices
            verts = _load_vertices(anatomy.mesh_path)
            if len(verts) > 0:
                self._bbox_min = verts.min(axis=0).astype(np.float32)
                self._bbox_max = verts.max(axis=0).astype(np.float32)
                self._obs_builder = ObservationBuilder(
                    self.cfg.obs, self._bbox_min, self._bbox_max,
                    path_scale=float(np.linalg.norm(
                        self._bbox_max - self._bbox_min
                    )),
                )
                self.observation_space = self._obs_builder.observation_space()
        except Exception as exc:
            logger.debug("Bbox update failed (using defaults): %s", exc)

    def _compute_path_remaining(
        self, tip: np.ndarray, target: np.ndarray
    ) -> float:
        if self._pathfinder is not None:
            return self._pathfinder.path_remaining(tip, target)
        return euclidean_path_remaining(tip, target)

    def _render_image(self) -> Optional[np.ndarray]:
        if not self.cfg.obs.use_wire_mask:
            return None
        wire_nodes = self._backend.get_dof_positions()
        return self._imager.render(wire_nodes, rng=self._rng)

    def _check_terminated(self) -> bool:
        if self._state is None or not self._state.devices:
            return False
        tip = self._state.devices[0].tip_position
        dist = float(np.linalg.norm(tip - self._state.target_position))
        return dist <= self.cfg.reward.target_threshold_mm

    def _check_truncated(self) -> bool:
        if self._state and self._state.simulation_error:
            return True
        return self._step_count >= self.cfg.episode.max_steps

    def _make_info(self) -> Dict:
        if self._state is None:
            return {}
        tip = self._state.devices[0].tip_position if self._state.devices else np.zeros(3)
        dist = float(np.linalg.norm(tip - self._state.target_position))
        return {
            "step": self._step_count,
            "episode": self._episode,
            "tip_position": tip.copy(),
            "target_distance_mm": dist,
            "path_remaining_mm": self._state.path_remaining,
            "target_reached": dist <= self.cfg.reward.target_threshold_mm,
            "sim_error": self._state.simulation_error,
        }

    @staticmethod
    def _build_anatomy_spec(cfg: EnvConfig) -> AnatomySpec:
        from ..anatomy.loader import load_anatomy as _load
        ac = cfg.anatomy
        return _load(
            mesh_path=ac.mesh_path if ac.mesh_path else "/dev/null",
            insertion_point=tuple(ac.insertion_point),
            insertion_direction=tuple(ac.insertion_direction),
            target_point=tuple(ac.target_point),
            centerline_path=ac.centerline_path,
            scale=tuple(ac.scale),
            rotation_yzx_deg=tuple(ac.rotation_yzx_deg),
            visu_mesh_path=ac.visu_mesh_path,
            name=ac.name,
        )

    @staticmethod
    def _build_device_spec(dc) -> DeviceSpec:
        from ..core.types import DeviceSpec as DS
        return DS(
            kind=dc.kind,
            total_length=dc.total_length,
            tip_length=dc.tip_length,
            tip_angle=dc.tip_angle,
            diameter=dc.diameter,
            stiffness=dc.stiffness,
            young_modulus=dc.young_modulus,
            poisson_ratio=dc.poisson_ratio,
        )
