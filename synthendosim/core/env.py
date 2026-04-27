"""
SynthEndoSim — Main Gymnasium environment.

Plugin-driven architecture: every subsystem is an injectable object.
Plugins can be passed directly (research mode) or loaded from config (production mode).

Subsystems (all optional with config-driven defaults):
  device        — interventional device(s) [Device]
  terminal      — success condition [Terminal]
  truncation    — failure/timeout condition [Truncation]
  start         — episode initialisation strategy [Start]
  interimtarget — intermediate waypoint manager [InterimTarget]
  pathfinder    — distance-to-goal computation [Pathfinder]
  reward        — reward function [CompositeReward]
  info          — info dict builder [InfoBuilder]
  renderer      — visualisation [Renderer]

Usage:
    import synthendosim as ses

    # Config-driven (preset)
    env = ses.make("SynthEndoSim-AorticArch-CAS-v1", anatomy_mesh="aorta.obj")

    # Direct plugin injection (research/custom)
    env = ses.SynthEndoEnv(
        cfg=cfg,
        terminal=TargetReachedTerminal(threshold_mm=5.0),
        interimtarget=CenterlineWaypointTarget(n_waypoints=6),
        renderer=VideoRenderer("run.mp4"),
    )
"""
from __future__ import annotations
import copy
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym

from .types import SimState, AnatomySpec, DeviceSpec
from ..config.schema import EnvConfig
from ..physics import make_backend, PhysicsBackend
from ..anatomy.loader import load_anatomy
from ..imaging import make_imager
from ..reward.components import build_reward, CompositeReward
from ..observation.builders import ObservationBuilder
from ..device.base import Device
from ..terminal.base import Terminal
from ..terminal.conditions import TargetReachedTerminal
from ..truncation.base import Truncation
from ..truncation.base import AnyTruncation
from ..truncation.conditions import MaxStepsTruncation, SimErrorTruncation
from ..start.base import Start
from ..start.strategies import InsertionPointStart
from ..interimtarget.base import InterimTarget, NoInterimTarget
from ..pathfinder.graph import Pathfinder, ManifoldPathfinder, EuclideanPathfinder
from ..info.builders import InfoBuilder
from ..visualisation.renderer import Renderer, NullRenderer

logger = logging.getLogger(__name__)


class SynthEndoEnv(gym.Env):
    """
    SynthEndoSim Gymnasium environment.

    Action space : Box(-1, 1, (2 * n_devices,))
                   [translation_scale, rotation_scale] per device
                   Actual velocity = action * [max_trans_mm_s, max_rot_rad_s]
    Obs space    : Dict (or Box if ObsConfig.flatten=True)

    All subsystems are injectable plugins — swap any component without
    touching the env code.
    """

    metadata = {"render_modes": ["rgb_array", "human", "none"]}

    MAX_TRANSLATION = 10.0   # mm/s — overridable via cfg.extras
    MAX_ROTATION    = np.pi  # rad/s

    def __init__(
        self,
        cfg: EnvConfig,
        *,
        # --- injectable plugins (all optional; fall back to config defaults) ---
        devices: Optional[List[Device]] = None,
        terminal: Optional[Terminal] = None,
        truncation: Optional[Truncation] = None,
        start: Optional[Start] = None,
        interimtarget: Optional[InterimTarget] = None,
        pathfinder: Optional[Pathfinder] = None,
        reward: Optional[CompositeReward] = None,
        info_builder: Optional[InfoBuilder] = None,
        renderer: Optional[Renderer] = None,
    ):
        super().__init__()
        self.cfg = cfg

        # --- Physics backend ---
        self._backend: PhysicsBackend = make_backend(
            cfg.physics.backend,
            dt_simulation=cfg.physics.dt_simulation,
            friction=cfg.physics.friction,
        )

        # --- Anatomy ---
        self._anatomy_spec: AnatomySpec = self._build_anatomy_spec(cfg)

        # --- Device specs (legacy typed DeviceSpec for SOFA backend) ---
        self._device_specs = [self._build_device_spec(cfg.device)]

        # --- Plugin: Device objects (optional higher-level wrappers) ---
        self._devices: Optional[List[Device]] = devices

        # --- Plugin: Pathfinder ---
        if pathfinder is not None:
            self._pathfinder: Pathfinder = pathfinder
        elif cfg.reward.use_manifold and self._anatomy_spec.centerline is not None:
            self._pathfinder = ManifoldPathfinder(
                self._anatomy_spec.centerline,
                curvature_weight=getattr(cfg.reward, "curvature_weight", 2.0),
            )
        else:
            self._pathfinder = EuclideanPathfinder()

        # --- Plugin: Terminal condition ---
        if terminal is not None:
            self._terminal: Terminal = terminal
        else:
            self._terminal = TargetReachedTerminal(
                threshold_mm=cfg.reward.target_threshold_mm
            )

        # --- Plugin: Truncation condition ---
        if truncation is not None:
            self._truncation: Truncation = truncation
        else:
            self._truncation = AnyTruncation([
                MaxStepsTruncation(max_steps=cfg.episode.max_steps),
                SimErrorTruncation(),
            ])

        # --- Plugin: Start strategy ---
        self._start: Start = start or InsertionPointStart()

        # --- Plugin: InterimTarget ---
        self._interimtarget: InterimTarget = interimtarget or NoInterimTarget()

        # --- Imager ---
        self._imager = make_imager(
            cfg.imaging.kind,
            lao_deg=cfg.imaging.lao_deg,
            lat_deg=cfg.imaging.lat_deg,
            image_size=tuple(cfg.imaging.image_size),
            n0_photons=cfg.imaging.n0_photons,
            mu_blood=cfg.imaging.mu_blood,
            mu_tissue=cfg.imaging.mu_tissue,
        )

        # --- Plugin: Reward ---
        self._reward_fn: CompositeReward = reward or build_reward(cfg.reward)

        # --- Plugin: Info builder ---
        self._info_builder: InfoBuilder = info_builder or InfoBuilder(
            target_threshold_mm=cfg.reward.target_threshold_mm
        )

        # --- Plugin: Renderer ---
        self._renderer: Renderer = renderer or NullRenderer()

        # --- State ---
        self._state: Optional[SimState] = None
        self._prev_state: Optional[SimState] = None
        self._step_count: int = 0
        self._episode: int = 0
        self._rng = np.random.default_rng()

        # --- Bounding box (updated on first reset) ---
        self._bbox_min = np.zeros(3, dtype=np.float32)
        self._bbox_max = np.ones(3, dtype=np.float32) * 200.0

        # --- Observation builder ---
        self._obs_builder = ObservationBuilder(
            cfg.obs, self._bbox_min, self._bbox_max
        )

        # --- Gym spaces ---
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 * len(self._device_specs),),
            dtype=np.float32,
        )
        self.observation_space = self._obs_builder.observation_space()

    # ------------------------------------------------------------------ #
    # Plugin hot-swap API                                                  #
    # ------------------------------------------------------------------ #

    def set_plugin(self, **kwargs) -> "SynthEndoEnv":
        """
        Hot-swap plugins at runtime (before or between episodes).

        env.set_plugin(
            interimtarget=CenterlineWaypointTarget(n_waypoints=6),
            renderer=VideoRenderer("out.mp4"),
        )
        """
        _map = {
            "terminal":      "_terminal",
            "truncation":    "_truncation",
            "start":         "_start",
            "interimtarget": "_interimtarget",
            "pathfinder":    "_pathfinder",
            "reward":        "_reward_fn",
            "info_builder":  "_info_builder",
            "renderer":      "_renderer",
        }
        for k, v in kwargs.items():
            if k not in _map:
                raise ValueError(f"Unknown plugin '{k}'. Valid: {list(_map)}")
            setattr(self, _map[k], v)
        return self

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

        # Resolve anatomy (per-reset overrides supported)
        anatomy = self._resolve_anatomy(options)

        # Start strategy: determine initial device state
        init_state = self._start.sample(anatomy, self._rng, episode=self._episode)

        # Reset backend with anatomy + initial state
        device_states = self._backend.reset(
            anatomy, self._device_specs, seed=seed,
        )

        # Update bounding box + obs builder
        self._update_bbox(anatomy)
        self._imager.set_anatomy_bounds(self._bbox_min, self._bbox_max)

        # Reset pathfinder with new centerline
        self._pathfinder.reset(anatomy.centerline)

        # Reset plugins
        self._interimtarget.reset(anatomy, self._episode)
        self._terminal.reset(anatomy, self._episode)
        self._truncation.reset(anatomy, self._episode)

        target = np.array(anatomy.target_point, dtype=np.float32)
        tip = device_states[0].tip_position if device_states else np.zeros(3)
        path_rem = self._pathfinder.path_remaining(tip, target)

        self._state = SimState(
            devices=device_states,
            step=0,
            episode=self._episode,
            target_position=target,
            interim_target=self._interimtarget.current(
                SimState(devices=device_states, step=0, episode=self._episode,
                         target_position=target)
            ),
            path_remaining=path_rem,
        )
        self._prev_state = None

        wire_img = self._render_dsa()
        obs_struct = self._obs_builder.build(self._state, wire_img)
        obs = self._obs_builder.filter(obs_struct)
        info = self._info_builder.reset_episode(self._episode, anatomy)
        self._renderer.render(self._state, wire_img)
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
        tip = device_states[0].tip_position if device_states else np.zeros(3)
        path_rem = self._pathfinder.path_remaining(tip, target)

        self._prev_state = self._state

        # Tentative next state (needed for interimtarget.advance)
        next_state = SimState(
            devices=device_states,
            step=self._step_count,
            episode=self._episode,
            target_position=target,
            path_remaining=path_rem,
            simulation_error=self._backend.simulation_error,
        )

        # Advance interim target and collect bonus
        interim_advanced = self._interimtarget.advance(next_state)
        next_state.interim_target = self._interimtarget.current(next_state)

        self._state = next_state

        # Compute reward (interim bonus injected via state.interim_target advancing)
        reward = float(self._reward_fn(self._state, self._prev_state))
        if interim_advanced:
            reward += getattr(self.cfg.reward, "waypoint_bonus", 1.0)

        terminated = self._terminal(self._state, self._prev_state)
        truncated  = self._truncation(self._state, self._prev_state)

        wire_img = self._render_dsa()
        obs_struct = self._obs_builder.build(self._state, wire_img)
        obs = self._obs_builder.filter(obs_struct)
        info = self._info_builder.build_step(
            self._state, self._prev_state,
            reward, terminated, truncated,
            interim_advanced=interim_advanced,
            interimtarget=self._interimtarget,
        )
        self._renderer.render(self._state, wire_img)
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Return (H, W, 3) uint8 RGB frame. Delegates to renderer plugin."""
        if self._state is None:
            return None
        wire_img = self._render_dsa()
        return self._renderer.render(self._state, wire_img)

    def close(self):
        self._backend.close()
        self._renderer.close()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _resolve_anatomy(self, options: Optional[dict]) -> AnatomySpec:
        """Support per-reset anatomy override via options dict."""
        if options and "anatomy_mesh" in options:
            spec = copy.copy(self._anatomy_spec)
            spec.mesh_path = options["anatomy_mesh"]
            return load_anatomy(
                mesh_path=spec.mesh_path,
                insertion_point=spec.insertion_point,
                insertion_direction=spec.insertion_direction,
                target_point=spec.target_point,
                name=options.get("anatomy_name", spec.name),
            )
        return self._anatomy_spec

    def _update_bbox(self, anatomy: AnatomySpec) -> None:
        try:
            from ..anatomy.loader import _load_vertices
            verts = _load_vertices(anatomy.mesh_path)
            if len(verts) > 0:
                self._bbox_min = verts.min(axis=0).astype(np.float32)
                self._bbox_max = verts.max(axis=0).astype(np.float32)
                self._obs_builder = ObservationBuilder(
                    self.cfg.obs, self._bbox_min, self._bbox_max,
                    path_scale=float(np.linalg.norm(self._bbox_max - self._bbox_min)),
                )
                self.observation_space = self._obs_builder.observation_space()
        except Exception as exc:
            logger.debug("Bbox update skipped: %s", exc)

    def _render_dsa(self) -> Optional[np.ndarray]:
        if not self.cfg.obs.use_wire_mask:
            return None
        wire_nodes = self._backend.get_dof_positions()
        return self._imager.render(wire_nodes, rng=self._rng)

    @staticmethod
    def _build_anatomy_spec(cfg: EnvConfig) -> AnatomySpec:
        ac = cfg.anatomy
        return load_anatomy(
            mesh_path=ac.mesh_path if ac.mesh_path else "",
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
        return DeviceSpec(
            kind=dc.kind,
            total_length=dc.total_length,
            tip_length=dc.tip_length,
            tip_angle=dc.tip_angle,
            diameter=dc.diameter,
            stiffness=dc.stiffness,
            young_modulus=dc.young_modulus,
            poisson_ratio=dc.poisson_ratio,
        )
