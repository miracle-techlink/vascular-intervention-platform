"""
SynthEndoSim — Visualisation / Renderer module.

Decoupled from the environment: the renderer receives a SimState + optional
DSA image and produces a visual output. Useful for:
  - Training dashboards (OpenCV window)
  - Notebook inspection (Matplotlib)
  - Video recording (frame buffer)
  - Headless CI (NullRenderer)

Implementations:
  NullRenderer        — no-op (default, training mode)
  OpenCVRenderer      — live window via cv2
  MatplotlibRenderer  — figure-based (Jupyter/script)
  VideoRenderer       — writes MP4 via imageio
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from ..core.types import SimState


class Renderer(ABC):
    """Abstract renderer interface."""

    @abstractmethod
    def render(
        self,
        state: SimState,
        dsa_image: Optional[np.ndarray] = None,
        extra: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """
        Render current state.
        Returns (H, W, 3) uint8 RGB frame or None.
        """

    def close(self) -> None:
        """Release resources."""


class NullRenderer(Renderer):
    """No-op renderer for headless training."""

    def render(self, state, dsa_image=None, extra=None):
        return None


class MatplotlibRenderer(Renderer):
    """
    Matplotlib-based renderer. Produces a figure with:
      - Left: DSA image (if provided)
      - Right: 3D tip trajectory + target sphere
    Returns (H, W, 3) uint8 numpy array via fig.canvas.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 5), dpi: int = 80):
        self.figsize = figsize
        self.dpi = dpi
        self._trajectory: List[np.ndarray] = []
        self._fig = None
        self._ax_dsa = None
        self._ax_3d = None

    def _ensure_fig(self, has_dsa: bool):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if self._fig is None:
            n = 2 if has_dsa else 1
            self._fig, axes = plt.subplots(1, n, figsize=self.figsize, dpi=self.dpi)
            if n == 1:
                axes = [axes]
            self._ax_dsa = axes[0] if has_dsa else None
            self._ax_3d = axes[-1]
        return self._fig

    def render(
        self,
        state: SimState,
        dsa_image: Optional[np.ndarray] = None,
        extra: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        if state.devices:
            self._trajectory.append(state.devices[0].tip_position.copy())

        has_dsa = dsa_image is not None
        fig = self._ensure_fig(has_dsa)

        if has_dsa and self._ax_dsa is not None:
            self._ax_dsa.clear()
            self._ax_dsa.imshow(dsa_image, cmap="gray", vmin=0, vmax=1)
            self._ax_dsa.set_title("DSA")
            self._ax_dsa.axis("off")

        ax = self._ax_3d
        ax.clear()
        if self._trajectory:
            traj = np.stack(self._trajectory, axis=0)
            ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=1.5, label="tip")
            tip = traj[-1]
            ax.plot(*tip[:2], "bo", markersize=6)
        tgt = state.target_position
        ax.plot(*tgt[:2], "r*", markersize=12, label="target")
        ax.set_title(f"Step {state.step} | dist={state.path_remaining:.1f}mm")
        ax.legend(fontsize=7)
        ax.set_aspect("equal")

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        h, w = fig.canvas.get_width_height()[::-1]
        return buf.reshape(h, w, 4)[:, :, :3]

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
        self._trajectory.clear()


class OpenCVRenderer(Renderer):
    """
    OpenCV live window. Falls back to NullRenderer if cv2 is not installed.
    """

    def __init__(self, window_name: str = "SynthEndoSim", wait_ms: int = 1):
        self.window_name = window_name
        self.wait_ms = wait_ms
        self._mpl = MatplotlibRenderer()

    def render(
        self,
        state: SimState,
        dsa_image: Optional[np.ndarray] = None,
        extra: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        try:
            import cv2
        except ImportError:
            return self._mpl.render(state, dsa_image, extra)

        frame = self._mpl.render(state, dsa_image, extra)
        if frame is None:
            return None
        bgr = frame[:, :, ::-1]
        cv2.imshow(self.window_name, bgr)
        cv2.waitKey(self.wait_ms)
        return frame

    def close(self):
        self._mpl.close()
        try:
            import cv2
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


class VideoRenderer(Renderer):
    """
    Writes episode frames to an MP4 file via imageio.
    Creates a new file per episode if `per_episode=True`.
    """

    def __init__(
        self,
        path: str = "episode.mp4",
        fps: int = 10,
        per_episode: bool = True,
    ):
        self.base_path = path
        self.fps = fps
        self.per_episode = per_episode
        self._writer = None
        self._mpl = MatplotlibRenderer()
        self._ep = 0

    def _open(self):
        try:
            import imageio
        except ImportError:
            return
        p = self.base_path
        if self.per_episode:
            stem, ext = p.rsplit(".", 1) if "." in p else (p, "mp4")
            p = f"{stem}_ep{self._ep:04d}.{ext}"
        self._writer = imageio.get_writer(p, fps=self.fps)

    def render(
        self,
        state: SimState,
        dsa_image: Optional[np.ndarray] = None,
        extra: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        # new episode
        if state.step == 0:
            self.close()
            self._ep += 1
            self._mpl = MatplotlibRenderer()
            self._open()

        frame = self._mpl.render(state, dsa_image, extra)
        if frame is not None and self._writer is not None:
            self._writer.append_data(frame)
        return frame

    def close(self):
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        self._mpl.close()


_REGISTRY = {
    "null":        NullRenderer,
    "matplotlib":  MatplotlibRenderer,
    "opencv":      OpenCVRenderer,
    "video":       VideoRenderer,
}


def make_renderer(kind: str = "null", **kwargs) -> Renderer:
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown renderer '{kind}'. Choose from: {list(_REGISTRY)}")
    return _REGISTRY[kind](**kwargs)


def register(name: str, cls):
    _REGISTRY[name] = cls
