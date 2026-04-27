"""
DRR (Digitally Reconstructed Radiograph) Fluoroscopy.

Physically-based synthetic X-ray using Beer-Lambert projection.
Replaces Pillow (2D line drawing) with proper DSA simulation.

Physics:
    I = I0 * exp(-∫ μ(x,y,z) dl)   [Beer-Lambert]
    DSA = -log(I_contrast / I_mask)  [Digital Subtraction Angiography]

Three attenuation layers:
    - Soft tissue background: μ ≈ 0.02  (always present)
    - Iodine contrast in vessel lumen: μ ≈ 0.8  (injected, vessel interior)
    - Metal guidewire: μ ≈ 8.0  (DOF positions from SOFA)

Usage (drop-in replacement for TrackingOnly or Pillow):
    fluoroscopy = DRRFluoroscopy(
        image_size=(256, 256),
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=7.5,
    )
"""

from typing import List, Optional, Tuple
import numpy as np
import gymnasium as gym
from scipy.ndimage import gaussian_filter

from .fluoroscopy import SimulatedFluoroscopy
from ..simulation import Simulation
from ..vesseltree import VesselTree
from ..vesseltree.util.voxelcube import VoxelCube, create_voxel_cube_from_mesh
from ...util.coordtransform import vessel_cs_to_tracking3d, tracking3d_to_2d


class DRRFluoroscopy(SimulatedFluoroscopy):
    """
    Physically-based DRR fluoroscopy for synthetic DSA generation.

    At reset():  build vessel attenuation volume + render mask image
    At step():   voxelize guidewire, project combined volume, subtract mask → DSA image

    The output image approximates what a DSA C-arm would produce:
    bright background, dark vessel lumen (contrast), bright-white guidewire.
    """

    def __init__(
        self,
        simulation: Simulation,
        vessel_tree: VesselTree,
        image_size: Tuple[int, int] = (256, 256),
        image_frequency: float = 7.5,
        image_rot_zx: Tuple[float, float] = (0.0, 0.0),
        image_center: Optional[Tuple[float, float, float]] = None,
        field_of_view: Optional[Tuple[float, float]] = None,
        projection_axis: int = 1,        # 0=X, 1=Y (AP view), 2=Z
        voxel_spacing: float = 1.0,      # mm per voxel (smaller = sharper, slower)
        mu_contrast: float = 0.8,        # iodine contrast attenuation coefficient
        mu_wire: float = 8.0,            # metal guidewire attenuation coefficient
        wire_radius: float = 0.4,        # guidewire radius in mm (0.014" ≈ 0.36mm clinical)
        noise_photons: float = 20000.0,  # simulated photon count (higher = less noise)
        blur_sigma: float = 1.2,         # detector PSF Gaussian sigma (pixels)
        dsa_gain: float = 6.0,           # contrast gain for display
        vignette_strength: float = 0.12, # image intensifier edge darkening (0=off)
        clinical_dsa_mask: bool = False, # True: zero pre-contrast mask → vessels visible in DSA
    ) -> None:
        self.simulation = simulation
        self.image_frequency = image_frequency
        self.image_rot_zx = image_rot_zx
        self.image_center = image_center or [0, 0, 0]
        self.field_of_view = field_of_view
        self.vessel_tree = vessel_tree
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.projection_axis = projection_axis
        self.voxel_spacing = voxel_spacing
        self.mu_contrast = mu_contrast
        self.mu_wire = mu_wire
        self.wire_radius = wire_radius
        self.noise_photons = noise_photons
        self.blur_sigma = blur_sigma
        self.dsa_gain = dsa_gain
        self.vignette_strength = vignette_strength
        self.clinical_dsa_mask = clinical_dsa_mask

        # Populated at reset()
        self._vessel_volume: Optional[np.ndarray] = None   # (X, Y, Z) float32
        self._voxel_origin: Optional[np.ndarray] = None    # world coords of [0,0,0]
        self._mask_proj: Optional[np.ndarray] = None       # pre-rendered mask projection
        self._bg_img: Optional[np.ndarray] = None          # anatomical background [0,1]
        self._image: Optional[np.ndarray] = None
        self._image_xray: Optional[np.ndarray] = None      # full X-ray (vessel+wire)

    # ------------------------------------------------------------------
    # Fluoroscopy interface
    # ------------------------------------------------------------------

    @property
    def image_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(0, 255, self.image_size, dtype=np.uint8)

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def tracking3d_space(self) -> gym.spaces.Box:
        low = vessel_cs_to_tracking3d(
            self.vessel_tree.coordinate_space.low,
            self.image_rot_zx, self.image_center, self.field_of_view)
        high = vessel_cs_to_tracking3d(
            self.vessel_tree.coordinate_space.high,
            self.image_rot_zx, self.image_center, self.field_of_view)
        return gym.spaces.Box(low=low, high=high)

    @property
    def tracking3d_space_episode(self) -> gym.spaces.Box:
        coords = self.vessel_tree.centerline_coordinates
        coords = vessel_cs_to_tracking3d(
            coords, self.image_rot_zx, self.image_center, self.field_of_view)
        low = np.min(coords, axis=0) * 1.1
        high = np.max(coords, axis=0) * 1.1
        return gym.spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32))

    @property
    def tracking2d_space(self) -> gym.spaces.Box:
        s = self.tracking3d_space
        return gym.spaces.Box(low=tracking3d_to_2d(s.low), high=tracking3d_to_2d(s.high))

    @property
    def tracking2d_space_episode(self) -> gym.spaces.Box:
        s = self.tracking3d_space_episode
        return gym.spaces.Box(low=tracking3d_to_2d(s.low), high=tracking3d_to_2d(s.high))

    @property
    def tracking3d(self) -> np.ndarray:
        return vessel_cs_to_tracking3d(
            self.simulation.dof_positions,
            self.image_rot_zx, self.image_center, self.field_of_view)

    @property
    def tracking2d(self) -> np.ndarray:
        return tracking3d_to_2d(self.tracking3d)

    @property
    def device_trackings3d(self) -> List[np.ndarray]:
        pos = self.tracking3d
        pos_flip = pos[::-1]
        lengths = self.simulation.inserted_lengths
        cum = np.cumsum(np.linalg.norm(np.diff(pos_flip, axis=0), axis=1))
        result = []
        for L in lengths:
            idx = np.argmin(np.abs(cum - L)) if len(cum) > 0 else 0
            result.append(pos_flip[:idx + 1][::-1])
        return result

    @property
    def device_trackings2d(self) -> List[np.ndarray]:
        return [tracking3d_to_2d(t) for t in self.device_trackings3d]

    # ------------------------------------------------------------------
    # Reset: build vessel volume + pre-render mask
    # ------------------------------------------------------------------

    def reset(self, episode_nr: int = 0) -> None:
        mesh_path = self.vessel_tree.mesh_path
        import pyvista as pv
        mesh = pv.read(mesh_path)

        sp = self.voxel_spacing
        vc = create_voxel_cube_from_mesh(mesh, spacing=(sp, sp, sp))

        # Vessel lumen attenuation volume (contrast agent)
        self._vessel_volume = vc.value_array * self.mu_contrast  # (X, Y, Z)
        self._voxel_origin = vc.world_offset.copy()
        self._voxel_spacing = np.array([sp, sp, sp], dtype=np.float32)

        # Pre-render mask
        # clinical_dsa_mask=True: zero mask (pre-contrast baseline) → both vessels
        #   and wire appear in DSA, matches real angiography appearance
        # clinical_dsa_mask=False: vessel+contrast mask → wire-only DSA (cleaner for RL)
        mask_proj = self._project(self._vessel_volume)
        self._mask_proj = np.zeros_like(mask_proj) if self.clinical_dsa_mask else mask_proj
        self._bg_img = self._build_bg_image()
        self._image = np.zeros(self.image_size, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Step: add wire, project, DSA subtract
    # ------------------------------------------------------------------

    def step(self) -> None:
        if self._vessel_volume is None:
            return

        # 1. Build guidewire attenuation volume (sparse update)
        wire_vol = np.zeros_like(self._vessel_volume)
        dof = self.simulation.dof_positions   # (N, 3) in vessel coords
        self._voxelize_wire(wire_vol, dof)

        # 2. Combined projection
        combined = self._vessel_volume + wire_vol
        proj = self._project(combined)

        # 3. Full X-ray image (vessel + wire, before DSA subtraction)
        #    Physics: Beer-Lambert, I = I0 * exp(-∫μdl)
        I_full = np.exp(-proj)  # [0,1], 1=transparent, 0=opaque
        xray_raw = np.clip(I_full * 255.0, 0.0, 255.0).astype(np.float32)

        # 4. DSA: digital subtraction — isolates wire signal
        #    proj > mask_proj where wire is → dsa > 0 at wire locations
        dsa = proj - self._mask_proj   # positive where wire adds attenuation

        # 5. Simulate Poisson X-ray noise on DSA signal
        I_mask = self.noise_photons * np.exp(-self._mask_proj)
        I_comb = self.noise_photons * np.exp(-proj)
        I_mask_noisy = np.random.poisson(np.maximum(I_mask, 1)).astype(np.float32)
        I_comb_noisy = np.random.poisson(np.maximum(I_comb, 1)).astype(np.float32)
        safe_ratio = np.maximum(I_comb_noisy, 1) / np.maximum(I_mask_noisy, 1)
        dsa_noisy = -np.log(safe_ratio)   # positive where wire/contrast present

        # 6. Blur (detector PSF)
        if self.blur_sigma > 0:
            xray_raw = gaussian_filter(xray_raw, sigma=self.blur_sigma)
            dsa_noisy = gaussian_filter(dsa_noisy, sigma=self.blur_sigma)

        # 7. Resize both to image_size
        xray_resized = self._resize(xray_raw)
        dsa_resized = self._resize(dsa_noisy)

        # Blend anatomical background after resize (now both are image_size)
        if self._bg_img is not None:
            xray_resized = xray_resized * (1.0 - 0.65 * self._bg_img)
        xray_resized = np.clip(xray_resized, 0.0, 255.0).astype(np.float32)

        # 8. Full X-ray: invert to clinical monitor convention
        #    (digital fluoroscopy: dark background, bright wire/dense objects)
        xray_inverted = 255.0 - xray_resized

        # Apply image intensifier vignetting (center bright, edges dark)
        if self.vignette_strength > 0:
            xray_inverted = self._apply_vignette(xray_inverted)

        xray_display = np.clip(xray_inverted, 0, 255).astype(np.uint8)

        # 8b. DSA: add residual motion artifact (imperfect mask/live alignment)
        motion_residual = gaussian_filter(
            np.random.randn(*dsa_resized.shape).astype(np.float32) * 0.015,
            sigma=10.0)
        dsa_resized = dsa_resized + motion_residual
        dsa_pos = np.clip(dsa_resized, 0, None)

        if self.clinical_dsa_mask:
            # Clinical DSA display: gray background, dark vessels, darkest = wire
            # Percentile normalization auto-scales to whatever is in the image
            p995 = float(np.percentile(dsa_pos, 99.5)) if dsa_pos.max() > 1e-6 else 1.0
            dsa_norm = np.clip(dsa_pos / max(p995, 1e-8), 0.0, 1.0)
            # Invert: signal → dark; background noise → medium gray (like real angio)
            gray_bg = 215.0
            dsa_display = np.clip(
                gray_bg * (1.0 - dsa_norm * self.dsa_gain * 0.22), 0, 255
            ).astype(np.uint8)
        else:
            # Wire-only mode: black background, bright wire (original RL training mode)
            dsa_ref = self.mu_wire * 2.0 * self.wire_radius * self.voxel_spacing
            dsa_norm = dsa_pos / (dsa_ref + 1e-8) * self.dsa_gain
            dsa_display = np.clip(dsa_norm * 255, 0, 255).astype(np.uint8)

        # 9. Default image = DSA (wire-only view, most useful for policy)
        self._image = dsa_display
        # Also expose full X-ray for visualization
        self._image_xray = xray_display

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_bg_image(self) -> np.ndarray:
        """Build anatomical chest background image, shape=image_size, range [0,1].

        Approximates: cardiac silhouette + lung texture + diaphragm + soft tissue body.
        Randomized per episode to simulate patient variability.
        """
        H, W = self.image_size
        y, x = np.ogrid[:H, :W]

        # Soft tissue body (elliptical torso cross-section)
        body = (((x - W * 0.5) / (W * 0.48))**2
                + ((y - H * 0.5) / (H * 0.46))**2 < 1).astype(np.float32) * 0.42

        # Cardiac silhouette (slightly left of center, soft gaussian blob)
        cx, cy = W * 0.47, H * 0.60
        cardiac = np.exp(-((x - cx)**2 / (W * 0.22)**2
                           + (y - cy)**2 / (H * 0.28)**2)) * 0.38

        # Diaphragm (soft horizontal band ~73% down)
        diaphragm = np.exp(-((y - H * 0.73)**2 / (H * 0.04)**2)) * 0.28

        # Lung field texture: multi-scale Gaussian noise for realistic mottling
        coarse = gaussian_filter(
            np.random.randn(H, W).astype(np.float32), sigma=22) * 0.06
        medium = gaussian_filter(
            np.random.randn(H, W).astype(np.float32), sigma=8) * 0.04
        fine = gaussian_filter(
            np.random.randn(H, W).astype(np.float32), sigma=2) * 0.02
        lung_texture = np.clip(coarse + medium + fine, 0, None)

        bg = body + cardiac + diaphragm + lung_texture
        return np.clip(bg, 0.0, 1.0).astype(np.float32)

    def _apply_vignette(self, img: np.ndarray) -> np.ndarray:
        """Image intensifier vignetting: center bright, edges progressively darker."""
        H, W = img.shape
        y, x = np.ogrid[:H, :W]
        d2 = ((x - W / 2.0)**2 + (y - H / 2.0)**2) / (min(H, W) / 2.0)**2
        mask = np.clip(1.0 - self.vignette_strength * d2, 0.15, 1.0).astype(np.float32)
        return img * mask

    def _voxelize_wire(self, wire_vol: np.ndarray, dof_positions: np.ndarray) -> None:
        """Mark guidewire DOF positions as high-attenuation voxels.

        Interpolates between consecutive DOF points so that the wire is rendered
        as a continuous tube, not just isolated spheres at sparse sample points.
        """
        if len(dof_positions) < 1:
            return

        shape = np.array(wire_vol.shape)
        sp = self._voxel_spacing
        origin = self._voxel_origin
        r_vox = int(np.ceil(self.wire_radius / sp[0]))
        r2 = r_vox * r_vox

        # Precompute offsets once
        offsets = []
        for dx in range(-r_vox, r_vox + 1):
            for dy in range(-r_vox, r_vox + 1):
                for dz in range(-r_vox, r_vox + 1):
                    if dx*dx + dy*dy + dz*dz <= r2:
                        offsets.append((dx, dy, dz))

        def _stamp(pt):
            idx = np.round((pt - origin) / sp).astype(int)
            for dx, dy, dz in offsets:
                ix, iy, iz = idx[0]+dx, idx[1]+dy, idx[2]+dz
                if 0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]:
                    wire_vol[ix, iy, iz] = self.mu_wire

        # Stamp each DOF point and interpolate along segments
        _stamp(dof_positions[0])
        for i in range(len(dof_positions) - 1):
            p0 = dof_positions[i].astype(np.float64)
            p1 = dof_positions[i + 1].astype(np.float64)
            seg_len = np.linalg.norm(p1 - p0)
            if seg_len < 1e-6:
                continue
            # Sample every half-voxel along the segment to guarantee no gaps
            n_samples = max(2, int(np.ceil(seg_len / (float(sp[0]) * 0.5))))
            for t in np.linspace(0.0, 1.0, n_samples):
                _stamp(p0 + t * (p1 - p0))

    def _project(self, volume: np.ndarray) -> np.ndarray:
        """
        Beer-Lambert projection along self.projection_axis.
        Returns integrated attenuation map of shape (H, W).
        """
        proj = np.sum(volume, axis=self.projection_axis) * self.voxel_spacing
        return proj.astype(np.float32)

    def _resize(self, proj: np.ndarray) -> np.ndarray:
        """Resize 2D projection to image_size using bilinear interpolation."""
        from scipy.ndimage import zoom
        h_target, w_target = self.image_size
        h_src, w_src = proj.shape
        if h_src == h_target and w_src == w_target:
            return proj
        zoom_h = h_target / h_src
        zoom_w = w_target / w_src
        return zoom(proj, (zoom_h, zoom_w), order=1)
