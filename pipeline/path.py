"""
Shortest path through vessel mesh from insertion point to LCCA target.
Algorithm: voxelize mesh interior → Dijkstra (skimage.graph) → smooth spline.
"""
import json
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from skimage.graph import route_through_array
import pyvista as pv


def compute_path(mesh_dir: Path, spacing: float = 2.0) -> list[list[float]]:
    nav = json.loads((mesh_dir / "nav_points.json").read_text())
    insertion = np.array(nav["insertion_position"], dtype=np.float32)
    lcca      = np.array(nav["target_lcca"],        dtype=np.float32)

    # ── 1. Load mesh and voxelize interior ────────────────────────────
    mesh_file = "vessel_combined_vtk.obj" if (mesh_dir / "vessel_combined_vtk.obj").exists() \
                else "vessel_combined.obj"
    mesh = pv.read(str(mesh_dir / mesh_file))

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    pad = spacing * 2
    nx = int((xmax - xmin + 2*pad) / spacing) + 1
    ny = int((ymax - ymin + 2*pad) / spacing) + 1
    nz = int((zmax - zmin + 2*pad) / spacing) + 1
    origin = np.array([xmin - pad, ymin - pad, zmin - pad], dtype=np.float32)

    # Sample grid points and test inside mesh
    xi = np.arange(nx) * spacing + origin[0]
    yi = np.arange(ny) * spacing + origin[1]
    zi = np.arange(nz) * spacing + origin[2]
    gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='ij')
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    pcloud = pv.PolyData(pts)
    selected = pcloud.select_enclosed_points(mesh.extract_surface(), check_surface=False)
    inside = selected['SelectedPoints'].reshape(nx, ny, nz).astype(bool)

    # Fill small holes for connectivity
    inside = binary_fill_holes(inside)

    # ── 2. World → voxel coordinate conversion ────────────────────────
    def world_to_vox(pt):
        idx = ((pt - origin) / spacing).astype(int)
        return tuple(np.clip(idx, 0, [nx-1, ny-1, nz-1]))

    def vox_to_world(idx):
        return np.array(idx, dtype=float) * spacing + origin

    start = world_to_vox(insertion)
    end   = world_to_vox(lcca)

    # If start/end outside vessel, find nearest inside voxel
    if not inside[start]:
        inside_pts = np.argwhere(inside)
        start = tuple(inside_pts[np.argmin(np.linalg.norm(inside_pts - start, axis=1))])
    if not inside[end]:
        inside_pts = np.argwhere(inside)
        end = tuple(inside_pts[np.argmin(np.linalg.norm(inside_pts - end, axis=1))])

    # ── 3. Dijkstra on voxel grid ─────────────────────────────────────
    # Cost = 1 inside vessel, high outside → path stays in vessel
    cost = np.where(inside, 1.0, 1e6).astype(np.float32)

    path_vox, _ = route_through_array(cost, start, end, fully_connected=True)
    path_vox = np.array(path_vox)

    # ── 4. Back to world + smooth ─────────────────────────────────────
    path_world = path_vox * spacing + origin  # (N, 3)

    # Downsample to ~1 point per 3mm, then smooth
    step = max(1, int(3.0 / spacing))
    path_world = path_world[::step]

    sigma = 3.0
    path_world[:, 0] = gaussian_filter1d(path_world[:, 0], sigma)
    path_world[:, 1] = gaussian_filter1d(path_world[:, 1], sigma)
    path_world[:, 2] = gaussian_filter1d(path_world[:, 2], sigma)

    # Save for caching
    out = {"path_points": path_world.tolist(), "n_points": len(path_world),
           "insertion": insertion.tolist(), "lcca": lcca.tolist()}
    (mesh_dir / "planned_path.json").write_text(json.dumps(out))

    return path_world.tolist()
