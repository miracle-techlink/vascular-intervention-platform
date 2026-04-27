"""mask → smoothed OBJ (from modeling_pipeline.py)"""
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from skimage.morphology import ball, closing as morpho_closing
from skimage.measure import marching_cubes
import trimesh
import trimesh.smoothing as tsmooth
from pathlib import Path
from typing import Optional

from .segment import VESSEL_LABELS


def _load_mask(nii_path: Path):
    img = nib.load(str(nii_path))
    data = img.get_fdata().astype(np.uint8)
    spacing = np.abs(np.diag(img.affine)[:3])
    return data, spacing


def _postprocess(mask: np.ndarray) -> np.ndarray:
    closed = morpho_closing(mask.astype(bool), ball(2)).astype(np.uint8)
    labeled, n = ndi.label(closed)
    if n == 0:
        return closed
    sizes = ndi.sum(closed, labeled, range(1, n + 1))
    return (labeled == int(np.argmax(sizes)) + 1).astype(np.uint8)


def _to_mesh(mask: np.ndarray, spacing: np.ndarray) -> Optional[trimesh.Trimesh]:
    if mask.sum() == 0:
        return None
    try:
        verts, faces, normals, _ = marching_cubes(mask.astype(float), level=0.5,
                                                  spacing=tuple(spacing))
    except Exception:
        return None
    if len(faces) == 0:
        return None
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    tsmooth.filter_taubin(mesh, iterations=10)
    if len(mesh.faces) > 5000:
        mesh = mesh.simplify_quadric_decimation(face_count=5000)
    return mesh


def _write_clean_obj(mesh: trimesh.Trimesh, path: Path):
    """Write OBJ without vertex colors — required for PyVista/stEVE."""
    with open(str(path), "w") as f:
        for v in mesh.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in mesh.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def build_meshes(seg_dir: Path, mesh_dir: Path) -> list[str]:
    """Returns list of vessel names that were successfully meshed."""
    mesh_dir.mkdir(parents=True, exist_ok=True)
    done = []
    combined_meshes = []

    for vessel in VESSEL_LABELS:
        nii = seg_dir / f"{vessel}.nii.gz"
        if not nii.exists():
            continue
        mask, spacing = _load_mask(nii)
        mesh = _to_mesh(_postprocess(mask), spacing)
        if mesh is None:
            continue
        _write_clean_obj(mesh, mesh_dir / f"{vessel}.obj")
        combined_meshes.append(mesh)
        done.append(vessel)

    if combined_meshes:
        combined = trimesh.util.concatenate(combined_meshes)
        _write_clean_obj(combined, mesh_dir / "vessel_combined.obj")

    return done
