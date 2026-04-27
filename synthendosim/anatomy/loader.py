"""
SynthEndoSim — Anatomy loader.

Loads OBJ/STL vessel meshes produced by our CT pipeline directly.
Auto-detects insertion point (lowest Y) and bounding box if not specified.
Supports optional centerline NPY file (N,3) for manifold reward.
"""
from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from ..core.types import AnatomySpec


def load_anatomy(
    mesh_path: str,
    insertion_point: Optional[Tuple[float, float, float]] = None,
    insertion_direction: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    target_point: Optional[Tuple[float, float, float]] = None,
    centerline_path: Optional[str] = None,
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    rotation_yzx_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    visu_mesh_path: Optional[str] = None,
    name: str = "custom",
) -> AnatomySpec:
    """
    Build an AnatomySpec from a mesh file.

    If insertion_point or target_point are None, they are auto-estimated
    from the mesh bounding box (entry = bottom, target = top centroid).
    """
    mesh_path = str(mesh_path)
    has_mesh = bool(mesh_path and mesh_path not in ("", "/dev/null"))

    if has_mesh:
        vertices = _load_vertices(mesh_path)

        if any(s != 1.0 for s in scale):
            vertices = vertices * np.array(scale)
            mesh_path = _save_transformed_mesh(mesh_path, scale, rotation_yzx_deg)

        if any(r != 0.0 for r in rotation_yzx_deg):
            vertices = _rotate_vertices(vertices, rotation_yzx_deg)
            if scale == (1.0, 1.0, 1.0):
                mesh_path = _save_transformed_mesh(mesh_path, scale, rotation_yzx_deg)

        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
    else:
        # No mesh: stub geometry (mock backend / testing without mesh file)
        bbox_min = np.zeros(3, dtype=np.float32)
        bbox_max = np.ones(3, dtype=np.float32) * 300.0

    if insertion_point is None:
        insertion_point = tuple(bbox_min.tolist())

    if target_point is None:
        if has_mesh:
            top_mask = vertices[:, 1] > bbox_max[1] - (bbox_max[1] - bbox_min[1]) * 0.1
            top_verts = vertices[top_mask]
            centroid = top_verts.mean(axis=0) if len(top_verts) > 0 else bbox_max
            target_point = tuple(centroid.tolist())
        else:
            target_point = tuple(bbox_max.tolist())

    centerline: Optional[np.ndarray] = None
    if centerline_path is not None:
        centerline = np.load(centerline_path).astype(np.float32)

    return AnatomySpec(
        mesh_path=mesh_path,
        insertion_point=tuple(insertion_point),
        insertion_direction=tuple(insertion_direction),
        target_point=tuple(target_point),
        centerline=centerline,
        name=name,
        scale=tuple(scale),
        rotation_yzx_deg=tuple(rotation_yzx_deg),
        visu_mesh_path=visu_mesh_path,
    )


def _load_vertices(mesh_path: str) -> np.ndarray:
    path = Path(mesh_path)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return _parse_obj_vertices(mesh_path)
    elif suffix in (".stl",):
        return _parse_stl_vertices(mesh_path)
    else:
        try:
            import trimesh
            m = trimesh.load(mesh_path, force="mesh")
            return np.array(m.vertices, dtype=np.float32)
        except ImportError:
            raise RuntimeError(
                f"Cannot load {suffix} mesh without trimesh. "
                "pip install trimesh"
            )


def _parse_obj_vertices(path: str) -> np.ndarray:
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float32)


def _parse_stl_vertices(path: str) -> np.ndarray:
    try:
        import trimesh
        m = trimesh.load(path, force="mesh")
        return np.array(m.vertices, dtype=np.float32)
    except ImportError:
        pass
    # Fallback: binary STL
    with open(path, "rb") as f:
        f.read(80)
        n_tri = int.from_bytes(f.read(4), "little")
        verts = []
        for _ in range(n_tri):
            f.read(12)  # normal
            for _ in range(3):
                x, y, z = np.frombuffer(f.read(12), dtype=np.float32)
                verts.append([x, y, z])
            f.read(2)  # attr
    return np.array(verts, dtype=np.float32)


def _rotate_vertices(
    verts: np.ndarray,
    rotation_yzx_deg: Tuple[float, float, float],
) -> np.ndarray:
    ry, rz, rx = [np.deg2rad(a) for a in rotation_yzx_deg]

    def Rx(a):
        return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    def Ry(a):
        return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    def Rz(a):
        return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])

    R = Ry(ry) @ Rz(rz) @ Rx(rx)
    return (R @ verts.T).T.astype(np.float32)


def _save_transformed_mesh(
    mesh_path: str,
    scale: Tuple[float, float, float],
    rotation_yzx_deg: Tuple[float, float, float],
) -> str:
    """Apply scale + rotation and save to temp OBJ. Returns new path."""
    try:
        import trimesh
        m = trimesh.load(mesh_path, force="mesh")
        m.vertices *= np.array(scale)
        ry, rz, rx = [np.deg2rad(a) for a in rotation_yzx_deg]
        if any(r != 0.0 for r in [ry, rz, rx]):
            import trimesh.transformations as tf
            R = tf.euler_matrix(rx, ry, rz, axes="sxyz")
            m.apply_transform(R)
        tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        m.export(tmp.name)
        return tmp.name
    except ImportError:
        # No trimesh: return original path (scale/rotation ignored for mesh file)
        return mesh_path
