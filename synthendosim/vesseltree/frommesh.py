"""
SynthEndoSim — VesselTreeFromMesh: load anatomy from OBJ/STL + optional branch list.
"""
from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np

from .base import VesselTree, Insertion
from .util.branch import (
    Branch, BranchWithRadii, BranchingPoint,
    calc_branching_with_radii, calc_branching,
    rotate_branches, scale_branches_xyz, rotate_array,
)


class VesselTreeFromMesh(VesselTree):
    """
    Wraps an existing OBJ/STL mesh as a VesselTree.

    Parameters
    ----------
    mesh_path : str
        Path to collision mesh (OBJ or STL).
    insertion_position : (x, y, z) in mm
    insertion_direction : unit vector (x, y, z)
    branches : list of Branch / BranchWithRadii, optional
        If provided, enables branch-aware target selection and pathfinding.
        If None, centerline is set to a stub and branching_points = [].
    approx_radii : float or list[float], optional
        Used to compute branching points when branches lack embedded radii.
    visu_mesh_path : str, optional
    rotation_yzx_deg : (y, z, x) degrees, optional
    scaling_xyz : (sx, sy, sz), optional
    """

    def __init__(
        self,
        mesh_path: str,
        insertion_position: Tuple[float, float, float],
        insertion_direction: Tuple[float, float, float],
        branches: Optional[List[Union[Branch, BranchWithRadii]]] = None,
        approx_radii: Optional[Union[float, List[float]]] = None,
        visu_mesh_path: Optional[str] = None,
        rotation_yzx_deg: Optional[Tuple[float, float, float]] = None,
        scaling_xyz: Optional[Tuple[float, float, float]] = None,
    ):
        self._raw_mesh = mesh_path
        self._raw_visu = visu_mesh_path
        self._raw_branches = list(branches) if branches else None
        self._approx_radii = approx_radii
        self._rot = rotation_yzx_deg or (0., 0., 0.)
        self._scale = scaling_xyz or (1., 1., 1.)

        ip = np.array(insertion_position, dtype=np.float32)
        id_ = np.array(insertion_direction, dtype=np.float32)
        if any(r != 0. for r in self._rot):
            ip  = rotate_array(ip,  *self._rot)
            id_ = rotate_array(id_, *self._rot)
        self.insertion = Insertion(ip, id_ / (np.linalg.norm(id_) + 1e-9))

        self.mesh_path      = mesh_path
        self.visu_mesh_path = visu_mesh_path
        self.branches       = None
        self.branching_points = []
        self.centerline_coordinates = np.zeros((1, 3), dtype=np.float32)
        self.bbox_low  = np.zeros(3, dtype=np.float32)
        self.bbox_high = np.ones(3, dtype=np.float32) * 300.
        self._initialized = False

    def reset(self, episode: int = 0, seed: Optional[int] = None) -> None:
        if self._initialized:
            return  # static anatomy — no need to reinitialise
        self._initialized = True
        bl = list(self._raw_branches or [])
        if any(s != 1. for s in self._scale):
            s = np.array(self._scale, dtype=np.float32)
            bl = [self._scale_branch(b, s) for b in bl]
        if any(r != 0. for r in self._rot) and bl:
            bl = list(rotate_branches(bl, self._rot))
        self.branches = tuple(bl) if bl else ()
        if bl:
            if isinstance(bl[0], BranchWithRadii):
                self.branching_points = calc_branching_with_radii(bl)
            else:
                self.branching_points = calc_branching(bl, self._approx_radii or 5.0)
            self.centerline_coordinates = np.concatenate([b.coordinates for b in bl])
            self.bbox_low  = np.min([b.low  for b in bl], axis=0).astype(np.float32)
            self.bbox_high = np.max([b.high for b in bl], axis=0).astype(np.float32)
        else:
            # derive bounds from mesh if trimesh available
            try:
                import trimesh
                m = trimesh.load(self._raw_mesh)
                self.bbox_low  = np.array(m.bounds[0], dtype=np.float32)
                self.bbox_high = np.array(m.bounds[1], dtype=np.float32)
            except Exception:
                pass

    def _scale_branch(self, b, s):
        c = b.coordinates * s
        return BranchWithRadii(b.name, c, b.radii) if isinstance(b, BranchWithRadii) else Branch(b.name, c)
