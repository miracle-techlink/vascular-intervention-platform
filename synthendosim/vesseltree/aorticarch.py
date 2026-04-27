"""
SynthEndoSim — Procedural Aortic Arch generator.

Generates 7 aortic arch anatomical variants (Type I–VII) using
Cubic Hermite Spline interpolation with per-patient random variation.

Arch types (ISR classification):
  I   : BCT → LCCA → LSA            (most common, ~75%)
  II  : BCT(RCCA+LCCA) → LSA        (bovine arch)
  IV  : RSA → CO(RCCA+LCCA) → LSA
  V   : BCT(RCCA+LCCA) → LSA → RSA
  VI  : BCT → CO(RSA+LSA)
  VII : RSA → RCCA → LCCA → LSA

Usage:
    tree = AorticArch(arch_type="I", seed=42)
    tree.reset()
    print(tree.branches)        # (aorta, bct, rcca, ...)
    print(tree.insertion)       # Insertion(position, direction)
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from .base import VesselTree, Insertion
from .util.branch import (
    BranchWithRadii, calc_branching_with_radii,
    rotate_branches, scale_branches_xyzd, calc_insertion_from_branch_start,
)
from .util.hermite import CHSPoint, chs_point_normal, chs_to_branch_points

ARCH_TYPES = ("I", "II", "IV", "V", "VI", "VII")


class AorticArch(VesselTree):
    """
    Procedurally generated aortic arch with randomised geometry.

    Parameters
    ----------
    arch_type : str
        One of "I", "II", "IV", "V", "VI", "VII".
    seed : int, optional
        Random seed for reproducibility.
    rotation_yzx_deg : (y, z, x) rotation angles in degrees.
    scaling_xyzd : (sx, sy, sz, sr) scaling for coords and radii.
    """

    def __init__(
        self,
        arch_type: str = "I",
        seed: Optional[int] = None,
        rotation_yzx_deg: Optional[Tuple[float, float, float]] = None,
        scaling_xyzd: Optional[Tuple[float, float, float, float]] = None,
    ):
        if arch_type not in ARCH_TYPES:
            raise ValueError(f"arch_type must be one of {ARCH_TYPES}")
        self.arch_type = arch_type
        self.seed = seed if seed is not None else int(np.random.randint(0, 2**31))
        self.rotation_yzx_deg = rotation_yzx_deg or (0., 0., 0.)
        self.scaling_xyzd = scaling_xyzd or (1., 1., 1., 1.)

        self.branches = None
        self.branching_points = None
        self.centerline_coordinates = None
        self.insertion = None
        self.bbox_low = np.zeros(3, dtype=np.float32)
        self.bbox_high = np.ones(3, dtype=np.float32) * 200.
        self.mesh_path = ""
        self.visu_mesh_path = None

    def reset(self, episode: int = 0, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)
        branches = self._generate(rng)
        if any(r != 0. for r in self.rotation_yzx_deg):
            branches = rotate_branches(branches, self.rotation_yzx_deg)
        if self.scaling_xyzd != (1., 1., 1., 1.):
            branches = scale_branches_xyzd(branches, self.scaling_xyzd)
        self.branches = tuple(branches)
        self.branching_points = calc_branching_with_radii(list(self.branches))
        self.centerline_coordinates = np.concatenate(
            [b.coordinates for b in self.branches]
        )
        pos, dir_ = calc_insertion_from_branch_start(self.branches[0])
        self.insertion = Insertion(pos, dir_)
        all_low  = np.min([b.low  for b in self.branches], axis=0)
        all_high = np.max([b.high for b in self.branches], axis=0)
        self.bbox_low  = all_low.astype(np.float32)
        self.bbox_high = all_high.astype(np.float32)
        self.mesh_path = self._make_mesh_path()

    # ── mesh path (uses anatomy loader temp file) ──────────────────────
    def _make_mesh_path(self) -> str:
        """Return empty string; SOFA backend can generate mesh on demand."""
        return ""

    # ── branch generators ──────────────────────────────────────────────

    def _generate(self, rng: np.random.Generator) -> List[BranchWithRadii]:
        n = rng.normal
        aorta = self._aorta(rng)
        res = 2  # mm resolution

        if self.arch_type == "I":
            return self._type_I(rng, n, aorta, res)
        elif self.arch_type == "II":
            return self._type_II(rng, n, aorta, res)
        elif self.arch_type == "IV":
            return self._type_IV(rng, n, aorta, res)
        elif self.arch_type == "V":
            return self._type_V(rng, n, aorta, res)
        elif self.arch_type == "VI":
            return self._type_VI(rng, n, aorta, res)
        elif self.arch_type == "VII":
            return self._type_VII(rng, n, aorta, res)
        return [aorta]

    def _make_branch(self, name: str, pts: List[CHSPoint], res: float) -> BranchWithRadii:
        coords, radii = chs_to_branch_points(pts, res)
        return BranchWithRadii(name, coords, radii)

    def _aorta(self, rng) -> BranchWithRadii:
        n = rng.normal
        km = dict
        p = [
            chs_point_normal(
                (0, -180, 0), (3, 3, 3),
                (0, 1, 0), (0.1, 0.1, 0.1), (100, 5),
                (13, 1), (0, 0.1), rng=rng,
            ),
            chs_point_normal(
                (-10, -100, 15), (5, 5, 5),
                (-0.2, 1, 0.1), (0.1, 0.1, 0.1), (80, 5),
                (12, 1), (0, 0.1), rng=rng,
            ),
            chs_point_normal(
                (-20, -30, 20), (5, 5, 5),
                (-0.5, 0.5, 0), (0.1, 0.1, 0.1), (60, 5),
                (13, 1), (0, 0.1), rng=rng,
            ),
            chs_point_normal(
                (-30, 30, 10), (5, 5, 5),
                (-0.3, 0.3, -0.2), (0.1, 0.1, 0.1), (50, 5),
                (14, 1), (0, 0.1), rng=rng,
            ),
            chs_point_normal(
                (-25, 80, 0), (5, 5, 5),
                (0, 1, -0.1), (0.1, 0.1, 0.1), (60, 5),
                (13, 1), (0, 0.1), rng=rng,
            ),
        ]
        return self._make_branch("aorta", p, 2.0)

    def _bct(self, origin: np.ndarray, rng) -> BranchWithRadii:
        p = [
            CHSPoint((*origin, 6.0), (0.1, 1, 0.1, 0)),
            chs_point_normal(
                (origin[0]+5, origin[1]+25, origin[2]+5), (3, 3, 3),
                (0.2, 1, 0.2), (0.1, 0.1, 0.1), (30, 3),
                (5, 0.5), (0, 0.1), rng=rng,
            ),
            chs_point_normal(
                (origin[0]+8, origin[1]+45, origin[2]+8), (3, 3, 3),
                (0.3, 1, 0.3), (0.1, 0.1, 0.1), (25, 3),
                (4.5, 0.5), (0, 0.1), rng=rng,
            ),
        ]
        return self._make_branch("brachiocephalic_trunk", p, 1.0)

    def _rcca(self, origin: np.ndarray, rng) -> BranchWithRadii:
        p = [
            CHSPoint((*origin, 3.5), (0.3, 1, 0.1, 0)),
            chs_point_normal(
                (origin[0]+15, origin[1]+40, origin[2]+5), (4, 4, 4),
                (0.4, 1, 0.1), (0.1, 0.1, 0.1), (40, 5),
                (3.0, 0.3), (0, 0.1), rng=rng,
            ),
        ]
        return self._make_branch("right_common_carotid", p, 1.0)

    def _lcca(self, origin: np.ndarray, rng) -> BranchWithRadii:
        p = [
            CHSPoint((*origin, 3.5), (-0.2, 1, -0.1, 0)),
            chs_point_normal(
                (origin[0]-12, origin[1]+40, origin[2]+5), (4, 4, 4),
                (-0.3, 1, 0.1), (0.1, 0.1, 0.1), (40, 5),
                (3.0, 0.3), (0, 0.1), rng=rng,
            ),
        ]
        return self._make_branch("left_common_carotid", p, 1.0)

    def _rsa(self, origin: np.ndarray, rng) -> BranchWithRadii:
        p = [
            CHSPoint((*origin, 4.5), (1, 0.2, 0.1, 0)),
            chs_point_normal(
                (origin[0]+40, origin[1]+10, origin[2]+5), (5, 3, 3),
                (1, 0.1, 0.1), (0.1, 0.1, 0.1), (45, 5),
                (3.5, 0.3), (0, 0.1), rng=rng,
            ),
        ]
        return self._make_branch("right_subclavian", p, 1.0)

    def _lsa(self, origin: np.ndarray, rng) -> BranchWithRadii:
        p = [
            CHSPoint((*origin, 4.5), (-1, 0.2, -0.1, 0)),
            chs_point_normal(
                (origin[0]-40, origin[1]+10, origin[2]+5), (5, 3, 3),
                (-1, 0.1, 0.1), (0.1, 0.1, 0.1), (45, 5),
                (3.5, 0.3), (0, 0.1), rng=rng,
            ),
        ]
        return self._make_branch("left_subclavian", p, 1.0)

    def _pick(self, aorta: BranchWithRadii, dist_from_end_mm: float, res: float) -> np.ndarray:
        cl = aorta.coordinates
        diffs = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        arc = np.concatenate([[0], np.cumsum(diffs)])
        total = arc[-1]
        t = max(0, total - dist_from_end_mm)
        idx = int(np.searchsorted(arc, t))
        return cl[min(idx, len(cl)-1)]

    def _type_I(self, rng, n, aorta, res):
        bct_origin = self._pick(aorta, n(36, 5), res)
        bct = self._bct(bct_origin, rng)
        tip = bct.coordinates[-1]
        rcca = self._rcca(tip, rng)
        rsa  = self._rsa(tip, rng)
        lcca_origin = self._pick(aorta, n(20, 4), res)
        lcca = self._lcca(lcca_origin, rng)
        lsa_origin = self._pick(aorta, n(8, 3), res)
        lsa  = self._lsa(lsa_origin, rng)
        return [aorta, bct, rcca, rsa, lcca, lsa]

    def _type_II(self, rng, n, aorta, res):
        bct_origin = self._pick(aorta, n(36, 3), res)
        bct = self._bct(bct_origin, rng)
        tip = bct.coordinates[-1]
        rcca = self._rcca(tip, rng)
        rsa  = self._rsa(tip, rng)
        lcca = self._lcca(tip, rng)
        lsa_origin = self._pick(aorta, n(8, 3), res)
        lsa  = self._lsa(lsa_origin, rng)
        return [aorta, bct, rcca, rsa, lcca, lsa]

    def _type_IV(self, rng, n, aorta, res):
        rsa_origin = self._pick(aorta, n(42, 5), res)
        rsa = self._rsa(rsa_origin, rng)
        co_origin = self._pick(aorta, n(20, 4), res)
        rcca = self._rcca(co_origin, rng)
        lcca = self._lcca(co_origin, rng)
        lsa_origin = self._pick(aorta, n(8, 3), res)
        lsa = self._lsa(lsa_origin, rng)
        return [aorta, rcca, rsa, lcca, lsa]

    def _type_V(self, rng, n, aorta, res):
        bct_origin = self._pick(aorta, n(36, 3), res)
        bct = self._bct(bct_origin, rng)
        tip = bct.coordinates[-1]
        rcca = self._rcca(tip, rng)
        lcca = self._lcca(tip, rng)
        lsa_origin = self._pick(aorta, n(8, 3), res)
        lsa = self._lsa(lsa_origin, rng)
        rsa_origin = self._pick(aorta, n(4, 2), res)
        rsa = self._rsa(rsa_origin, rng)
        return [aorta, bct, rcca, rsa, lcca, lsa]

    def _type_VI(self, rng, n, aorta, res):
        bct_origin = self._pick(aorta, n(36, 3), res)
        bct = self._bct(bct_origin, rng)
        tip = bct.coordinates[-1]
        rcca = self._rcca(tip, rng)
        lcca = self._lcca(tip, rng)
        co_origin = self._pick(aorta, n(10, 2), res)
        rsa = self._rsa(co_origin, rng)
        lsa = self._lsa(co_origin, rng)
        return [aorta, bct, rcca, rcca, lcca, rsa, lsa]

    def _type_VII(self, rng, n, aorta, res):
        rsa_origin = self._pick(aorta, n(38, 3.5), res)
        rsa = self._rsa(rsa_origin, rng)
        rcca_origin = self._pick(aorta, n(20, 3), res)
        rcca = self._rcca(rcca_origin, rng)
        lcca_origin = self._pick(aorta, n(20, 3), res)
        lcca = self._lcca(lcca_origin, rng)
        lsa_origin = self._pick(aorta, n(8, 3), res)
        lsa = self._lsa(lsa_origin, rng)
        return [aorta, rcca, rsa, lcca, lsa]
