"""Compute insertion point + LCCA target (from compute_nav_points.py)"""
import json
import numpy as np
import nibabel as nib
from pathlib import Path


def compute_nav(seg_dir: Path, mesh_dir: Path) -> dict:
    def load_vox(label: str):
        p = seg_dir / f"{label}.nii.gz"
        if not p.exists():
            return None, None
        img = nib.load(str(p))
        sp = float(img.header.get_zooms()[0])
        return np.argwhere(img.get_fdata() > 0.5), sp

    ao_vox, sp = load_vox("aorta")
    if ao_vox is None or len(ao_vox) == 0:
        return {"error": "aorta not found"}

    z_lo = ao_vox[:, 2].min() + (ao_vox[:, 2].max() - ao_vox[:, 2].min()) * 0.05
    bot = ao_vox[ao_vox[:, 2] <= z_lo].mean(axis=0) * sp
    top_ao = ao_vox[ao_vox[:, 2].argmax()].astype(float) * sp
    direction = top_ao - bot
    direction = direction / np.linalg.norm(direction)

    lcca_vox, _ = load_vox("common_carotid_artery_left")
    if lcca_vox is not None and len(lcca_vox) >= 10:
        z_hi = lcca_vox[:, 2].max() - (lcca_vox[:, 2].max() - lcca_vox[:, 2].min()) * 0.2
        target = lcca_vox[lcca_vox[:, 2] >= z_hi].mean(axis=0) * sp
    else:
        target = top_ao

    nav = {
        "insertion_position": [round(v, 2) for v in bot.tolist()],
        "insertion_direction": [round(v, 4) for v in direction.tolist()],
        "target_lcca": [round(v, 2) for v in target.tolist()],
        "spacing_mm": sp,
    }
    (mesh_dir / "nav_points.json").write_text(json.dumps(nav, indent=2))
    return nav
