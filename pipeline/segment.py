"""TotalSegmentator: NIfTI → vessel mask NIfTIs (from batch_pipeline.py)"""
import nibabel as nib
from pathlib import Path

VESSEL_LABELS = [
    "aorta",
    "common_carotid_artery_left",
    "common_carotid_artery_right",
    "brachiocephalic_trunk",
    "subclavian_artery_left",
    "subclavian_artery_right",
]


def segment(input_nii: Path, out_dir: Path) -> Path:
    if (out_dir / "aorta.nii.gz").exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    from totalsegmentator.python_api import totalsegmentator as tseg
    img = nib.load(str(input_nii))
    tseg(img, str(out_dir), fast=True, device="cpu", quiet=True,
         roi_subset=VESSEL_LABELS)
    return out_dir
