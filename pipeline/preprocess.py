"""NRRD/NIfTI → isotropic NIfTI (from batch_pipeline.py)"""
import numpy as np
import SimpleITK as sitk
from pathlib import Path


def preprocess(ct_path: Path, out_path: Path, new_spacing: float = 1.5) -> Path:
    if out_path.exists():
        return out_path
    img = sitk.ReadImage(str(ct_path))
    arr = sitk.GetArrayFromImage(img).astype(np.int32)
    offset = 1024 if arr.min() >= 0 else 0
    arr = (arr - offset).astype(np.int16)
    img_hu = sitk.GetImageFromArray(arr)
    img_hu.CopyInformation(img)
    orig_sp = img_hu.GetSpacing()
    orig_sz = img_hu.GetSize()
    new_sz = [int(orig_sz[i] * orig_sp[i] / new_spacing + 0.5) for i in range(3)]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing([new_spacing] * 3)
    rs.SetSize(new_sz)
    rs.SetOutputDirection(img_hu.GetDirection())
    rs.SetOutputOrigin(img_hu.GetOrigin())
    rs.SetDefaultPixelValue(-1024)
    rs.SetInterpolator(sitk.sitkLinear)
    img_rs = rs.Execute(img_hu)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img_rs, str(out_path))
    return out_path
