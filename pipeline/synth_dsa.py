"""
Synthetic DSA video generator.

Pipeline:
  1. Load vessel OBJ + nav_points + planned_path
  2. Compute Dijkstra path if not cached
  3. Project 3D → 2D with a simulated C-arm (LAO 30°, caudal 10°)
  4. Render frame-by-frame:
       - background: dark grey X-ray plate
       - vessels: semi-transparent white wire silhouette (DRR-style)
       - catheter: bright white tube growing step-by-step
       - tip: glowing disc
       - HUD: stage label, progress bar, tip coords
  5. Write MP4 + annotations JSON
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# ── optional: compute path if missing ─────────────────────────────────────────
def _ensure_path(mesh_dir: Path, spacing: float = 2.0):
    cached = mesh_dir / "planned_path.json"
    if cached.exists():
        return json.loads(cached.read_text())["path_points"]
    print("[synth_dsa] computing Dijkstra path …")
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from pipeline.path import compute_path
    return compute_path(mesh_dir, spacing)


# ── C-arm projection ───────────────────────────────────────────────────────────
def _build_projection(pts_3d: np.ndarray,
                      lao_deg: float = 30.0, caudal_deg: float = 10.0,
                      canvas: int = 512) -> tuple:
    """
    Return (R, scale, offset) so that  p2d = (R @ pt)[[0,1]] * scale + offset
    covers the canvas comfortably.
    """
    lao = np.radians(lao_deg)
    caudal = np.radians(caudal_deg)

    # rotate around Z (LAO/RAO) then X (caudal/cranial)
    Rz = np.array([[ np.cos(lao), np.sin(lao), 0],
                   [-np.sin(lao), np.cos(lao), 0],
                   [ 0,           0,            1]])
    Rx = np.array([[1, 0,              0           ],
                   [0, np.cos(caudal), np.sin(caudal)],
                   [0,-np.sin(caudal), np.cos(caudal)]])
    R = Rx @ Rz

    proj = (R @ pts_3d.T).T  # (N, 3)
    xy   = proj[:, :2]       # ignore depth for orthographic projection

    mn, mx = xy.min(axis=0), xy.max(axis=0)
    span = (mx - mn).max()
    margin = 0.10
    scale  = canvas * (1 - 2 * margin) / span
    offset = canvas * margin - mn * scale

    return R, scale, offset


def project(pts_3d: np.ndarray, R, scale, offset) -> np.ndarray:
    proj = (R @ pts_3d.T).T[:, :2]
    return (proj * scale + offset).astype(np.float32)


# ── vessel silhouette (pre-render once) ───────────────────────────────────────
def _load_vessel_verts(obj_path: Path) -> np.ndarray:
    verts = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(list(map(float, line.split()[1:4])))
    return np.array(verts, dtype=np.float32)


def _load_vessel_faces(obj_path: Path, verts: np.ndarray) -> list:
    """Return list of (i,j,k) 0-based face indices."""
    faces = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                if len(idx) >= 3:
                    faces.append(idx[:3])
    return faces


def _render_vessel_layer(obj_path: Path, R, scale, offset, canvas: int,
                         alpha: float = 0.18) -> np.ndarray:
    """
    Return a float32 [H,W] greyscale image: vessel silhouette blurred.
    """
    verts = _load_vessel_verts(obj_path)
    if len(verts) == 0:
        return np.zeros((canvas, canvas), np.float32)

    faces = _load_vessel_faces(obj_path, verts)
    p2d   = project(verts, R, scale, offset).astype(np.int32)

    img = np.zeros((canvas, canvas), np.float32)
    # draw edge lines for every face
    for f in faces:
        for a, b in [(f[0],f[1]),(f[1],f[2]),(f[2],f[0])]:
            pa = p2d[a]; pb = p2d[b]
            cv2.line(img, tuple(pa), tuple(pb), color=1.0, thickness=1,
                     lineType=cv2.LINE_AA)

    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    return np.clip(img, 0, 1) * alpha


# ── stage classification ───────────────────────────────────────────────────────
def _classify_stage(step: int, total: int, path_3d: np.ndarray,
                    insertion: np.ndarray, target: np.ndarray) -> tuple[str, str]:
    frac = step / max(total - 1, 1)
    tip  = path_3d[step]

    dist_to_target = np.linalg.norm(tip - target)
    height_frac    = (tip[2] - insertion[2]) / max(target[2] - insertion[2], 1)

    if frac < 0.08:
        return "INSERTING",     "#00ff88"
    elif height_frac < 0.35:
        return "DESCENDING AORTA", "#00e5ff"
    elif height_frac < 0.62:
        return "AORTIC ARCH",   "#ffb300"
    elif dist_to_target > 20:
        return "NAVIGATING",    "#29b6f6"
    else:
        return "ARRIVED [OK]",  "#ff4444"


# ── HUD helpers ────────────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX

def _put_text(img, text, xy, color=(255,255,255), scale=0.45, thickness=1):
    cv2.putText(img, text, xy, FONT, scale, color, thickness, cv2.LINE_AA)

def _progress_bar(img, frac, canvas):
    bw, bh = canvas - 40, 6
    bx, by = 20, canvas - 22
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (60,60,60), -1)
    cv2.rectangle(img, (bx, by), (bx + int(bw * frac), by + bh), (80,180,255), -1)


# ── main render ────────────────────────────────────────────────────────────────
def generate_dsa_video(
    mesh_dir: Path,
    output_path: Path,
    fps: int = 25,
    canvas: int = 512,
    lao: float = 30.0,
    caudal: float = 10.0,
    catheter_step: int = 1,   # path points per frame
    vessel_alpha: float = 0.22,
    noise_sigma: float = 2.5,
    extra_hold_frames: int = 40,  # freeze on final frame
):
    import imageio.v2 as iio

    # ── 1. data ────────────────────────────────────────────────────────────────
    nav  = json.loads((mesh_dir / "nav_points.json").read_text())
    insertion = np.array(nav["insertion_position"], np.float32)
    target    = np.array(nav["target_lcca"],        np.float32)
    path_3d   = np.array(_ensure_path(mesh_dir),    np.float32)  # (N,3)
    total     = len(path_3d)

    print(f"[synth_dsa] path={total} pts  insertion={insertion.round(1)}  target={target.round(1)}")

    # ── 2. projection ──────────────────────────────────────────────────────────
    all_pts = np.vstack([path_3d, insertion.reshape(1,3), target.reshape(1,3)])
    R, scale, offset = _build_projection(all_pts, lao, caudal, canvas)

    # ── 3. vessel layer (static) ──────────────────────────────────────────────
    vessel_obj = mesh_dir / "vessel_combined.obj"
    if not vessel_obj.exists():
        vessel_obj = mesh_dir / "vessel_combined_vtk.obj"
    print("[synth_dsa] rendering vessel silhouette …")
    vessel_layer = _render_vessel_layer(vessel_obj, R, scale, offset, canvas, vessel_alpha)

    # ── 4. special-point projections ──────────────────────────────────────────
    ins_2d = project(insertion.reshape(1,3), R, scale, offset)[0].astype(int)
    tgt_2d = project(target.reshape(1,3),    R, scale, offset)[0].astype(int)
    path_2d = project(path_3d, R, scale, offset)  # (N,2) float

    # ── 5. per-frame render ────────────────────────────────────────────────────
    annotations = []
    writer      = iio.get_writer(str(output_path), format="FFMPEG",
                                 fps=fps, codec="libx264",
                                 output_params=["-crf","18","-pix_fmt","yuv420p"])

    steps = list(range(0, total, catheter_step))
    if steps[-1] != total - 1:
        steps.append(total - 1)

    for frame_idx, step in enumerate(steps):
        # ── base plate: dark grey with slight noise ────────────────────────────
        noise  = np.random.normal(0, noise_sigma / 255.0, (canvas, canvas)).astype(np.float32)
        plate  = np.clip(0.06 + noise, 0, 1)

        # ── vessel silhouette ─────────────────────────────────────────────────
        plate  = np.clip(plate + vessel_layer, 0, 1)

        # convert to BGR
        frame  = (plate * 255).astype(np.uint8)
        frame  = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # ── catheter tube (bright white) ─────────────────────────────────────
        visible = path_2d[:step+1].astype(np.int32)
        if len(visible) >= 2:
            for i in range(1, len(visible)):
                frac_along = i / len(visible)
                bright = int(180 + 75 * frac_along)
                cv2.line(frame,
                         tuple(visible[i-1]), tuple(visible[i]),
                         (bright, bright, bright), 2, cv2.LINE_AA)

        # ── glowing tip ───────────────────────────────────────────────────────
        tip_xy = tuple(path_2d[step].astype(int))
        for r, a in [(9, 40), (5, 100), (3, 200)]:
            overlay = frame.copy()
            cv2.circle(overlay, tip_xy, r, (255, 255, 255), -1)
            cv2.addWeighted(overlay, a/255.0, frame, 1 - a/255.0, 0, frame)

        # ── insertion marker (green cross) ────────────────────────────────────
        cv2.drawMarker(frame, tuple(ins_2d), (0, 200, 100), cv2.MARKER_CROSS, 12, 1)

        # ── target marker (red diamond) ───────────────────────────────────────
        cv2.drawMarker(frame, tuple(tgt_2d), (60, 60, 255), cv2.MARKER_DIAMOND, 14, 2)

        # ── HUD ───────────────────────────────────────────────────────────────
        stage_name, stage_color_hex = _classify_stage(step, total, path_3d, insertion, target)
        stage_bgr = tuple(int(stage_color_hex.lstrip("#")[i:i+2], 16) for i in (4,2,0))

        tip_world = path_3d[step]
        dist_mm   = float(np.linalg.norm(tip_world - target))
        progress  = step / max(total - 1, 1)

        # top-left: stage + distance
        cv2.rectangle(frame, (8,8), (220, 58), (0,0,0), -1)
        cv2.rectangle(frame, (8,8), (220, 58), stage_bgr, 1)
        _put_text(frame, stage_name, (14, 28), color=stage_bgr, scale=0.52, thickness=1)
        _put_text(frame, f"Dist to LCCA: {dist_mm:.1f} mm", (14, 48), scale=0.38)

        # top-right: frame / total
        _put_text(frame, f"Frame {frame_idx+1}/{len(steps)}", (canvas-130, 24), scale=0.38)

        # bottom-right: tip coords
        coord_str = f"({tip_world[0]:.0f},{tip_world[1]:.0f},{tip_world[2]:.0f})"
        _put_text(frame, coord_str, (canvas-170, canvas-30), scale=0.36)

        # labels
        _put_text(frame, "INS", (ins_2d[0]+5, ins_2d[1]-5), color=(0,200,100), scale=0.32)
        _put_text(frame, "LCCA", (tgt_2d[0]+5, tgt_2d[1]-5), color=(60,60,255), scale=0.32)

        # bottom bar: progress
        _progress_bar(frame, progress, canvas)

        # ── write frame ───────────────────────────────────────────────────────
        writer.append_data(frame[:, :, ::-1])  # BGR→RGB for imageio

        annotations.append({
            "frame_idx": frame_idx,
            "path_step": int(step),
            "progress":  float(round(progress, 4)),
            "stage":     stage_name,
            "tip_world": tip_world.tolist(),
            "tip_2d":    [int(tip_xy[0]), int(tip_xy[1])],
            "dist_to_target_mm": float(round(dist_mm, 2)),
        })

    # hold on final frame
    for _ in range(extra_hold_frames):
        writer.append_data(frame[:, :, ::-1])

    writer.close()
    print(f"[synth_dsa] wrote {output_path}  ({len(steps)} frames @ {fps}fps)")

    # ── 6. save annotations ───────────────────────────────────────────────────
    ann_path = output_path.with_suffix(".json")
    ann_path.write_text(json.dumps({
        "video":       str(output_path),
        "fps":         fps,
        "canvas":      canvas,
        "lao_deg":     lao,
        "caudal_deg":  caudal,
        "n_frames":    len(annotations),
        "nav": {
            "insertion_position": insertion.tolist(),
            "target_lcca":        target.tolist(),
        },
        "stages": {
            "INSERTING":       "catheter entering vessel at groin",
            "DESCENDING AORTA":"traversing descending thoracic aorta",
            "AORTIC ARCH":     "navigating the aortic arch (highest risk zone)",
            "NAVIGATING":      "approaching LCCA ostium",
            "ARRIVED [OK]":    "tip within threshold of LCCA target",
        },
        "frames": annotations,
    }, indent=2))
    print(f"[synth_dsa] annotations → {ann_path}")
    return str(output_path), str(ann_path)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case",    default="KiTS_K1",
                    help="case folder name under vascular-v2/data/")
    ap.add_argument("--out",     default=None,
                    help="output mp4 path (default: data/<case>/synth_dsa.mp4)")
    ap.add_argument("--fps",     type=int,   default=25)
    ap.add_argument("--canvas",  type=int,   default=512)
    ap.add_argument("--lao",     type=float, default=30.0)
    ap.add_argument("--caudal",  type=float, default=10.0)
    ap.add_argument("--step",    type=int,   default=1,
                    help="path points per frame (higher = faster animation)")
    ap.add_argument("--spacing", type=float, default=2.0,
                    help="voxel spacing for path planning (mm)")
    args = ap.parse_args()

    DATA_ROOT = Path(__file__).parent.parent / "data"
    mesh_dir  = DATA_ROOT / args.case / "meshes"
    out_path  = Path(args.out) if args.out else DATA_ROOT / args.case / "synth_dsa.mp4"

    generate_dsa_video(
        mesh_dir   = mesh_dir,
        output_path= out_path,
        fps        = args.fps,
        canvas     = args.canvas,
        lao        = args.lao,
        caudal     = args.caudal,
        catheter_step = args.step,
        noise_sigma   = 2.5,
    )
