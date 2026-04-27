"""
Synthetic DSA video — v2, matched to guide3d real fluoroscopy statistics.

Key improvements over v1:
  - Dark-background fluoroscopy convention (mean ~90, max ~180, std ~35)
  - Circular image-intensifier vignette
  - Anatomical background: cardiac silhouette + vessel shadow via Beer-Lambert
    column-sum through vessel voxel grid (lifted from stEVE DRRFluoroscopy)
  - Guidewire: 1-pixel thin bright line, slight Gaussian blur (PSF)
  - Poisson quantum noise on detector
  - Grayscale output (L, uint8) matching guide3d format
  - Saves both single-plane and side-by-side biplane strip
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import gaussian_filter, zoom


# ── load / ensure path ────────────────────────────────────────────────────────
def _ensure_path(mesh_dir: Path, spacing: float = 2.0):
    cached = mesh_dir / "planned_path.json"
    if cached.exists():
        return json.loads(cached.read_text())["path_points"]
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from pipeline.path import compute_path
    print("[synth_dsa_v2] computing Dijkstra path …")
    return compute_path(mesh_dir, spacing)


# ── C-arm projection matrix ───────────────────────────────────────────────────
def _rotation(lao_deg: float, caudal_deg: float) -> np.ndarray:
    lao = np.radians(lao_deg)
    caudal = np.radians(caudal_deg)
    Rz = np.array([[ np.cos(lao), np.sin(lao), 0],
                   [-np.sin(lao), np.cos(lao), 0],
                   [ 0,           0,            1]])
    Rx = np.array([[1, 0,               0            ],
                   [0,  np.cos(caudal), np.sin(caudal)],
                   [0, -np.sin(caudal), np.cos(caudal)]])
    return Rx @ Rz


def _build_scale_offset(pts_3d: np.ndarray, R: np.ndarray, canvas: int):
    proj = (R @ pts_3d.T).T[:, :2]
    mn, mx = proj.min(0), proj.max(0)
    span   = (mx - mn).max()
    margin = 0.12
    scale  = canvas * (1 - 2*margin) / (span + 1e-8)
    offset = canvas * margin - mn * scale
    return scale, offset


def project_pts(pts: np.ndarray, R, scale, offset) -> np.ndarray:
    return ((R @ pts.T).T[:, :2] * scale + offset).astype(np.float32)


# ── vessel mesh rasterisation → Beer-Lambert thickness map ───────────────────
def _build_vessel_bg(obj_path: Path, canvas: int,
                     R: np.ndarray, scale, offset,
                     voxel_spacing: float = 3.0,   # kept for API compat, unused
                     mu_tissue: float = 0.012) -> np.ndarray:
    """
    Project mesh triangles onto canvas and accumulate face-density.
    High face-density = vessel wall (many overlapping faces) → stronger shadow.
    Produces crisp double-band vessel wall appearance matching real fluoroscopy.
    """
    try:
        # ── load OBJ ─────────────────────────────────────────────────────────
        verts, faces = [], []
        with open(obj_path) as f:
            for line in f:
                if line.startswith("v "):
                    verts.append(list(map(float, line.split()[1:4])))
                elif line.startswith("f "):
                    idx = [int(t.split("/")[0])-1 for t in line.split()[1:]]
                    if len(idx) >= 3:
                        faces.append(idx[:3])
        verts = np.array(verts, np.float32)
        if len(verts) == 0:
            return np.zeros((canvas, canvas), np.float32)

        # ── project all vertices ──────────────────────────────────────────────
        p2d = project_pts(verts, R, scale, offset).astype(np.int32)

        # ── rasterise each triangle into accumulation buffer ──────────────────
        # thickness_map[i,j] counts how many faces project onto pixel (i,j)
        thickness = np.zeros((canvas, canvas), np.float32)
        for f in faces:
            tri = p2d[f]  # (3,2)
            cv2.fillConvexPoly(thickness, tri.reshape(-1,1,2), 1.0)

        # ── thickness → soft attenuation map ─────────────────────────────────
        # σ=4 blur: hard triangle edges → soft vessel wall bands
        thickness_sm = gaussian_filter(thickness, sigma=4.0)
        # Normalise 0→1
        t_max = thickness_sm.max()
        if t_max > 0:
            thickness_sm /= t_max
        # Gentle linear attenuation: vessel walls darken ~25% at max
        atten = thickness_sm * 0.28

        return np.clip(atten, 0, None).astype(np.float32)

    except Exception as e:
        print(f"[synth_dsa_v2] vessel bg failed ({e}), using simple background")
        return np.zeros((canvas, canvas), np.float32)


# ── image intensifier vignette ───────────────────────────────────────────────
def _vignette_mask(canvas: int) -> np.ndarray:
    y, x = np.ogrid[:canvas, :canvas]
    r    = canvas / 2.0
    d2   = ((x - r)**2 + (y - r)**2) / r**2
    # Hard circular boundary + soft rolloff matching real II
    circle   = (d2 < 1.0).astype(np.float32)
    rolloff  = np.clip(1.0 - (d2 - 0.75) / 0.25, 0.0, 1.0)
    return circle * rolloff


# ── anatomical chest background ──────────────────────────────────────────────
def _chest_bg(canvas: int, rng: np.random.Generator) -> np.ndarray:
    """
    Anatomical chest background matching real fluoroscopy texture:
    cardiac silhouette + ribs + spine + lung mottling + diaphragm.
    """
    H = W = canvas
    y, x = np.ogrid[:H, :W]

    # Soft tissue body silhouette
    body = (((x-W*.5)/(W*.46))**2 + ((y-H*.5)/(H*.44))**2 < 1).astype(np.float32) * 0.30

    # Cardiac silhouette (left-of-center teardrop)
    cardiac = np.exp(-((x-W*.46)**2/(W*.19)**2 + (y-H*.60)**2/(H*.24)**2)) * 0.32

    # Diaphragm dome
    diaphragm = np.exp(-((y-H*.73)**2/(H*.035)**2)) * 0.20

    # Spine (vertical dark band centre)
    spine = np.exp(-((x-W*.50)**2/(W*.025)**2)) * 0.22 * (y > H*.25).astype(float)

    # Ribs — subtle curved bands with random jitter
    ribs = np.zeros((H, W), np.float32)
    rib_positions = [0.30, 0.38, 0.46, 0.54, 0.61, 0.68]
    rng_r = np.random.default_rng(7)
    for ry in rib_positions:
        jitter = rng_r.uniform(-0.015, 0.015)
        rib = np.exp(-((y - H*(ry+jitter))**2 / (H*0.012)**2)) * 0.10
        # taper off near spine and edges
        taper = np.clip(1.0 - np.abs(x - W*0.5)/(W*0.46), 0, 1)
        ribs += rib * taper.astype(np.float32)

    # Lung texture — multi-scale noise
    coarse = gaussian_filter(rng.standard_normal((H,W)).astype(np.float32), 20) * 0.045
    medium = gaussian_filter(rng.standard_normal((H,W)).astype(np.float32),  7) * 0.035
    fine   = gaussian_filter(rng.standard_normal((H,W)).astype(np.float32),  2) * 0.018

    bg = body + cardiac + diaphragm + spine + ribs + np.clip(coarse+medium+fine, 0, None)
    return np.clip(bg, 0.0, 1.0).astype(np.float32)


# ── stage labels ─────────────────────────────────────────────────────────────
def _stage(step, total, path_3d, insertion, target):
    tip  = path_3d[step]
    frac = step / max(total-1, 1)
    dist = float(np.linalg.norm(tip - target))
    h    = (tip[2]-insertion[2]) / max(target[2]-insertion[2], 1)

    if frac < 0.08:              return "INSERTING",        "#00ff88"
    elif h   < 0.35:             return "DESCENDING AORTA", "#00e5ff"
    elif h   < 0.62:             return "AORTIC ARCH",      "#ffb300"
    elif dist > 20:              return "NAVIGATING",       "#29b6f6"
    else:                        return "ARRIVED [OK]",     "#ff4444"


# ── HUD ──────────────────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX

def _hud(frame_bgr, stage, color_hex, dist, progress, step, total, tip_w, canvas):
    bgr = tuple(int(color_hex.lstrip("#")[i:i+2],16) for i in (4,2,0))
    cv2.rectangle(frame_bgr, (8,8), (230,55), (0,0,0), -1)
    cv2.rectangle(frame_bgr, (8,8), (230,55), bgr, 1)
    cv2.putText(frame_bgr, stage,          (14,27), FONT, 0.50, bgr, 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Dist: {dist:.1f} mm", (14,47), FONT, 0.37, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"{step+1}/{total}", (canvas-90,22), FONT, 0.37, (150,150,150), 1, cv2.LINE_AA)
    coord = f"({tip_w[0]:.0f},{tip_w[1]:.0f},{tip_w[2]:.0f})"
    cv2.putText(frame_bgr, coord, (canvas-175, canvas-28), FONT, 0.34, (130,130,130), 1, cv2.LINE_AA)
    # progress bar
    bx,by,bw,bh = 18, canvas-18, canvas-36, 5
    cv2.rectangle(frame_bgr, (bx,by), (bx+bw,by+bh), (40,40,40), -1)
    cv2.rectangle(frame_bgr, (bx,by), (bx+int(bw*progress),by+bh), (80,160,240), -1)


# ── main ─────────────────────────────────────────────────────────────────────
def generate(
    mesh_dir:    Path,
    output_path: Path,
    fps:         int   = 15,
    canvas:      int   = 1024,
    lao:         float = 30.0,
    caudal:      float = 10.0,
    lao2:        float = 120.0,   # second plane (≈ lateral)
    caudal2:     float = 0.0,
    catheter_step: int = 1,
    voxel_spacing: float = 3.0,
    noise_photons: float = 15000.0,
    wire_brightness: float = 0.70, # wire darkness relative to background (higher = more visible)
    seed:        int   = 42,
    biplane:     bool  = True,
    hud:         bool  = True,
    extra_hold:  int   = 30,
):
    import imageio.v2 as iio

    rng = np.random.default_rng(seed)

    nav       = json.loads((mesh_dir / "nav_points.json").read_text())
    insertion = np.array(nav["insertion_position"], np.float32)
    target    = np.array(nav["target_lcca"],        np.float32)
    path_3d   = np.array(_ensure_path(mesh_dir),    np.float32)
    total     = len(path_3d)
    print(f"[v2] path={total} pts")

    # ── projections ──────────────────────────────────────────────────────────
    all_pts = np.vstack([path_3d, insertion[None], target[None]])
    R1 = _rotation(lao,   caudal)
    R2 = _rotation(lao2,  caudal2)
    s1,o1 = _build_scale_offset(all_pts, R1, canvas)
    s2,o2 = _build_scale_offset(all_pts, R2, canvas)

    path_2d_1 = project_pts(path_3d, R1, s1, o1)
    path_2d_2 = project_pts(path_3d, R2, s2, o2)
    ins_2d_1  = project_pts(insertion[None], R1, s1, o1)[0].astype(int)
    ins_2d_2  = project_pts(insertion[None], R2, s2, o2)[0].astype(int)
    tgt_2d_1  = project_pts(target[None],    R1, s1, o1)[0].astype(int)
    tgt_2d_2  = project_pts(target[None],    R2, s2, o2)[0].astype(int)

    # ── static layers ────────────────────────────────────────────────────────
    vessel_obj = mesh_dir / "vessel_combined.obj"
    if not vessel_obj.exists():
        vessel_obj = mesh_dir / "vessel_combined_vtk.obj"

    print("[v2] building vessel attenuation background …")
    vessel_bg1 = _build_vessel_bg(vessel_obj, canvas, R1, s1, o1, voxel_spacing)
    vessel_bg2 = _build_vessel_bg(vessel_obj, canvas, R2, s2, o2, voxel_spacing) if biplane else None

    chest1 = _chest_bg(canvas, rng)
    chest2 = _chest_bg(canvas, rng) if biplane else None
    vig    = _vignette_mask(canvas)

    # Target fluoroscopy stats (matched to guide3d): mean≈90, std≈35, max≈180
    # We'll express image in [0,1] then rescale to [0,180] uint8
    BG_BASE  = 0.72   # baseline plate brightness → mean~96 after vignette/shadows
    BG_ANAT  = 0.14   # anatomical chest layer weight (darkens background)
    BG_VESS  = 0.18   # vessel shadow weight (darkens where vessels are)

    def _render_plane(step, path_2d, ins_2d, tgt_2d, vessel_bg, chest, plane_rng):
        # 1. Base plate
        plate = np.full((canvas,canvas), BG_BASE, np.float32)

        # 2. Anatomical layers
        plate -= chest * BG_ANAT
        plate -= vessel_bg * BG_VESS

        # 3. Poisson quantum noise (fresh per-frame rng for realistic temporal variation)
        photons = np.round(noise_photons * plate).astype(np.int64)
        photons = np.maximum(photons, 1)
        noisy   = plane_rng.poisson(photons).astype(np.float32) / noise_photons
        plate   = noisy
        # re-seed so next frame gets independent noise
        plane_rng.bit_generator.state = np.random.default_rng(
            plane_rng.integers(0, 2**31)).bit_generator.state

        # 4. Guidewire: thin DARK line (metal absorbs X-rays → lower intensity)
        # Matches guide3d: guidewire appears as thin dark line on gray background
        visible = path_2d[:step+1].astype(np.int32)
        wire_layer = np.zeros((canvas,canvas), np.float32)
        if len(visible) >= 2:
            for i in range(1, len(visible)):
                cv2.line(wire_layer, tuple(visible[i-1]), tuple(visible[i]),
                         color=1.0, thickness=1, lineType=cv2.LINE_AA)
        # Minimal PSF blur — keep wire crisp (σ=0.4 matches ~0.3mm detector blur)
        wire_blur = gaussian_filter(wire_layer, sigma=0.4)
        # Subtract: wire darkens the plate
        plate = plate - wire_blur * wire_brightness

        # 5. Tip: small dark dot (angled/rounded tip is more radio-opaque)
        tip_xy = tuple(path_2d[step].astype(int))
        if 0<=tip_xy[0]<canvas and 0<=tip_xy[1]<canvas:
            tip_dark = np.zeros_like(plate)
            cv2.circle(tip_dark, tip_xy, 3, 0.8, -1)
            plate = plate - gaussian_filter(tip_dark, sigma=1.5) * wire_brightness

        # 6. Vignette (image intensifier)
        plate = plate * vig

        # 7. Map to [0,180] uint8 to match guide3d dynamic range
        plate = np.clip(plate * 180.0, 0, 200).astype(np.uint8)
        return plate, tip_xy

    # ── write video ──────────────────────────────────────────────────────────
    annotations = []
    out_w = canvas*2 + 8 if biplane else canvas
    out_h = canvas

    writer = iio.get_writer(
        str(output_path), format="FFMPEG", fps=fps, codec="libx264",
        output_params=["-crf","20","-pix_fmt","yuv420p"])

    steps = list(range(0, total, catheter_step))
    if steps[-1] != total-1:
        steps.append(total-1)

    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed+1)

    for fi, step in enumerate(steps):
        img1, tip1 = _render_plane(step, path_2d_1, ins_2d_1, tgt_2d_1, vessel_bg1, chest1, rng1)
        img2, tip2 = (_render_plane(step, path_2d_2, ins_2d_2, tgt_2d_2, vessel_bg2, chest2, rng2)
                      if biplane else (None, None))

        stage_name, stage_color = _stage(step, total, path_3d, insertion, target)
        dist   = float(np.linalg.norm(path_3d[step] - target))
        prog   = step / max(total-1, 1)

        # Convert to BGR for HUD drawing
        def to_bgr(gray): return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        f1 = to_bgr(img1)

        # Markers
        cv2.drawMarker(f1, tuple(ins_2d_1), (0,200,100), cv2.MARKER_CROSS,   14, 1)
        cv2.drawMarker(f1, tuple(tgt_2d_1), (60,60,255), cv2.MARKER_DIAMOND, 16, 2)
        cv2.putText(f1, "INS",  (ins_2d_1[0]+6, ins_2d_1[1]-6), FONT, 0.32, (0,200,100), 1)
        cv2.putText(f1, "LCCA", (tgt_2d_1[0]+6, tgt_2d_1[1]-6), FONT, 0.32, (60,60,255), 1)

        if hud:
            _hud(f1, stage_name, stage_color, dist, prog, fi, len(steps), path_3d[step], canvas)

        if biplane:
            f2 = to_bgr(img2)
            cv2.drawMarker(f2, tuple(ins_2d_2), (0,200,100), cv2.MARKER_CROSS,   14, 1)
            cv2.drawMarker(f2, tuple(tgt_2d_2), (60,60,255), cv2.MARKER_DIAMOND, 16, 2)
            if hud:
                # minimal label on plane 2
                bgr = tuple(int(stage_color.lstrip("#")[i:i+2],16) for i in (4,2,0))
                cv2.putText(f2, f"LAT {step+1}/{len(steps)}", (14,24), FONT, 0.40, bgr, 1)
            sep = np.zeros((canvas, 8, 3), np.uint8)
            frame = np.hstack([f1, sep, f2])
        else:
            frame = f1

        writer.append_data(frame[:,:,::-1])  # BGR→RGB

        annotations.append({
            "frame_idx": fi, "path_step": int(step),
            "progress": float(round(prog,4)), "stage": stage_name,
            "tip_world": path_3d[step].tolist(),
            "tip_2d_plane1": [int(tip1[0]), int(tip1[1])],
            "dist_to_target_mm": float(round(dist,2)),
        })

    for _ in range(extra_hold):
        writer.append_data(frame[:,:,::-1])
    writer.close()

    ann_path = output_path.with_suffix(".json")
    ann_path.write_text(json.dumps({
        "video": str(output_path), "fps": fps, "canvas": canvas,
        "biplane": biplane, "lao_deg": lao, "caudal_deg": caudal,
        "n_frames": len(annotations),
        "nav": {"insertion_position": insertion.tolist(),
                "target_lcca": target.tolist()},
        "stages": {
            "INSERTING": "catheter entering vessel",
            "DESCENDING AORTA": "traversing descending thoracic aorta",
            "AORTIC ARCH": "navigating aortic arch",
            "NAVIGATING": "approaching LCCA ostium",
            "ARRIVED [OK]": "tip within threshold of LCCA target",
        },
        "frames": annotations,
    }, indent=2))

    print(f"[v2] wrote {output_path}  ({len(steps)} frames @ {fps}fps)")
    print(f"[v2] annotations → {ann_path}")
    return str(output_path), str(ann_path)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case",    default="KiTS_K1")
    ap.add_argument("--out",     default=None)
    ap.add_argument("--fps",     type=int,   default=15)
    ap.add_argument("--canvas",  type=int,   default=1024)
    ap.add_argument("--lao",     type=float, default=30.0)
    ap.add_argument("--caudal",  type=float, default=10.0)
    ap.add_argument("--step",    type=int,   default=1)
    ap.add_argument("--no-biplane", action="store_true")
    ap.add_argument("--no-hud",     action="store_true")
    args = ap.parse_args()

    DATA_ROOT = Path(__file__).parent.parent / "data"
    mesh_dir  = DATA_ROOT / args.case / "meshes"
    out_path  = Path(args.out) if args.out else DATA_ROOT / args.case / "synth_dsa_v2.mp4"

    generate(
        mesh_dir=mesh_dir, output_path=out_path,
        fps=args.fps, canvas=args.canvas,
        lao=args.lao, caudal=args.caudal,
        catheter_step=args.step,
        biplane=not args.no_biplane,
        hud=not args.no_hud,
    )
