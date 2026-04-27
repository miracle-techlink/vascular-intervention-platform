import os
# RTX 5090 (sm_120) not supported by stable PyTorch — must be set before any torch import
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import uuid
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pipeline.preprocess import preprocess
from pipeline.segment import segment
from pipeline.mesh import build_meshes
from pipeline.nav import compute_nav
from pipeline.infer import run_infer
from pipeline.path import compute_path

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DIST_DIR = Path(__file__).parent.parent / "frontend" / "dist"


def case_dir(case_id: str) -> Path:
    return DATA_DIR / case_id


# ── 1. Upload ────────────────────────────────────────────────────────────────

@app.post("/cases/upload")
async def upload(file: UploadFile = File(...)):
    case_id = str(uuid.uuid4())[:8]
    d = case_dir(case_id)
    d.mkdir()
    dest = d / file.filename
    dest.write_bytes(await file.read())
    (d / "status.json").write_text(json.dumps({"status": "uploaded", "file": file.filename}))
    return {"case_id": case_id}


# ── 2. Run pipeline (SSE) ────────────────────────────────────────────────────

@app.get("/cases/{case_id}/run")
async def run_pipeline(case_id: str):
    d = case_dir(case_id)
    if not d.exists():
        raise HTTPException(404, "case not found")

    async def generate():
        def emit(phase: str, msg: str):
            return f"data: {json.dumps({'phase': phase, 'msg': msg})}\n\n"

        try:
            # find uploaded file
            files = [f for f in d.iterdir() if f.suffix in (".nrrd", ".nii", ".gz", ".zip", ".dcm")]
            if not files:
                yield emit("error", "no input file found")
                return
            src = files[0]

            yield emit("preprocess", "resampling to 1.5mm isotropic...")
            nii = d / "input.nii.gz"
            await asyncio.to_thread(preprocess, src, nii)
            yield emit("preprocess", "done")

            yield emit("segment", "running TotalSegmentator...")
            seg_dir = d / "segmented"
            await asyncio.to_thread(segment, nii, seg_dir)
            yield emit("segment", "done")

            yield emit("mesh", "building vessel meshes...")
            mesh_dir = d / "meshes"
            vessels = await asyncio.to_thread(build_meshes, seg_dir, mesh_dir)
            yield emit("mesh", f"done: {vessels}")

            yield emit("nav", "computing navigation points...")
            nav = await asyncio.to_thread(compute_nav, seg_dir, mesh_dir)
            yield emit("nav", f"done: insertion={nav.get('insertion_position')}")

            (d / "status.json").write_text(json.dumps({"status": "ready", "vessels": vessels}))
            yield emit("ready", json.dumps({"vessels": vessels, "nav": nav}))

        except Exception as e:
            yield emit("error", str(e))

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── 3. Viewer data ───────────────────────────────────────────────────────────

@app.get("/cases/{case_id}/mesh/{vessel}")
def get_mesh(case_id: str, vessel: str):
    path = case_dir(case_id) / "meshes" / f"{vessel}.obj"
    if not path.exists():
        raise HTTPException(404, "mesh not found")
    return FileResponse(path, media_type="text/plain")


@app.get("/cases/{case_id}/nav")
def get_nav(case_id: str):
    path = case_dir(case_id) / "meshes" / "nav_points.json"
    if not path.exists():
        raise HTTPException(404, "nav not ready")
    return json.loads(path.read_text())


@app.get("/cases/{case_id}/planned_path")
async def get_planned_path(case_id: str, spacing: float = 2.0):
    mesh_dir = case_dir(case_id) / "meshes"
    if not (mesh_dir / "nav_points.json").exists():
        raise HTTPException(400, "pipeline not complete")
    # return cached if exists
    cached = mesh_dir / "planned_path.json"
    if cached.exists():
        return json.loads(cached.read_text())
    # compute (slow, run in thread)
    pts = await asyncio.to_thread(compute_path, mesh_dir, spacing)
    return {"path_points": pts, "n_points": len(pts)}


@app.get("/cases/{case_id}/status")
def get_status(case_id: str):
    path = case_dir(case_id) / "status.json"
    if not path.exists():
        raise HTTPException(404, "case not found")
    return json.loads(path.read_text())


@app.get("/cases")
def list_cases():
    cases = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        s = d / "status.json"
        status = json.loads(s.read_text()) if s.exists() else {}
        cases.append({"case_id": d.name, **status})
    return cases


# ── 4. Library (built-in AVT dataset) ────────────────────────────────────────

AVT_ROOT = Path("/home/liuyue/Research/血管介入手术机器人/data/AVT")

def _scan_library() -> list[dict]:
    entries = []
    for center, glob in [("KiTS", "KiTS/*/"), ("Rider", "Rider/*/")]:
        for case_path in sorted(AVT_ROOT.glob(glob)):
            nrrd = next(case_path.glob("*.nrrd"), None)
            if nrrd is None or ".seg" in nrrd.name:
                nrrd = next((f for f in case_path.glob("*.nrrd") if ".seg" not in f.name), None)
            if nrrd is None:
                continue
            name = f"{center}_{case_path.name}"
            # check if already built
            cached = DATA_DIR / name
            status = "ready" if (cached / "meshes" / "nav_points.json").exists() else "not_built"
            entries.append({"name": name, "center": center, "file": str(nrrd), "status": status})
    return entries


@app.get("/library")
def get_library():
    return _scan_library()


@app.post("/library/{name}/load")
def load_from_library(name: str):
    """Create a case dir pointing to AVT source file. Returns case_id (= name)."""
    entry = next((e for e in _scan_library() if e["name"] == name), None)
    if entry is None:
        raise HTTPException(404, "case not in library")

    d = DATA_DIR / name
    d.mkdir(exist_ok=True)

    # If meshes already built, return ready immediately
    if (d / "meshes" / "nav_points.json").exists():
        status = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
        return {"case_id": name, "cached": True, **status}

    # Symlink source file so /run endpoint can find it
    src = Path(entry["file"])
    link = d / src.name
    if not link.exists():
        link.symlink_to(src)
    (d / "status.json").write_text(json.dumps({"status": "uploaded", "file": src.name, "source": "library"}))
    return {"case_id": name, "cached": False}


# ── 5. SAC inference (SSE) ───────────────────────────────────────────────────

@app.get("/cases/{case_id}/infer")
async def infer(case_id: str, max_steps: int = 800,
                threshold_mm: float = 15.0, friction: float = 0.01):
    mesh_dir = case_dir(case_id) / "meshes"
    if not (mesh_dir / "nav_points.json").exists():
        raise HTTPException(400, "pipeline not complete")

    async def generate():
        loop  = asyncio.get_event_loop()
        queue = asyncio.Queue()

        # capture loop BEFORE entering thread — get_event_loop() in thread is unreliable
        def _worker():
            try:
                for frame in run_infer(mesh_dir, max_steps, threshold_mm, friction):
                    loop.call_soon_threadsafe(queue.put_nowait, frame)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"error": str(e)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _worker)

        while True:
            frame = await queue.get()
            if frame is None:
                break
            yield f"data: {json.dumps(frame)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── 5. Serve frontend (must be last) ─────────────────────────────────────────

if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        return FileResponse(DIST_DIR / "index.html")
