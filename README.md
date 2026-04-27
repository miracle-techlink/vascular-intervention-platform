# Vascular Intervention Planning Platform

CT scan → 3D vessel model → catheter path planning → SOFA physics simulation.

Web-based surgical planning tool for endovascular interventions (TAVI, TEVAR, carotid stenting). Upload a CT scan and get an interactive 3D vessel map with an optimal catheter insertion path in minutes.

---

## Pipeline

```
CT (.nrrd / .nii.gz)
  └─ preprocess.py      resample to 1.5 mm isotropic (SimpleITK)
  └─ segment.py         vessel label extraction (TotalSegmentator)
                        aorta, carotid L/R, brachiocephalic trunk, subclavian L/R
  └─ mesh.py            marching cubes → smoothed OBJ meshes (trimesh)
  └─ nav.py             compute insertion point (aorta bottom) + LCCA target
  └─ path.py            voxelized Dijkstra → spline smooth → planned_path.json
  └─ infer.py           SOFA BeamAdapter catheter simulation (SAC policy)
  └─ synth_dsa_v2.py    synthetic DSA fluoroscopy video generation
```

---

## Run

**Requirements:** conda envs `dsa3d` (API + pipeline) and `sofa` (SOFA simulation)

```bash
# Build frontend (first time only)
bash build.sh

# Start server
bash start.sh          # default port 8000
bash start.sh 8001     # custom port
```

Open **http://localhost:8000**

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/cases/upload` | Upload CT scan |
| `GET` | `/cases/{id}/run` | Run full pipeline (SSE stream) |
| `GET` | `/cases/{id}/mesh/{vessel}` | Fetch OBJ mesh |
| `GET` | `/cases/{id}/nav` | Insertion point + LCCA target |
| `GET` | `/cases/{id}/planned_path` | Dijkstra catheter path |
| `GET` | `/cases/{id}/infer` | SOFA simulation (SSE stream) |
| `GET` | `/library` | List built-in AVT dataset cases |
| `POST` | `/library/{name}/load` | Load a library case |

---

## Data

Built-in library uses the [AVT dataset](https://github.com/numisveinsson/BloodVesselML3D) (KiTS + Rider centers). Processed cases are cached under `data/`.

External dependencies:
- `stEVE/` — RL environment + SAC policy model (`models_sac_curriculum/Stage3-5mm_final.zip`)
- `deps/sofa/` — SOFA v23.06 Linux binary

---

## Stack

- **Backend:** FastAPI + uvicorn
- **Frontend:** React 18 + Three.js (`@react-three/fiber`)
- **Pipeline:** SimpleITK · TotalSegmentator · trimesh · PyVista · scikit-image · SOFA BeamAdapter
- **Simulation:** Stable-Baselines3 SAC policy
