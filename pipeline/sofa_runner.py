"""
Standalone script — runs inside miniforge3/envs/sofa Python.
Reads mesh_dir from argv, streams JSON per step to stdout.
Usage: sofa_python sofa_runner.py <mesh_dir> [max_steps] [threshold_mm] [friction]
"""
import sys, json, os
import numpy as np
from pathlib import Path

STEVE_DIR = Path("/home/liuyue/Research/血管介入手术机器人/stEVE")
SOFA_BASE = Path("/home/liuyue/Research/血管介入手术机器人/deps/sofa/SOFA_v23.06.00_Linux")
MODEL     = STEVE_DIR / "models_sac_curriculum/Stage3-5mm_final.zip"

sys.path.insert(0, str(STEVE_DIR))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# SOFA_ROOT and PYTHONPATH/LD_LIBRARY_PATH are set by parent process (infer.py)

mesh_dir      = Path(sys.argv[1])
max_steps     = int(sys.argv[2])   if len(sys.argv) > 2 else 800
threshold_mm  = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
friction      = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01

nav = json.loads((mesh_dir / "nav_points.json").read_text())
mesh_path   = str(mesh_dir / "vessel_combined.obj")
insert      = nav["insertion_position"]
direction   = nav["insertion_direction"]
lcca_target = np.array(nav["target_lcca"], dtype=np.float32)

import eve
from stable_baselines3 import SAC

vessel_tree = eve.intervention.vesseltree.FromMesh(
    mesh=mesh_path,
    insertion_position=tuple(insert),
    insertion_direction=tuple(direction),
    rotation_yzx_deg=(0.0, 0.0, 0.0),
)
device     = eve.intervention.device.JShaped()
simulation = eve.intervention.simulation.SofaBeamAdapter(friction=friction)
fluoroscopy= eve.intervention.fluoroscopy.TrackingOnly(
    simulation=simulation, vessel_tree=vessel_tree
)
target = eve.intervention.target.Manual(
    targets_vessel_cs=[lcca_target], threshold=threshold_mm, fluoroscopy=fluoroscopy
)
intervention = eve.intervention.MonoPlaneStatic(
    vessel_tree=vessel_tree, devices=[device],
    simulation=simulation, fluoroscopy=fluoroscopy, target=target,
)
env = eve.Env(
    intervention=intervention,
    observation=eve.observation.TipState(intervention=intervention),
    reward=eve.reward.TargetReached(intervention=intervention, factor=100.0),
    terminal=eve.terminal.TargetReached(intervention=intervention),
    truncation=eve.truncation.Combination([
        eve.truncation.MaxSteps(max_steps),
        eve.truncation.VesselEnd(intervention=intervention),
    ]),
)

model = SAC.load(str(MODEL), device="cpu",
                 custom_objects={"observation_space": env.observation_space,
                                 "action_space": env.action_space})

insert_np = np.array(insert, dtype=np.float32)

def norm_obs(o):
    """Shift tip to insertion-relative coords so values match training distribution."""
    n = o.copy()
    n[:3] = o[:3] - insert_np
    return n

obs, _ = env.reset()
path  = [obs[:3].tolist()]
total = 0.0

for step in range(max_steps):
    action, _ = model.predict(norm_obs(obs), deterministic=True)
    obs, rew, term, trunc, _ = env.step(action)
    tip = obs[:3].tolist()
    path.append(tip)
    total += float(rew)
    dist = float(np.linalg.norm(np.array(tip) - lcca_target))
    print(json.dumps({"step": step+1, "tip": tip, "reward": round(float(rew),3),
                      "dist_mm": round(dist,1), "done": False}), flush=True)
    if term or trunc:
        break

env.close()
final = np.array(path[-1])
print(json.dumps({
    "done": True, "final": True,
    "path_points": path,
    "total_reward": round(total, 2),
    "dist_to_lcca_mm": round(float(np.linalg.norm(final - lcca_target)), 1),
    "target_reached": float(np.linalg.norm(final - lcca_target)) < threshold_mm,
    "n_steps": len(path) - 1,
}), flush=True)
