"""
SynthEndoSim — Multi-anatomy generalisation ablation.

Trains 3 conditions:
  single   : one anatomy (K1)
  multi5   : 5 anatomies (K1-K5)
  multi10  : 10 anatomies (K1-K10)

Usage:
    python train_ablation.py --mesh-dir data/meshes/ --mode multi5 --steps 200000
"""
import argparse
import os
import glob
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import synthendosim as ses

parser = argparse.ArgumentParser()
parser.add_argument("--mesh-dir", required=True,
                    help="Directory with aorta_K1.obj ... aorta_K10.obj")
parser.add_argument("--mode", choices=["single", "multi5", "multi10"],
                    default="multi5")
parser.add_argument("--steps", type=int, default=200_000)
parser.add_argument("--insertion", nargs=3, type=float,
                    default=[-23.0, -180.0, -15.0])
parser.add_argument("--target", nargs=3, type=float,
                    default=[-31.7, 70.3, 11.6])
parser.add_argument("--output-dir", default="models_ablation")
parser.add_argument("--eval-seeds", nargs="+", type=int,
                    default=[1000, 1001, 1002, 1003, 1004])
args = parser.parse_args()

n_meshes = {"single": 1, "multi5": 5, "multi10": 10}[args.mode]
meshes = sorted(glob.glob(os.path.join(args.mesh_dir, "*.obj")))[:n_meshes]
if not meshes:
    raise FileNotFoundError(f"No OBJ meshes found in {args.mesh_dir}")

print(f"Mode: {args.mode}  |  Anatomies: {len(meshes)}")

out_dir = os.path.join(args.output_dir, args.mode)
os.makedirs(out_dir, exist_ok=True)

base_cfg = {
    "anatomy": {
        "insertion_point": args.insertion,
        "insertion_direction": [0.0, 1.0, 0.0],
        "target_point": args.target,
    },
    "episode": {"max_steps": 300},
    "physics": {"backend": "sofa"},
}

# ── Training env: rotate through meshes each episode via wrapper ───────────

class RotatingMeshWrapper:
    """Wraps SynthEndoEnv and cycles through mesh list on each reset."""
    def __init__(self, env, mesh_list):
        self._env = env
        self._meshes = mesh_list
        self._idx = 0
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        mesh = self._meshes[self._idx % len(self._meshes)]
        self._idx += 1
        return self._env.reset(options={"anatomy_mesh": mesh}, **kwargs)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        self._env.close()

    # SB3 compatibility
    def __getattr__(self, name):
        return getattr(self._env, name)


train_base = ses.make_env(config_dict=base_cfg)
train_env = RotatingMeshWrapper(train_base, meshes)

eval_base = ses.make_env(config_dict={**base_cfg,
                                       "anatomy": {**base_cfg["anatomy"],
                                                    "mesh_path": meshes[0]}})
eval_cb = EvalCallback(
    eval_base,
    best_model_save_path=out_dir,
    log_path=out_dir,
    eval_freq=5_000,
    n_eval_episodes=len(args.eval_seeds),
    deterministic=True,
)

model = SAC(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log=f"{out_dir}/tb",
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
)

model.learn(total_timesteps=args.steps, callback=eval_cb)
model.save(os.path.join(out_dir, "final_model"))
train_env.close()
eval_base.close()
print(f"Saved to {out_dir}/")
