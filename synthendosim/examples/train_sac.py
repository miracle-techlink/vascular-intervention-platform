"""
SynthEndoSim — SAC training example (SOFA backend).

Usage:
    conda activate sofa
    export SOFA_ROOT=/opt/sofa/SOFA_v23.06.00_Linux
    export PYTHONPATH=$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH
    python train_sac.py --mesh data/aorta_K1.obj --steps 200000
"""
import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
import synthendosim as ses

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", required=True)
parser.add_argument("--target", nargs=3, type=float,
                    default=[-31.7, 70.3, 11.6])
parser.add_argument("--insertion", nargs=3, type=float,
                    default=[-23.0, -180.0, -15.0])
parser.add_argument("--steps", type=int, default=200_000)
parser.add_argument("--eval-seeds", nargs="+", type=int,
                    default=[1000, 1001, 1002, 1003, 1004])
parser.add_argument("--output", default="models_sac")
args = parser.parse_args()

# ── Training env ──────────────────────────────────────────────────────────────
train_env = ses.from_mesh(
    mesh=args.mesh,
    insertion_point=tuple(args.insertion),
    target_point=tuple(args.target),
    backend="sofa",
    max_steps=300,
)

check_env(train_env, warn=True)

# ── Eval env (held-out seeds) ─────────────────────────────────────────────────
eval_env = ses.from_mesh(
    mesh=args.mesh,
    insertion_point=tuple(args.insertion),
    target_point=tuple(args.target),
    backend="sofa",
    max_steps=300,
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=args.output,
    log_path=args.output,
    eval_freq=5_000,
    n_eval_episodes=len(args.eval_seeds),
    deterministic=True,
)

# ── Train ─────────────────────────────────────────────────────────────────────
model = SAC(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log=f"{args.output}/tb",
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
)

model.learn(total_timesteps=args.steps, callback=eval_cb)
model.save(f"{args.output}/final_model")
train_env.close()
eval_env.close()
print(f"Saved to {args.output}/")
