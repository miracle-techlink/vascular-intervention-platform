"""
Demo recorder for Experiment 7 (TD3 hierarchical RL: nav + fine-positioning).

Runs N tries, picks the best successful episode, saves:
  - demo_exp7.gif   (animated)
  - demo_exp7.mp4   (video)
  - demo_exp7_final.png

Usage:
    cd /home/liuyue/Research/血管介入手术机器人/stEVE
    CUDA_VISIBLE_DEVICES="" \\
    SOFA_ROOT=... PYTHONPATH=... LD_LIBRARY_PATH=... \\
    /home/liuyue/miniconda3/envs/sofa/bin/python3.8 examples/demo_exp7.py \\
        --fine_model models_exp7/run2/E7_best \\
        --nav_model  models_exp5/E5-S4-7mm_final \\
        --tries 30 --seed 42
"""

import os, argparse, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import imageio

import eve
from stable_baselines3 import TD3, SAC
import gymnasium as gym

# ── constants ────────────────────────────────────────────────────────────────
LCCA_ENTRY_DIST = 25.0
NAV_MAX_STEPS   = 160
NAV_MAX_RETRY   = 5


class _NullInterimTarget:
    @property
    def coordinates3d(self): return None
    def step(self): pass
    def reset(self, *a, **kw): pass


# ── env (identical to train_exp7.py) ─────────────────────────────────────────
def make_env(seed: int = 42):
    vessel_tree = eve.intervention.vesseltree.AorticArch(
        seed=seed, scaling_xyzd=[1.0, 1.0, 1.0, 0.75])
    device      = eve.intervention.device.JShaped()
    simulation  = eve.intervention.simulation.SofaBeamAdapter(friction=0.01)
    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation, vessel_tree=vessel_tree,
        image_frequency=7.5, image_rot_zx=[0, 0])
    target = eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree, fluoroscopy=fluoroscopy,
        threshold=5.0, branches=["lcca"])
    intervention = eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree, devices=[device], simulation=simulation,
        fluoroscopy=fluoroscopy, target=target, normalize_action=True)
    pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)
    state  = eve.observation.TipStateWithGoal(intervention=intervention)
    reward = eve.reward.Combination([
        eve.reward.TargetReached(intervention=intervention, factor=100.0),
        eve.reward.TipToTargetDistDelta(
            factor=1.0, intervention=intervention,
            interim_target=_NullInterimTarget()),
        eve.reward.FailurePenalty(intervention=intervention, factor=-50.0),
    ])
    terminal   = eve.terminal.TargetReached(intervention=intervention)
    truncation = eve.truncation.Combination([
        eve.truncation.MaxSteps(200),
        eve.truncation.VesselEnd(intervention=intervention),
        eve.truncation.SimError(intervention=intervention),
    ])
    base = eve.Env(intervention=intervention, observation=state,
                   reward=reward, terminal=terminal,
                   truncation=truncation, pathfinder=pathfinder)

    class FlatActionEnv(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            orig = env.action_space
            if len(orig.shape) > 1:
                self.action_space = gym.spaces.Box(
                    low=orig.low.flatten(), high=orig.high.flatten(), dtype=orig.dtype)
                self._act_shape = orig.shape
            else:
                self._act_shape = None
        def step(self, action):
            if self._act_shape is not None:
                action = np.array(action).reshape(self._act_shape)
            return self.env.step(action)

    env = FlatActionEnv(base)
    return env, intervention, pathfinder


def env_step(env, action):
    return env.step(action)


# ── branch colors ─────────────────────────────────────────────────────────────
BRANCH_COLORS = {
    "aorta":      "#8B8B8B",
    "brachio":    "#A0A0A0",
    "lcca":       "#FF6B6B",
    "lsa":        "#A0A0A0",
    "rcca":       "#B0B0B0",
    "rsa":        "#B0B0B0",
    "innominate": "#A0A0A0",
}


# ── frame renderer ────────────────────────────────────────────────────────────
def render_frame(ax, intervention, pathfinder, nav_trail, fine_trail,
                 phase, step, ep_reward, terminated, target_2d, nav_entry_dist):
    ax.clear()
    ax.set_facecolor("#0D0D0D")

    vessel_tree = intervention.vessel_tree
    fluoro      = intervention.fluoroscopy

    # vessel branches
    for branch in vessel_tree.branches:
        coords = branch.coordinates
        x, y = coords[:, 0], coords[:, 1]
        name  = branch.name.lower()
        color = next((v for k, v in BRANCH_COLORS.items() if k in name), "#888888")
        lw    = 4 if "aorta" in name else 2.5
        alpha = 0.75 if "lcca" in name else 0.45
        ax.plot(x, y, color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")
        mid = len(coords) // 2
        ax.text(x[mid], y[mid], branch.name, fontsize=6,
                color="#CCCCCC", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="#0D0D0D", ec="none", alpha=0.6))

    # pathfinder path
    if pathfinder.path_points3d is not None and len(pathfinder.path_points3d) > 1:
        pp = pathfinder.path_points3d
        ax.plot(pp[:, 0], pp[:, 1], color="#FFD700", linewidth=1.0,
                alpha=0.4, linestyle="--")

    # nav trail (gray dashed)
    if len(nav_trail) > 1:
        tr = np.array(nav_trail)
        ax.plot(tr[:, 0], tr[:, 1], color="#888888",
                linewidth=0.8, alpha=0.35, linestyle=":")

    # fine-positioning trail (cyan)
    if len(fine_trail) > 1:
        tr = np.array(fine_trail)
        ax.plot(tr[:, 0], tr[:, 1], color="#00BFFF",
                linewidth=0.9, alpha=0.5)

    # guidewire body
    tracking = fluoro.tracking3d
    if len(tracking) > 1:
        ax.plot(tracking[:, 0], tracking[:, 1],
                color="#00CFFF", linewidth=2.5, alpha=0.95,
                solid_capstyle="round", zorder=5)
    tip = tracking[0]
    ax.plot(tip[0], tip[1], "o", color="#FFFFFF",
            markersize=7, zorder=6,
            markeredgecolor="#00CFFF", markeredgewidth=1.5)

    # target
    tx, ty = target_2d
    ax.add_patch(plt.Circle((tx, ty), 5, color="#FF4444",
                             fill=False, linewidth=2, linestyle="--", zorder=7))
    ax.plot(tx, ty, "*", color="#FF4444", markersize=14, zorder=8)
    ax.text(tx + 3, ty + 3, "LCCA\ntarget", fontsize=7,
            color="#FF8888", fontweight="bold")

    # phase + status overlay
    dist = float(np.linalg.norm(tracking[0] - intervention.target.coordinates3d))
    if terminated:
        status_txt = "✓  SUCCESS"
        status_col = "#00FF88"
    elif phase == "nav":
        status_txt = f"[NAV]  dist={dist:.0f}mm"
        status_col = "#FFAA00"
    else:
        status_txt = f"[FINE] step {step:>3d}/200"
        status_col = "#FFFFFF"

    # entry marker (dashed circle at 25mm from target)
    ax.add_patch(plt.Circle((tx, ty), 25, color="#FFAA00",
                             fill=False, linewidth=0.8, linestyle="--",
                             alpha=0.4, zorder=4))
    ax.text(tx + 18, ty + 20, "entry\n25mm", fontsize=5.5,
            color="#FFAA00", alpha=0.6, ha="center")

    info = (f"{status_txt}\n"
            f"Reward: {ep_reward:+.1f}\n"
            f"Dist:   {dist:.1f} mm")
    ax.text(0.02, 0.97, info, transform=ax.transAxes,
            fontsize=9, color=status_col, va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#000000",
                      ec=status_col, alpha=0.75, linewidth=1.2))

    # phase legend
    phase_col = "#FFAA00" if phase == "nav" else "#00BFFF"
    ax.text(0.98, 0.97, f"{'Navigation' if phase == 'nav' else 'Fine-Pos (TD3)'}",
            transform=ax.transAxes, fontsize=8, color=phase_col,
            ha="right", va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#000000",
                      ec=phase_col, alpha=0.7, linewidth=1.0))

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig_to_pil(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).copy()


# ── run one episode ───────────────────────────────────────────────────────────
def run_episode(env, intervention, pathfinder, nav_model, fine_model):
    obs, _ = env.reset()
    target_3d = intervention.target.coordinates3d
    target_2d = target_3d[:2]

    fig, ax = plt.subplots(figsize=(7, 9))
    fig.patch.set_facecolor("#0D0D0D")

    frames    = []
    nav_trail = []
    fine_trail= []
    ep_reward = 0.0
    nav_entry_dist = None

    # ── Phase 1: Navigation ───────────────────────────────────────────────
    entered = False
    for attempt in range(NAV_MAX_RETRY):
        if attempt > 0:
            obs, _ = env.reset()
            nav_trail.clear()
        for s in range(NAV_MAX_STEPS):
            tip = intervention.fluoroscopy.tracking3d[0]
            dist = float(np.linalg.norm(tip - target_3d))
            nav_trail.append(tip[:2].copy())

            # render every 4 nav steps
            if s % 4 == 0:
                render_frame(ax, intervention, pathfinder, nav_trail, fine_trail,
                             "nav", s, ep_reward, False, target_2d, None)
                frames.append(fig_to_pil(fig, dpi=90))

            if dist < LCCA_ENTRY_DIST:
                entered = True
                nav_entry_dist = dist
                break

            action, _ = nav_model.predict(obs, deterministic=False)
            obs, rew, term, trunc, _ = env_step(env, action)
            if term or trunc:
                break

        if entered:
            break

    if not entered:
        plt.close(fig)
        return False, 0, 0.0, []

    # ── Phase 2: Fine-positioning (TD3) ───────────────────────────────────
    done = False
    step = 0
    terminated = False
    while not done:
        tip = intervention.fluoroscopy.tracking3d[0]
        fine_trail.append(tip[:2].copy())

        action, _ = fine_model.predict(obs, deterministic=True)
        obs, rew, terminated, truncated, _ = env_step(env, action)
        ep_reward += rew
        step += 1
        done = terminated or truncated

        if step % 2 == 0 or done:
            render_frame(ax, intervention, pathfinder, nav_trail, fine_trail,
                         "fine", step, ep_reward, terminated, target_2d, nav_entry_dist)
            frames.append(fig_to_pil(fig, dpi=90))

    plt.close(fig)
    return terminated, step, ep_reward, frames


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine_model", required=True, help="TD3 fine model path (no .zip)")
    ap.add_argument("--nav_model",  default="models_exp5/E5-S4-7mm_final")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--tries",      type=int, default=30)
    ap.add_argument("--out_dir",    default="demo_output_exp7")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading nav model:  {args.nav_model}")
    nav_model = SAC.load(args.nav_model)
    print(f"Loading fine model: {args.fine_model}")
    fine_model = TD3.load(args.fine_model)

    env, intervention, pathfinder = make_env(seed=args.seed)

    best_frames, best_reward, best_steps = None, -9999, 0
    success_count = 0

    for trial in range(args.tries):
        print(f"  Trial {trial+1}/{args.tries} ...", end=" ", flush=True)
        success, steps, rew, frames = run_episode(
            env, intervention, pathfinder, nav_model, fine_model)
        print(f"{'SUCCESS' if success else 'fail':<8}  "
              f"steps={steps:>3d}  reward={rew:+.1f}  frames={len(frames)}")

        if success:
            success_count += 1
            if rew > best_reward:
                best_reward = rew
                best_steps  = steps
                best_frames = frames
        if success_count >= 3:
            break

    env.close()

    if best_frames is None:
        print("\n没有成功 episode，保存最后一次尝试。")
        best_frames = frames if frames else []

    if not best_frames:
        print("无帧可保存，退出。"); return

    print(f"\nBest: {best_steps} steps  reward={best_reward:.1f}  "
          f"success_rate={success_count}/{trial+1}")

    # GIF
    gif_path = os.path.join(args.out_dir, "demo_exp7.gif")
    best_frames[0].save(gif_path, save_all=True, append_images=best_frames[1:],
                        duration=100, loop=0, optimize=True)
    print(f"Saved GIF:  {gif_path}  ({len(best_frames)} frames)")

    # MP4
    mp4_path = os.path.join(args.out_dir, "demo_exp7.mp4")
    try:
        writer = imageio.get_writer(mp4_path, fps=12, codec="libx264",
                                    quality=8, macro_block_size=1)
        for f in best_frames:
            writer.append_data(np.array(f.convert("RGB")))
        writer.close()
        print(f"Saved MP4:  {mp4_path}")
    except Exception as e:
        print(f"MP4 failed ({e}), GIF only.")

    # final frame PNG
    png_path = os.path.join(args.out_dir, "demo_exp7_final.png")
    best_frames[-1].save(png_path)
    print(f"Saved PNG:  {png_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
