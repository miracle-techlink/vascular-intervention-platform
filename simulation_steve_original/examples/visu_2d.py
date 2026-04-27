"""
2D real-time visualization of guidewire navigation.
All coordinates unified to tracking2D space.
"""
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import eve
from eve.util.coordtransform import vessel_cs_to_tracking3d, tracking3d_to_2d
from gymnasium.wrappers import FlattenObservation
import argparse

def make_env(target_branch="lcca", seed=42):
    vessel_tree = eve.intervention.vesseltree.AorticArch(
        seed=seed, scaling_xyzd=[1.0, 1.0, 1.0, 0.75])
    device = eve.intervention.device.JShaped()
    simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.01)
    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation, vessel_tree=vessel_tree,
        image_frequency=7.5, image_rot_zx=[0, 0])
    target = eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree, fluoroscopy=fluoroscopy,
        threshold=5, branches=[target_branch])
    intervention = eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree, devices=[device],
        simulation=simulation, fluoroscopy=fluoroscopy,
        target=target, normalize_action=True)
    pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)
    position = eve.observation.Tracking2D(intervention=intervention, n_points=5)
    position_norm = eve.observation.wrapper.NormalizeTracking2DEpisode(position, intervention)
    target_obs = eve.observation.Target2D(intervention=intervention)
    target_norm = eve.observation.wrapper.NormalizeTracking2DEpisode(target_obs, intervention)
    insertion = eve.observation.InsertionLengths(intervention=intervention)
    state = eve.observation.ObsTuple([position_norm, target_norm, insertion])
    target_reward = eve.reward.TargetReached(intervention=intervention, factor=10.0)
    path_delta = eve.reward.PathLengthDelta(pathfinder=pathfinder, factor=0.01)
    step_penalty = eve.reward.Step(factor=-0.01)
    reward = eve.reward.Combination([target_reward, path_delta, step_penalty])
    terminal = eve.terminal.TargetReached(intervention=intervention)
    max_steps = eve.truncation.MaxSteps(300)
    env = eve.Env(
        intervention=intervention, observation=state,
        reward=reward, terminal=terminal, truncation=max_steps,
        pathfinder=pathfinder)
    return FlattenObservation(env), intervention, pathfinder

def vessel_to_2d(coords, fluoroscopy):
    """Convert vessel coordinates to 2D tracking coordinates."""
    coords_3d = vessel_cs_to_tracking3d(
        coords,
        fluoroscopy.image_rot_zx,
        fluoroscopy.image_center,
        fluoroscopy.field_of_view,
    )
    return tracking3d_to_2d(coords_3d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--target", type=str, default="lcca")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env, intervention, pathfinder = make_env(args.target)
    fluoro = intervention.fluoroscopy
    
    policy = None
    if args.model:
        from stable_baselines3 import SAC
        policy = SAC.load(args.model)
        title_prefix = f"SAC Model: {args.model}"
    else:
        title_prefix = "Random Policy"
        print("Using RANDOM policy")

    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(7, 9))

    for ep in range(args.episodes):
        obs, _ = env.reset()
        vessel_tree = intervention.vessel_tree

        # Convert all vessel centerlines to tracking 2D
        branch_lines_2d = {}
        for bn in vessel_tree.keys():
            coords = vessel_tree[bn].coordinates
            coords_2d = vessel_to_2d(coords, fluoro)
            branch_lines_2d[bn] = coords_2d

        # Target is already in tracking3d, convert to 2d
        tgt_2d = tracking3d_to_2d(intervention.target.coordinates3d)

        tip_trail_x, tip_trail_y = [], []
        ep_reward = 0
        step = 0
        done = False

        while not done:
            if policy:
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            obs, rew, terminated, truncated, info = env.step(action)
            ep_reward += rew
            step += 1
            done = terminated or truncated

            try:
                tracking_2d = intervention.fluoroscopy.tracking2d
                if len(tracking_2d) > 0:
                    tip = tracking_2d[-1]
                    tip_trail_x.append(tip[0])
                    tip_trail_y.append(tip[1])
                    
                    if step % 3 == 0 or done:
                        ax.clear()
                        # Vessel tree (now in same coord system!)
                        for bn, c2d in branch_lines_2d.items():
                            if len(c2d) > 0:
                                color = "limegreen" if bn == args.target else "silver"
                                lw = 4 if bn == args.target else 2.5
                                ax.plot(c2d[:, 0], c2d[:, 1], color=color, alpha=0.5, linewidth=lw)
                                mid = len(c2d) // 2
                                ax.annotate(bn.upper(), (c2d[mid, 0], c2d[mid, 1]),
                                           fontsize=8, color="dimgray", fontweight="bold")

                        # Target (red star + circle)
                        circle = Circle((tgt_2d[0], tgt_2d[1]), 5,
                                       fill=False, color="red", linewidth=2, linestyle="--")
                        ax.add_patch(circle)
                        ax.plot(tgt_2d[0], tgt_2d[1], "r*", markersize=15, label="Target")

                        # Guidewire (blue)
                        ax.plot(tracking_2d[:, 0], tracking_2d[:, 1],
                               "dodgerblue", linewidth=2.5, alpha=0.9, label="Guidewire")
                        ax.plot(tip[0], tip[1], "bo", markersize=10, zorder=5)

                        # Tip trail
                        if len(tip_trail_x) > 1:
                            ax.plot(tip_trail_x, tip_trail_y, "cyan", alpha=0.3, linewidth=0.8)

                        path_len = pathfinder.path_length
                        status = "SUCCESS!" if terminated else f"Step {step}/300"
                        ax.set_title(f"Ep {ep+1} | {status} | R={ep_reward:.2f} | Path={path_len:.1f}mm",
                                    fontsize=12)
                        ax.set_aspect("equal")
                        ax.legend(loc="upper right", fontsize=9)
                        ax.grid(True, alpha=0.2)
                        fig.suptitle(f"stEVE: {title_prefix} -> {args.target.upper()}", 
                                    fontsize=14, fontweight="bold")
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                        plt.pause(0.02)
            except Exception as e:
                if step <= 2:
                    print(f"Warning at step {step}: {e}")

        result = "SUCCESS" if terminated else "TIMEOUT"
        print(f"Ep {ep+1}: {result} in {step} steps, reward={ep_reward:.3f}")
        plt.pause(2.0)

    print("Done! Close the window to exit.")
    plt.ioff()
    plt.show()
    env.close()

if __name__ == "__main__":
    main()
