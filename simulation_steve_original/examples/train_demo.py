"""
EndoSim Demo: Train SAC to navigate guidewire to target.
Headless training - no visualization.

Usage:
    python examples/train_demo.py --steps 100000 --target lcca
"""
import argparse
import os
import numpy as np
import eve
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from gymnasium.wrappers import FlattenObservation


class PrintCallback(BaseCallback):
    """Print training progress every N steps."""
    def __init__(self, print_freq=5000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self):
        if self.n_calls % self.print_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                mean_len = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
                print(f"  Step {self.n_calls}: mean_reward={mean_rew:.3f}, mean_ep_len={mean_len:.0f}")
        return True


def make_env(target_branch="lcca", seed=42):
    """Create a headless stEVE environment."""
    vessel_tree = eve.intervention.vesseltree.AorticArch(
        seed=seed,
        scaling_xyzd=[1.0, 1.0, 1.0, 0.75],
    )

    device = eve.intervention.device.JShaped()

    simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.01)

    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=7.5,
        image_rot_zx=[0, 0],
    )

    target = eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=5,
        branches=[target_branch],
    )

    intervention = eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
        normalize_action=True,
    )

    pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)

    # Observation: 2D tracking (normalized) + target + insertion length
    position = eve.observation.Tracking2D(intervention=intervention, n_points=5)
    position_norm = eve.observation.wrapper.NormalizeTracking2DEpisode(position, intervention)
    target_obs = eve.observation.Target2D(intervention=intervention)
    target_norm = eve.observation.wrapper.NormalizeTracking2DEpisode(target_obs, intervention)
    insertion = eve.observation.InsertionLengths(intervention=intervention)

    state = eve.observation.ObsTuple([position_norm, target_norm, insertion])

    # Reward
    target_reward = eve.reward.TargetReached(intervention=intervention, factor=10.0)
    path_delta = eve.reward.PathLengthDelta(pathfinder=pathfinder, factor=0.01)
    step_penalty = eve.reward.Step(factor=-0.01)
    reward = eve.reward.Combination([target_reward, path_delta, step_penalty])

    # Terminal / Truncation
    terminal = eve.terminal.TargetReached(intervention=intervention)
    max_steps = eve.truncation.MaxSteps(300)

    env = eve.Env(
        intervention=intervention,
        observation=state,
        reward=reward,
        terminal=terminal,
        truncation=max_steps,
        pathfinder=pathfinder,
    )

    return FlattenObservation(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--target", type=str, default="lcca",
                        choices=["lcca", "rcca", "lsa", "rsa", "bct"])
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"sac_{args.target}_{args.steps}")

    print(f"=== EndoSim Demo Training ===")
    print(f"Target: {args.target}")
    print(f"Steps: {args.steps}")
    print(f"Save to: {save_path}")

    env = make_env(target_branch=args.target, seed=args.seed)
    print(f"Obs space: {env.observation_space.shape}")
    print(f"Act space: {env.action_space.shape}")

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        verbose=0,
        tensorboard_log=os.path.join(args.save_dir, "tb_logs"),
    )

    print(f"\nTraining started...")
    model.learn(
        total_timesteps=args.steps,
        callback=PrintCallback(print_freq=5000),
        progress_bar=False,
    )

    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    # Quick evaluation
    print(f"\n=== Quick Evaluation (20 episodes) ===")
    successes = 0
    total_rewards = []
    for ep in range(20):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            ep_reward += rew
            done = terminated or truncated
        if terminated:
            successes += 1
        total_rewards.append(ep_reward)

    print(f"Success rate: {successes}/20 = {successes/20*100:.0f}%")
    print(f"Mean reward: {np.mean(total_rewards):.3f}")
    env.close()
    print("=== DONE ===")


if __name__ == "__main__":
    main()
