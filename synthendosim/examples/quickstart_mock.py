"""
SynthEndoSim — Quickstart (mock backend, no SOFA required).

Demonstrates the full API without SOFA installed.
Replace backend="mock" with backend="sofa" for real physics.
"""
import numpy as np
import synthendosim as ses

print(f"SynthEndoSim v{ses.__version__}")

# ── Build env with mock physics (no SOFA needed) ──────────────────────────────
env = ses.make_env(config_dict={
    "physics": {"backend": "mock"},
    "anatomy": {
        "insertion_point": [-23.0, -180.0, -15.0],
        "insertion_direction": [0.0, 1.0, 0.0],
        "target_point": [-31.7, 70.3, 11.6],
        "mesh_path": "",
        "name": "mock_aortic_arch",
    },
    "reward": {
        "manifold_distance_weight": 1.0,
        "target_reached_bonus": 10.0,
        "step_penalty": -0.01,
        "use_manifold": False,           # no centerline in mock mode
    },
    "obs": {
        "use_tip_3d": True,
        "use_path_remaining": True,
        "use_wire_mask": False,
    },
    "episode": {"max_steps": 50},
})

print(f"Action space : {env.action_space}")
print(f"Obs space    : {env.observation_space}")

# ── Run 3 episodes ─────────────────────────────────────────────────────────
for ep in range(3):
    obs, info = env.reset(seed=ep)
    total_reward = 0.0
    for t in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(
        f"Episode {ep+1}: steps={t+1:3d}  "
        f"total_reward={total_reward:7.3f}  "
        f"target_dist={info['target_distance_mm']:.1f}mm  "
        f"reached={info['target_reached']}"
    )

env.close()
print("Done.")
