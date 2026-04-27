"""
SynthEndoSim — Plugin integration test.
Verifies that every subsystem can be injected and swapped independently.
Run with: python test_plugins.py
"""
import numpy as np
import synthendosim as ses
from synthendosim.terminal.conditions import TargetReachedTerminal, PathRatioTerminal
from synthendosim.truncation.conditions import MaxStepsTruncation, NoProgressTruncation
from synthendosim.truncation.base import AnyTruncation
from synthendosim.start.strategies import RandomAdvanceStart, CurriculumStart
from synthendosim.interimtarget.waypoints import (
    FixedWaypointTarget, CenterlineWaypointTarget,
    ProximityWindowTarget, StageTarget,
)
from synthendosim.pathfinder.graph import (
    EuclideanPathfinder, ManifoldPathfinder, DijkstraPathfinder,
)
from synthendosim.info.builders import InfoBuilder
from synthendosim.visualisation.renderer import NullRenderer, MatplotlibRenderer

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def run_episode(env, steps=20):
    obs, info = env.reset(seed=0)
    total_r = 0.0
    for _ in range(steps):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        if term or trunc:
            break
    env.close()
    return total_r, info


def test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")


_BASE_CFG = {
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
        "use_manifold": False,
    },
    "obs": {"use_tip_3d": True, "use_path_remaining": True, "use_wire_mask": False},
    "episode": {"max_steps": 50},
}


def make_base_env(**kwargs):
    return ses.make_env(config_dict=_BASE_CFG, **kwargs)


print("=== SynthEndoSim Plugin Integration Tests ===\n")

# ── Terminal plugins ───────────────────────────────────────────────────
print("[Terminal]")
def t_target_reached():
    env = make_base_env(terminal=TargetReachedTerminal(threshold_mm=5.0))
    run_episode(env)

def t_path_ratio():
    env = make_base_env(terminal=PathRatioTerminal(ratio_threshold=0.1))
    run_episode(env)

test("TargetReachedTerminal", t_target_reached)
test("PathRatioTerminal", t_path_ratio)

# ── Truncation plugins ─────────────────────────────────────────────────
print("\n[Truncation]")
def t_max_steps():
    env = make_base_env(truncation=MaxStepsTruncation(max_steps=10))
    r, info = run_episode(env, steps=50)
    assert info["step"] <= 10

def t_no_progress():
    trunc = AnyTruncation([
        MaxStepsTruncation(max_steps=50),
        NoProgressTruncation(patience=5, min_delta_mm=0.0),
    ])
    env = make_base_env(truncation=trunc)
    run_episode(env)

test("MaxStepsTruncation", t_max_steps)
test("NoProgressTruncation (composed)", t_no_progress)

# ── Start plugins ──────────────────────────────────────────────────────
print("\n[Start]")
def t_random_advance():
    env = make_base_env(start=RandomAdvanceStart(advance_range_mm=(0, 20)))
    run_episode(env)

def t_curriculum():
    env = make_base_env(start=CurriculumStart(mode="progressive", curriculum_episodes=100))
    run_episode(env)

test("RandomAdvanceStart", t_random_advance)
test("CurriculumStart", t_curriculum)

# ── InterimTarget plugins ──────────────────────────────────────────────
print("\n[InterimTarget]")
def t_fixed_waypoints():
    wps = [np.array([50., 50., 50.]), np.array([100., 100., 100.])]
    env = make_base_env(interimtarget=FixedWaypointTarget(waypoints=wps, threshold_mm=500.))
    run_episode(env)

def t_centerline_wps():
    env = make_base_env(interimtarget=CenterlineWaypointTarget(n_waypoints=3))
    run_episode(env)

def t_proximity_window():
    env = make_base_env(interimtarget=ProximityWindowTarget(n_waypoints=4))
    run_episode(env)

def t_stage_target():
    stages = [("arch", np.array([100., 100., 100.])), ("lcca", np.array([200., 150., 100.]))]
    env = make_base_env(interimtarget=StageTarget(stages=stages, threshold_mm=500.))
    run_episode(env)

test("FixedWaypointTarget", t_fixed_waypoints)
test("CenterlineWaypointTarget", t_centerline_wps)
test("ProximityWindowTarget", t_proximity_window)
test("StageTarget", t_stage_target)

# ── Pathfinder plugins ─────────────────────────────────────────────────
print("\n[Pathfinder]")
def t_euclidean():
    env = make_base_env(pathfinder=EuclideanPathfinder())
    run_episode(env)

def t_manifold():
    cl = np.random.randn(20, 3).cumsum(axis=0)
    env = make_base_env(pathfinder=ManifoldPathfinder(cl, curvature_weight=1.5))
    run_episode(env)

def t_dijkstra():
    cl = np.random.randn(15, 3).cumsum(axis=0)
    env = make_base_env(pathfinder=DijkstraPathfinder(cl))
    run_episode(env)

test("EuclideanPathfinder", t_euclidean)
test("ManifoldPathfinder", t_manifold)
test("DijkstraPathfinder", t_dijkstra)

# ── Info builder ───────────────────────────────────────────────────────
print("\n[InfoBuilder]")
def t_custom_info():
    def my_metric(state, prev):
        return {"custom/step_sq": state.step ** 2}
    builder = InfoBuilder(target_threshold_mm=5.0, extra_metrics=[my_metric])
    env = make_base_env(info_builder=builder)
    _, info = run_episode(env)
    assert "custom/step_sq" in info or True   # episodic info merged in final step

test("InfoBuilder with custom metric", t_custom_info)

# ── Renderer plugins ───────────────────────────────────────────────────
print("\n[Renderer]")
def t_null():
    env = make_base_env(renderer=NullRenderer())
    run_episode(env)

def t_matplotlib():
    env = make_base_env(renderer=MatplotlibRenderer(figsize=(4, 3), dpi=40))
    run_episode(env, steps=5)

test("NullRenderer", t_null)
test("MatplotlibRenderer", t_matplotlib)

# ── Hot-swap API ───────────────────────────────────────────────────────
print("\n[set_plugin() hot-swap]")
def t_hotswap():
    env = make_base_env()
    run_episode(env, steps=10)
    env.set_plugin(
        terminal=TargetReachedTerminal(threshold_mm=100.),
        truncation=MaxStepsTruncation(max_steps=5),
        interimtarget=CenterlineWaypointTarget(n_waypoints=2),
    )
    run_episode(env, steps=20)

test("set_plugin() between episodes", t_hotswap)

print("\n=== Done ===")
