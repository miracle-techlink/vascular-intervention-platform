"""
消融实验：多血管几何训练 vs 单血管训练
========================================
验证论文核心 Claim：
  1. 多解剖训练 → 更强跨患者泛化（held-out SR ↑）
  2. 在新血管上的收敛更快（sample efficiency ↑）

三组对比：
  --mode single   只在 seed=42 上训练（模拟 guide3d 单病人场景）
  --mode multi5   在 5 个 seed 上轮训
  --mode multi10  在 10 个 seed 上轮训

Eval：固定在 5 个从未见过的 held-out seeds 上评估 SR
     held-out = [1000, 1001, 1002, 1003, 1004]

输出：
  models_ablation/<mode>/
    ├── best.zip
    ├── ckpt_<step>.zip
    └── eval_log.csv   ← 每次eval记录 step, SR_train, SR_holdout

启动：
  source activate_sofa_5090.sh
  # 三组并行跑（不同mode）
  python examples/train_ablation_generalization.py --mode single   > logs/ablation_single.log 2>&1 &
  python examples/train_ablation_generalization.py --mode multi5   > logs/ablation_multi5.log 2>&1 &
  python examples/train_ablation_generalization.py --mode multi10  > logs/ablation_multi10.log 2>&1 &
"""

import os, csv, time, argparse, itertools
import numpy as np
import gymnasium
import eve
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


class _NullInterimTarget:
    @property
    def coordinates3d(self): return None
    def step(self): pass
    def reset(self, *a, **kw): pass

# ── 配置 ───────────────────────────────────────────────────────────────────
TOTAL_STEPS    = 200_000
EVAL_FREQ      = 5_000
EVAL_EPISODES  = 20
THRESHOLD_MM   = 10.0
NAV_WEIGHTS    = "models_sac_curriculum/Stage3-5mm_final.zip"

# 训练 seeds（每组使用前 N 个）
TRAIN_SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]

# Held-out seeds（从未出现在训练中）
HOLDOUT_SEEDS = [1000, 1001, 1002, 1003, 1004]

MODES = {
    "single":  TRAIN_SEEDS[:1],   # 只用 seed=42
    "multi5":  TRAIN_SEEDS[:5],
    "multi10": TRAIN_SEEDS[:10],
}


# ── 环境构建 ────────────────────────────────────────────────────────────────
def make_env(seed: int, threshold: float = THRESHOLD_MM):
    vessel_tree = eve.intervention.vesseltree.AorticArch(
        seed=seed, scaling_xyzd=[1.0, 1.0, 1.0, 0.75])
    device      = eve.intervention.device.JShaped()
    simulation  = eve.intervention.simulation.SofaBeamAdapter(friction=0.01)
    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation, vessel_tree=vessel_tree,
        image_frequency=7.5, image_rot_zx=[0, 0])
    target = eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree, fluoroscopy=fluoroscopy,
        threshold=threshold, branches=["lcca"])
    intervention = eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree, devices=[device],
        simulation=simulation, fluoroscopy=fluoroscopy,
        target=target, normalize_action=True)
    obs    = eve.observation.TipState(intervention=intervention)
    reward = eve.reward.Combination([
        eve.reward.TargetReached(intervention=intervention, factor=100.0),
        eve.reward.TipToTargetDistDelta(factor=1.0, intervention=intervention,
                                        interim_target=_NullInterimTarget()),
        eve.reward.FailurePenalty(intervention=intervention, factor=-50.0),
    ])
    terminal   = eve.terminal.TargetReached(intervention=intervention)
    truncation = eve.truncation.Combination([
        eve.truncation.MaxSteps(300),
        eve.truncation.VesselEnd(intervention=intervention),
        eve.truncation.SimError(intervention=intervention),
    ])
    return eve.Env(intervention=intervention, observation=obs,
                   reward=reward, terminal=terminal, truncation=truncation)


# ── 多血管训练 Wrapper ──────────────────────────────────────────────────────
class MultiAnatomyWrapper(gymnasium.Env):
    """
    episode 结束时自动切换到下一个 seed 的血管几何。
    对 SB3 来说行为和普通 Gymnasium Env 完全一样。
    """
    metadata = {}

    def __init__(self, seeds: list, threshold: float = THRESHOLD_MM):
        super().__init__()
        self.seeds    = seeds
        self._cycle   = itertools.cycle(seeds)
        self._cur_seed = next(self._cycle)
        self._env      = make_env(self._cur_seed, threshold)
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space
        self._ep_count = 0
        self._seed_counts = {s: 0 for s in seeds}

    def reset(self, **kwargs):
        self._cur_seed = next(self._cycle)
        self._seed_counts[self._cur_seed] += 1
        self._ep_count += 1
        if self._ep_count % 50 == 0:
            dist_str = "  ".join(f"s{s}:{n}" for s, n in self._seed_counts.items())
            print(f"  [MultiAnatomy] ep={self._ep_count}  {dist_str}", flush=True)
        self._env.close()
        self._env = make_env(self._cur_seed)
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space
        return self._env.reset(**kwargs)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        self._env.close()

    def render(self):
        pass

    @property
    def current_seed(self):
        return self._cur_seed


# ── 评估（多血管）──────────────────────────────────────────────────────────
def evaluate_on_seeds(model, seeds: list, n_ep: int = EVAL_EPISODES) -> dict:
    """
    在给定的每个 seed 上各跑 n_ep 次，返回每个 seed 的 SR 和平均 SR。
    """
    results = {}
    for seed in seeds:
        env = make_env(seed)
        ok  = 0
        for _ in range(n_ep):
            obs, _ = env.reset()
            done   = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
            if term:
                ok += 1
        env.close()
        results[seed] = ok / n_ep
    results["mean"] = float(np.mean(list(results.values())))
    return results


# ── Callback ────────────────────────────────────────────────────────────────
class AblationCallback(BaseCallback):
    def __init__(self, train_seeds, holdout_seeds, save_dir, csv_path):
        super().__init__(verbose=0)
        self.train_seeds   = train_seeds
        self.holdout_seeds = holdout_seeds
        self.save_dir      = save_dir
        self.csv_path      = csv_path
        self.best_sr_ho    = 0.0
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["step", "sr_train_mean"] + \
                     [f"sr_train_s{s}" for s in self.train_seeds[:5]] + \
                     ["sr_holdout_mean"] + \
                     [f"sr_holdout_s{s}" for s in self.holdout_seeds]
            w.writerow(header)

    def _on_step(self):
        if self.n_calls % EVAL_FREQ != 0:
            return True

        step = self.n_calls

        # 在训练 seeds（最多取前5个）上评估
        train_eval_seeds = self.train_seeds[:5]
        tr = evaluate_on_seeds(self.model, train_eval_seeds, n_ep=10)
        ho = evaluate_on_seeds(self.model, self.holdout_seeds, n_ep=10)

        # 保存最佳（按 held-out SR 排）
        if ho["mean"] > self.best_sr_ho:
            self.best_sr_ho = ho["mean"]
            self.model.save(os.path.join(self.save_dir, "best"))

        self.model.save(os.path.join(self.save_dir, f"ckpt_{step}"))

        # 写 CSV
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            row = [step, round(tr["mean"], 4)] + \
                  [round(tr.get(s, 0.0), 4) for s in train_eval_seeds] + \
                  [round(ho["mean"], 4)] + \
                  [round(ho.get(s, 0.0), 4) for s in self.holdout_seeds]
            w.writerow(row)

        msg = (f"[Ablation] step={step}\n"
               f"SR-train: {tr['mean']*100:.0f}%  "
               f"SR-holdout: {ho['mean']*100:.0f}%  "
               f"(best-ho: {self.best_sr_ho*100:.0f}%)")
        print(f"  {msg}", flush=True)
        feishu_notify(msg)
        return True


# ── Feishu ───────────────────────────────────────────────────────────────────
def feishu_notify(msg):
    try:
        import json, urllib.request
        td  = json.dumps({"app_id": "cli_a92716abb1f8dceb",
                          "app_secret": "PXN08uaAlsUhmMqM8VOfAhMVCniRYqGX"}).encode()
        req = urllib.request.Request(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            data=td, headers={"Content-Type": "application/json"})
        token = json.loads(urllib.request.urlopen(req).read()).get("tenant_access_token", "")
        md  = json.dumps({"receive_id": "oc_76e5fd8ef8133aeb835687f69fc90443",
                          "msg_type": "text",
                          "content": json.dumps({"text": msg})}).encode()
        req = urllib.request.Request(
            "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
            data=md, headers={"Authorization": f"Bearer {token}",
                               "Content-Type": "application/json"})
        urllib.request.urlopen(req)
    except Exception as e:
        print(f"  [feishu] {e}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi5", "multi10"],
                        default="multi5")
    args = parser.parse_args()

    train_seeds = MODES[args.mode]
    save_dir    = f"models_ablation/{args.mode}"
    csv_path    = f"{save_dir}/eval_log.csv"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("=" * 60)
    print(f"  消融实验：{args.mode}")
    print(f"  训练 seeds: {train_seeds}")
    print(f"  Held-out seeds: {HOLDOUT_SEEDS}")
    print(f"  总步数: {TOTAL_STEPS // 1000}k  eval 每 {EVAL_FREQ // 1000}k 步")
    print("=" * 60)

    # 训练环境
    if args.mode == "single":
        train_env = make_env(train_seeds[0])
    else:
        train_env = MultiAnatomyWrapper(train_seeds)

    # SAC 模型
    model = SAC(
        "MlpPolicy", train_env,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=2000,
        verbose=0,
        device="cpu",
    )

    # 加载预训练权重（如果有，加速收敛）
    if os.path.exists(NAV_WEIGHTS):
        pretrained = SAC.load(NAV_WEIGHTS, device="cpu")
        model.policy.actor.load_state_dict(
            pretrained.policy.actor.state_dict(), strict=False)
        print(f"  预训练权重已加载: {NAV_WEIGHTS}")

    cb = AblationCallback(
        train_seeds=train_seeds,
        holdout_seeds=HOLDOUT_SEEDS,
        save_dir=save_dir,
        csv_path=csv_path,
    )

    feishu_notify(
        f"[Ablation] {args.mode} 开始\n"
        f"train_seeds={train_seeds}\n"
        f"holdout={HOLDOUT_SEEDS}\n"
        f"总步数={TOTAL_STEPS // 1000}k"
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb,
                reset_num_timesteps=True, progress_bar=False)

    # 最终全量评估
    print("\n  最终 held-out 全量评估（20 ep/seed）...", flush=True)
    final_tr = evaluate_on_seeds(model, train_seeds[:5], n_ep=EVAL_EPISODES)
    final_ho = evaluate_on_seeds(model, HOLDOUT_SEEDS,   n_ep=EVAL_EPISODES)

    summary = (
        f"[Ablation] {args.mode} 完成\n"
        f"SR-train:   {final_tr['mean']*100:.0f}%\n"
        f"SR-holdout: {final_ho['mean']*100:.0f}%  (泛化)\n"
        f"耗时: {(time.time()-t0)/3600:.1f}h\n"
        f"Holdout per-seed: " +
        "  ".join(f"s{s}:{final_ho[s]*100:.0f}%" for s in HOLDOUT_SEEDS)
    )
    print(f"\n{summary}", flush=True)
    feishu_notify(summary)

    model.save(f"{save_dir}/final")
    train_env.close()


if __name__ == "__main__":
    main()
