"""
Train a source policy for FrozenLake Task 1 using PPO with early stopping (deterministic reward ≥ 1.0).

For further experiments, one can:
1) fine-tune it on the optimal trajectory to keep the optimal actions
2) and on safety-critical states off the trajectory take only safe actions.
3) with 100 % multi-label accuracy requirement

Usage:
  python train_source_policy.py --seed 42
  python train_source_policy.py --seed 42 --cfg standard_4x4 --output-dir out/
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

os.environ.setdefault("MPLBACKEND", "Agg")

# ── path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent          # frozen_lake/
_EXP_DIR = _SCRIPT_DIR.parent                          # experiments/
_PROJECT_ROOT = _EXP_DIR.parent.parent                 # CertifiedContinualLearning/
_RL_DIR = _PROJECT_ROOT / "rl_project"
for p in (_EXP_DIR, str(_RL_DIR), str(_PROJECT_ROOT), str(_SCRIPT_DIR)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from frozenlake_utils import (
    create_frozenlake_safety_rashomon_dataset,
    get_all_unsafe_state_action_pairs,
    make_frozenlake_env,
    observation_to_position,
    one_hot_encode_state,
)
from rl_project.utils.ppo_utils import PPOConfig, ppo_train
from rl_project.utils.gymnasium_utils import plot_episode
from rl_project.experiments.frozen_lake.frozenlake_utils import finetune_policy

# ── constants ───────────────────────────────────────────────────────────────
N_ACTIONS = 4

# ── helpers ─────────────────────────────────────────────────────────────────
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_actor(obs_dim: int, hidden: int = 64) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, N_ACTIONS),
    )


def _make_critic(obs_dim: int, hidden: int = 64) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    )

def train_ppo(
    env_map: list[str],
    seed: int,
    total_steps: int,
    is_slippery: bool,
    hidden: int,
) -> tuple[torch.nn.Sequential, torch.nn.Sequential, dict]:
    env = make_frozenlake_env(env_map, task_num=0, is_slippery=is_slippery)
    obs_dim = env.observation_space.shape[0]  # num_states + 1 # type: ignore
    actor = _make_actor(obs_dim, hidden)
    critic = _make_critic(obs_dim, hidden)
    cfg = PPOConfig(
        seed=seed,
        total_timesteps=total_steps,
        eval_episodes=1, # NOTE: use one episode because env is deterministic
        rollout_steps=256,
        update_epochs=8,
        minibatch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        lr=3e-4,
        max_grad_norm=0.5,
        early_stop=True,
        early_stop_min_steps=0,
        early_stop_deterministic_total_reward_threshold=1.0,
        early_stop_deterministic_eval_episodes=1,
        device="cpu",
    )
    actor, critic, training_data = ppo_train( # type: ignore
        env=env, cfg=cfg,
        actor_warm_start=actor, critic_warm_start=critic,
        return_training_data=True,
    )
    env.close()
    return actor.cpu(), critic.cpu(), training_data # type: ignore

# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train an omnisafe source policy for FrozenLake Task 1.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", type=str, default="standard_4x4", help="Config key in configs.yaml.")
    parser.add_argument("--safety-finetuning", type=bool, default=True, help="Whether to do the safety fine-tuning step.")
    parser.add_argument("--total-steps", type=int, default=500_000, help="Max PPO timesteps.")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save outputs (default: outputs/<cfg>/<seed>/source).")
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Directory for plots (default: outputs/<cfg>/<seed>/plots).",
    )
    args = parser.parse_args()

    # ── load config ──
    with open(_SCRIPT_DIR / "configs.yaml") as f:
        all_cfgs = yaml.safe_load(f)
    if args.cfg not in all_cfgs:
        raise ValueError(f"Config '{args.cfg}' not in configs.yaml. Available: {list(all_cfgs)}")
    cfg = all_cfgs[args.cfg]
    env_map: list[str] = cfg["env1_map"]
    is_slippery: bool = bool(cfg.get("is_slippery", False))

    out_dir = Path(args.output_dir) if args.output_dir else _SCRIPT_DIR / "outputs" / args.cfg / str(args.seed) / "source"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
    else:
        plots_dir = (out_dir.parent / "plots") if out_dir.name == "source" else (out_dir / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    _set_seeds(args.seed)

    # ── Step 1: Train PPO on Task 1 ──
    print("=" * 60)
    print(f"Step 1 — PPO training  (seed={args.seed}, cfg={args.cfg})")
    print("=" * 60)
    actor, critic, training_data = train_ppo(env_map, args.seed, args.total_steps, is_slippery, args.hidden)

    # Quick verification
    env = make_frozenlake_env(env_map, task_num=0, is_slippery=is_slippery)
    obs, _ = env.reset(seed=args.seed)
    done, total_reward = False, 0.0
    while not done:
        with torch.no_grad():
            action = int(torch.argmax(actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)), 1).item())
        obs, r, term, trunc, _ = env.step(action)
        total_reward += r # type: ignore
        done = term or trunc
    env.close()
    print(f"  PPO deterministic total reward = {total_reward:.1f}")
    if total_reward < 1.0:
        raise RuntimeError("PPO did not learn the optimal policy (reward < 1.0).")
    
    # ── Step 2: Fine-tune for safety (OPTIONAL) ──
    if args.safety_finetuning:
        print("\n" + "=" * 60)
        print("Step 2 — Safety fine-tuning")
        print("=" * 60)
        print("  Finetuning source policy for safety …")
        safety_rashomon_dataset = create_frozenlake_safety_rashomon_dataset(
            make_frozenlake_env(env_map, task_num=0, is_slippery=is_slippery), task_flag=0.0,
            use_shield=True,
        )
        finetuning_result_dct = finetune_policy(
            policy=actor,
            dataset=safety_rashomon_dataset,
            env=make_frozenlake_env(env_map=env_map, task_num=0, is_slippery=is_slippery),
            overlap_mode="policy",
            required_accuracy=1.0,
        )
        actor = finetuning_result_dct['policy']
        if not finetuning_result_dct['reached_target']:
            raise ValueError(
                f"Safety finetuning did not reach required accuracy. "
                f"Final accuracy: {finetuning_result_dct['final_accuracy']:.4f}, "
                f"required: {finetuning_result_dct['target_accuracy']:.4f}"
            )

    # Plot the learned policy
    render_env = make_frozenlake_env(
        env_map, task_num=0, is_slippery=is_slippery, render_mode='rgb_array'
    )
    plot_episode(
        env=render_env,
        actor=actor.cpu(),
        seed=args.seed,
        save_path=str(plots_dir / "source_policy_trajectory.png"),
        figsize_per_frame=(3.0, 3.0),
        title='Source Policy Trajectory (Task 1)'
    )
    render_env.close()
    plt.close("all")

    # # Plot policy arrows on top of the environment frame
    # plot_policy_arrows(actor, env_map, is_slippery, save_path=str(out_dir / "source_policy_arrows.png"))

    # ── Step 3: Save model ──
    print("\n" + "=" * 60)
    print("Step 4 — Saving neural policy")
    print("=" * 60)
    model_path = out_dir / "source_policy.pt"
    torch.save(actor.state_dict(), model_path)
    print(f"  Saved model → {model_path}")
    print("\nDone.")

    # (Optional) Save critic as well, for potential future use in downstream fine-tuning.
    critic_path = out_dir / "source_critic.pt"
    torch.save(critic.state_dict(), critic_path)
    print(f"  Saved critic → {critic_path}")

    # Save training data (states, actions, etc.) for downstream use (e.g. EWC Fisher computation).
    training_data_path = out_dir / "source_training_data.pt"
    torch.save(training_data, training_data_path)
    print(f"  Saved training data → {training_data_path}")


if __name__ == "__main__":
    main()
