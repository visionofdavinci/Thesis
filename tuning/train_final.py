"""
Train PPO escape agent with the best hyperparameters found on Snellius.
Loads best_hparams.json and runs a full training for a configurable number
of episodes, saving the final policy.
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engines.train_ppo_escape import (
    RunningMeanStd,
    PBRSRewardShaper,
    CurriculumManager,
    EscapeEnvironment,
    make_engine,
)
from engines.ppo_policy import PPOAgent


FIXED = {
    "gamma":         0.99,
    "clip_epsilon":  0.2,
    "value_coeff":   0.5,
    "max_grad_norm": 0.5,
    "pbrs_clip":     5.0,
}


def load_best_hparams(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def train_final(hparams: dict, n_episodes: int = 50000,
                save_dir: str = "ppo_escape_final", seed: int = 42,
                verbose: bool = True):
    np.random.seed(seed)

    goal = np.array([2.0, 2.0, 2.0])
    engine = make_engine(goal)

    shaper = PBRSRewardShaper(
        engine,
        gamma=hparams.get("gamma", FIXED["gamma"]),
        potential_scale=hparams["potential_scale"],
        clip_bound=hparams.get("pbrs_clip", FIXED["pbrs_clip"]),
    )

    curriculum = CurriculumManager(
        promotion_threshold=0.45,
        early_promotion_threshold=0.35,
        window_size=50,
        min_episodes_per_stage=60,
    )

    env = EscapeEnvironment(
        field_engine=engine,
        reward_shaper=shaper,
        max_escape_speed=1.5,
        max_episode_steps=200,
        chaser_idx=-1,
        chaser_speed=0.0,
        progress_weight=hparams["progress_weight"],
    )

    h = hparams["hidden_size"]
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        act_dim=3,
        lr_actor=hparams["lr_actor"],
        lr_critic=hparams["lr_critic"],
        gamma=hparams.get("gamma", FIXED["gamma"]),
        gae_lambda=hparams["gae_lambda"],
        clip_epsilon=hparams.get("clip_epsilon", FIXED["clip_epsilon"]),
        entropy_coeff=hparams["entropy_coeff"],
        value_coeff=hparams.get("value_coeff", FIXED["value_coeff"]),
        max_grad_norm=hparams.get("max_grad_norm", FIXED["max_grad_norm"]),
        n_epochs=hparams["n_epochs"],
        batch_size=hparams["batch_size"],
        hidden_sizes=[h, h],
    )
    agent.policy.log_std[:] = hparams["init_log_std"]

    return_normaliser = RunningMeanStd()
    steps_per_update = hparams["steps_per_update"]

    if verbose:
        print("=" * 65)
        print("PPO ESCAPE FINAL TRAINING — Best Hyperparameters")
        print("=" * 65)
        print(f"  episodes         : {n_episodes}")
        print(f"  seed             : {seed}")
        print(f"  hidden           : [{h}, {h}]")
        print(f"  lr_actor         : {hparams['lr_actor']:.6f}")
        print(f"  lr_critic        : {hparams['lr_critic']:.6f}")
        print(f"  entropy_coeff    : {hparams['entropy_coeff']:.6f}")
        print(f"  gae_lambda       : {hparams['gae_lambda']:.4f}")
        print(f"  n_epochs         : {hparams['n_epochs']}")
        print(f"  batch_size       : {hparams['batch_size']}")
        print(f"  steps_per_update : {steps_per_update}")
        print(f"  potential_scale  : {hparams['potential_scale']:.4f}")
        print(f"  progress_weight  : {hparams['progress_weight']:.4f}")
        print(f"  init_log_std     : {hparams['init_log_std']:.4f}")
        print(f"  save_dir         : {save_dir}")
        print()

    episode_rewards = []
    episode_goals = 0
    episode_collisions = 0
    total_steps = 0

    for episode in range(n_episodes):
        meta = curriculum.configure_engine(engine, goal)
        env.chaser_idx = meta["chaser_idx"]
        env.chaser_speed = curriculum.chaser_speed
        env.max_episode_steps = curriculum.max_episode_steps
        env.ws_lo = meta["ws_lo"]
        env.ws_hi = meta["ws_hi"]
        engine.set_workspace(meta["ws_lo"], meta["ws_hi"])

        for o in engine.obstacles:
            o.position += np.random.randn(3) * 0.15

        obs = env.reset(agent_pos=meta["start_pos"])
        ep_reward = 0.0
        info = {}

        for _ in range(env.max_episode_steps):
            action, log_prob, value, pre_tanh = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.buffer.add(obs, action, pre_tanh, reward, value, log_prob, done)

            obs = next_obs
            ep_reward += reward
            total_steps += 1

            if (total_steps % steps_per_update == 0
                    and len(agent.buffer.observations) >= agent.batch_size):
                _, _, last_val, _ = agent.select_action(obs)
                agent.buffer.compute_returns_and_advantages(agent, last_val)
                if agent.buffer.returns is not None:
                    return_normaliser.update(agent.buffer.returns)
                    agent.buffer.returns = return_normaliser.normalize(
                        agent.buffer.returns)
                agent.update()

            if done:
                if info.get("collision", False):
                    episode_collisions += 1
                if info.get("goal_reached", False):
                    episode_goals += 1
                break

        episode_rewards.append(ep_reward)

        promoted = curriculum.record_episode(
            info.get("goal_reached", False),
            info.get("collision", False),
            info.get("d_goal", 99.0),
            info.get("survived", False),
        )

        if verbose and (episode % 500 == 0 or promoted):
            recent_r = (np.mean(episode_rewards[-100:])
                        if len(episode_rewards) >= 100
                        else np.mean(episode_rewards))
            recent_results = curriculum.episode_results[-50:]
            success_rate = sum(recent_results) / max(len(recent_results), 1)

            print(f"  Ep {episode:6d}  r={ep_reward:+8.1f}  avg={recent_r:+8.1f}  "
                  f"goals={episode_goals}  coll={episode_collisions}  "
                  f"sr={success_rate:.2f}  stage={curriculum.stage_config_name}")
            if promoted:
                print(f"    >>> PROMOTED to {curriculum.stage_config_name}!")

    # flush remaining buffer
    if len(agent.buffer.observations) >= agent.batch_size:
        _, _, last_val, _ = agent.select_action(obs)
        agent.buffer.compute_returns_and_advantages(agent, last_val)
        if agent.buffer.returns is not None:
            return_normaliser.update(agent.buffer.returns)
            agent.buffer.returns = return_normaliser.normalize(agent.buffer.returns)
        agent.update()

    os.makedirs(save_dir, exist_ok=True)
    agent.save(save_dir)

    # also save the hparams used for reproducibility
    with open(os.path.join(save_dir, "hparams_used.json"), "w") as f:
        json.dump(hparams, f, indent=2)

    if verbose:
        print(f"\n  Done. {n_episodes} ep, {total_steps} steps.")
        print(f"  Goals: {episode_goals}/{n_episodes} "
              f"({100 * episode_goals / max(n_episodes, 1):.1f}%)")
        print(f"  Collisions: {episode_collisions}/{n_episodes} "
              f"({100 * episode_collisions / max(n_episodes, 1):.1f}%)")
        print(f"  Final stage: {curriculum.stage_config_name}")
        for sh in curriculum.stage_history:
            print(f"    {sh['from_stage']}->{sh['to_stage']} at ep "
                  f"{sh['episodes']} (rate={sh['success_rate']:.2f})")
        print(f"  Saved to {save_dir}/")

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO escape with best hyperparameters")
    parser.add_argument("--hparams", type=str, default="tuning/best_hparams.json",
                        help="Path to best_hparams.json")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--save-dir", type=str, default="ppo_escape_final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    hparams = load_best_hparams(args.hparams)
    train_final(
        hparams=hparams,
        n_episodes=args.episodes,
        save_dir=args.save_dir,
        seed=args.seed,
        verbose=not args.quiet,
    )
