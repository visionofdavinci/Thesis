"""
Hyperparameter search for PPO Escape agent using Optuna + TPE.

Designed for Snellius (SLURM array jobs). Each SLURM task is one worker
that contributes `--n-trials` trials to a shared Optuna study stored in a
SQLite journal file on the shared filesystem.

Reference: Bergstra & Bengio (2012) — TPE.
           Andrychowicz et al. (2021) — PPO sensitivity analysis.
"""

import argparse
import json
import os
import sys
import numpy as np

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

#these live in the same directory as this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engines.train_ppo_escape import (
    RunningMeanStd,
    PBRSRewardShaper,
    CurriculumManager,
    EscapeEnvironment,
    make_engine,
)
from engines.ppo_policy import PPOAgent

optuna.logging.set_verbosity(optuna.logging.WARNING)


#search space

#parameters kept FIXED - based on strong literature priors / safety bounds
FIXED = {
    "gamma":         0.99,   # Schulman et al. standard
    "clip_epsilon":  0.2,    # PPO paper default
    "value_coeff":   0.5,    # standard
    "max_grad_norm": 0.5,    # standard
    "pbrs_clip":     5.0,    # safety bound, not sensitive
}

def suggest_params(trial: optuna.Trial) -> dict:
    """Define the 11-parameter search space for Optuna TPE."""
    return {
        #PPO algorithm
        "lr_actor":         trial.suggest_float("lr_actor",         1e-5,  5e-4,  log=True),
        "lr_critic":        trial.suggest_float("lr_critic",        1e-4,  5e-3,  log=True),
        "entropy_coeff":    trial.suggest_float("entropy_coeff",    3e-3,  5e-2,  log=True),
        "gae_lambda":       trial.suggest_float("gae_lambda",       0.90,  0.99),
        "n_epochs":         trial.suggest_int(  "n_epochs",         5,     15),
        "batch_size":       trial.suggest_categorical("batch_size", [32, 64, 128]),
        "steps_per_update": trial.suggest_categorical("steps_per_update", [1024, 2048, 4096]),
        #network
        "hidden_size":      trial.suggest_categorical("hidden_size", [64, 128]),
        #reward
        "progress_weight":  trial.suggest_float("progress_weight",  0.5,   3.0),
        "potential_scale":  trial.suggest_float("potential_scale",  0.3,   5.0),
        # exploration
        "init_log_std":     trial.suggest_float("init_log_std",    -1.2,   0.0),
    }


#single-trial training function

def train_trial(config: dict, n_episodes: int = 3000, seed: int = 0) -> dict:
    """
    Runs one full training with the given hyperparameter config.
    Returns evaluation metrics dict.
    """
    np.random.seed(seed)

    goal   = np.array([2.0, 2.0, 2.0])
    engine = make_engine(goal)

    shaper = PBRSRewardShaper(
        engine,
        gamma=FIXED["gamma"],
        potential_scale=config["potential_scale"],
        clip_bound=FIXED["pbrs_clip"],
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
        progress_weight=config["progress_weight"],
    )

    h = config["hidden_size"]
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        act_dim=3,
        lr_actor=config["lr_actor"],
        lr_critic=config["lr_critic"],
        gamma=FIXED["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_epsilon=FIXED["clip_epsilon"],
        entropy_coeff=config["entropy_coeff"],
        value_coeff=FIXED["value_coeff"],
        max_grad_norm=FIXED["max_grad_norm"],
        n_epochs=config["n_epochs"],
        batch_size=config["batch_size"],
        hidden_sizes=[h, h],
    )
    # set initial exploration level
    agent.policy.log_std[:] = config["init_log_std"]

    return_normaliser  = RunningMeanStd()
    steps_per_update   = config["steps_per_update"]

    goals_per_ep = []
    total_steps  = 0
    collisions   = 0

    for episode in range(n_episodes):
        meta = curriculum.configure_engine(engine, goal)
        env.chaser_idx        = meta["chaser_idx"]
        env.chaser_speed      = curriculum.chaser_speed
        env.max_episode_steps = curriculum.max_episode_steps
        env.ws_lo             = meta["ws_lo"]
        env.ws_hi             = meta["ws_hi"]
        engine.set_workspace(meta["ws_lo"], meta["ws_hi"])

        for o in engine.obstacles:
            o.position += np.random.randn(3) * 0.15

        obs      = env.reset(agent_pos=meta["start_pos"])
        info     = {}

        for _ in range(env.max_episode_steps):
            action, log_prob, value, pre_tanh = agent.select_action(obs)
            next_obs, reward, done, info      = env.step(action)
            agent.buffer.add(obs, action, pre_tanh, reward, value, log_prob, done)
            obs          = next_obs
            total_steps += 1

            if (total_steps % steps_per_update == 0
                    and len(agent.buffer.observations) >= agent.batch_size):
                _, _, last_val, _ = agent.select_action(obs)
                agent.buffer.compute_returns_and_advantages(agent, last_val)
                if agent.buffer.returns is not None:
                    return_normaliser.update(agent.buffer.returns)
                    agent.buffer.returns = return_normaliser.normalize(
                        agent.buffer.returns
                    )
                agent.update()

            if done:
                if info.get("collision", False):
                    collisions += 1
                break

        goals_per_ep.append(1 if info.get("goal_reached", False) else 0)
        curriculum.record_episode(
            info.get("goal_reached", False),
            info.get("collision",    False),
            info.get("d_goal",       99.0),
            info.get("survived",     False),
        )

    # flush remaining buffer
    if len(agent.buffer.observations) >= agent.batch_size:
        _, _, last_val, _ = agent.select_action(obs)
        agent.buffer.compute_returns_and_advantages(agent, last_val)
        if agent.buffer.returns is not None:
            return_normaliser.update(agent.buffer.returns)
            agent.buffer.returns = return_normaliser.normalize(agent.buffer.returns)
        agent.update()

    tail_n         = max(n_episodes // 5, 1)
    goal_rate_tail = float(np.mean(goals_per_ep[-tail_n:]))
    goal_rate_all  = float(np.mean(goals_per_ep))

    return {
        "goal_rate_tail":   goal_rate_tail,   # primary metric
        "goal_rate_all":    goal_rate_all,
        "final_stage":      curriculum.current_stage,
        "total_goals":      int(sum(goals_per_ep)),
        "total_collisions": collisions,
        "total_steps":      total_steps,
    }


#optuna objective factory

def make_objective(n_episodes: int, n_seeds: int = 2):
    """
    Returns an Optuna objective function closed over n_episodes and n_seeds.

    Each config is evaluated on `n_seeds` independent seeds and the scores
    are averaged. -> to reduce the impact of environment randomness on
    the Bayesian model.
    """
    def objective(trial: optuna.Trial) -> float:
        config = suggest_params(trial)
        scores      = []
        all_metrics = []

        for seed in range(n_seeds):
            try:
                m = train_trial(config, n_episodes=n_episodes, seed=seed)
            except Exception as exc:
                raise optuna.exceptions.TrialPruned(
                    f"Trial {trial.number} seed {seed} crashed: {exc}"
                )

            score = m["goal_rate_tail"] + 0.05 * (m["final_stage"] / 8.0)
            scores.append(score)
            all_metrics.append(m)

        mean_score       = float(np.mean(scores))
        mean_goal_rate   = float(np.mean([m["goal_rate_tail"] for m in all_metrics]))
        mean_final_stage = float(np.mean([m["final_stage"]    for m in all_metrics]))

        # store for --report later
        trial.set_user_attr("goal_rate_tail",   mean_goal_rate)
        trial.set_user_attr("final_stage",       mean_final_stage)
        trial.set_user_attr("config_json",       json.dumps(config, indent=2))

        return mean_score   #optuna direction="maximize"

    return objective


# entry point

def build_storage(storage_path: str):
    """Creates an NFS-safe JournalFileBackend (Optuna 4.x) from a path string.
    Uses JournalFileOpenLock - because I work in wondows.
    """
    backend = JournalFileBackend(storage_path, lock_obj=JournalFileOpenLock(storage_path))
    return JournalStorage(backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPO Escape Hyperparameter Search (Optuna + TPE)"
    )
    parser.add_argument("--study-name",   type=str, default="ppo_escape_search",
                        help="Name of the Optuna study.")
    parser.add_argument("--storage",      type=str, required=True,
                        help="Path to the Optuna journal log file. "
                             "Must be on a shared filesystem (home/project dir).")
    parser.add_argument("--n-trials",     type=int, default=5,
                        help="Number of trials this worker will run.")
    parser.add_argument("--n-episodes",   type=int, default=3000,
                        help="Training episodes per trial per seed.")
    parser.add_argument("--n-seeds",      type=int, default=2,
                        help="Random seeds per trial (averaged for robustness).")
    parser.add_argument("--create-study", action="store_true",
                        help="Create the study and exit. Run this once before "
                             "submitting the SLURM array.")
    parser.add_argument("--report",       action="store_true",
                        help="Print the best trial and top-10 results, then exit.")
    parser.add_argument("--seed-offset",  type=int, default=0,
                        help="Base seed for this worker. Set to "
                             "$SLURM_ARRAY_TASK_ID in the job script.")
    args = parser.parse_args()

    storage = build_storage(args.storage)

    # create-study mode 
    if args.create_study:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            storage=storage,
            sampler=TPESampler(seed=42),
            load_if_exists=True,
        )
        print(f"Study '{args.study_name}' ready.")
        print(f"  Storage  : {args.storage}")
        print(f"  Direction: maximize  (metric = goal_rate_tail + curriculum bonus)")
        print(f"\nFixed hyperparameters:")
        for k, v in FIXED.items():
            print(f"  {k:16s} = {v}")
        print(f"\nSearching over 11 parameters:")
        for name in ["lr_actor","lr_critic","entropy_coeff","gae_lambda","n_epochs",
                     "batch_size","steps_per_update","hidden_size",
                     "progress_weight","potential_scale","init_log_std"]:
            print(f"  {name}")
        sys.exit(0)

    #report mode
    if args.report:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"\nStudy: '{args.study_name}'")
        print(f"  Completed trials : {len(completed)} / {len(study.trials)}")

        if not completed:
            print("  No completed trials yet.")
            sys.exit(0)

        best = study.best_trial
        print(f"\nBest trial  : #{best.number}  score={best.value:.4f}")
        print(f"  goal_rate_tail : {best.user_attrs.get('goal_rate_tail', 'N/A'):.4f}")
        print(f"  final_stage    : {best.user_attrs.get('final_stage', 'N/A'):.2f}/8")
        print(f"\n  Hyperparameters:")
        for k, v in best.params.items():
            print(f"    {k:22s} = {v}")

        # top-10 sorted by score
        top = sorted(completed, key=lambda t: t.value, reverse=True)[:10]
        print(f"\nTop-10 trials:")
        print(f"  {'#':>4}  {'score':>7}  {'goal_rate':>9}  {'stage':>5}  lr_actor")
        for t in top:
            print(f"  {t.number:4d}  {t.value:7.4f}  "
                  f"{t.user_attrs.get('goal_rate_tail', 0):9.4f}  "
                  f"{t.user_attrs.get('final_stage', 0):5.1f}  "
                  f"{t.params.get('lr_actor', 0):.2e}")

        # save best config to JSON for easy re-use in train_ppo_escape.py
        best_config = {**FIXED, **best.params}
        with open("best_hparams.json", "w") as f:
            json.dump(best_config, f, indent=2)
        print(f"\n  Best config saved to best_hparams.json")
        sys.exit(0)

    # normal worker mode 
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        sampler=TPESampler(seed=42 + args.seed_offset),
        load_if_exists=True,
    )

    print(f"Worker (seed_offset={args.seed_offset}): starting {args.n_trials} trials "
          f"({args.n_episodes} ep × {args.n_seeds} seeds each).")

    objective = make_objective(n_episodes=args.n_episodes, n_seeds=args.n_seeds)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    n_done = len([t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Worker done. Study now has {n_done} completed trials.")
    if study.best_trial is not None:
        print(f"  Current best score : {study.best_value:.4f}")
        print(f"  Best params so far : {study.best_params}")