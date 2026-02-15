"""RL training utilities: trainer config, result processing, CSV export."""
import csv
import os
from typing import Any, Dict, List, Optional

import numpy as np
import ray
from ray.rllib.algorithms import impala, pg, ppo
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

import config


def _get_learner_time_ms_impala(train_results: Dict[str, Any]) -> float:
    """Extract total learner time in seconds for Impala."""
    t = train_results["info"]["timing_breakdown"]
    return (
        t["learner_grad_time_ms"]
        + t["learner_load_time_ms"]
        + t["learner_load_wait_time_ms"]
        + t["learner_dequeue_time_ms"]
    ) / 1000


def _get_learner_loss(train_results: Dict[str, Any], algo_name: str) -> float:
    """Extract learner loss; returns 0 if key missing (e.g. Impala sometimes)."""
    try:
        learner = train_results["info"]["learner"]
        if DEFAULT_POLICY_ID not in learner:
            return 0.0
        stats = learner[DEFAULT_POLICY_ID]["learner_stats"]
        if algo_name == "pg":
            return float(stats["policy_loss"])
        return float(stats["total_loss"])
    except (KeyError, TypeError):
        return 0.0


def log_csv_path(env_name: str, algo_name: str, csv_suffix: str = "") -> str:
    """Path for experiment log CSV (same convention as export_csv and plot)."""
    return os.path.join(config.LOGS_DIR, f"{env_name}~{algo_name}~{csv_suffix}.csv")


def init_trainer_config(algo_name: str, env_name: str):
    """Build RLlib trainer config for the given algo and env."""
    num_rollout_workers = config.num_rollout_workers
    num_envs_per_worker = config.num_envs_per_worker
    rollout_fragment_length = config.envs[env_name]["rollout_fragment_length"]
    train_batch_size = (
        num_rollout_workers * num_envs_per_worker * rollout_fragment_length
    )

    if algo_name == "pg":
        trainer_config = pg.PGConfig()
    elif algo_name == "impala":
        trainer_config = impala.ImpalaConfig()
        trainer_config.estimate_batch_size = num_envs_per_worker * rollout_fragment_length
    elif algo_name == "ppo":
        trainer_config = ppo.PPOConfig()
        trainer_config.estimate_batch_size = num_envs_per_worker * rollout_fragment_length
    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    trainer_config = (
        trainer_config
        .framework(framework=config.framework)
        .environment(env=env_name)
        .resources(
            num_gpus=config.num_gpus_for_local_worker,
            num_cpus_for_local_worker=config.num_cpus_for_local_worker,
            num_cpus_per_worker=config.num_cpus_per_worker,
            num_gpus_per_worker=config.num_gpus_per_worker,
        )
        .rollouts(
            rollout_fragment_length=rollout_fragment_length,
            num_rollout_workers=num_rollout_workers,
            num_envs_per_worker=num_envs_per_worker,
            batch_mode="truncate_episodes",
        )
        .debugging(
            log_level="ERROR",
            logger_config={"type": ray.tune.logger.NoopLogger},
            log_sys_usage=False,
        )
        .reporting(min_time_s_per_iteration=config.min_time_s_per_iteration)
        .experimental(_enable_new_api_stack=config._enable_new_api_stack)
        .training(train_batch_size=train_batch_size)
        .evaluation(
            evaluation_interval=config.evaluation_interval,
            evaluation_num_workers=config.evaluation_num_workers,
            evaluation_duration=config.evaluation_num_workers,
        )
    )

    if algo_name == "ppo":
        trainer_config = trainer_config.training(
            sgd_minibatch_size=train_batch_size,
        )

    return trainer_config


def process_train_results(
    algo_name: str,
    round_id: int,
    train_results: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Extract metrics from one training iteration. Returns None if inputs invalid."""
    if round_id is None or train_results is None:
        return None

    if algo_name == "impala":
        learner_time = _get_learner_time_ms_impala(train_results)
    else:
        learner_time = train_results["timers"]["learn_time_ms"] / 1000

    actor_time = train_results["timers"]["sample_time_ms"] / 1000

    episode_reward = train_results["evaluation"]["hist_stats"]["episode_reward"]
    if len(episode_reward) == 0:
        eval_reward_max = eval_reward_mean = eval_reward_min = 0.0
    else:
        eval_reward_max = float(np.max(episode_reward))
        eval_reward_mean = float(np.mean(episode_reward))
        eval_reward_min = float(np.min(episode_reward))

    duration = train_results["time_this_iter_s"]
    cost_per_round = duration * (
        config.server_learner_per_s + config.server_actor_per_s
    )

    return {
        "round_id": round_id,
        "episodes_this_iter": train_results["episodes_this_iter"],
        "duration": duration,
        "learner_time": learner_time,
        "actor_time": actor_time,
        "eval_reward_max": eval_reward_max,
        "eval_reward_mean": eval_reward_mean,
        "eval_reward_min": eval_reward_min,
        "learner_loss": _get_learner_loss(train_results, algo_name),
        "episode_reward": episode_reward,
        "cost_per_round": cost_per_round,
    }


def export_csv(
    env_name: str,
    algo_name: str,
    csv_suffix: str,
    csv_file: List[List],
) -> None:
    """Write experiment log CSV to config.LOGS_DIR."""
    path = log_csv_path(env_name, algo_name, csv_suffix)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_file)
