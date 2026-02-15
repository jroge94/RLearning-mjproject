"""Run RL experiments: train per (algo, env) and log metrics to CSV."""
import logging
from typing import List, Optional, Sequence

import ray

import config
import utils

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CSV columns (must match order of row data in _append_round_row)
CSV_HEADER = [
    "round_id",
    "duration",
    "episodes_this_iter",
    "learner_time",
    "actor_time",
    "eval_reward_max",
    "eval_reward_mean",
    "eval_reward_min",
    "learner_loss",
    "cost",
]


def _append_round_row(info: dict) -> List:
    """Build one CSV row from process_train_results info."""
    return [
        info["round_id"],
        info["duration"],
        info["episodes_this_iter"],
        info["learner_time"],
        info["actor_time"],
        info["eval_reward_max"],
        info["eval_reward_mean"],
        info["eval_reward_min"],
        info["learner_loss"],
        info["cost_per_round"],
    ]


def run_experiment(
    algo_name: str,
    env_name: str,
    max_rounds: Optional[int] = None,
) -> None:
    """Train one (algo, env) for max_rounds iterations and save CSV to logs/."""
    rounds = max_rounds if max_rounds is not None else config.stop_max_round
    csv_rows: List[List] = [CSV_HEADER]

    trainer_config = utils.init_trainer_config(
        algo_name=algo_name,
        env_name=env_name,
    )
    trainer = trainer_config.build()

    try:
        for round_id in range(1, rounds + 1):
            train_results = trainer.train()
            info = utils.process_train_results(
                algo_name=algo_name,
                round_id=round_id,
                train_results=train_results,
            )
            if info is None:
                continue

            csv_rows.append(_append_round_row(info))

            logger.info(
                "algo=%s env=%s round=%s duration=%.2f eval_reward_mean=%.2f cost=%.4f",
                algo_name,
                env_name,
                info["round_id"],
                info["duration"],
                info["eval_reward_mean"],
                info["cost_per_round"],
            )

        utils.export_csv(
            env_name=env_name,
            algo_name=algo_name,
            csv_suffix="",
            csv_file=csv_rows,
        )
    finally:
        trainer.stop()


def main(
    algos: Optional[Sequence[str]] = None,
    envs: Optional[Sequence[str]] = None,
    max_rounds: Optional[int] = None,
) -> None:
    """Run (algo, env) experiments with a single Ray session."""
    algo_list = list(algos) if algos is not None else config.get_algo_names()
    env_list = list(envs) if envs is not None else config.get_env_names()
    logger.info("Starting experiments (algos=%s, envs=%s)", algo_list, env_list)

    ray.init(
        log_to_driver=False,
        configure_logging=True,
        logging_level=logging.ERROR,
    )

    try:
        for algo_name in algo_list:
            for env_name in env_list:
                run_experiment(
                    algo_name=algo_name,
                    env_name=env_name,
                    max_rounds=max_rounds,
                )
    finally:
        ray.shutdown()

    logger.info("Experiments finished.")


if __name__ == "__main__":
    main()
