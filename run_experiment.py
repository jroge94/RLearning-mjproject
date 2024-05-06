import numpy as np
import collections
import logging
import ray
import config
import utils


def run_experiment(
    algo_name,
    env_name,
):
    csv_round = [
        [
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
    ]

    # Init trainer
    trainer_config = utils.init_trainer_config(
        algo_name=algo_name,
        env_name=env_name,
    )
    trainer = trainer_config.build()

    # Start training
    round_id = 1
    while round_id <= config.stop_max_round:
        # Train one round
        train_results = trainer.train()
        # print(train_results)

        # Process results
        info = utils.process_train_results(
            algo_name=algo_name,
            round_id=round_id,
            train_results=train_results,
        )

        # Log as CSV
        csv_round.append(
            [
                round_id,
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
        )

        print("")
        print("******************")
        print("******************")
        print("******************")
        print("")
        print("Running algo {}, env {}".format(algo_name, env_name))
        print("round_id: {}".format(info["round_id"]))
        print("duration: {}".format(info["duration"]))
        print("eval_reward_mean: {}".format(info["eval_reward_mean"]))
        print("cost: {}".format(info["cost_per_round"]))

        round_id = round_id + 1

    # Export CSV
    utils.export_csv(
        env_name=env_name, 
        algo_name=algo_name, 
        csv_name="",
        csv_file=csv_round
    )

    trainer.stop()

    
if __name__ == '__main__':
    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")
    ray.init(
        log_to_driver=False,
        configure_logging=True,
        logging_level=logging.ERROR
    )

    for algo_name in config.algos:
        for env_name in config.envs.keys():
            run_experiment(
                algo_name=algo_name,
                env_name=env_name,
            )

    ray.shutdown()
    print("")
    print("**********")
    print("**********")
    print("**********")