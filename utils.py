from ray.rllib.algorithms import pg, impala, ppo
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
import ray
import config
import csv
import numpy as np

def init_trainer_config(
    algo_name,
    env_name,
):
    # Init with the same number of actors
    num_rollout_workers = config.num_rollout_workers
    num_envs_per_worker = config.num_envs_per_worker
    rollout_fragment_length = config.envs[env_name]['rollout_fragment_length']
    train_batch_size = num_rollout_workers * num_envs_per_worker * rollout_fragment_length

    if algo_name == "pg":
        trainer_config = pg.PGConfig()
    elif algo_name == "impala":
        trainer_config = impala.ImpalaConfig()
        trainer_config.estimate_batch_size = num_envs_per_worker * rollout_fragment_length
    elif algo_name == "ppo":
        trainer_config = ppo.PPOConfig()
        trainer_config.estimate_batch_size = num_envs_per_worker * rollout_fragment_length
    
    # Configure trainer
    trainer_config = (
        trainer_config
        .framework(framework=config.framework)
        # .callbacks(callbacks_class=CustomCallbacks)
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
            # batch_mode="complete_episodes",
            batch_mode="truncate_episodes",
        )
        .debugging(
            log_level="ERROR",
            logger_config={"type": ray.tune.logger.NoopLogger},
            log_sys_usage=False
        ) # Disable all loggings to save time
    )

    # Configure report time to avoid learner thread dies: https://discuss.ray.io/t/impala-bugs-and-some-other-observations/9863/7
    trainer_config = trainer_config.reporting(
        min_time_s_per_iteration=config.min_time_s_per_iteration,
    )

    # Configure experimental settings
    trainer_config = trainer_config.experimental(
        _enable_new_api_stack=config._enable_new_api_stack,
    )

    # Configure train batch
    trainer_config = trainer_config.training(
        train_batch_size=train_batch_size,
    )
    if algo_name == "ppo":
        trainer_config = trainer_config.training(
            sgd_minibatch_size=train_batch_size,
        )

    # Configure evaluation
    trainer_config = trainer_config.evaluation(
        evaluation_interval=config.evaluation_interval,
        evaluation_num_workers=config.evaluation_num_workers,
        evaluation_duration=config.evaluation_num_workers,
    )

    return trainer_config

def process_train_results(
    algo_name,
    round_id,
    train_results,
):
    if round_id is not None and train_results is not None:
        # Learner time
        if algo_name == "impala":
            learner_time = (train_results["info"]["timing_breakdown"]["learner_grad_time_ms"] + \
                train_results["info"]["timing_breakdown"]["learner_load_time_ms"] + \
                train_results["info"]["timing_breakdown"]["learner_load_wait_time_ms"] + \
                train_results["info"]["timing_breakdown"]["learner_dequeue_time_ms"]) / 1000
        else:
            learner_time = train_results["timers"]["learn_time_ms"] / 1000
        
        # Actor time
        actor_time = train_results["timers"]["sample_time_ms"] / 1000

        # Eval rewards
        episode_reward = train_results['evaluation']["hist_stats"]["episode_reward"]
        if len(episode_reward) == 0:
            eval_reward_max = 0
            eval_reward_mean = 0
            eval_reward_min = 0
        else:
            # episode_reward = utils.remove_outliers(episode_reward)
            eval_reward_max = np.max(episode_reward)
            eval_reward_mean = np.mean(episode_reward)
            eval_reward_min = np.min(episode_reward)
        
        episodes_this_iter = train_results['episodes_this_iter']
        num_steps_trained_this_iter = train_results['num_steps_trained_this_iter']

        # Learner info
        if algo_name == "pg":
            learner_loss = train_results["info"]['learner'][DEFAULT_POLICY_ID]['learner_stats']['policy_loss']
        elif algo_name == "impala":
            if DEFAULT_POLICY_ID in train_results["info"]['learner']:
                learner_loss = train_results["info"]['learner'][DEFAULT_POLICY_ID]['learner_stats']['total_loss']
            else:
                learner_loss = 0
        elif algo_name == "ppo":
            learner_loss = train_results["info"]['learner'][DEFAULT_POLICY_ID]['learner_stats']['total_loss']

        # Duration
        duration = train_results["time_this_iter_s"]

        # Cost
        cost_per_round = duration * (config.server_learner_per_s + config.server_actor_per_s)

        info = {
            "round_id": round_id,
            "episodes_this_iter": episodes_this_iter,
            "duration": duration,
            "learner_time": learner_time,
            "actor_time": actor_time,
            "eval_reward_max": eval_reward_max,
            "eval_reward_mean": eval_reward_mean,
            "eval_reward_min": eval_reward_min,
            "learner_loss": learner_loss,
            "episode_reward": episode_reward,
            "cost_per_round": cost_per_round,
        }
    else:
        info = None

    return info

def export_csv(
    env_name, 
    algo_name, 
    csv_name,
    csv_file
):
    with open(
        "logs/{}~{}~{}.csv".format(
            env_name, 
            algo_name, 
            csv_name,
        ), 
        "w", 
        newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerows(csv_file)