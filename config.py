# Ray cluster
num_rollout_workers = 16
num_envs_per_worker = 4
num_cpus_for_local_worker = 2
num_gpus_for_local_worker = 1
num_cpus_per_worker = 1
num_gpus_per_worker = 0
evaluation_num_workers = 6
min_time_s_per_iteration = None
_enable_new_api_stack = False

# Training
max_exp = 1
framework = "torch"
envs = {
    "Hopper-v3": {
        "rollout_fragment_length": 512,
    },
    "Humanoid-v3": {
        "rollout_fragment_length": 512,
    },
    "Walker2d-v3": {
        "rollout_fragment_length": 512,
    },
}
algos = [
    # "pg",
    # "impala",
    "ppo",
]

# Stop criteria
stop_max_round = 50

# Evaluate
evaluation_interval = 1

# Pricing units
server_learner_per_s = (3.0600 + 17.92 / 30 / 24) / 60 / 60 # vm + ip + disk
server_actor_per_s = (0.68 + 17.92 / 30 / 24) / 60 / 60 # vm + ip + disk
