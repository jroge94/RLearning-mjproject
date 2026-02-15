"""Plot reward curves and cost comparisons from experiment logs."""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

import config


def process_file(file_path: str) -> Tuple[pd.DataFrame, float]:
    """Load one log CSV and return reward columns and total cost."""
    df = pd.read_csv(file_path)
    rewards = df[["round_id", "eval_reward_max", "eval_reward_mean", "eval_reward_min"]]
    total_cost = df["cost"].sum()
    return rewards, total_cost


def save_rewards_plot(
    file_path: str,
    env: str,
    model: str,
    img_dir: str,
) -> None:
    """Save reward metrics plot for one env-model to img_dir."""
    rewards_data, _ = process_file(file_path)
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_data["round_id"], rewards_data["eval_reward_max"], label="Max Reward")
    plt.plot(rewards_data["round_id"], rewards_data["eval_reward_mean"], label="Mean Reward")
    plt.plot(rewards_data["round_id"], rewards_data["eval_reward_min"], label="Min Reward")
    plt.xlabel("Round Number")
    plt.ylabel("Reward")
    plt.title(f"Reward Metrics Over Rounds for {env}-{model}")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(img_dir, f"{env}-{model}-rewards.png")
    plt.savefig(out_path)
    plt.close()


def process_and_plot_all_files(
    logs_dir: str,
    img_dir: str,
    envs: List[str],
    models: List[str],
) -> Dict[str, float]:
    """Plot rewards for each log file and return total costs per env-model."""
    total_costs: Dict[str, float] = {}
    for env in envs:
        for model in models:
            path = os.path.join(logs_dir, f"{env}~{model}~.csv")
            if os.path.exists(path):
                rewards_data, total_cost = process_file(path)
                save_rewards_plot(path, env, model, img_dir)
                total_costs[f"{env}-{model}"] = total_cost
            else:
                print(f"File not found: {path}")
    return total_costs


def plot_total_costs_bar_graph(total_costs: Dict[str, float], img_dir: str) -> None:
    """Save bar chart of total costs per env-model to img_dir."""
    plt.figure(figsize=(14, 7))
    labels = list(total_costs.keys())
    costs = list(total_costs.values())
    plt.bar(labels, costs, color="skyblue")
    plt.xlabel("Environment-Model Combinations")
    plt.ylabel("Total Cost")
    plt.title("Total Costs for Each Environment-Model Combination")
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.2)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "total_costs_comparison.png"))
    plt.close()


def plot_env_comparison(
    logs_dir: str,
    img_dir: str,
    env: str,
    models: List[str],
) -> None:
    """Compare mean reward across models for one environment."""
    plt.figure(figsize=(12, 6))
    for model in models:
        path = os.path.join(logs_dir, f"{env}~{model}~.csv")
        if os.path.exists(path):
            rewards_data, _ = process_file(path)
            plt.plot(
                rewards_data["round_id"],
                rewards_data["eval_reward_mean"],
                label=model,
            )
    plt.xlabel("Round Number")
    plt.ylabel("Mean Reward")
    plt.title(f"Mean Reward Comparison Across Models for {env}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(img_dir, f"{env}-model-comparison.png"))
    plt.close()


def plot_model_comparison(
    logs_dir: str,
    img_dir: str,
    envs: List[str],
    model: str,
) -> None:
    """Compare mean reward across environments for one model."""
    plt.figure(figsize=(12, 6))
    for env in envs:
        path = os.path.join(logs_dir, f"{env}~{model}~.csv")
        if os.path.exists(path):
            rewards_data, _ = process_file(path)
            plt.plot(
                rewards_data["round_id"],
                rewards_data["eval_reward_mean"],
                label=env,
            )
    plt.xlabel("Round Number")
    plt.ylabel("Mean Reward")
    plt.title(f"Mean Reward Comparison Across Environments for {model}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(img_dir, f"{model}-env-comparison.png"))
    plt.close()


def main(
    logs_dir: Optional[str] = None,
    imgs_dir: Optional[str] = None,
    envs: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
) -> None:
    """Generate plots from experiment logs. Uses config defaults when args are None."""
    logs_dir = logs_dir or config.LOGS_DIR
    imgs_dir = imgs_dir or config.IMGS_DIR
    env_list = list(envs) if envs is not None else config.get_env_names()
    model_list = list(models) if models is not None else config.get_algo_names()

    os.makedirs(imgs_dir, exist_ok=True)

    total_costs = process_and_plot_all_files(
        logs_dir, imgs_dir, env_list, model_list
    )
    plot_total_costs_bar_graph(total_costs, imgs_dir)

    for env in env_list:
        plot_env_comparison(logs_dir, imgs_dir, env, model_list)
    for model in model_list:
        plot_model_comparison(logs_dir, imgs_dir, env_list, model)


if __name__ == "__main__":
    main()
