import matplotlib.pyplot as plt
import pandas as pd
import os

# Function to process a single file and extract required data
def process_file(file_path):
    df = pd.read_csv(file_path)
    rewards_data = df[['round_id', 'eval_reward_max', 'eval_reward_mean', 'eval_reward_min']]
    total_cost = df['cost'].sum()
    return rewards_data, total_cost

# Updated directory paths
logs_directory = 'logs/'
img_directory = 'imgs/'

# Ensure the img/ directory exists
if not os.path.exists(img_directory):
    os.makedirs(img_directory)

# Function to save the plot of rewards for each file
def save_rewards_plot(file_path, env, model):
    rewards_data, _ = process_file(file_path)
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_data['round_id'], rewards_data['eval_reward_max'], label='Max Reward')
    plt.plot(rewards_data['round_id'], rewards_data['eval_reward_mean'], label='Mean Reward')
    plt.plot(rewards_data['round_id'], rewards_data['eval_reward_min'], label='Min Reward')
    plt.xlabel('Round Number')
    plt.ylabel('Reward')
    plt.title(f'Reward Metrics Over Rounds for {env}-{model}')
    plt.legend()
    plt.grid(True)
    plot_filename = f'{img_directory}/{env}-{model}-rewards.png'
    plt.savefig(plot_filename)
    plt.close()

# Function to process all files, save plots, and collect total costs
def process_and_plot_all_files(logs_directory, envs, models):
    total_costs = {}
    for env in envs:
        for model in models:
            file_name = f'{env}~{model}~.csv'
            file_path = os.path.join(logs_directory, file_name)
            if os.path.exists(file_path):
                rewards_data, total_cost = process_file(file_path)
                save_rewards_plot(file_path, env, model)
                total_costs[f'{env}-{model}'] = total_cost
            else:
                print(f"File not found: {file_name}")
    return total_costs

# Function to plot a bar graph for total costs
def plot_total_costs_bar_graph(total_costs):
    plt.figure(figsize=(14, 7))  # Increase figure size
    labels = list(total_costs.keys())
    costs = list(total_costs.values())
    
    plt.bar(labels, costs, color='skyblue')
    plt.xlabel('Environment-Model Combinations')
    plt.ylabel('Total Cost')
    plt.title('Total Costs for Each Environment-Model Combination')
    plt.xticks(rotation=90)  # Rotate labels to 90 degrees
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin
    plt.grid(axis='y')
    
    plt.tight_layout()  # Ensure everything fits without overlapping
    bar_graph_filename = f'{img_directory}/total_costs_comparison.png'
    plt.savefig(bar_graph_filename)
    plt.close()


# Function to plot comparison of mean rewards across different models for each environment
def plot_env_comparison(logs_directory, env, models, img_directory):
    plt.figure(figsize=(12, 6))
    for model in models:
        file_name = f'{env}~{model}~.csv'
        file_path = os.path.join(logs_directory, file_name)
        if os.path.exists(file_path):
            rewards_data, _ = process_file(file_path)
            plt.plot(rewards_data['round_id'], rewards_data['eval_reward_mean'], label=f'{model}')
    
    plt.xlabel('Round Number')
    plt.ylabel('Mean Reward')
    plt.title(f'Mean Reward Comparison Across Models for {env}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{img_directory}/{env}-model-comparison.png')
    plt.close()

# Function to plot comparison of mean rewards across different environments for each model
def plot_model_comparison(logs_directory, envs, model, img_directory):
    plt.figure(figsize=(12, 6))
    for env in envs:
        file_name = f'{env}~{model}~.csv'
        file_path = os.path.join(logs_directory, file_name)
        if os.path.exists(file_path):
            rewards_data, _ = process_file(file_path)
            plt.plot(rewards_data['round_id'], rewards_data['eval_reward_mean'], label=f'{env}')
    
    plt.xlabel('Round Number')
    plt.ylabel('Mean Reward')
    plt.title(f'Mean Reward Comparison Across Environments for {model}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{img_directory}/{model}-env-comparison.png')
    plt.close()

# Run the function
envs = ['Hopper-v3', 'Humanoid-v3', 'Walker2d-v3']
models = ['impala', 'pg', 'ppo']
total_costs = process_and_plot_all_files(logs_directory, envs, models)
# Plot and save the bar graph
plot_total_costs_bar_graph(total_costs)

for env in envs:
    plot_env_comparison(logs_directory, env, models, img_directory)
for model in models:
    plot_model_comparison(logs_directory, envs, model, img_directory)

