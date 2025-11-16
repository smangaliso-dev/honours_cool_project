#%%
from pathlib import Path
from itertools import product
from minigrid.core.world_object import Ball, Box, Key
from minigrid.core.constants import COLOR_TO_IDX
from enum import Enum

from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from compute_auc import find_files, find_folders

# Base directory
BASE_DIR = Path(".")

# Subdirectories
MODEL_DIR = BASE_DIR    / "models"
LOG_DIR = BASE_DIR      / "logs"
VIDEO_DIR = BASE_DIR    / "videos"
DATA_DIR = BASE_DIR     / "data"
DOCS_DIR = BASE_DIR     / "docs"
IMG_DIR = BASE_DIR     / "images"

# Training parameters
SAVE_FREQUENCY: int = 100_000
NUM_ENVS = 5
DEFAULT_SIZE = 8

# Distributions
class DistType(Enum):
    NO_DIST             = 0
    FULL                = 1
    BASIC_SKILLS        = 2
    BASIC_SKILLS_EXCL   = 3 


OBJ_TYPES = [Ball, Box, Key,None]
COLORS = list(COLOR_TO_IDX.keys()) + [None]

FULL_DIST = list(product(OBJ_TYPES,COLORS))
FULL_DIST.remove((None,None))

BASIC_SKILLS_DIST = list(set(product(OBJ_TYPES,[None])) ^ set(product([None],COLORS)))
# the third distribution is similar to BASIC_SKILLS_DIST but will exclude evaluation combinations from training


# def evaluation_data(
#     model, env, 
#     n_episodes=50, 
#     deterministic=True, 
#     window_approx_steps=2048,
#     model_name="",
#     dist_type="",
#     seed=60
# ):
#     """
#     Evaluate a PPO agent and log episode-level performance metrics.
#     Includes:
#         - episodic rewards
#         - cumulative mean rewards
#         - sliding window mean rewards (≈ TensorBoard's ep_rew_mean)
#     """

#     results = []
#     episode_counter = 0

#     for ep in trange(n_episodes, desc=f"Evaluating PPO({dist_type}) agent"):
#         obs, info = env.reset()
#         done, truncated = False, False
#         ep_reward = 0.0
#         steps = 0
#         correct_pickup = 0

#         while not (done or truncated):
#             action, _ = model.predict(obs, deterministic=deterministic)
#             obs, reward, done, truncated, info = env.step(action)
#             ep_reward += reward
#             steps += 1

#             if "correct_pickup" in info:
#                 correct_pickup = int(info["correct_pickup"])

#         episode_counter += 1
#         success = correct_pickup

#         results.append({
#             "episode": episode_counter,
#             "total_reward": ep_reward,
#             "steps": steps,
#             "success": success
#         })
        
#     env.close()    

#     df = pd.DataFrame(results)
#     # df["mean_reward_so_far"] = df["total_reward"].expanding().mean()

#     # --- Simulate TensorBoard's ep_rew_mean ---
#     # Estimate how many episodes ≈ 2048 steps (for window size)
#     avg_ep_len = df["steps"].mean()
#     if avg_ep_len > 0:
#         window_size = max(1, int(window_approx_steps // avg_ep_len))
#     else:
#         window_size = 1

#     df["rolling_ep_rew_mean"] = df["total_reward"].rolling(window=window_size, min_periods=1).mean()

#     # Add summary statistics
#     df.attrs["avg_reward_final"] = df["total_reward"].mean()
#     df.attrs["success_rate_final"] = df["success"].mean()

#     df.to_csv(f"./data/{model_name}_{dist_type}_{seed}_evaluation_data.csv", index=False)

#     return df


# def evaluation_plots(data: pd.DataFrame,model_name: str, dist_type: str,seed: int):

# 	plt.figure(figsize=(12, 6))

# 	# Total Reward per Episode
# 	plt.subplot(2, 1, 1)
# 	plt.plot(data["episode"], data["total_reward"], label="Total Reward per Episode", alpha=0.25)
# 	plt.plot(data["episode"], data["rolling_ep_rew_mean"], label="Rolling Mean Reward", color='orange')
# 	plt.xlabel("Episode")
# 	plt.ylabel("Reward")
# 	plt.title("Episode Rewards and Rolling Mean")
# 	plt.legend()

# 	# Success Rate
# 	plt.subplot(2, 1, 2)
# 	plt.plot(data["episode"], data["success"].cumsum() / data["episode"], label="Cumulative Success Rate", color='green')
# 	plt.xlabel("Episode")
# 	plt.ylabel("Success Rate")
# 	plt.title("Cumulative Success Rate Over Episodes")
# 	plt.legend()

# 	plt.tight_layout()
# 	plt.savefig(f"./images/evaluation/{model_name}_{dist_type}_{seed}_evaluation_plots.png")	
# 	plt.show()

# -------------------------------------------------------------------------------------------------------

def evaluation_data_across_seeds(
    env_fn,
    model_type: str,
    distractors: int,
    dist_type: str,
    n_episodes: int = 50,
    deterministic: bool = True,
    window_approx_steps: int = 2048,
    obj_type=Ball,
    obj_color="green",
    model_steps = 1000000,
    seeds = [42,51,60],
    render = False
):
    """
    Evaluate PPO agents trained with different seeds, aggregate results, and compute mean statistics.

    Args:
        env_fn: callable that returns a fresh environment for each evaluation.
        model_type: str, type of model (e.g., "gpt2", "cnn")
        distractors: int, number of distractor objects
        dist_type: str, type of distractor setup
        n_episodes: int, number of episodes per run
        deterministic: bool, use deterministic policy actions
        window_approx_steps: int, used to estimate rolling mean window
        seeds: list of seeds (default [42, 51, 60])

    Returns:
        mean_df: pd.DataFrame containing aggregated mean results across seeds
        all_dfs: dict of per-seed dataframes
    """

    all_dfs = {}
    
    print(obj_color,obj_type)

    for seed in seeds:
        model_folders =find_folders(f"models/ppo/{'simple/' if model_type == 'trans' else ''}", f"minigrid_{model_type}_{distractors}_{dist_type}_{seed}_*")
        model_paths = find_files(f"models/ppo/{'simple/' if model_type == 'trans' else ''}" + model_folders[0] + "/", f"{model_type}_baseline_{distractors}_8_{model_steps}_steps*")
        print(f"{model_folders[0]=}, {model_paths=}")
        if not model_paths:
            print(f"⚠️ No model found for seed {seed} at {f'models/ppo/minigrid_{model_type}_{distractors}_{dist_type}_{seed}_*'}")
            continue

        model_path = model_paths[0]
        print(f"✅ Loading model for seed {seed} from {model_path}")

        # Load PPO model
        env = env_fn(
            seed=seed,
            size=DEFAULT_SIZE,
            num_objects=1+distractors,
            max_steps=None,
            target_type=obj_type,
            target_color=obj_color,
            render_mode='human' if render else 'rgb_array',)()

        model = PPO.load(model_path, env=env)

        results = []
        episode_counter = 0

        for ep in trange(n_episodes, desc=f"Evaluating {model_type}({dist_type}) seed={seed}"):
            obs, info = env.reset()
            done, truncated = False, False
            ep_reward = 0.0
            steps = 0
            correct_pickup = 0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                steps += 1

                if "correct_pickup" in info:
                    correct_pickup = int(info["correct_pickup"])

            episode_counter += 1
            results.append({
                "episode": episode_counter,
                "total_reward": ep_reward,
                "steps": steps,
                "success": correct_pickup,
                "seed": seed,
            })

        env.close()

        df = pd.DataFrame(results)

        # Compute rolling mean similar to TensorBoard
        avg_ep_len = df["steps"].mean()
        window_size = max(1, int(window_approx_steps // avg_ep_len)) if avg_ep_len > 0 else 1
        df["rolling_ep_rew_mean"] = df["total_reward"].rolling(window=window_size, min_periods=1).mean()
        df["success_rate"] = np.cumsum(df["success"]) / (np.arange(len(df)) + 1)

        df.attrs["avg_reward_final"] = df["total_reward"].mean()
        df.attrs["success_rate_final"] = df["success"].mean()

        # df.to_csv(f"./data/{model_type}_{dist_type}_{distractors}_{seed}_evaluation_data.csv", index=False)
        all_dfs[seed] = df

    # --- Aggregate results across seeds ---
    all_dfs_list = []
    for seed, df in all_dfs.items():
        df_seed = df.copy()
        df_seed["seed"] = seed
        all_dfs_list.append(df_seed)

    combined = pd.concat(all_dfs_list)
    grouped = combined.groupby("episode").agg(
        mean_reward=("total_reward", "mean"),
        std_reward=("total_reward", "std"),
        episode_len=("steps", "mean"),
        std_episode_len=("steps","std"),
        success_rate=("success_rate", "mean"),
        std_success_rate=("success_rate", "std"), 
        success=("success", "mean"),
    ).reset_index()

    grouped["rolling_ep_rew_mean"] = grouped["mean_reward"].rolling(window=5, min_periods=1).mean()
    grouped["success_indicator"] = (grouped["success"] > .5).astype('int16')

    grouped.to_csv(f"./data/{model_type}_{dist_type}_{distractors}_aggregated_evaluation.csv", index=False)

    return grouped, all_dfs


def evaluation_plots_across_seeds(mean_dfs, dist_type: str, distractors: int, all_dfs = None, model_type = None):
    """
    Plot evaluation curves across 3 seeds:
      - Subplot 1: Total reward per episode (mean + std)
      - Subplot 2: Success rate per episode (mean + std)
    """

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    type_colors = ("green", "orange")
    models = ("transformer", "mpl")

    # --- 1️⃣ Total Reward per Episode ---
    ax = axes[0]
    # for seed, df in all_dfs.items():
    #     ax.plot(
    #         df["episode"], df["total_reward"],
    #         label=f"Seed {seed}", alpha=0.25, linestyle="--"
    #     )
    for c,mean_df in enumerate(mean_dfs): 
        ax.plot(mean_df["episode"], mean_df["mean_reward"],
                label=f"Mean Reward ({models[c]})", color=type_colors[c], linewidth=2)
        ax.fill_between(
            mean_df["episode"],
            mean_df["mean_reward"] - mean_df["std_reward"].fillna(0),
            mean_df["mean_reward"] + mean_df["std_reward"].fillna(0),
            color=type_colors[c], alpha=0.3, label=f"Std Dev ({models[c]})"
        )

        ax.set_ylabel("Reward", fontsize=12)
        ax.set_title(f"Episode Rewards — ({dist_type}), {distractors} distractors", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # --- 2️⃣ Success Rate per Episode ---
    ax = axes[1]

    # for seed, df in all_dfs.items():
    #     cum_success = np.cumsum(df["success_rate"]) / (np.arange(len(df)) + 1)
    #     ax.plot(df["episode"], cum_success, alpha=0.25, linestyle="--")

    # Add aggregated mean + std for success rate
    # ax.plot(mean_df["episode"], mean_df["success"],
    #         label="Success indicator", color="gray", linewidth=2)
    for c,mean_df in enumerate(mean_dfs):
        ax.plot(mean_df["episode"], mean_df["success_rate"],
                label=f"Mean Success Rate ({models[c]})", color=type_colors[c], linewidth=2)
        ax.fill_between(
            mean_df["episode"],
            mean_df["success_rate"] - mean_df["std_success_rate"].fillna(0),
            mean_df["success_rate"] + mean_df["std_success_rate"].fillna(0),
            color=type_colors[c], alpha=0.3, label=f"Std Dev ({models[c]})"
        )

        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Success Rate", fontsize=12)
        ax.set_title("Cumulative Success Rate Over Episodes", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        
    plt.savefig(f"./images/evaluation/{dist_type}_{distractors}_aggregated_rewards_success.png", dpi=200)
    plt.show()


