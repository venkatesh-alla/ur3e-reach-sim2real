import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3 import PPO, SAC

import gymnasium_robotics
import numpy as np


from typing import Callable, Optional
import os
from gymnasium.envs.registration import register
from rl_zoo3.wrappers import MaskVelocityWrapper
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise


import sys
sys.path.append('/misis/project_multi_skill_rl/rl_skills/envs')
sys.path.append('/misis/project_multi_skill_rl/rl_skills/envs/real')

import mujoco_UR3e_reach
import ur3e_reach_real

try:
    import UR3e
except ImportError:
    pass

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""

    register(
        id=f"UR3eReach{suffix}-v1",
        entry_point="mujoco_UR3e_reach:UR3eReachEnv",
        kwargs={"reward_type": reward_type,},
        max_episode_steps=100,
    )

    register(
        id=f"UR3eReachReal{suffix}-v1",
        entry_point="ur3e_reach_real:UR3eReachRealEnv",
        kwargs={"reward_type": reward_type,},
        max_episode_steps=100,
    )


from evaluation_logger import EvaluationLogger
from real.tests.test_robot_client import FakeRobotClient
from real.robot_interface.ur3e_rtde_sync import UR3eRTDESyncClient

# ROBOT_HOST = "172.17.0.2"      # URSim
ROBOT_HOST = "192.168.123.3" # real robot

ROBOT_PORT = 30004

CONFIG_FILE = "/misis/project_multi_skill_rl/rl_skills/envs/real/robot_interface/sync_control_loop_configuration.xml"

client = UR3eRTDESyncClient(
    robot_host=ROBOT_HOST,
    robot_port=ROBOT_PORT,
    config_filename=CONFIG_FILE,
)

client.connect()

# --------------------------------------------------------
# Training seed
# --------------------------------------------------------

trained_seed = 42


# --------------------------------------------------------
# All models trained
# --------------------------------------------------------

MODELS = {

    # Models trained with seed 42 

    "ppo_sp_5cm_acc_42": "ppo/UR3eReach-v1_1/best_model",
    "ppo_sp_1cm_acc_42": "ppo/UR3eReach-v1_6/best_model",
    "ppo_de_1cm_acc_42": "ppo/UR3eReachDense-v1_6/best_model",

    "tqc_sp_her_5cm_acc_42": "tqc/UR3eReach-v1_2/best_model",
    "tqc_sp_her_1cm_acc_42": "tqc/UR3eReach-v1_14/best_model",

    "tqc_sp_noher_5cm_acc_42": "tqc/UR3eReach-v1_24/best_model",
    "tqc_sp_noher_1cm_acc_42": "tqc/UR3eReach-v1_25/best_model",

    "tqc_de_her_5cm_acc_42": "tqc/UR3eReachDense-v1_1/best_model",
    "tqc_de_her_1cm_acc_42": "tqc/UR3eReachDense-v1_2/best_model",

    "tqc_de_noher_5cm_acc_42": "tqc/UR3eReachDense-v1_21/best_model",
    "tqc_de_noher_1cm_acc_42": "tqc/UR3eReachDense-v1_22/best_model",

    "sac_sp_noher_5cm_acc_42": "sac/UR3eReach-v1_16/best_model",
    "sac_sp_noher_1cm_acc_42": "sac/UR3eReach-v1_6/best_model",
    "sac_sp_her_1cm_acc_42": "sac/UR3eReach-v1_1/best_model",
    "sac_de_noher_1cm_acc_42": "sac/UR3eReachDense-v1_6/best_model",
    "sac_de_her_1cm_acc_42": "sac/UR3eReachDense-v1_1/best_model",

    # Models trained with seed 43
    "ppo_de_1cm_acc_43": "ppo/UR3eReachDense-v1_7/best_model",

    # Models trained with seed 44
    "ppo_de_1cm_acc_44": "ppo/UR3eReachDense-v1_8/best_model",

}

# --------------------------------------------------------
# Model used
# --------------------------------------------------------

# model_name = f"ppo_de_1cm_acc_{trained_seed}"
# model_name = f"tqc_sp_noher_5cm_acc_{trained_seed}"
# model_name = f"sac_de_noher_1cm_acc_{trained_seed}"
# model_name = f"tqc_de_noher_1cm_acc_{trained_seed}"
# model_name = f"ppo_de_1cm_acc_{trained_seed}"

model_name = f"tqc_sp_her_1cm_acc_{trained_seed}"


model_used = MODELS[model_name]

# --------------------------------------------------------
# Create base environment
# --------------------------------------------------------
env = gym.make("UR3eReachReal-v1")
env.unwrapped.robot_client = client

# Wrap in VecEnv for stable-baselines3 compatibility
env = DummyVecEnv([lambda: env])
# --------------------------------------------------------
# Evaluation logging flag
# --------------------------------------------------------

EVAL_FLAG = False
# --------------------------------------------------------
# Create logger only if evaluation logging is enabled
# --------------------------------------------------------

logger = None

if EVAL_FLAG:

    logger = EvaluationLogger(
        model_name=f"{model_name}"
    )

# --------------------------------------------------------
# Seeds
# --------------------------------------------------------

# seeds = [100, 200, 300, 400, 500]
seeds = [100]

# Store cumulative statistics across seeds
all_seed_results = []

# --------------------------------------------------------
# Evaluation Loop
# --------------------------------------------------------

for seed in seeds:

    print("\n")
    print("=" * 70)
    print(f"STARTING EVALUATION FOR SEED: {seed}")
    print("=" * 70)
    print("\n")

    np.random.seed(seed)

    if "tqc" in model_name:

        model = TQC.load(
            f"/misis/project_multi_skill_rl/rl_skills/final_sim_models/{model_used}",
            env=env,
            seed=seed
        )

    elif "ppo" in model_name:

        model = PPO.load(
            f"/misis/project_multi_skill_rl/rl_skills/final_sim_models/{model_used}",
            env=env,
            seed=seed
        )

    elif "sac" in model_name:

        model = SAC.load(
            f"/misis/project_multi_skill_rl/rl_skills/final_sim_models/{model_used}",
            env=env,
            seed=seed
        )

    # Make the environment and goal sampling deterministic/reproducible for evaluation
    env.seed(seed)
    obs = env.reset()

    num_episodes = 5
    max_steps = 100

    episode_rewards = []
    episode_lengths = []
    successes = []

    for episode in range(num_episodes):

        done = False
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

        ep_reward = 0.0
        ep_len = 0
        episode_success = False

        # --------------------------------------------------
        # Store goal position
        # --------------------------------------------------

        if EVAL_FLAG:

            goal = obs["desired_goal"][0]

            logger.log_goal(
                evaluation_seed=seed,
                episode=episode + 1,
                goal=goal,
            )

        while not done:

            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )

            try:

                # action *= 0.6 # Corrective action only for ppo
                obs, reward, done, info = env.step(action)

                episode_start = done

                ep_reward += float(reward[0])
                ep_len += 1

                # --------------------------------------------------
                # SUCCESS = environment terminated naturally
                # --------------------------------------------------

                if done and ep_len < max_steps:
                    episode_success = True

            except RuntimeError as e:

                print("\n")
                print("=" * 60)
                print("ROBOT EXECUTION FAILURE DETECTED")
                print(str(e))
                print("Marking current episode as FAILURE")
                print("=" * 60)
                print("\n")

                episode_success = False

                # ---------------------------------
                # Manual recovery reset
                # ---------------------------------

                obs = env.reset()
                done = True

        success = episode_success

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        successes.append(success)

        # --------------------------------------------------
        # Log episode result
        # --------------------------------------------------

        if EVAL_FLAG:

            logger.log_episode_result(
                evaluation_seed=seed,
                episode=episode + 1,
                reward=ep_reward,
                length=ep_len,
                success=success,
            )

        print("\n")
        print("=" * 50)
        print(
            f"Seed: {seed} | "
            f"Episode: {episode + 1}/{num_episodes} | "
            f"Reward: {ep_reward:.3f} | "
            f"Length: {ep_len} | "
            f"Success: {success}"
        )
        print("=" * 50)
        print("\n")

    # =====================================================
    # CUMULATIVE STATISTICS FOR CURRENT SEED
    # =====================================================

    mean_episode_reward = np.mean(episode_rewards)
    mean_episode_length = np.mean(episode_lengths)

    success_rate = np.mean(successes)

    # Store seed-level results
    all_seed_results.append({
        "seed": seed,
        "mean_reward": mean_episode_reward,
        "mean_length": mean_episode_length,
        "success_rate": success_rate,
    })

    # --------------------------------------------------
    # Log seed summary
    # --------------------------------------------------

    if EVAL_FLAG:

        logger.log_seed_summary(
            evaluation_seed=seed,
            num_episodes=num_episodes,
            mean_reward=mean_episode_reward,
            mean_length=mean_episode_length,
            success_rate=success_rate,
        )

    print("\n")
    print("=" * 70)
    print(f"SEED {seed} EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Episodes run          : {num_episodes}")
    print(f"Mean episode reward   : {mean_episode_reward:.3f}")
    print(f"Mean episode length   : {mean_episode_length:.3f}")
    print(f"Success rate          : {success_rate * 100:.2f}%")

    print("=" * 70)
    print("\n")

# =========================================================
# FINAL RESULTS ACROSS ALL SEEDS
# =========================================================

print("\n")
print("=" * 90)
print("FINAL RESULTS ACROSS ALL SEEDS")
print("=" * 90)

header = (
    f"{'Seed':<10}"
    f"{'Mean Reward':<20}"
    f"{'Mean Length':<20}"
    f"{'Success Rate':<20}"
)

print(header)
print("-" * 90)

for result in all_seed_results:

    print(
        f"{result['seed']:<10}"
        f"{result['mean_reward']:<20.3f}"
        f"{result['mean_length']:<20.3f}"
        f"{result['success_rate'] * 100:<20.2f}"
    )

print("-" * 90)

overall_mean_reward = np.mean(
    [r["mean_reward"] for r in all_seed_results]
)

overall_mean_length = np.mean(
    [r["mean_length"] for r in all_seed_results]
)

overall_success_rate = np.mean(
    [r["success_rate"] for r in all_seed_results]
)

# --------------------------------------------------
# Log final aggregate results
# --------------------------------------------------

if EVAL_FLAG:

    logger.log_final_results(
        overall_mean_reward=overall_mean_reward,
        overall_mean_length=overall_mean_length,
        overall_success_rate=overall_success_rate,
    )

print(f"Overall Mean Reward  : {overall_mean_reward:.3f}")
print(f"Overall Mean Length  : {overall_mean_length:.3f}")
print(f"Overall Success Rate : {overall_success_rate * 100:.2f}%")

print("=" * 90)
print("\n")

env.close()

