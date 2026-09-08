import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env # validates Gym/GoalEnv API compliance

from ur3e_reach_real import UR3eReachRealEnv
from tests.test_robot_client import FakeRobotClient


from typing import Callable, Optional
import os

import gymnasium as gym
from gymnasium.envs.registration import register

from rl_zoo3.wrappers import MaskVelocityWrapper
from gymnasium.wrappers import FlattenObservation

from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.env_util import make_vec_env
#from gymnasium import wrapper

import sys
sys.path.append('/misis/project_multi_skill_rl/rl_skills/envs/real')
#sys.path.append('/content/drive/MyDrive/project_multi_skill_rl/rl_skills/envs')
import ur3e_reach_real

try:
    import UR3e
except ImportError:
    pass

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    register(
        id=f"UR3eReachReal{suffix}-v1",
        entry_point="ur3e_reach_real:UR3eReachRealEnv",
        kwargs={"reward_type": reward_type,},
        max_episode_steps=100,
        )


print("=== 1. Create & Inject Mock ===")
# env = UR3eReachRealEnv(reward_type="dense")  # Triggers __init__ (spaces, params)
env = gym.make("UR3eReachReal-v1")  # Triggers __init__ (spaces, params)
env.unwrapped.robot_client = FakeRobotClient()  # Dependency injection

print("Action space:", env.action_space)
print("Obs space:", env.observation_space)
print("Distance threshold:", env.unwrapped.distance_threshold)

print("\n=== 2. Gymnasium Checker (API Validation) ===")
# check_env(env.unwrapped, warn=True)  # Tests reset/step/spaces/GoalEnv.compute_reward
print("✅ Passed if no errors!")

print("\n=== 3. Manual Reset/Step Loop ===")
obs, info = env.reset(seed=42)
print("Reset obs keys:", obs.keys())
print("Initial goal:", obs['desired_goal'])
print("Initial achieved:", obs['achieved_goal'])

terminated, truncated = False, False
step_count = 0
while not (terminated or truncated) and step_count < 150:  # Short episode
    action = env.action_space.sample()  # Random [-1,1]^4
    obs, reward, terminated, truncated, info = env.step(action)
    step_count += 1
    print(f"Step {step_count}: reward={reward:.3f}, success={info['is_success']}, "
          f"dist={np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']):.3f}")

print(f"Episode done: steps={step_count}, final success={info['is_success']}")

print("\n=== 4. Reward Consistency Check ===")
# GoalEnv test: manual compute_reward must match step
manual_reward = env.unwrapped.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
assert np.isclose(reward, manual_reward), f"Mismatch: {reward} vs {manual_reward}"
print("✅ Rewards match!")
