import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env # validates Gym/GoalEnv API compliance

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

from robot_interface.ur3e_rtde_sync import UR3eRTDESyncClient
from ur3e_reach_real import UR3eReachRealEnv

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    register(
        id=f"UR3eReachReal{suffix}-v1",
        entry_point="ur3e_reach_real:UR3eReachRealEnv",
        kwargs={"reward_type": reward_type,},
        max_episode_steps=100,
        )




client = UR3eRTDESyncClient(
    robot_host="172.17.0.2",
    robot_port=30004,
    config_filename="/misis/project_multi_skill_rl/rl_skills/envs/real/robot_interface/sync_control_loop_configuration.xml"
)

client.connect()

env = gym.make("UR3eReachReal-v1")  # Triggers __init__ (spaces, params)
env.unwrapped.robot_client = client  # Dependency injection

# ------------------------------------------------------------------
# Level 1: Test reset
# ------------------------------------------------------------------
obs, info = env.reset()

# ------------------------------------------------------------------
# Level 2: Single step motion test
# ------------------------------------------------------------------
# print("TCP:", env.unwrapped._last_robot_state.actual_TCP_pose[:3])
# action = np.array([0.0, 0.0, -0.2, 0.0])
# obs, reward, terminated, truncated, info = env.step(action)
# print("TCP:", env.unwrapped._last_robot_state.actual_TCP_pose[:3])
# print("Obs achieved:", obs["achieved_goal"])

# ------------------------------------------------------------------
# Level 2: Multiple step motion test
# ------------------------------------------------------------------
# print("TCP:", env.unwrapped._last_robot_state.actual_TCP_pose[:3])
# action = np.array([0.0, 0.0, -0.05, 0.0])
# for _ in range(10):
#     obs, reward, _, _, _ = env.step(np.array(action))
#     print(obs["achieved_goal"])

# print("TCP:", env.unwrapped._last_robot_state.actual_TCP_pose[:3])
# print("Obs achieved:", obs["achieved_goal"])

# ------------------------------------------------------------------
# Level 2: until terminated or truncated (full)
# ------------------------------------------------------------------
print("TCP:", env.unwrapped._last_robot_state.actual_TCP_pose[:3])
action = np.random.uniform(-0.0005, 0, size=(4,))
action[3] = 0.0  # No gripper command
# action = np.array([0.0, 0.0, z_action, 0.0])
terminated, truncated = False, False
ep_reward = 0.0
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(action)
    ep_reward += reward 
    print(obs["achieved_goal"])

print("TCP:", env.unwrapped._last_robot_state.actual_TCP_pose[:3])
print("Obs achieved:", obs["achieved_goal"])
print("Episode reward:", ep_reward)