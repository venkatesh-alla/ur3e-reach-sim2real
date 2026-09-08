"""
Real-robot reach task for UR3e.

This environment configures task-specific parameters and goal sampling.
All control-loop and Gym logic is inherited from BaseRealRobotEnv.
"""

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle
from transforms import (
    real_to_mujoco_position,
    mujoco_to_real_position
)
from base_real_robot_env import BaseRealRobotEnv

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class UR3eReachRealEnv(BaseRealRobotEnv):
    """
    Reach task using the real UR3e robot.
    """

    def __init__(
        self,
        reward_type: str = "sparse",
        **kwargs
    ):
        n_actions = 4  # dx, dy, dz, gripper command
        obs_dim = 10
        goal_dim = 3
        # ------------------------------------------------------------------
        # Action space
        # ------------------------------------------------------------------
        # Same as simulation: Cartesian delta (dx, dy, dz) + gripper command
        action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_actions,),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Observation space
        # ------------------------------------------------------------------
        # This must EXACTLY match the trained policy expectations.
        observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=(goal_dim,), dtype="float64"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=(goal_dim,), dtype="float64"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=(obs_dim,), dtype="float64"
                ),
            )
        )

        # Initial robot joint positions (example values) ##########################
        initial_qpos = {
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }

        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            distance_threshold=0.01,
            reward_type=reward_type,
            has_object=False,
            block_gripper=True,
            gripper_extra_height=0.35,
            target_in_the_air=True,
            target_offset=0.0,
            target_range=0.15,
            initial_qpos=initial_qpos,
            **kwargs,
        )

        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

    
    def compute_terminated(self, achieved_goal, desired_goal, info):
        # Terminate if within distance threshold
        dist = goal_distance(achieved_goal, desired_goal)
        print(f"Sampled goal - Mujoco frame: {desired_goal} | Robot frame: {mujoco_to_real_position(desired_goal)}")
        print(f"Achieved goal - Mujoco frame: {achieved_goal} | Robot frame: {mujoco_to_real_position(achieved_goal)}")
        print("Distance to goal:", dist)

        return dist < self.distance_threshold
        