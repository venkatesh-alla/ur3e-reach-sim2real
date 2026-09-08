"""
Base class for real-robot Gymnasium environments.

This class is the real-robot analogue of BaseUR3eEnv (simulation),
but it does NOT depend on MuJoCo. It only defines:
- the Gym / GoalEnv interface
- the control-loop structure
- reward and success logic

"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium_robotics.core import GoalEnv
import time

from transforms import (
    real_to_mujoco_position,
    mujoco_delta_to_real
)

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class BaseRealRobotEnv(GoalEnv):
    """
    Task-agnostic base environment for real-robot control.

    Subclasses are responsible for:
    - goal sampling
    - task-specific configuration
    - defining observation/action spaces
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        action_space: spaces.Space,
        observation_space: spaces.Space,
        distance_threshold: float,
        reward_type: str,
        has_object: bool,
        block_gripper: bool,
        gripper_extra_height: float,
        target_in_the_air: bool,
        target_offset: float,
        target_range: float,
        initial_qpos,
        **kwargs,
    ):
        """
        Parameters
        ----------
        action_space:
            Gym action space (must match the trained policy).
        observation_space:
            Gym observation space (Dict with observation/achieved_goal/desired_goal).
        distance_threshold:
            Threshold for goal success.
        reward_type:
            "sparse" or "dense".
        max_episode_steps:
            Episode length cap.
        """

        super().__init__()

        self.action_space = action_space
        self.observation_space = observation_space

        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.has_object = has_object
        self.block_gripper = block_gripper
        self.gripper_extra_height = gripper_extra_height
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.target_range = target_range
        self.initial_qpos = initial_qpos

        self.robot_client = None  # injected synchronous RTDE client
        self._last_robot_state = None

        self.np_random = super().np_random  # Inherit from GoalEnv

        self.start = 0
        self.goal = np.zeros(0)

        # Robot communication objects (RTDE, etc.) should be
        # created outside or injected later.
        # This class does NOT own sockets or connections.
    

    # ---------------------------------------------------------------------
    # Gymnasium methods
    # ---------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:

        super().reset(seed=seed)

        self.start += 1
        if self.start > 1:
            time.sleep(3)  # Short Wait to demonstrate success from first ep ends

        print("\n=== Resetting Environment and Starting new episode ===\n")

        # Reset robot to a safe initial configuration
        self._reset_robot()

        time.sleep(5)  # Ensure robot to reach home position after reset is called

        # Sample a new goal (task-specific)
        self.goal = self._sample_goal().copy() # In real robot frame

        # Read robot state and build observation
        robot_state = self._read_robot_state()
        obs = self._get_obs(robot_state)

        return obs, {}

    def step(self, action):

        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._apply_action(action)

        # Read updated robot state
        robot_state = self._read_robot_state()
        obs = self._get_obs(robot_state)

        achieved_goal = obs["achieved_goal"]
        desired_goal = obs["desired_goal"]

        reward = self.compute_reward(achieved_goal, desired_goal, {})
        success = bool(self._is_success(achieved_goal, desired_goal))

        terminated = self.compute_terminated(achieved_goal, desired_goal, {})
        truncated = self.compute_truncated(achieved_goal, desired_goal, {})
        info: Dict[str, Any] = {
            "is_success": success,
        }

        return obs, reward, terminated, truncated, info
    
    def compute_terminated(self, achieved_goal, desired_goal, info):
        return False  # NEVER terminate (episodes run to time limit)

    def compute_truncated(self, achieved_goal, desired_goal, info):
        return False  # Delegate to TimeLimit wrapper only

    # ---------------------------------------------------------------------
    # GoalEnv methods
    # ---------------------------------------------------------------------
        
    def compute_reward(self, achieved_goal, goal, info):

        # Reward function for reach task matching the gymnasium-mujoco setup

        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _is_success(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> np.ndarray:
        """
        Determine if the goal has been achieved.
        """

        d = goal_distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float32)    



    # ---------------------------------------------------------------------
    # Robot data acquisition and control methods
    # ---------------------------------------------------------------------

    # State and observation related methods
    def _read_robot_state(self):
        """
        Return the most recent robot state received from RTDE.
        """

        # In synchronous mode, state is updated by `_apply_action`
        return self._last_robot_state
    
    def _build_observation_vector(self, robot_state):
        """
        Build observation vector with MuJoCo-equivalent layout - replacement for generate_mujoco_observations.

        TODO:
        - Match EXACT ordering from UR3e MuJoCo `_get_obs()`
        - Use TCP pose for grip_pos
        - Velocities must be finite-differenced or read from RTDE
        """

        # grip_pos = np.array(robot_state.actual_TCP_pose[:3], dtype=np.float32)
        grip_pos_real = np.array(robot_state.actual_TCP_pose[:3], dtype=np.float32)
        grip_pos = real_to_mujoco_position(grip_pos_real).astype(np.float32)
        
        # Making everything except grip_pos zero in the observation vector for reach task
        gripper_state = np.array([0.0, 0.0], dtype=np.float32)
        grip_velp = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        gripper_vel = np.array([0.0, 0.0], dtype=np.float32)

        # PLACEHOLDERS 
        # TODO: Replace with real object state if applicable for the future tasks involving objects
        if self.has_object:
            # Object position (replace with real object tracking)
            object_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # TODO: Replace with real object pose
            object_rel_pos = object_pos - grip_pos

            # Object rotation (Euler angles, placeholder)
            object_rot = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # TODO: Replace with real object rotation

            # Object velocities (placeholders)
            object_velp = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # TODO: Replace with real object linear velocity
            object_velr = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # TODO: Replace with real object angular velocity

        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)

        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        )
    
    def _get_obs(self, robot_state: Any) -> Dict[str, np.ndarray]:
        """
        Construct the Gym observation dict.

        This is the real-robot analogue of `_get_obs()` in the MuJoCo env.
        The CONTENT of `robot_state` is intentionally undefined here.
        """
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self._build_observation_vector(robot_state)
        
        if not self.has_object:
                achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        
        obs = np.concatenate(
                [
                    grip_pos,
                    object_pos.ravel(),
                    object_rel_pos.ravel(),
                    gripper_state,
                    object_rot.ravel(),
                    object_velp.ravel(),
                    object_velr.ravel(),
                    grip_velp.ravel(),
                    gripper_vel,
                ]
        )

        desired_goal_mujoco = real_to_mujoco_position(self.goal.copy())

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": desired_goal_mujoco.copy(),
        }
    
    # Action related methods
    def _set_action(self, action: np.ndarray):
        """
        Convert the policy action into separate control commands for position, orientation, and gripper.
        """
        assert action.shape == (4,)
        action = action.copy() # Ensure we don't modify the action outside this scope

        pos_ctrl, gripper_ctrl = action[:3], action[3]
        pos_ctrl *= 0.05  # Scale position control

        # UR3e gripper expects only one value instead of two values like fetch
        # Rescale gripper action from [-1, 1] to [1, 255]
        gripper_ctrl = np.clip(127 * gripper_ctrl + 128, 1, 255).astype(np.uint8)
        gripper_ctrl = np.array([gripper_ctrl])  # shape (1,)

        assert gripper_ctrl.shape == (1,)
        if self.block_gripper:
            gripper_ctrl = np.array([255]) # fully closed gripper
        
        processed_action = np.concatenate([pos_ctrl, gripper_ctrl]) #(4,)
        return processed_action
    
    def _action_to_target_pose(self, action: np.ndarray):
        """
        Generate a target TCP pose from the processed action from _set_action method.
        """

        assert self._last_robot_state is not None

        # Current TCP pose from RTDE
        current_pose = list(self._last_robot_state.actual_TCP_pose)

        # --- Position delta ---

        delta_pos_mujoco = action[:3] # Relative position update in MuJoCo frame
        delta_pos = mujoco_delta_to_real(delta_pos_mujoco) # Convert to real robot frame

        target_pose = current_pose.copy()
        target_pose[:3] += delta_pos # Relative position update

        # --- Orientation ---
        # Fixed orientation, exactly like MuJoCo
        target_pose[3:] = current_pose[3:]

        print(f"Waypoint: {target_pose}")

        return target_pose

    def _apply_action(self, action: np.ndarray) -> None:
        """
        Send waypoint to robot and block until execution completes.
        """

        assert self.robot_client is not None
        
        processed_action = self._set_action(action)
        target_pose = self._action_to_target_pose(processed_action)
        
        # Blocking MoveJ (synchronous)
        self._last_robot_state = self.robot_client.moveJ_blocking(target_pose)


    # Sample goal and reset methods
    
    # Reset method includes a simple homing behavior to a predefined pose using moveJ_blocking.
    def _reset_robot(self) -> None:
        """
        Minimal mock reset: Home pose via moveJ_blocking (safe start).
        """
        assert self.robot_client is not None, "Inject robot_client first!"
        
        # Target home pose list

        # Robot's initial position equivalent to mujooco's initial position
        home_pose = [-0.26915, 0.118, 0.38223, 0.1, 3.041, 0]

        self._last_robot_state = self.robot_client.moveJ_blocking(home_pose)
        self.initial_gripper_xpos = self._last_robot_state.actual_TCP_pose.copy()  # Store for goal sampling

    def _sample_goal(self):
        """
        Sample a new goal within a cubic task space defined relative to the robot base.

        """
        while True:
            if self.has_object:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)
            else:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
                goal_pose = list(goal) + list(self.initial_gripper_xpos[3:])  # Keep orientation same as initial    

                in_safe_workspace = goal[2] > 0.25 # Additional check to makesure goal is well above the table surface                

                if self.robot_client.check_reachability(goal_pose) and in_safe_workspace:
                    print(f"Sampled goal in robot frame: {goal}, corresponding to Mujoco frame: {real_to_mujoco_position(goal)}")
                    initial_distance_to_goal = goal_distance(np.array(self.initial_gripper_xpos[:3]), np.array(goal))
                    print(f"Initial distance to goal: {initial_distance_to_goal} m") 
                    print("\n")
                    return goal.copy()
                else:
                    print("\n")
                    print(f"Sampled unreachable/unsafe goal pose: {goal_pose}, resampling...")
                    continue
        

    
        
        