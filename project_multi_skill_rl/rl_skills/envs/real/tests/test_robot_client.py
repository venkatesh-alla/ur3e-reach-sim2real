# robot_interface/test_robot_client.py
import numpy as np

class FakeRobotClient:
    """
    Mock RTDE client for Gym env testing.
    Simulates moveJ_blocking: applies relative deltas, clips workspace, updates state.
    """
    def __init__(self, initial_tcp_pose=[0.5, 0.0, 0.4, 0.0, 3.14, 0.0]):
        # UR3e home-like TCP pose (x,y,z,rx,ry,rz)
        self.actual_TCP_pose = np.array(initial_tcp_pose)
        self.actual_q = np.zeros(6)  # Joints (unused but exposed)
        self.gripper = np.array([255])  # Closed

    def moveJ_blocking(self, target_pose):
        """
        Called by env._apply_action with 6D list [x,y,z,rx,ry,rz].
        Applies as relative delta to current (matching _action_to_target_pose logic).
        Clips pos to safe UR3e workspace.
        Returns fresh state object.
        """
        # Convert list to array
        target_pose = np.array(target_pose)
        assert len(target_pose) == 6

        # Relative update (env already scaled dx/dy/dz by 0.05)
        self.actual_TCP_pose += target_pose  # Pos + ori deltas

        #### Clipping code to simulate safe workspace
        # # # # Clip safe workspace (adjust to your UR3e setup)
        # # # self.actual_TCP_pose[:3] = np.clip(self.actual_TCP_pose[:3], [0.3, -0.3, 0.2], [0.8, 0.3, 0.6])
        # # # # Gripper from target[3] (already 1-255 uint8 via _set_action)
        # # # self.gripper = np.array([int(np.clip(target_pose[3], 1, 255))])

        # Return new state object (must have .actual_TCP_pose etc.)
        class RobotState:
            actual_TCP_pose = self.actual_TCP_pose.copy()
            actual_q = self.actual_q.copy()
            gripper = self.gripper.copy()
        return RobotState()
