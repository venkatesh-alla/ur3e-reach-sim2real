# test_injection.py (in your_folder)
from tests.test_robot_client import FakeRobotClient

client = FakeRobotClient()
state = client.moveL_blocking([0.01, 0.0, 0.0, 0, 0, 0])  # Small dx, close grip
print(state.actual_TCP_pose[:3])  # ~[0.51, 0.0, 0.4]
print(state.gripper)  # [255]
