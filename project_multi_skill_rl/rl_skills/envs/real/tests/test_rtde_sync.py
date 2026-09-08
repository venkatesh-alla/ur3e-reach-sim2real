import time
import os

from real.robot_interface.ur3e_rtde_sync import UR3eRTDESyncClient


ROBOT_HOST = "172.17.0.2"      # URSim
# ROBOT_HOST = "192.168.123.3" # real robot
ROBOT_PORT = 30004
CONFIG_FILE = "/misis/project_multi_skill_rl/rl_skills/envs/real/robot_interface/sync_control_loop_configuration.xml"


# def main():
#     print("Connecting to URSim...")

#     client = UR3eRTDESyncClient(
#         robot_host=ROBOT_HOST,
#         robot_port=ROBOT_PORT,
#         config_filename=CONFIG_FILE,
#     )

#     client.connect()
#     print("Connected.")

#     try:
#         new_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#         while True:
#             # ---- Move 1 ----
#             pose1 = [-0.12, -0.23, 0.14, 0, 3.11, 0.04]
#             pose2 = [-0.12, -0.21, 0.21, 0, 3.11, 0.04]


#             new_pose = pose1 if new_pose == pose2 else pose2
#             print("Sending new pose:", new_pose)
#             state = client.moveJ_blocking(new_pose)
#             print("Move complete.")
#             time.sleep(3)
#             print("TCP:", state.actual_TCP_pose)
#             print("\n---\n")

#     finally:
#         print("Disconnecting...")
#         client.disconnect()
#         print("Done.")




def main():
    print("Connecting to URSim...")

    client = UR3eRTDESyncClient(
        robot_host=ROBOT_HOST,
        robot_port=ROBOT_PORT,
        config_filename=CONFIG_FILE,
    )

    client.connect()
    print("Connected.")

    try:
        # pose1 = [0.12, -0.21, 0.31, 0, 3.11, 0.04]
        # pose1 = [-0.15161103, -0.28426025, 0.34262234, 0, 3.11, 0.04]
        
        # pose1 = [-0.17818961, -0.04172079,  0.46724845, -1.1710377612251675e-14, 3.1100000000000043, 0.0400000000081]
        # pose2 = [0.6, 0.6, 0.6, 0, 0, 0]  # intentionally unreachable
        # pose3 = [-0.12, -0.23, 0.34, 0, 3.11, 0.04]

        pose1 = [-49.07e-03,-460.72e-03, 482.64e-03, 0.111, -2.230, 2.320]
        pose2 = [-49.07e-03,-460.72e-03, 300.0e-03, 0.111, -2.230, 2.320]
        pose3 = [0.6, 0.6, 0.6, 0, 0, 0]  # intentionally unreachable
        pose4 = [0.510045550391078, -0.11995395086705694, 0.2500000175833701, 0.1, 3.041, 0]


        test_poses = [pose1, pose2, pose3, pose4]

        for pose in test_poses:
            print("\nTesting pose:", pose)

            reachable = client.check_reachability(pose)
            print("Reachable:", reachable)

            if reachable:
                print("Executing motion...")
                state = client.moveJ_blocking(pose)
                print("Motion complete.")
                print("TCP:", state.actual_TCP_pose)
                time.sleep(5)
            else:
                print("Skipping motion (unreachable).")

            

    finally:
        print("Disconnecting...")
        client.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()