"""
Synchronous RTDE communication client for UR robots.

Implements the exact handshake pattern from the official
Universal Robots RTDE example:

- Wait for robot ready flag (output_int_register_0 == 1)
- Send new pose
- Wait for motion confirmation (output_int_register_0 == 0)
- Acknowledge
- Return final robot state

Designed for deterministic Gym integration.
"""

import sys
import logging
from typing import List, Optional
import time

import real.robot_interface.rtde.rtde as rtde
import real.robot_interface.rtde.rtde_config as rtde_config


class UR3eRTDESyncClient:
    """
    Strict synchronous RTDE MoveJ client.
    """
    def __init__(
        self,
        robot_host: str,
        robot_port: int,
        config_filename: str,
    ):
        self.robot_host = robot_host
        self.robot_port = robot_port
        self.config_filename = config_filename

        self.con: Optional[rtde.RTDE] = None
        self.setp = None
        self.watchdog = None
        self.reachability_setp = None
        self.reachability_flag = None

        logging.getLogger().setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        conf = rtde_config.ConfigFile(self.config_filename)

        state_names, state_types = conf.get_recipe("state")
        setp_names, setp_types = conf.get_recipe("setp")
        watchdog_names, watchdog_types = conf.get_recipe("watchdog")
        reachability_setp_names, reachability_setp_types = conf.get_recipe("reachability_setp")
        reachability_flag_names, reachability_flag_types = conf.get_recipe("reachability_flag")

        self.con = rtde.RTDE(self.robot_host, self.robot_port)
        self.con.connect()
        self.con.get_controller_version()

        self.con.send_output_setup(state_names, state_types)
        self.setp = self.con.send_input_setup(setp_names, setp_types)
        self.watchdog = self.con.send_input_setup(watchdog_names, watchdog_types)
        self.reachability_setp = self.con.send_input_setup(reachability_setp_names, reachability_setp_types)
        self.reachability_flag = self.con.send_input_setup(reachability_flag_names, reachability_flag_types)

        # Initialize setpoint registers to zero
        for i in range(6):
            setattr(self.setp, f"input_double_register_{i}", 0.0)

        # Reachability pose
        for i in range(6, 12):
            setattr(self.reachability_setp, f"input_double_register_{i}", 0.0)

        # Watchdog register
        self.watchdog.input_int_register_0 = 0
        
        # Reachability flag
        self.reachability_flag.input_int_register_1 = 0

        

        if not self.con.send_start():
            raise RuntimeError("RTDE synchronization start failed")

    def disconnect(self) -> None:
        if self.con is not None:
            self.con.send_pause()
            self.con.disconnect()
            self.con = None

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def setp_to_list(sp):
        sp_list = []
        for i in range(0, 6):
            sp_list.append(sp.__dict__["input_double_register_%i" % i])
        return sp_list

    @staticmethod
    def list_to_setp(sp, list):
        for i in range(0, 6):
            sp.__dict__["input_double_register_%i" % i] = list[i]
        return sp
    
        # ------------------------------------------------------------------
    # Timeout-safe RTDE wait helper
    # ------------------------------------------------------------------

    def _wait_for_register_value(
        self,
        register_name: str,
        expected_value: int,
        timeout: float,
        timeout_message: str,
    ):
        """
        Wait until a robot output register reaches the expected value.

        Raises:
            RuntimeError: if RTDE connection is lost or timeout occurs.
        """

        assert self.con is not None

        start_time = time.time()

        while True:

            state = self.con.receive()

            if state is None:
                raise RuntimeError("RTDE connection lost")

            current_value = getattr(state, register_name)

            if current_value == expected_value:
                return state

            if time.time() - start_time > timeout:

                # ------------------------------------------
                # Clear watchdog trigger
                # ------------------------------------------
                if self.watchdog is not None:
                    self.watchdog.input_int_register_0 = 0
                    self.con.send(self.watchdog)

                # ------------------------------------------
                # Flush stale pose registers
                # ------------------------------------------
                if self.setp is not None:
                    zero_pose = [0.0] * 6
                    self.list_to_setp(self.setp, zero_pose)
                    self.con.send(self.setp)

                raise RuntimeError(timeout_message)

    # ------------------------------------------------------------------
    # Synchronous control primitive
    # ------------------------------------------------------------------

    # Adding timeout as well to prevent infinite blocking in case of communication issues
        
    def moveJ_blocking(self, target_pose: List[float]):
        """
        Blocking MoveJ with deterministic RTDE handshake and
        client-side pose flushing to avoid stale setpoints.

        Assumes:
        - Robot sets output_int_register_0 = 1 when ready
        - Robot consumes pose once and sets it to 0
        - Thread_1 continuously mirrors pose registers
        """

        assert self.con is not None
        assert self.setp is not None
        assert self.watchdog is not None
        assert len(target_pose) == 6

        # --------------------------------------------------
        # 1. Wait until robot is ready for a new command
        # --------------------------------------------------
        state = self._wait_for_register_value(
            register_name="output_int_register_0",
            expected_value=1,
            timeout=5.0,
            timeout_message=(
                "Timeout waiting for robot ready signal. "
                "Polyscope program may have stopped."
            ),
        )

        # --------------------------------------------------
        # 2. Write target pose to RTDE registers
        # --------------------------------------------------
        self.list_to_setp(self.setp, target_pose)
        self.con.send(self.setp)

        # --------------------------------------------------
        # 3. Trigger motion (edge)
        # --------------------------------------------------
        self.watchdog.input_int_register_0 = 1
        self.con.send(self.watchdog)

        # --------------------------------------------------
        # 4. Wait for motion completion
        # --------------------------------------------------
        state = self._wait_for_register_value(
            register_name="output_int_register_0",
            expected_value=0,
            timeout=20.0,
            timeout_message=(
                "Timeout waiting for motion completion. "
                "Polyscope program may have stopped."
            ),
        )

        # --------------------------------------------------
        # 5. ACK completion
        # --------------------------------------------------
        self.watchdog.input_int_register_0 = 0
        self.con.send(self.watchdog)

        # --------------------------------------------------
        # 6. FLUSH pose registers (CRITICAL)
        # --------------------------------------------------
        zero_pose = [0.0] * 6
        self.list_to_setp(self.setp, zero_pose)
        self.con.send(self.setp)

        return state


    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_actual_TCP_pose(state) -> List[float]:
        """
        Extract TCP pose from RTDE state.
        """
        return list(state.actual_TCP_pose)

    @staticmethod
    def get_target_q(state) -> List[float]:
        return list(state.target_q)

    @staticmethod
    def get_target_qd(state) -> List[float]:
        return list(state.target_qd)

    # ------------------------------------------------------------------
    # Reachability check primitive
    # ------------------------------------------------------------------

    def wait_for_robot_ready(self):
        assert self.con is not None

        while True:
            state = self.con.receive()
            if state is None:
                raise RuntimeError("RTDE connection lost")

            if state.output_int_register_2 == 12345:
                return

    
    
    def check_reachability(self, pose):
        assert self.con is not None
        assert self.reachability_setp is not None
        assert self.reachability_flag is not None
        assert len(pose) == 6

        # --- Ensure Polyscope program is running ---
        self.wait_for_robot_ready()

        # --- Load pose into input registers 6–11 ---
        for i in range(6):
            setattr(
                self.reachability_setp,
                f"input_double_register_{6 + i}",
                pose[i]
            )
        self.con.send(self.reachability_setp)

        # --- Ensure request line LOW (clear stale edge) ---
        self.reachability_flag.input_int_register_1 = 0
        self.con.send(self.reachability_flag)

        # --- Raise request (0 → 1 edge) ---
        self.reachability_flag.input_int_register_1 = 1
        self.con.send(self.reachability_flag)

        # --- Wait for sticky result ---
        while True:
            state = self.con.receive()
            if state is None:
                raise RuntimeError("RTDE connection lost")

            result_flag = state.output_int_register_1

            if result_flag == 0:
                result = True
                break
            elif result_flag == -1:
                result = False
                break
            # result_flag == 1 → still idle / computing

        # --- Acknowledge result (allow robot to reset) ---
        self.reachability_flag.input_int_register_1 = 0
        self.con.send(self.reachability_flag)

        return result


