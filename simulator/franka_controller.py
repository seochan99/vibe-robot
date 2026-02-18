"""Franka Panda controller — IK/FK, motion primitives, grasping.

Uses roboticstoolbox-python for IK and provides high-level motion primitives
(move_to, pick, place, open_gripper, close_gripper) that the planner can chain.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional

import mujoco
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from simulator.mujoco_env import MuJoCoEnv


class GripperState(Enum):
    OPEN = "open"
    CLOSED = "closed"


class MotionResult:
    """Result of a motion primitive execution."""

    def __init__(self, success: bool, message: str = "", trajectory: list = None):
        self.success = success
        self.message = message
        self.trajectory = trajectory or []

    def __repr__(self) -> str:
        return f"MotionResult(success={self.success}, msg='{self.message}')"


class FrankaController:
    """High-level controller for Franka Panda arm in MuJoCo.

    Provides motion primitives that the Code-as-Policies planner uses:
    - move_to(pos, quat): Move end-effector to target pose
    - pick(object_name): Grasp an object
    - place(object_name, target_pos): Place an object at a target position
    - open_gripper / close_gripper
    """

    # Franka joint limits (radians)
    JOINT_LIMITS_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    JOINT_LIMITS_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    GRIPPER_OPEN = 0.04
    GRIPPER_CLOSED = 0.001
    GRASP_APPROACH_HEIGHT = 0.08   # approach from above
    KINEMATIC_SNAP_DIST = 0.045    # for kinematic attachment fallback

    def __init__(self, env: MuJoCoEnv):
        self.env = env
        self._gripper_state = GripperState.OPEN
        self._attached_object: Optional[str] = None
        self.realtime = False  # When True, pace execution to wall-clock time

        # roboticstoolbox Panda model for IK
        self._rtb_panda = rtb.models.Panda()

    def get_joint_positions(self) -> np.ndarray:
        """Current 7-DOF arm joint positions."""
        return self.env.data.qpos[:7].copy()

    def get_ee_pos(self) -> np.ndarray:
        """Current end-effector position."""
        return self.env.get_state().ee_pos

    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Solve inverse kinematics for a target pose.

        Args:
            target_pos: Target (x, y, z) in world frame.
            target_quat: Target orientation as quaternion (w, x, y, z). Uses top-down if None.

        Returns:
            7-DOF joint configuration or None if IK fails.
        """
        if target_quat is None:
            # Default: gripper pointing down
            T = SE3(target_pos) * SE3.Rx(np.pi)
        else:
            # Convert MuJoCo quaternion (w,x,y,z) to rotation matrix
            R = np.zeros((3, 3))
            mujoco.mju_quat2Mat(R.ravel(), target_quat)
            T = SE3.Rt(R, target_pos)

        q0 = self.get_joint_positions()
        sol = self._rtb_panda.ikine_LM(T, q0=q0, ilimit=500, slimit=100)
        if sol.success:
            return sol.q
        return None

    def move_to_joint(
        self,
        target_q: np.ndarray,
        duration: float = 2.0,
        record: bool = True,
    ) -> MotionResult:
        """Move arm joints to target configuration using interpolation.

        Uses smooth (cosine) interpolation for natural-looking motion.
        """
        current_q = self.get_joint_positions()
        n_steps = int(duration / self.env._control_dt)

        gripper_ctrl = self.GRIPPER_OPEN if self._gripper_state == GripperState.OPEN else self.GRIPPER_CLOSED

        wall_start = time.monotonic()

        for i in range(n_steps):
            # Cosine interpolation for smooth motion
            t = 0.5 * (1 - np.cos(np.pi * (i + 1) / n_steps))
            q_interp = current_q + t * (target_q - current_q)

            ctrl = np.zeros(self.env.N_ACTUATORS)
            ctrl[:7] = q_interp
            ctrl[7:] = gripper_ctrl

            self.env.step_control(ctrl)

            if record:
                self.env.record_frame()

            # Update kinematic attachment if holding an object
            if self._attached_object:
                self._update_attachment()

            # Real-time pacing: sleep to match wall-clock time
            if self.realtime:
                sim_elapsed = (i + 1) * self.env._control_dt
                wall_elapsed = time.monotonic() - wall_start
                sleep_time = sim_elapsed - wall_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        return MotionResult(success=True, message="Joint motion complete")

    def move_to(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray] = None,
        duration: float = 2.0,
        record: bool = True,
    ) -> MotionResult:
        """Move end-effector to a Cartesian target pose.

        Args:
            target_pos: (x, y, z) in world frame.
            target_quat: Orientation quaternion. Top-down if None.
            duration: Motion duration in seconds.
            record: Whether to record trajectory frames.
        """
        target_q = self.solve_ik(target_pos, target_quat)
        if target_q is None:
            return MotionResult(success=False, message=f"IK failed for target {target_pos}")

        # Check workspace bounds
        if not self.env.check_workspace_bounds(target_pos):
            return MotionResult(success=False, message=f"Target {target_pos} outside workspace")

        return self.move_to_joint(target_q, duration=duration, record=record)

    def open_gripper(self, duration: float = 0.5) -> MotionResult:
        """Open the gripper fingers."""
        self._gripper_state = GripperState.OPEN
        self._attached_object = None
        current_q = self.get_joint_positions()
        ctrl = np.zeros(self.env.N_ACTUATORS)
        ctrl[:7] = current_q
        ctrl[7:] = self.GRIPPER_OPEN

        wall_start = time.monotonic()
        n_steps = int(duration / self.env._control_dt)
        for i in range(n_steps):
            self.env.step_control(ctrl)
            if self.realtime:
                sim_elapsed = (i + 1) * self.env._control_dt
                sleep_time = sim_elapsed - (time.monotonic() - wall_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return MotionResult(success=True, message="Gripper opened")

    def close_gripper(self, duration: float = 0.5) -> MotionResult:
        """Close the gripper fingers."""
        self._gripper_state = GripperState.CLOSED
        current_q = self.get_joint_positions()
        ctrl = np.zeros(self.env.N_ACTUATORS)
        ctrl[:7] = current_q
        ctrl[7:] = self.GRIPPER_CLOSED

        wall_start = time.monotonic()
        n_steps = int(duration / self.env._control_dt)
        for i in range(n_steps):
            self.env.step_control(ctrl)
            if self.realtime:
                sim_elapsed = (i + 1) * self.env._control_dt
                sleep_time = sim_elapsed - (time.monotonic() - wall_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return MotionResult(success=True, message="Gripper closed")

    def pick(self, object_name: str, approach_height: float = None) -> MotionResult:
        """Pick up a named object.

        Sequence: approach from above → descend → close gripper → lift.
        Uses small offset retries and a staged vertical lift for stability.
        """
        approach_h = approach_height or self.GRASP_APPROACH_HEIGHT

        obj_pos = self.env.get_object_pos(object_name)
        if obj_pos is None:
            return MotionResult(success=False, message=f"Object '{object_name}' not found")

        # 1. Ensure open gripper before approach to avoid pushing/tipping objects.
        self.open_gripper()

        # 2. Retry grasp with slight lateral offsets to recover from unstable contacts.
        grasp_offsets = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.008, 0.0]),
            np.array([0.0, -0.008, 0.0]),
            np.array([0.008, 0.0, 0.0]),
            np.array([-0.008, 0.0, 0.0]),
        ]

        attached = False
        for attempt_i, offset in enumerate(grasp_offsets):
            candidate = obj_pos + offset

            approach_pos = candidate.copy()
            approach_pos[2] += approach_h
            result = self.move_to(approach_pos, duration=0.9 if attempt_i > 0 else 1.0)
            if not result.success:
                continue

            grasp_pos = candidate.copy()
            grasp_pos[2] += 0.012
            result = self.move_to(grasp_pos, duration=1.1)
            if not result.success:
                continue

            self.close_gripper(duration=0.7)
            if self._try_kinematic_attach(object_name, snap_scale=1.0 if attempt_i == 0 else 1.25):
                attached = True
                break

            # Re-open before trying a new approach.
            self.open_gripper(duration=0.35)

        if not attached:
            return MotionResult(
                success=False,
                message=f"Failed to grasp '{object_name}' after {len(grasp_offsets)} attempts",
            )

        # 3. Controlled two-stage vertical lift to reduce wobble right after grasp.
        ee_pos = self.get_ee_pos()
        mid_lift = ee_pos.copy()
        mid_lift[2] += max(0.04, approach_h * 0.55)
        result = self.move_to(mid_lift, duration=0.9)
        if not result.success:
            self._attached_object = None
            return MotionResult(
                success=False,
                message=f"Grasped '{object_name}' but mid-lift failed: {result.message}",
            )

        final_lift = mid_lift.copy()
        final_lift[2] += max(0.03, approach_h * 0.45)
        result = self.move_to(final_lift, duration=1.0)
        if not result.success:
            self._attached_object = None
            return MotionResult(
                success=False,
                message=f"Grasped '{object_name}' but final lift failed: {result.message}",
            )

        return MotionResult(success=True, message=f"Picked '{object_name}' with stabilized lift")

    def _try_kinematic_attach(self, object_name: str, snap_scale: float = 1.0) -> bool:
        """Attach when gripper/object distance is close enough."""
        ee_pos = self.get_ee_pos()
        current_obj_pos = self.env.get_object_pos(object_name)
        if current_obj_pos is None:
            return False

        dist = np.linalg.norm(ee_pos - current_obj_pos)
        if dist < self.KINEMATIC_SNAP_DIST * snap_scale:
            self._attached_object = object_name
            return True
        return False

    def place(
        self,
        object_name: str,
        target_pos: np.ndarray,
        approach_height: float = None,
    ) -> MotionResult:
        """Place the held object at a target position.

        Sequence: move above target → descend → open gripper → retreat.
        """
        approach_h = approach_height or self.GRASP_APPROACH_HEIGHT

        target_pos = np.asarray(target_pos, dtype=float)

        # 1. Move above target
        above_pos = target_pos.copy()
        above_pos[2] += approach_h
        result = self.move_to(above_pos)
        if not result.success:
            return result

        # 2. Descend to place height
        place_pos = target_pos.copy()
        place_pos[2] += 0.01
        result = self.move_to(place_pos, duration=1.0)
        if not result.success:
            return result

        # 3. Open gripper (releases object)
        self.open_gripper()
        self._attached_object = None

        # 4. Retreat upward
        retreat_pos = place_pos.copy()
        retreat_pos[2] += approach_h
        result = self.move_to(retreat_pos, duration=1.0)

        return MotionResult(
            success=True,
            message=f"Placed '{object_name}' at {target_pos.tolist()}",
        )

    def home(self) -> MotionResult:
        """Return to home configuration."""
        return self.move_to_joint(self.env.HOME_QPOS[:7], duration=2.0)

    def _update_attachment(self) -> None:
        """Move attached object to follow end-effector (kinematic snap)."""
        if self._attached_object is None:
            return
        body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, self._attached_object,
        )
        if body_id < 0:
            return

        ee_pos = self.get_ee_pos()
        # Find the joint address for this free body
        jnt_id = self.env.model.body_jntadr[body_id]
        if jnt_id >= 0 and self.env.model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
            qpos_adr = self.env.model.jnt_qposadr[jnt_id]
            # Set position to track end-effector (with small offset below)
            self.env.data.qpos[qpos_adr:qpos_adr + 3] = ee_pos - np.array([0, 0, 0.02])
            # Zero velocity
            qvel_adr = self.env.model.jnt_dofadr[jnt_id]
            self.env.data.qvel[qvel_adr:qvel_adr + 6] = 0

    @property
    def gripper_state(self) -> GripperState:
        return self._gripper_state

    @property
    def is_holding(self) -> bool:
        return self._attached_object is not None

    @property
    def held_object(self) -> Optional[str]:
        return self._attached_object
