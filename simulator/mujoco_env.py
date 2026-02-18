"""MuJoCo environment wrapper for VibeRobot.

Manages simulation lifecycle: load scene, step physics, read state, apply control.
Supports both headless and rendered modes.
"""

from __future__ import annotations

import tempfile
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from config import (
    CONTROL_TIMESTEP,
    FRANKA_DIR,
    FRANKA_SCENE_XML,
    MAX_SIM_DURATION,
    SIM_TIMESTEP,
    WORKSPACE_BOUNDS,
)


@dataclass
class SimState:
    """Snapshot of the simulation state."""

    time: float
    qpos: np.ndarray  # joint positions
    qvel: np.ndarray  # joint velocities
    ee_pos: np.ndarray  # end-effector position (3,)
    ee_quat: np.ndarray  # end-effector orientation (4,)
    object_states: dict  # {name: {"pos": (3,), "quat": (4,)}}


class MuJoCoEnv:
    """MuJoCo simulation environment for Franka Panda pick-and-place tasks."""

    # Franka Panda: 7 arm + 1 coupled finger actuator = 8 actuators
    # (qpos has 9 entries: 7 arm + 2 finger joint positions)
    N_ARM_JOINTS = 7
    N_FINGER_JOINTS = 2
    N_ACTUATORS = 8

    # Default home position for Franka Panda
    HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])

    def __init__(
        self,
        scene_xml: Optional[str] = None,
        xml_string: Optional[str] = None,
        timestep: float = SIM_TIMESTEP,
    ):
        """Initialize the MuJoCo environment.

        Args:
            scene_xml: Path to MJCF XML file. Uses default Franka scene if None.
            xml_string: Raw MJCF XML string. Takes precedence over scene_xml.
            timestep: Physics timestep in seconds.
        """
        if xml_string is not None:
            # Write to temp file in Franka dir so includes/meshes resolve
            self._tmp_xml = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".xml",
                dir=str(FRANKA_DIR),
                delete=False,
            )
            self._tmp_xml.write(xml_string)
            self._tmp_xml.flush()
            self.model = mujoco.MjModel.from_xml_path(self._tmp_xml.name)
        elif scene_xml is not None:
            self.model = mujoco.MjModel.from_xml_path(scene_xml)
        else:
            self.model = mujoco.MjModel.from_xml_path(FRANKA_SCENE_XML)

        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self._timestep = timestep
        self._control_dt = CONTROL_TIMESTEP
        self._steps_per_control = int(self._control_dt / self._timestep)
        # MuJoCo renderer objects are not safe to share across threads.
        # Keep a per-thread renderer to avoid cross-thread OpenGL/context hangs.
        self._renderer_local = threading.local()
        self._lock = threading.RLock()
        self._trajectory: list[dict] = []
        # Frame capture mode for GIF generation
        self._capture_frames = False
        self._capture_every: int = 10
        self._capture_counter: int = 0
        self._captured_frames: list[np.ndarray] = []
        self._capture_width: int = 320
        self._capture_height: int = 240

    def reset(self, qpos: Optional[np.ndarray] = None) -> SimState:
        """Reset simulation to initial or specified state."""
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            if qpos is not None:
                self.data.qpos[: len(qpos)] = qpos
            else:
                self.data.qpos[: len(self.HOME_QPOS)] = self.HOME_QPOS
            mujoco.mj_forward(self.model, self.data)
            self._trajectory = []
            return self.get_state()

    def step(self, ctrl: Optional[np.ndarray] = None, n_steps: int = 1) -> SimState:
        """Step the simulation forward.

        Args:
            ctrl: Control signal for actuators. None keeps current control.
            n_steps: Number of physics steps.
        """
        with self._lock:
            if ctrl is not None:
                self.data.ctrl[: len(ctrl)] = ctrl
            for _ in range(n_steps):
                mujoco.mj_step(self.model, self.data)
            return self.get_state()

    def step_control(self, ctrl: Optional[np.ndarray] = None) -> SimState:
        """Step one control cycle (multiple physics steps)."""
        result = self.step(ctrl, n_steps=self._steps_per_control)
        if self._capture_frames:
            self._capture_counter += 1
            if self._capture_counter % self._capture_every == 0:
                self._captured_frames.append(
                    self.render_offscreen(self._capture_width, self._capture_height)
                )
        return result

    def enable_frame_capture(
        self, every_n: int = 10, width: int = 320, height: int = 240
    ) -> None:
        """Start auto-capturing frames every N control steps."""
        self._capture_frames = True
        self._capture_every = every_n
        self._capture_counter = 0
        self._captured_frames = []
        self._capture_width = width
        self._capture_height = height

    def disable_frame_capture(self) -> list[np.ndarray]:
        """Stop capturing and return all captured frames."""
        self._capture_frames = False
        frames = self._captured_frames
        self._captured_frames = []
        return frames

    def get_state(self) -> SimState:
        """Get current simulation state."""
        with self._lock:
            ee_pos, ee_quat = self._get_ee_pose()
            obj_states = self._get_object_states()
            return SimState(
                time=self.data.time,
                qpos=self.data.qpos.copy(),
                qvel=self.data.qvel.copy(),
                ee_pos=ee_pos,
                ee_quat=ee_quat,
                object_states=obj_states,
            )

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get end-effector (hand) position and orientation."""
        # Try to find the hand site/body
        for name in ["hand", "grip_site", "end_effector"]:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                pos = self.data.xpos[body_id].copy()
                quat = self.data.xquat[body_id].copy()
                return pos, quat
        # Fallback: use the last link before fingers
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link8")
            if body_id >= 0:
                return self.data.xpos[body_id].copy(), self.data.xquat[body_id].copy()
        except Exception:
            pass
        # Ultimate fallback
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def _get_object_states(self) -> dict:
        """Get positions/orientations of all free-floating objects."""
        states = {}
        for i in range(self.model.njnt):
            jnt_type = self.model.jnt_type[i]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                body_id = self.model.jnt_bodyid[i]
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if name:
                    states[name] = {
                        "pos": self.data.xpos[body_id].copy(),
                        "quat": self.data.xquat[body_id].copy(),
                    }
        return states

    def get_object_pos(self, name: str) -> Optional[np.ndarray]:
        """Get position of a named object."""
        with self._lock:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return self.data.xpos[body_id].copy()
            return None

    def check_contact(self, name1: str, name2: str) -> bool:
        """Check if two named geoms are in contact."""
        with self._lock:
            geom1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name1)
            geom2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name2)
            if geom1 < 0 or geom2 < 0:
                return False
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if {c.geom1, c.geom2} == {geom1, geom2}:
                    return True
            return False

    def check_workspace_bounds(self, pos: np.ndarray) -> bool:
        """Check if a position is within the workspace bounds."""
        for i, axis in enumerate(["x", "y", "z"]):
            lo, hi = WORKSPACE_BOUNDS[axis]
            if pos[i] < lo or pos[i] > hi:
                return False
        return True

    def record_frame(self) -> None:
        """Record current state for trajectory playback."""
        with self._lock:
            state = self.get_state()
            self._trajectory.append(
                {
                    "time": state.time,
                    "qpos": state.qpos.tolist(),
                    "ee_pos": state.ee_pos.tolist(),
                    "objects": {
                        k: {"pos": v["pos"].tolist(), "quat": v["quat"].tolist()}
                        for k, v in state.object_states.items()
                    },
                }
            )

    def get_trajectory(self) -> list[dict]:
        """Return recorded trajectory frames."""
        with self._lock:
            return list(self._trajectory)

    def render_offscreen(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Render the scene to an RGB image (offscreen)."""
        with self._lock:
            renderer = self._get_thread_renderer(width, height)
            renderer.update_scene(self.data)
            return renderer.render()

    def _get_thread_renderer(self, width: int, height: int):
        """Get or create a renderer bound to the current thread."""
        renderer = getattr(self._renderer_local, "renderer", None)
        renderer_size = getattr(self._renderer_local, "renderer_size", None)

        if renderer is not None and renderer_size == (width, height):
            return renderer

        if renderer is not None:
            close = getattr(renderer, "close", None)
            if callable(close):
                close()

        renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer_local.renderer = renderer
        self._renderer_local.renderer_size = (width, height)
        return renderer

    @property
    def sim_time(self) -> float:
        return self.data.time

    @property
    def n_joints(self) -> int:
        return self.model.njnt

    @property
    def n_actuators(self) -> int:
        return self.model.nu
