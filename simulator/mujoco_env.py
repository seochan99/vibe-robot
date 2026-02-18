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
        # Interactive free camera state used by MJPEG rendering.
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._camera)
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.azimuth = 120.0
        self._camera.elevation = -22.0
        self._camera.distance = 2.0
        self._camera.lookat[:] = np.array([0.45, 0.0, 0.35], dtype=float)

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
        # Prefer explicit EE/gripper sites if available.
        for name in ["gripper", "grip_site", "ee_site", "attachment_site", "end_effector"]:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                pos = self.data.site_xpos[site_id].copy()
                quat = np.zeros(4, dtype=float)
                mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id].copy())
                return pos, quat

        # Use midpoint between left/right fingers when possible (best grasp center proxy).
        left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        if left_id >= 0 and right_id >= 0:
            pos = 0.5 * (self.data.xpos[left_id] + self.data.xpos[right_id])
            hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
            if hand_id >= 0:
                quat = self.data.xquat[hand_id].copy()
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            return pos.copy(), quat

        # Try to find the hand body.
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

    def get_object_geom_size(self, name: str) -> Optional[np.ndarray]:
        """Get representative geom size for a named object body."""
        with self._lock:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                return None
            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] == body_id:
                    return self.model.geom_size[geom_id].copy()
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
            renderer.update_scene(self.data, camera=self._camera)
            return renderer.render()

    def get_camera_state(self) -> dict:
        """Get current free-camera parameters."""
        with self._lock:
            return {
                "azimuth": float(self._camera.azimuth),
                "elevation": float(self._camera.elevation),
                "distance": float(self._camera.distance),
                "lookat": [float(v) for v in self._camera.lookat],
            }

    def set_camera_state(
        self,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
        distance: Optional[float] = None,
        lookat: Optional[np.ndarray] = None,
    ) -> dict:
        """Set free-camera parameters with safe clamping."""
        with self._lock:
            if azimuth is not None:
                self._camera.azimuth = float(azimuth)
            if elevation is not None:
                self._camera.elevation = float(np.clip(elevation, -89.0, 89.0))
            if distance is not None:
                self._camera.distance = float(np.clip(distance, 0.4, 6.0))
            if lookat is not None and len(lookat) == 3:
                lookat_arr = np.asarray(lookat, dtype=float).copy()
                for i, axis in enumerate(("x", "y", "z")):
                    lo, hi = WORKSPACE_BOUNDS[axis]
                    margin = 0.35 if axis != "z" else 0.4
                    lookat_arr[i] = np.clip(lookat_arr[i], lo - margin, hi + margin)
                self._camera.lookat[:] = lookat_arr
            return self.get_camera_state()

    def set_object_pos(self, name: str, pos: np.ndarray) -> bool:
        """Move a free-body object directly without rebuilding the whole scene."""
        with self._lock:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                return False

            jnt_id = self.model.body_jntadr[body_id]
            if jnt_id < 0 or self.model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_FREE:
                return False

            qpos_adr = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[qpos_adr:qpos_adr + 3] = np.asarray(pos, dtype=float)[:3]
            self.data.qvel[self.model.jnt_dofadr[jnt_id]:self.model.jnt_dofadr[jnt_id] + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            return True

    def project_screen_to_plane(
        self,
        nx: float,
        ny: float,
        plane_z: float = 0.35,
        aspect: float = 4.0 / 3.0,
    ) -> Optional[np.ndarray]:
        """Project normalized screen coord to a world point on z=plane_z."""
        with self._lock:
            nx = float(np.clip(nx, 0.0, 1.0))
            ny = float(np.clip(ny, 0.0, 1.0))
            az = np.deg2rad(float(self._camera.azimuth))
            el = np.deg2rad(float(self._camera.elevation))
            dist = max(0.4, float(self._camera.distance))
            lookat = np.array(self._camera.lookat, dtype=float)

            # Direction from camera position toward lookat.
            forward = np.array(
                [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
                dtype=float,
            )
            cam_pos = lookat - forward * dist

            world_up = np.array([0.0, 0.0, 1.0], dtype=float)
            right = np.cross(forward, world_up)
            right_norm = float(np.linalg.norm(right))
            if right_norm < 1e-6:
                right = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                right /= right_norm
            up = np.cross(right, forward)
            up /= max(1e-6, float(np.linalg.norm(up)))

            x_ndc = (nx - 0.5) * 2.0
            y_ndc = (0.5 - ny) * 2.0
            fovy_rad = np.deg2rad(float(self.model.vis.global_.fovy))
            tan_half = np.tan(fovy_rad * 0.5)

            ray = forward + right * x_ndc * tan_half * aspect + up * y_ndc * tan_half
            ray_norm = float(np.linalg.norm(ray))
            if ray_norm < 1e-9:
                return None
            ray /= ray_norm

            if abs(ray[2]) < 1e-7:
                return None
            t = (float(plane_z) - cam_pos[2]) / ray[2]
            if t <= 0.0:
                return None
            return cam_pos + ray * t

    def project_world_to_screen(
        self,
        world_pos: np.ndarray,
        aspect: float = 4.0 / 3.0,
    ) -> Optional[tuple[float, float, float]]:
        """Project world point to normalized screen coord and depth."""
        with self._lock:
            az = np.deg2rad(float(self._camera.azimuth))
            el = np.deg2rad(float(self._camera.elevation))
            dist = max(0.4, float(self._camera.distance))
            lookat = np.array(self._camera.lookat, dtype=float)

            forward = np.array(
                [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
                dtype=float,
            )
            cam_pos = lookat - forward * dist

            world_up = np.array([0.0, 0.0, 1.0], dtype=float)
            right = np.cross(forward, world_up)
            right_norm = float(np.linalg.norm(right))
            if right_norm < 1e-6:
                right = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                right /= right_norm
            up = np.cross(right, forward)
            up /= max(1e-6, float(np.linalg.norm(up)))

            rel = np.asarray(world_pos, dtype=float) - cam_pos
            depth = float(np.dot(rel, forward))
            if depth <= 1e-6:
                return None

            x_cam = float(np.dot(rel, right))
            y_cam = float(np.dot(rel, up))
            fovy_rad = np.deg2rad(float(self.model.vis.global_.fovy))
            tan_half = np.tan(fovy_rad * 0.5)
            x_ndc = x_cam / (depth * tan_half * max(1e-6, float(aspect)))
            y_ndc = y_cam / (depth * tan_half)

            nx = 0.5 + x_ndc * 0.5
            ny = 0.5 - y_ndc * 0.5
            return float(nx), float(ny), depth

    def pick_object_from_screen(
        self,
        object_names: list[str],
        nx: float,
        ny: float,
        aspect: float = 4.0 / 3.0,
        pick_radius: float = 0.08,
    ) -> Optional[dict]:
        """Pick the nearest visible object to a screen coord."""
        with self._lock:
            nx = float(np.clip(nx, 0.0, 1.0))
            ny = float(np.clip(ny, 0.0, 1.0))
            radius = max(0.005, float(pick_radius))

            best = None
            for name in object_names:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id < 0:
                    continue
                pos = self.data.xpos[body_id].copy()
                screen = self.project_world_to_screen(pos, aspect=aspect)
                if screen is None:
                    continue
                sx, sy, depth = screen
                dist = float(np.hypot(sx - nx, sy - ny))
                if dist > radius:
                    continue

                if best is None:
                    best = (name, pos, sx, sy, dist, depth)
                    continue

                _, _, _, _, best_dist, best_depth = best
                if dist < best_dist or (abs(dist - best_dist) < 1e-6 and depth < best_depth):
                    best = (name, pos, sx, sy, dist, depth)

            if best is None:
                return None

            name, pos, sx, sy, dist, depth = best
            return {
                "name": name,
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                "screen": [float(sx), float(sy)],
                "distance": float(dist),
                "depth": float(depth),
            }

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
