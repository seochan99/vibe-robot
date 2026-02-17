"""Simulation preview recording and playback.

Records MuJoCo simulation frames as images for Gradio preview display.
Supports saving to MP4/GIF for CHI paper demos.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from simulator.mujoco_env import MuJoCoEnv


class PreviewRecorder:
    """Records simulation frames for preview playback."""

    def __init__(self, env: MuJoCoEnv, width: int = 640, height: int = 480):
        self.env = env
        self.width = width
        self.height = height
        self._frames: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._renderer = mujoco.Renderer(env.model, height=height, width=width)

    def capture_frame(self) -> np.ndarray:
        """Capture current frame as RGB array."""
        self._renderer.update_scene(self.env.data)
        frame = self._renderer.render()
        self._frames.append(frame.copy())
        self._timestamps.append(self.env.sim_time)
        return frame

    def get_frames(self) -> list[np.ndarray]:
        """Return all captured frames."""
        return self._frames

    def get_timestamps(self) -> list[float]:
        """Return timestamps for each frame."""
        return self._timestamps

    def reset(self) -> None:
        """Clear recorded frames."""
        self._frames = []
        self._timestamps = []

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    @property
    def duration(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        return self._timestamps[-1] - self._timestamps[0]

    def save_frames_as_images(self, output_dir: str, prefix: str = "frame") -> list[str]:
        """Save frames as individual PNG images. Returns list of paths."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow required for saving frames: pip install Pillow")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, frame in enumerate(self._frames):
            path = out / f"{prefix}_{i:04d}.png"
            Image.fromarray(frame).save(str(path))
            paths.append(str(path))
        return paths

    def save_gif(self, output_path: str, fps: int = 20) -> str:
        """Save frames as an animated GIF."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow required for GIF export: pip install Pillow")

        if not self._frames:
            raise ValueError("No frames recorded")

        images = [Image.fromarray(f) for f in self._frames]
        duration_ms = int(1000 / fps)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
        return output_path

    def get_thumbnail(self, frame_idx: int = 0) -> Optional[np.ndarray]:
        """Get a single frame as thumbnail."""
        if frame_idx < len(self._frames):
            return self._frames[frame_idx]
        return None


class TrajectoryPlayer:
    """Replays a recorded trajectory in MuJoCo for preview."""

    def __init__(self, env: MuJoCoEnv, trajectory: list[dict]):
        """
        Args:
            env: MuJoCo environment.
            trajectory: List of state dicts from MuJoCoEnv.get_trajectory().
        """
        self.env = env
        self.trajectory = trajectory
        self._recorder = PreviewRecorder(env)

    def replay(self, record: bool = True) -> list[np.ndarray]:
        """Replay trajectory and optionally capture frames.

        Returns list of RGB frames if record=True.
        """
        self._recorder.reset()
        frames = []

        for state in self.trajectory:
            qpos = np.array(state["qpos"])
            self.env.data.qpos[:len(qpos)] = qpos
            mujoco.mj_forward(self.env.model, self.env.data)

            if record:
                frame = self._recorder.capture_frame()
                frames.append(frame)

        return frames

    def save_preview(self, output_path: str, fps: int = 20) -> str:
        """Replay and save as GIF."""
        self.replay(record=True)
        return self._recorder.save_gif(output_path, fps=fps)

    @property
    def recorder(self) -> PreviewRecorder:
        return self._recorder
