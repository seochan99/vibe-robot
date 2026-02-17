"""Smoke tests — verify MuJoCo + Franka basic functionality."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FRANKA_SCENE_XML


class TestMuJoCoSmoke:
    """Basic MuJoCo installation and model loading tests."""

    def test_mujoco_import(self):
        import mujoco
        assert hasattr(mujoco, "MjModel")

    def test_load_franka_scene(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(FRANKA_SCENE_XML)
        data = mujoco.MjData(model)
        assert model.nq > 0  # Has joints
        assert model.nu > 0  # Has actuators

    def test_franka_step(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(FRANKA_SCENE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        assert data.time > 0

    def test_franka_actuator_count(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(FRANKA_SCENE_XML)
        # Franka Panda: 7 arm joints + 1 finger actuator = 8 actuators
        assert model.nu == 8


class TestMuJoCoEnv:
    """Test MuJoCoEnv wrapper."""

    def test_create_env(self):
        from simulator.mujoco_env import MuJoCoEnv
        env = MuJoCoEnv()
        state = env.reset()
        assert state.time == 0.0 or state.time < 0.01

    def test_step_env(self):
        from simulator.mujoco_env import MuJoCoEnv
        env = MuJoCoEnv()
        env.reset()
        state = env.step()
        assert state.time > 0

    def test_get_state(self):
        from simulator.mujoco_env import MuJoCoEnv
        env = MuJoCoEnv()
        env.reset()
        state = env.get_state()
        assert state.qpos is not None
        assert state.ee_pos is not None
        assert len(state.ee_pos) == 3
