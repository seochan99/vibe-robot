"""VibeRobot configuration — environment-aware settings."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MENAGERIE_DIR = ASSETS_DIR / "mujoco_menagerie"
FRANKA_DIR = MENAGERIE_DIR / "franka_emika_panda"

ENV = os.getenv("VIBEROBOT_ENV", "local")  # "local" or "server"

CONFIG = {
    "local": {
        "vlm_backend": "api",           # GPT-4o API
        "vlm_model": "gpt-4o",
        "llm_model": "gpt-4o",
        "mujoco_renderer": "cpu",
        "sim_parallelism": 1,
        "gradio_host": "127.0.0.1",
        "gradio_port": 7860,
    },
    "server": {
        "vlm_backend": "local",          # Local VLM (LLaVA/CogVLM)
        "vlm_model": "llava-v1.6",
        "llm_model": "gpt-4o",
        "mujoco_renderer": "gpu",
        "sim_parallelism": 8,
        "gradio_host": "0.0.0.0",
        "gradio_port": 7860,
    },
}


def get_config() -> dict:
    """Return config for the current environment."""
    return CONFIG[ENV]


# API keys — loaded from env vars, never hardcoded
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# MuJoCo model paths
FRANKA_XML = str(FRANKA_DIR / "panda.xml")
FRANKA_SCENE_XML = str(FRANKA_DIR / "scene.xml")

# Simulation defaults
SIM_TIMESTEP = 0.002       # 500 Hz physics
CONTROL_TIMESTEP = 0.02    # 50 Hz control
MAX_SIM_DURATION = 30.0    # seconds

# Safety defaults
WORKSPACE_BOUNDS = {
    "x": (-0.5, 0.8),
    "y": (-0.5, 0.5),
    "z": (0.0, 0.8),
}
MAX_EE_VELOCITY = 0.5      # m/s
MAX_JOINT_TORQUE = 87.0    # Nm
