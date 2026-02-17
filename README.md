# VibeRobot: Vibe-to-Verify for Safe Robotic Manipulation

> Bridging the gap between natural language "vibe coding" and safe physical robot execution.

## Overview

VibeRobot enables users to control a robot arm using vague, natural language commands (e.g., "prevent it from flying away!!"). The system:

1. **Infers intent** using Theory of Mind — understanding what users *mean*, not just what they *say*
2. **Discovers affordances** using Gibson's ecological psychology — finding latent object capabilities activated by context
3. **Generates safe plans** with Code-as-Policies style primitive chaining
4. **Enforces safety** through pre/invariant/post conditions and tripwire contracts
5. **Requires preview approval** — simulation must be reviewed before physical execution

## Architecture

```
Layer 1: VLM Interface (Intent Inference + Scene Understanding + Task Spec)
    ↓
Layer 2: Convertor (Affordance Engine + Plan Generator + Safety Contract)
    ↓
Layer 3: Robot Arm Simulator (MuJoCo + Franka Panda + Preview System)
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/seochan99/vibe-robot.git
cd vibe-robot

# Create virtual environment (Python 3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download Franka Panda model
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git assets/mujoco_menagerie

# Run tests
pytest tests/ -v

# Launch UI
python -m ui.gradio_app
```

## Key Demo: Wind + Paper Scenario

```
User: "날라가지 않게 막아!!" (Prevent it from flying away!!)
  ↓
Intent: Secure loose papers against wind disturbance
  ↓
Affordance: Book's latent "weight_provider" affordance activated
  ↓
Plan: pick(book) → place(book, on=papers)
  ↓
Safety Contract → Sim Preview → User Approval → Execute
```

## Tech Stack

| Component | Choice |
|-----------|--------|
| Simulator | MuJoCo 3.5 |
| Robot Model | Franka Panda (MuJoCo Menagerie) |
| IK/FK | roboticstoolbox-python |
| VLM/LLM | ChatGPT OAuth / GPT-4o API (pluggable) |
| UI | Gradio |

## Authentication: ChatGPT Login (No API Key Needed)

VibeRobot supports **ChatGPT Plus/Pro subscription login** — no API key required.
This uses the same OAuth mechanism as OpenCode, OpenClaw, and OpenAI's Codex CLI.

```bash
# Login with ChatGPT account (opens browser)
python auth_cli.py login

# For SSH/headless environments
python auth_cli.py login --device

# Check auth status
python auth_cli.py status

# List available providers
python auth_cli.py providers
```

### Provider Priority (auto mode)

| Priority | Provider | Auth Method |
|----------|----------|-------------|
| 1 | `chatgpt_oauth` | ChatGPT Plus/Pro login (browser OAuth) |
| 2 | `codex_cli` | OpenAI Codex CLI (`npm i -g @openai/codex`) |
| 3 | `openai_api` | Standard API key (`$OPENAI_API_KEY`) |

The system auto-detects the first available authenticated provider.

## Project Structure

```
viberobot/
├── providers/          # LLM provider abstraction (ChatGPT OAuth, OpenAI API, Codex CLI)
├── simulator/          # MuJoCo environment, Franka controller, scene builder
├── core/               # AI pipeline: intent, affordance, planner, safety
├── ui/                 # Gradio app, failure cards
├── study/              # User study: tasks, instruments, logging
└── tests/              # Unit + integration tests (47 tests)
```

## Tests

```bash
pytest tests/ -v  # 47 tests covering smoke, affordance, pipeline, and providers
```
