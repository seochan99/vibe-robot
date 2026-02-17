"""FastAPI backend for VibeRobot! web interface.

Exposes the Python pipeline as a REST API consumed by the Next.js frontend.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline import PipelineResult, PipelineStage, VibeRobotPipeline
from simulator.franka_controller import FrankaController
from simulator.mujoco_env import MuJoCoEnv
from simulator.scene_builder import TASK_PRESETS, build_scene_xml
from study.logger import InteractionLogger
from ui.failure_cards import FailureCardGenerator

app = FastAPI(title="VibeRobot! API", version="0.1.0")

# CORS for Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── State ─────────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.pipeline: Optional[VibeRobotPipeline] = None
        self.current_result: Optional[PipelineResult] = None
        self.scene_name: str = ""
        self.logger = InteractionLogger()
        self.failure_gen = FailureCardGenerator()

    def init_pipeline(self, scene_name: str):
        self.scene_name = scene_name
        objects = TASK_PRESETS.get(scene_name, TASK_PRESETS["wind_paper"])
        xml = build_scene_xml(objects, include_wind="wind" in scene_name)
        env = MuJoCoEnv(xml_string=xml)
        controller = FrankaController(env)
        env.reset()
        self.pipeline = VibeRobotPipeline(env, controller, scene_objects=objects)


state = AppState()


# ── Request / Response models ────────────────────────────────────────────────

class CommandRequest(BaseModel):
    command: str
    scene: str = "wind_paper"
    model: str = "rule_based"


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "viberobot"}


@app.post("/api/command")
def run_command(req: CommandRequest):
    """Run the pipeline for a user command."""
    if state.pipeline is None or state.scene_name != req.scene:
        state.init_pipeline(req.scene)

    state.logger.log_command(req.command)
    conditions = ["wind_present"] if "wind" in req.scene else []
    result = state.pipeline.run_sync(req.command, conditions=conditions, auto_approve=False)
    state.current_result = result

    return _serialize_result(result)


@app.post("/api/approve")
def approve_execution():
    """Approve and execute the current plan."""
    if state.current_result is None or state.pipeline is None:
        return {"success": False, "error": "No plan to execute", "results": []}

    state.logger.log_approval(approved=True)
    result = state.pipeline.approve_and_execute(state.current_result)
    state.current_result = result
    state.logger.log_execution_result(result.execution_results)

    return {
        "success": result.success,
        "results": result.execution_results,
        "error": result.error,
    }


@app.post("/api/reject")
def reject_execution():
    """Reject the current plan."""
    state.logger.log_approval(approved=False)
    state.current_result = None
    return {"status": "rejected"}


@app.get("/api/scenes")
def list_scenes():
    """List available scenes."""
    return {
        "scenes": [
            {"id": k, "objects": len(v)}
            for k, v in TASK_PRESETS.items()
        ]
    }


@app.get("/api/models")
def list_models():
    """List available AI models."""
    models = [
        {"id": "rule_based", "name": "Rule-Based (No AI)", "requires_auth": False},
        {"id": "gpt-5.3-codex", "name": "GPT-5.3 Codex", "requires_auth": True},
        {"id": "gpt-5.2-codex", "name": "GPT-5.2 Codex", "requires_auth": True},
        {"id": "gpt-5-codex-mini", "name": "GPT-5 Codex Mini", "requires_auth": True},
        {"id": "gpt-4o", "name": "GPT-4o", "requires_auth": True},
        {"id": "o3", "name": "o3", "requires_auth": True},
        {"id": "o4-mini", "name": "o4-mini", "requires_auth": True},
    ]
    return {"models": models}


# ── Auth routes ──────────────────────────────────────────────────────────────

@app.get("/api/auth/status")
def auth_status():
    """Check ChatGPT OAuth status."""
    try:
        from providers.chatgpt_auth import ChatGPTAuth
        auth = ChatGPTAuth()
        return auth.status
    except Exception:
        return {"authenticated": False, "message": "Auth module unavailable"}


@app.post("/api/auth/login")
def auth_login():
    """Trigger browser OAuth login."""
    try:
        from providers.chatgpt_auth import ChatGPTAuth
        auth = ChatGPTAuth()
        tokens = auth.login(headless=False)
        return {
            "authenticated": True,
            "plan": tokens.plan_type or "unknown",
            "account_id": tokens.account_id[:8] + "..." if tokens.account_id else "",
        }
    except Exception as e:
        return {"authenticated": False, "message": str(e)}


@app.post("/api/auth/logout")
def auth_logout():
    """Remove stored credentials."""
    try:
        from providers.chatgpt_auth import ChatGPTAuth
        ChatGPTAuth().logout()
        return {"status": "logged_out"}
    except Exception:
        return {"status": "ok"}


# ── Serialization ────────────────────────────────────────────────────────────

def _serialize_result(result: PipelineResult) -> dict:
    """Convert PipelineResult to JSON-safe dict."""
    out: dict = {"stage": result.stage.value, "error": result.error}

    if result.intent:
        i = result.intent
        out["intent"] = {
            "raw_command": i.raw_command,
            "literal_meaning": i.literal_meaning,
            "intended_meaning": i.intended_meaning,
            "immediate_goal": i.immediate_goal,
            "deep_goal": i.deep_goal,
            "target_objects": i.target_objects,
            "implicit_constraints": i.implicit_constraints,
            "confidence": i.confidence,
            "reasoning": i.reasoning or "",
        }

    if result.affordance:
        a = result.affordance
        out["affordance"] = {
            "recommended_object": a.recommended_object or "",
            "recommended_affordance": a.recommended_affordance or "",
            "objects": [
                {
                    "object_name": oa.object_name,
                    "object_category": oa.object_category,
                    "active_affordances": [
                        {"name": af.name, "score": af.score}
                        for af in oa.active_affordances
                    ],
                }
                for oa in a.objects
            ],
            "reasoning": a.reasoning or "",
        }

    if result.plan:
        p = result.plan
        out["plan"] = {
            "plan_description": p.plan_description,
            "steps": [
                {
                    "action": s.action,
                    "target": s.target,
                    "params": s.params,
                    "description": s.description or "",
                }
                for s in p.steps
            ],
            "estimated_duration": p.estimated_duration,
            "risk_level": p.risk_level,
        }

    if result.safety_contract:
        d = result.safety_contract.to_display_dict()
        out["safety"] = d

    # Encode preview frame as base64 PNG
    if result.preview_frames:
        try:
            from PIL import Image
            frame = result.preview_frames[-1]
            if isinstance(frame, np.ndarray):
                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                out["preview_image"] = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            pass

    return out


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
