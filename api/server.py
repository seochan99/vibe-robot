"""FastAPI backend for VibeRobot! web interface.

Exposes the Python pipeline as a REST API consumed by the Next.js frontend.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline import PipelineResult, PipelineStage, VibeRobotPipeline
from api.context_resolver import (
    _analyze_command_context,
    _build_context_hint,
)
from simulator.franka_controller import FrankaController
from simulator.mujoco_env import MuJoCoEnv
from simulator.scene_builder import (
    TASK_PRESETS,
    SceneObject,
    build_scene_xml,
    get_scene_objects_info,
)
from study.logger import InteractionLogger
from ui.failure_cards import FailureCardGenerator
from config import WORKSPACE_BOUNDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="VibeRobot! API", version="0.1.0")

# CORS for Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://viberob0t.vercel.app",
    ],
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
        self.last_target_object: Optional[str] = None
        self.retry_count: int = 0
        self.last_command: Optional[str] = None
        self.last_plan_description: Optional[str] = None
        self.last_failure_summary: Optional[str] = None
        self.turn_history: list[dict] = []
        self.logger = InteractionLogger()
        self.failure_gen = FailureCardGenerator()

    def init_pipeline(self, scene_name: str):
        self.scene_name = scene_name
        self.last_target_object = None
        self.retry_count = 0
        self.last_command = None
        self.last_plan_description = None
        self.last_failure_summary = None
        self.turn_history = []
        objects = TASK_PRESETS.get(scene_name, TASK_PRESETS["wind_paper"])
        xml = build_scene_xml(objects, include_wind="wind" in scene_name)
        env = MuJoCoEnv(xml_string=xml)
        controller = FrankaController(env)
        env.reset()
        self.pipeline = VibeRobotPipeline(env, controller, scene_objects=objects)

    def append_history(self, entry: dict):
        self.turn_history.append(entry)
        # Keep recent context compact.
        self.turn_history = self.turn_history[-12:]


state = AppState()


# ── Request / Response models ────────────────────────────────────────────────


class CommandRequest(BaseModel):
    command: str
    scene: str = "wind_paper"
    model: str = "rule_based"
    access_token: Optional[str] = None
    account_id: Optional[str] = None


def _infer_primary_target(result: PipelineResult) -> Optional[str]:
    if not result.plan:
        return None
    for step in result.plan.steps:
        target = (step.target or "").strip()
        if target and target != "held_object":
            return target
    return None


def _summarize_execution_issue(result: PipelineResult) -> Optional[str]:
    if result.error:
        return result.error
    failed = [r for r in result.execution_results if not r.get("success", True)]
    if failed:
        head = failed[0]
        return f"{head.get('action')}({head.get('target')}): {head.get('message')}"
    return None


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "viberobot"}


@app.post("/api/command")
async def run_command(req: CommandRequest):
    """Run the pipeline for a user command (non-streaming fallback)."""
    logger.info("/api/command scene=%s model=%s", req.scene, req.model)
    if state.pipeline is None or state.scene_name != req.scene:
        state.init_pipeline(req.scene)
    if state.pipeline is not None:
        state.pipeline.sync_scene_objects_from_env()

    state.logger.log_command(req.command)
    conditions = ["wind_present"] if "wind" in req.scene else []
    scene_objects = state.pipeline._scene_objects if state.pipeline else []
    provider = None
    if req.model != "rule_based" and req.access_token:
        from providers.chatgpt_provider import ChatGPTOAuthProvider

        provider = ChatGPTOAuthProvider(
            access_token=req.access_token,
            account_id=req.account_id or "",
            model=req.model,
        )

    decision = await _analyze_command_context(
        command=req.command,
        scene_objects=scene_objects,
        last_target_object=state.last_target_object,
        retry_count=state.retry_count,
        last_failure_summary=state.last_failure_summary,
        last_plan_description=state.last_plan_description,
        turn_history=state.turn_history,
        provider=provider,
    )
    context_hint = _build_context_hint(
        scene_objects=scene_objects,
        last_target_object=state.last_target_object,
        decision=decision,
        retry_count=state.retry_count,
        last_failure_summary=state.last_failure_summary,
    )
    preferred_target = decision.resolved_target or (
        state.last_target_object if decision.lock_previous_target else None
    )
    if context_hint:
        logger.info(
            "Applying context hint target=%s source=%s",
            preferred_target,
            decision.source,
        )

    loop = asyncio.get_event_loop()

    if provider is not None:
        async def _run_async():
            return await state.pipeline.run_async(
                req.command,
                provider=provider,
                conditions=conditions,
                auto_approve=False,
                context_hint=context_hint,
                preferred_target=preferred_target,
            )

        result = await loop.run_in_executor(None, lambda: asyncio.run(_run_async()))
    else:
        result = await loop.run_in_executor(
            None,
            lambda: state.pipeline.run_sync(
                req.command,
                conditions=conditions,
                auto_approve=False,
                context_hint=context_hint,
                preferred_target=preferred_target,
            ),
        )

    state.current_result = result
    state.last_command = req.command
    state.last_plan_description = result.plan.plan_description if result.plan else None
    prev_target = state.last_target_object
    target = _infer_primary_target(result)
    if target:
        state.last_target_object = target
        if prev_target != target:
            state.retry_count = 0
    issue = _summarize_execution_issue(result)
    if result.stage == PipelineStage.FAILED and issue:
        state.last_failure_summary = issue
    elif not decision.is_retry_feedback:
        state.last_failure_summary = None
    state.append_history(
        {
            "ts": time.time(),
            "stage": result.stage.value,
            "command": req.command,
            "target": target or preferred_target,
            "note": issue or result.error or (result.plan.plan_description if result.plan else ""),
        }
    )
    logger.info("/api/command done stage=%s error=%s", result.stage.value, result.error)
    return _serialize_result(result)


@app.post("/api/command/stream")
async def run_command_stream(req: CommandRequest):
    """Run the pipeline with SSE streaming — emits stage updates as they happen."""
    import json as _json
    import queue

    logger.info("/api/command/stream start scene=%s model=%s", req.scene, req.model)
    if state.pipeline is None or state.scene_name != req.scene:
        state.init_pipeline(req.scene)
    if state.pipeline is not None:
        state.pipeline.sync_scene_objects_from_env()

    state.logger.log_command(req.command)
    conditions = ["wind_present"] if "wind" in req.scene else []
    scene_objects = state.pipeline._scene_objects if state.pipeline else []
    provider = None
    if req.model != "rule_based" and req.access_token:
        from providers.chatgpt_provider import ChatGPTOAuthProvider

        provider = ChatGPTOAuthProvider(
            access_token=req.access_token,
            account_id=req.account_id or "",
            model=req.model,
        )

    decision = await _analyze_command_context(
        command=req.command,
        scene_objects=scene_objects,
        last_target_object=state.last_target_object,
        retry_count=state.retry_count,
        last_failure_summary=state.last_failure_summary,
        last_plan_description=state.last_plan_description,
        turn_history=state.turn_history,
        provider=provider,
    )
    context_hint = _build_context_hint(
        scene_objects=scene_objects,
        last_target_object=state.last_target_object,
        decision=decision,
        retry_count=state.retry_count,
        last_failure_summary=state.last_failure_summary,
    )
    preferred_target = decision.resolved_target or (
        state.last_target_object if decision.lock_previous_target else None
    )
    if context_hint:
        logger.info(
            "Applying context hint target=%s source=%s",
            preferred_target,
            decision.source,
        )

    stage_queue: queue.Queue = queue.Queue()
    if decision.reason and preferred_target:
        stage_queue.put(
            {
                "stage": "reference_resolution",
                "message": f'Resolved reference to "{preferred_target}"',
            }
        )

    def on_stage(stage: str, message: str, _result):
        stage_queue.put({"stage": stage, "message": message})

    loop = asyncio.get_event_loop()

    # Start pipeline in a thread
    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    if provider is not None:
        async def _run_async():
            return await state.pipeline.run_async(
                req.command,
                provider=provider,
                conditions=conditions,
                auto_approve=False,
                on_stage=on_stage,
                context_hint=context_hint,
                preferred_target=preferred_target,
            )

        future = executor.submit(lambda: asyncio.run(_run_async()))
    else:
        future = executor.submit(
            state.pipeline.run_sync,
            req.command,
            conditions,
            False,
            on_stage,
            context_hint,
            preferred_target,
        )

    async def event_stream():
        last_emit = time.monotonic()
        heartbeat_every_s = 2.0

        try:
            # Yield stage events while pipeline runs
            while not future.done():
                emitted = False

                # Drain all available stage events
                while not stage_queue.empty():
                    try:
                        evt = stage_queue.get_nowait()
                        yield f"event: stage\ndata: {_json.dumps(evt)}\n\n"
                        emitted = True
                    except queue.Empty:
                        break

                now = time.monotonic()
                if emitted:
                    last_emit = now
                elif now - last_emit >= heartbeat_every_s:
                    hb = {"stage": "heartbeat", "message": "Still running..."}
                    yield f"event: heartbeat\ndata: {_json.dumps(hb)}\n\n"
                    last_emit = now

                await asyncio.sleep(0.05)

            # Drain remaining events
            while not stage_queue.empty():
                try:
                    evt = stage_queue.get_nowait()
                    yield f"event: stage\ndata: {_json.dumps(evt)}\n\n"
                except queue.Empty:
                    break

            # Get final result
            try:
                result = future.result()
                state.current_result = result
                state.last_command = req.command
                state.last_plan_description = (
                    result.plan.plan_description if result.plan else None
                )
                prev_target = state.last_target_object
                target = _infer_primary_target(result)
                if target:
                    state.last_target_object = target
                    if prev_target != target:
                        state.retry_count = 0
                issue = _summarize_execution_issue(result)
                if result.stage == PipelineStage.FAILED and issue:
                    state.last_failure_summary = issue
                elif not decision.is_retry_feedback:
                    state.last_failure_summary = None
                state.append_history(
                    {
                        "ts": time.time(),
                        "stage": result.stage.value,
                        "command": req.command,
                        "target": target or preferred_target,
                        "note": issue or result.error or (result.plan.plan_description if result.plan else ""),
                    }
                )
                yield (
                    "event: stage\ndata: "
                    + _json.dumps(
                        {
                            "stage": "finalizing",
                            "message": "Preparing preview output...",
                        }
                    )
                    + "\n\n"
                )
                serialized = _serialize_result(result)
                logger.info(
                    "/api/command/stream done stage=%s error=%s",
                    result.stage.value,
                    result.error,
                )
                yield f"event: result\ndata: {_json.dumps(serialized)}\n\n"
            except Exception as e:
                logger.exception("/api/command/stream failed")
                yield f"event: error\ndata: {_json.dumps({'error': str(e)})}\n\n"
        finally:
            executor.shutdown(wait=False, cancel_futures=False)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/approve")
async def approve_execution():
    """Approve and execute the current plan (real-time, non-blocking)."""
    if state.current_result is None or state.pipeline is None:
        return {"success": False, "error": "No plan to execute", "results": []}

    state.logger.log_approval(approved=True)

    # Run in a thread so the MJPEG stream keeps rendering during execution
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        state.pipeline.approve_and_execute,
        state.current_result,
    )
    state.current_result = result
    state.last_plan_description = result.plan.plan_description if result.plan else None
    target = _infer_primary_target(result)
    if target:
        state.last_target_object = target
    issue = _summarize_execution_issue(result)
    if issue:
        state.last_failure_summary = issue
        state.retry_count += 1
    else:
        state.last_failure_summary = None
        state.retry_count = 0
    state.append_history(
        {
            "ts": time.time(),
            "stage": "approved_execution",
            "command": state.last_command or "(approve)",
            "target": target,
            "note": issue or "Execution completed",
        }
    )
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
    if state.current_result:
        target = _infer_primary_target(state.current_result)
        if target:
            if state.last_target_object != target:
                state.retry_count = 0
            state.last_target_object = target
        if state.current_result.plan:
            desc = state.current_result.plan.plan_description
            tgt = target or "unknown target"
            state.last_failure_summary = (
                f'User rejected preview for "{tgt}". Planned behavior: {desc}'
            )
            state.append_history(
                {
                    "ts": time.time(),
                    "stage": "rejected_preview",
                    "command": state.last_command or "(reject)",
                    "target": tgt,
                    "note": desc,
                }
            )
        state.retry_count += 1
    state.current_result = None
    return {"status": "rejected"}


@app.get("/api/scenes")
def list_scenes():
    """List available scenes."""
    return {"scenes": [{"id": k, "objects": len(v)} for k, v in TASK_PRESETS.items()]}


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


# ── Session routes (renamed from /auth/* to avoid ad-blocker detection) ──────


@app.get("/api/session/status")
def session_status():
    """Check ChatGPT OAuth status."""
    try:
        from providers.chatgpt_auth import ChatGPTAuth

        auth = ChatGPTAuth()
        return auth.status
    except Exception:
        return {"authenticated": False, "message": "Auth module unavailable"}


@app.post("/api/session/connect")
def session_connect():
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


@app.post("/api/session/disconnect")
def session_disconnect():
    """Remove stored credentials."""
    try:
        from providers.chatgpt_auth import ChatGPTAuth

        ChatGPTAuth().logout()
        return {"status": "logged_out"}
    except Exception:
        return {"status": "ok"}


# ── Scene management models ──────────────────────────────────────────────────


class SceneInitRequest(BaseModel):
    scene: str = "wind_paper"


class AddObjectRequest(BaseModel):
    obj_type: str  # "book", "cup", "pen", "paper", "apple", "banana", "orange", "cube"
    position: list[float] = [0.5, 0.0, 0.35]
    name: Optional[str] = None


class MoveObjectRequest(BaseModel):
    name: str
    position: list[float]


class RemoveObjectRequest(BaseModel):
    name: str


class SimCameraRequest(BaseModel):
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    distance: Optional[float] = None
    lookat: Optional[list[float]] = None


class SimProjectRequest(BaseModel):
    nx: float
    ny: float
    plane_z: float = 0.35
    aspect: float = 4.0 / 3.0


class SimPickRequest(BaseModel):
    nx: float
    ny: float
    aspect: float = 4.0 / 3.0
    radius: float = 0.08


# ── Scene management routes ─────────────────────────────────────────────────

# Predefined object templates for the palette
_OBJECT_TEMPLATES: dict[str, dict] = {
    "book": {
        "obj_type": "box",
        "size": (0.08, 0.06, 0.015),
        "rgba": (0.2, 0.3, 0.7, 1.0),
        "mass": 0.5,
        "properties": {"graspable": True, "heavy": True, "weight_kg": 0.5},
    },
    "cup": {
        "obj_type": "cylinder",
        "size": (0.03, 0.05),
        "rgba": (0.9, 0.85, 0.8, 1.0),
        "mass": 0.2,
        "properties": {"graspable": True, "container": True},
    },
    "pen": {
        "obj_type": "cylinder",
        "size": (0.005, 0.07),
        "rgba": (0.1, 0.1, 0.1, 1.0),
        "mass": 0.01,
        "properties": {"graspable": True, "thin": True},
    },
    "paper": {
        "obj_type": "box",
        "size": (0.1, 0.07, 0.001),
        "rgba": (1.0, 1.0, 0.95, 1.0),
        "mass": 0.005,
        "friction": (0.3, 0.001, 0.0001),
        "properties": {"graspable": True, "loose": True, "fragile": True},
    },
    "apple": {
        "obj_type": "sphere",
        "size": (0.03,),
        "rgba": (0.9, 0.15, 0.1, 1.0),
        "mass": 0.15,
        "properties": {"graspable": True, "fruit": True},
    },
    "banana": {
        "obj_type": "cylinder",
        "size": (0.015, 0.06),
        "rgba": (1.0, 0.9, 0.2, 1.0),
        "mass": 0.12,
        "properties": {"graspable": True, "fruit": True},
    },
    "orange": {
        "obj_type": "sphere",
        "size": (0.035,),
        "rgba": (1.0, 0.6, 0.0, 1.0),
        "mass": 0.2,
        "properties": {"graspable": True, "fruit": True},
    },
    "cube": {
        "obj_type": "box",
        "size": (0.025, 0.025, 0.025),
        "rgba": (0.9, 0.1, 0.1, 1.0),
        "mass": 0.05,
        "properties": {"graspable": True},
    },
}


@app.post("/api/scene/init")
def scene_init(req: SceneInitRequest):
    """Initialize or reinitialize the scene."""
    state.init_pipeline(req.scene)
    return {"status": "ok", "scene": req.scene}


@app.get("/api/scene/objects")
def scene_objects():
    """Get current scene objects."""
    if state.pipeline is None:
        return {"objects": []}
    objs = get_scene_objects_info(state.pipeline._scene_objects)
    return {"objects": objs}


@app.post("/api/scene/add-object")
def scene_add_object(req: AddObjectRequest):
    """Add an object to the current scene."""
    if state.pipeline is None:
        state.init_pipeline(state.scene_name or "wind_paper")

    template = _OBJECT_TEMPLATES.get(req.obj_type, _OBJECT_TEMPLATES["cube"])

    # Generate unique name
    existing_names = {o.name for o in state.pipeline._scene_objects}
    base_name = req.name or req.obj_type
    name = base_name
    counter = 1
    while name in existing_names:
        counter += 1
        name = f"{base_name}_{counter:02d}"

    pos = tuple(req.position) if len(req.position) == 3 else (0.5, 0.0, 0.35)

    new_obj = SceneObject(
        name=name,
        obj_type=template["obj_type"],
        size=template["size"],
        pos=pos,
        rgba=template["rgba"],
        mass=template["mass"],
        friction=template.get("friction", (1.0, 0.005, 0.0001)),
        properties=template["properties"].copy(),
    )

    # Rebuild scene with new object
    state.pipeline._scene_objects.append(new_obj)
    _rebuild_env()

    return {"name": name, "status": "ok"}


@app.post("/api/scene/move-object")
def scene_move_object(req: MoveObjectRequest):
    """Move an object to a new position."""
    if state.pipeline is None:
        return {"status": "error", "error": "No active scene"}

    for obj in state.pipeline._scene_objects:
        if obj.name == req.name:
            target = np.asarray(req.position[:3], dtype=float)
            # Keep object motion inside workspace bounds to avoid odd teleports.
            for i, axis in enumerate(("x", "y", "z")):
                lo, hi = WORKSPACE_BOUNDS[axis]
                target[i] = float(np.clip(target[i], lo, hi))
            obj.pos = tuple(target.tolist())

            moved = False
            if state.pipeline.env is not None:
                moved = state.pipeline.env.set_object_pos(req.name, target)
            if not moved:
                _rebuild_env()
            return {"status": "ok"}

    return {"status": "error", "error": f"Object '{req.name}' not found"}


@app.delete("/api/scene/remove-object")
def scene_remove_object(req: RemoveObjectRequest):
    """Remove an object from the scene."""
    if state.pipeline is None:
        return {"status": "error", "error": "No active scene"}

    original_len = len(state.pipeline._scene_objects)
    state.pipeline._scene_objects = [
        o
        for o in state.pipeline._scene_objects
        if o.name != req.name or o.properties.get("fixed")
    ]

    if len(state.pipeline._scene_objects) < original_len:
        _rebuild_env()
        return {"status": "ok"}
    return {"status": "error", "error": f"Object '{req.name}' not found or is fixed"}


def _rebuild_env():
    """Rebuild MuJoCo env from current scene objects."""
    if state.pipeline is None:
        return
    previous_camera = None
    if state.pipeline.env is not None:
        try:
            previous_camera = state.pipeline.env.get_camera_state()
        except Exception:
            previous_camera = None

    objects = state.pipeline._scene_objects
    xml = build_scene_xml(objects, include_wind="wind" in state.scene_name)
    env = MuJoCoEnv(xml_string=xml)
    controller = FrankaController(env)
    env.reset()
    if previous_camera:
        env.set_camera_state(
            azimuth=previous_camera.get("azimuth"),
            elevation=previous_camera.get("elevation"),
            distance=previous_camera.get("distance"),
            lookat=previous_camera.get("lookat"),
        )
    state.pipeline.env = env
    state.pipeline.controller = controller


# ── Interactive camera / projection ─────────────────────────────────────────


@app.get("/api/sim/camera")
def sim_get_camera():
    """Get current simulation free-camera state."""
    if state.pipeline is None:
        state.init_pipeline(state.scene_name or "wind_paper")
    return {"camera": state.pipeline.env.get_camera_state()}


@app.post("/api/sim/camera")
def sim_set_camera(req: SimCameraRequest):
    """Update simulation free-camera state."""
    if state.pipeline is None:
        state.init_pipeline(state.scene_name or "wind_paper")
    camera = state.pipeline.env.set_camera_state(
        azimuth=req.azimuth,
        elevation=req.elevation,
        distance=req.distance,
        lookat=req.lookat,
    )
    return {"status": "ok", "camera": camera}


@app.post("/api/sim/project")
def sim_project_to_world(req: SimProjectRequest):
    """Project normalized screen coordinate onto tabletop plane."""
    if state.pipeline is None:
        state.init_pipeline(state.scene_name or "wind_paper")
    point = state.pipeline.env.project_screen_to_plane(
        req.nx,
        req.ny,
        plane_z=req.plane_z,
        aspect=max(0.25, float(req.aspect)),
    )
    if point is None:
        return {"status": "error", "error": "projection_failed"}

    for i, axis in enumerate(("x", "y", "z")):
        lo, hi = WORKSPACE_BOUNDS[axis]
        point[i] = np.clip(point[i], lo, hi)
    return {"status": "ok", "point": [float(point[0]), float(point[1]), float(point[2])]}


@app.post("/api/sim/pick-object")
def sim_pick_object(req: SimPickRequest):
    """Pick nearest movable object from screen coordinate."""
    if state.pipeline is None:
        state.init_pipeline(state.scene_name or "wind_paper")

    movable_names = [
        obj.name
        for obj in state.pipeline._scene_objects
        if not obj.properties.get("fixed", False)
    ]
    picked = state.pipeline.env.pick_object_from_screen(
        movable_names,
        req.nx,
        req.ny,
        aspect=max(0.25, float(req.aspect)),
        pick_radius=max(0.005, float(req.radius)),
    )
    if not picked:
        return {"status": "none"}
    return {"status": "ok", "object": picked}


# ── MJPEG streaming ─────────────────────────────────────────────────────────


@app.get("/api/sim/stream")
async def sim_stream():
    """MJPEG streaming endpoint — renders MuJoCo at ~10fps."""

    async def generate():
        frozen_stages = {
            PipelineStage.INTENT_INFERENCE,
            PipelineStage.SCENE_UNDERSTANDING,
            PipelineStage.AFFORDANCE_ANALYSIS,
            PipelineStage.PLAN_GENERATION,
            PipelineStage.SAFETY_CONTRACT,
            PipelineStage.SIMULATION_PREVIEW,
            PipelineStage.AWAITING_APPROVAL,
        }
        while True:
            if state.pipeline and state.pipeline.env:
                try:
                    if state.pipeline.current_stage in frozen_stages:
                        # Keep right panel visually frozen until explicit approval.
                        await asyncio.sleep(0.2)
                        continue
                    is_executing = (
                        state.pipeline.current_stage == PipelineStage.EXECUTING
                    )
                    width, height = (560, 420) if is_executing else (640, 480)
                    frame = await asyncio.to_thread(
                        state.pipeline.env.render_offscreen,
                        width,
                        height,
                    )
                    from PIL import Image

                    img = Image.fromarray(frame)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=75)
                    jpg_bytes = buf.getvalue()

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpg_bytes)).encode() + b"\r\n"
                        b"\r\n" + jpg_bytes + b"\r\n"
                    )
                except Exception:
                    # Yield a small placeholder if render fails
                    await asyncio.sleep(0.5)
                    continue
            else:
                await asyncio.sleep(0.5)
                continue

            # During live execution prioritize control-thread progress over render FPS.
            await asyncio.sleep(0.12 if is_executing else 0.05)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


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
            "reasoning": p.reasoning or "",
        }

    if result.safety_contract:
        d = result.safety_contract.to_display_dict()
        out["safety"] = d

    # Encode preview frames as animated GIF (or single PNG fallback)
    if result.preview_frames:
        try:
            from PIL import Image

            valid = [f for f in result.preview_frames if isinstance(f, np.ndarray)]
            if len(valid) > 1:
                # Sample frames to keep GIF encoding responsive.
                max_frames = 8
                if len(valid) > max_frames:
                    stride = max(1, (len(valid) + max_frames - 1) // max_frames)
                    valid = valid[::stride]
                # Animated GIF
                images = [Image.fromarray(f) for f in valid]
                buf = io.BytesIO()
                images[0].save(
                    buf,
                    format="GIF",
                    save_all=True,
                    append_images=images[1:],
                    duration=90,  # lightweight preview animation
                    loop=0,
                )
                out["preview_image"] = base64.b64encode(buf.getvalue()).decode()
                out["preview_format"] = "gif"
            elif valid:
                # Single frame PNG fallback
                img = Image.fromarray(valid[0])
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                out["preview_image"] = base64.b64encode(buf.getvalue()).decode()
                out["preview_format"] = "png"
        except Exception:
            pass

    return out


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
