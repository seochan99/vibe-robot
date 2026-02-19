"""End-to-end Vibe-to-Verify pipeline orchestrator.

Coordinates the full workflow:
  Command → Intent Inference → Scene Understanding → Affordance Engine →
  Plan Generation → Safety Contract → Simulation Preview → User Approval → Execution
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional

import numpy as np

from config import WORKSPACE_BOUNDS
from core.affordance_engine import AffordanceAnalysis, AffordanceEngine
from core.intent_inference import IntentInference, UserIntent
from core.planner import ExecutionPlan, PlanStep, Planner
from core.safety_contract import (
    SafetyContract,
    SafetyContractGenerator,
    SafetyMonitor,
)
from core.scene_understanding import SceneState, SceneUnderstanding
from simulator.franka_controller import FrankaController, MotionResult
from simulator.mujoco_env import MuJoCoEnv
from simulator.scene_builder import SceneObject, get_scene_objects_info

logger = logging.getLogger(__name__)

_LOW_CLEARING_POSITIONS = (
    [0.35, 0.42, 0.06],
    [0.45, 0.42, 0.06],
    [0.55, 0.42, 0.06],
    [0.65, 0.42, 0.06],
    [0.35, -0.42, 0.06],
    [0.45, -0.42, 0.06],
    [0.55, -0.42, 0.06],
    [0.65, -0.42, 0.06],
)


class PipelineStage(Enum):
    IDLE = "idle"
    INTENT_INFERENCE = "intent_inference"
    SCENE_UNDERSTANDING = "scene_understanding"
    AFFORDANCE_ANALYSIS = "affordance_analysis"
    PLAN_GENERATION = "plan_generation"
    SAFETY_CONTRACT = "safety_contract"
    SIMULATION_PREVIEW = "simulation_preview"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineResult:
    """Complete result from one pipeline run."""

    stage: PipelineStage
    intent: Optional[UserIntent] = None
    scene: Optional[SceneState] = None
    affordance: Optional[AffordanceAnalysis] = None
    plan: Optional[ExecutionPlan] = None
    safety_contract: Optional[SafetyContract] = None
    preview_frames: list = field(default_factory=list)
    execution_results: list = field(default_factory=list)
    error: Optional[str] = None
    timestamps: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.stage == PipelineStage.COMPLETED and self.error is None


class VibeRobotPipeline:
    """Main pipeline orchestrating all VibeRobot components."""

    def __init__(
        self,
        env: MuJoCoEnv,
        controller: FrankaController,
        scene_objects: list[SceneObject] = None,
    ):
        self.env = env
        self.controller = controller
        self._scene_objects = scene_objects or []

        # Core components
        self.intent_engine = IntentInference()
        self.scene_engine = SceneUnderstanding()
        self.affordance_engine = AffordanceEngine()
        self.planner = Planner()
        self.safety_gen = SafetyContractGenerator()

        self._current_stage = PipelineStage.IDLE
        self._result = PipelineResult(stage=PipelineStage.IDLE)

    @property
    def current_stage(self) -> PipelineStage:
        return self._current_stage

    def sync_scene_objects_from_env(self) -> None:
        """Refresh cached scene-object positions from the live MuJoCo state."""
        if not self._scene_objects or self.env is None:
            return
        for obj in self._scene_objects:
            if obj.properties.get("fixed"):
                continue
            pos = self.env.get_object_pos(obj.name)
            if pos is not None and len(pos) == 3:
                obj.pos = (float(pos[0]), float(pos[1]), float(pos[2]))

    def _is_low_clearing_intent(self, intent: UserIntent) -> bool:
        goal = (intent.immediate_goal or "").strip().lower()
        cmd = (intent.raw_command or "").lower()
        if goal in {"clean", "clear", "sort", "organize"}:
            return True
        low_words = ("under", "below", "bottom", "밑", "아래", "하부", "바닥")
        return any(w in cmd for w in low_words)

    def _clamp_position(self, position: list[float]) -> list[float]:
        pos = np.asarray(position[:3], dtype=float).copy()
        for i, axis in enumerate(("x", "y", "z")):
            lo, hi = WORKSPACE_BOUNDS[axis]
            pos[i] = float(np.clip(pos[i], lo, hi))
        return pos.tolist()

    def _resolve_plan_target_name(
        self,
        target: str,
        scene: SceneState,
    ) -> str:
        if not target:
            return target
        resolved = self.planner._resolve_object_name(target, scene)  # noqa: SLF001
        return resolved or target

    def _normalize_plan_for_execution(
        self,
        plan: ExecutionPlan,
        scene: SceneState,
        intent: UserIntent,
        notify=None,
    ) -> ExecutionPlan:
        """Normalize planner output into controller-ready primitives.

        LLM plans are allowed to be semantic. This adapter guarantees executable
        params for primitives expected by the controller.
        """
        low_clearing = self._is_low_clearing_intent(intent)
        normalized: list[PlanStep] = []
        held_target: str | None = None
        placement_index = 0
        object_pos: dict[str, list[float]] = {
            obj.name: list(obj.position) for obj in scene.objects
        }

        resolved_targets: list[str] = [
            self._resolve_plan_target_name(step.target, scene) for step in plan.steps
        ]

        for idx, step in enumerate(plan.steps):
            target = self._resolve_plan_target_name(step.target, scene)
            params = dict(step.params or {})

            if step.action == "pick":
                held_target = target

            if (
                step.action == "move_to"
                and "position" not in params
                and target in object_pos
            ):
                obj_pos = object_pos[target]
                next_action = plan.steps[idx + 1].action if idx + 1 < len(plan.steps) else ""
                next_target = resolved_targets[idx + 1] if idx + 1 < len(resolved_targets) else ""

                # Interpret semantic move_to(object) steps into concrete Cartesian goals.
                if held_target and target == held_target:
                    # Move held object upward to a safe transport height.
                    pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.18]
                elif next_action == "close_gripper" and next_target == target:
                    # Pre-close descent near object center.
                    pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.02]
                else:
                    # Generic approach above object.
                    pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.12]

                params["position"] = self._clamp_position(pos)
                if notify:
                    notify(
                        "plan_normalization",
                        f"Auto-filled move_to target for {target}: {params['position']}",
                    )

            if "position" in params and isinstance(params.get("position"), (list, tuple)):
                pos = list(params.get("position", []))
                if len(pos) >= 3:
                    params["position"] = self._clamp_position(pos)

            if step.action == "place":
                if (not target or target == "held_object") and held_target:
                    target = held_target

                if "position" not in params and "on" not in params:
                    if low_clearing:
                        pos = list(_LOW_CLEARING_POSITIONS[placement_index % len(_LOW_CLEARING_POSITIONS)])
                        placement_index += 1
                    else:
                        pos = [0.5, 0.0, 0.35]
                    params["position"] = self._clamp_position(pos)
                    if notify:
                        notify(
                            "plan_normalization",
                            f"Auto-filled place target for {target}: {params['position']}",
                        )
                held_target = None

            if step.action == "open_gripper":
                held_target = None
            elif step.action == "close_gripper" and target:
                held_target = target

            normalized.append(
                PlanStep(
                    action=step.action,
                    target=target,
                    params=params,
                    description=step.description,
                )
            )

        return ExecutionPlan(
            steps=normalized,
            plan_description=plan.plan_description,
            estimated_duration=plan.estimated_duration,
            risk_level=plan.risk_level,
            reasoning=plan.reasoning,
            requires_approval=plan.requires_approval,
        )

    def run_sync(
        self,
        command: str,
        conditions: list[str] = None,
        auto_approve: bool = False,
        on_stage=None,
        context_hint: str | None = None,
        preferred_target: str | None = None,
    ) -> PipelineResult:
        """Run the full pipeline synchronously (for development/testing).

        Args:
            command: Natural language user command.
            conditions: Environmental conditions (e.g. ["wind_present"]).
            auto_approve: Skip user approval step (for automated testing).
            on_stage: Optional callback(stage_name: str, message: str, result_so_far: PipelineResult).
        """
        result = PipelineResult(stage=PipelineStage.IDLE)
        result.timestamps["start"] = time.time()

        def _notify(stage: str, msg: str):
            if on_stage:
                on_stage(stage, msg, result)

        try:
            # Keep semantic scene metadata aligned with current simulator state.
            self.sync_scene_objects_from_env()

            # Stage 1: Intent Inference
            self._current_stage = PipelineStage.INTENT_INFERENCE
            _notify("intent_inference", "Analyzing your command...")
            result.timestamps["intent_start"] = time.time()
            scene_summary = self._get_scene_summary(context_hint=context_hint)
            result.intent = self.intent_engine.infer_sync(command, scene_summary)
            if preferred_target:
                result.intent.target_objects = [preferred_target]
                _notify("retry_context", f"Keeping target object: {preferred_target}")
            result.timestamps["intent_end"] = time.time()
            _notify("intent_done", f"Intent: {result.intent.intended_meaning}")

            # Stage 2: Scene Understanding
            self._current_stage = PipelineStage.SCENE_UNDERSTANDING
            _notify("scene_understanding", "Understanding the scene...")
            result.timestamps["scene_start"] = time.time()
            objects_info = get_scene_objects_info(self._scene_objects)
            result.scene = self.scene_engine.analyze_from_metadata(
                objects_info,
                conditions=conditions,
            )
            result.timestamps["scene_end"] = time.time()
            _notify(
                "scene_done", f"Scene: {len(result.scene.objects)} objects detected"
            )

            # Stage 3: Affordance Analysis
            self._current_stage = PipelineStage.AFFORDANCE_ANALYSIS
            _notify("affordance_analysis", "Analyzing object affordances...")
            result.timestamps["affordance_start"] = time.time()
            result.affordance = self.affordance_engine.analyze(
                result.scene, result.intent
            )
            result.timestamps["affordance_end"] = time.time()
            rec = result.affordance.recommended_object or "none"
            _notify("affordance_done", f"Recommended: {rec}")

            # Stage 4: Plan Generation
            self._current_stage = PipelineStage.PLAN_GENERATION
            _notify("plan_generation", "Generating execution plan...")
            result.timestamps["plan_start"] = time.time()
            result.plan = self.planner.generate_plan_sync(
                result.scene,
                result.intent,
                result.affordance,
            )
            result.plan = self._apply_retry_target_guardrail(
                result.plan,
                result.intent,
                result.scene,
                result.affordance,
                preferred_target,
                _notify,
            )
            result.plan = self._normalize_plan_for_execution(
                result.plan,
                result.scene,
                result.intent,
                _notify,
            )
            result.timestamps["plan_end"] = time.time()

            if not result.plan.steps:
                result.stage = PipelineStage.FAILED
                result.error = f"Planning failed: {result.plan.plan_description}"
                _notify("failed", result.error)
                return result

            steps_desc = ", ".join(f"{s.action}({s.target})" for s in result.plan.steps)
            _notify("plan_done", f"Plan: {steps_desc}")

            # Stage 5: Safety Contract
            self._current_stage = PipelineStage.SAFETY_CONTRACT
            _notify("safety_contract", "Generating safety constraints...")
            result.safety_contract = self.safety_gen.generate(result.plan)
            _notify("safety_done", "Safety contract ready")

            # Stage 6: Simulation Preview (chat preview before approval)
            self._current_stage = PipelineStage.SIMULATION_PREVIEW
            _notify(
                "simulation_preview",
                "Generating chat preview from simulation...",
            )
            result.timestamps["sim_start"] = time.time()
            preview_results = self._run_simulation_preview(
                result.plan, result.safety_contract
            )
            result.preview_frames = preview_results.get("frames", [])
            result.timestamps["sim_end"] = time.time()
            _notify("preview_done", "Preview complete")

            # Check if simulation had safety violations
            if preview_results.get("violations"):
                result.error = (
                    f"Safety violations in preview: {preview_results['violations']}"
                )
                result.stage = PipelineStage.FAILED
                _notify("failed", result.error)
                return result

            # Stage 7: Awaiting Approval (real robot motion still pending)
            if not auto_approve:
                self._current_stage = PipelineStage.AWAITING_APPROVAL
                result.stage = PipelineStage.AWAITING_APPROVAL
                _notify(
                    "awaiting_approval",
                    "Preview ready. Approve to run in live simulation.",
                )
                return result

            # Stage 8: Execute
            self._current_stage = PipelineStage.EXECUTING
            result.timestamps["exec_start"] = time.time()
            result.execution_results = self._execute_plan(
                result.plan, result.safety_contract
            )
            result.timestamps["exec_end"] = time.time()
            self._apply_execution_outcome(result)
            result.timestamps["end"] = time.time()

        except Exception as e:
            result.stage = PipelineStage.FAILED
            result.error = str(e)
            _notify("failed", str(e))

        self._current_stage = result.stage
        self._result = result
        return result

    async def run_async(
        self,
        command: str,
        provider=None,
        conditions: list[str] = None,
        auto_approve: bool = False,
        on_stage=None,
        context_hint: str | None = None,
        preferred_target: str | None = None,
    ) -> PipelineResult:
        """Run the full pipeline using an LLM provider for intent/planning.

        Falls back to rule-based if LLM call fails.

        Args:
            command: Natural language user command.
            provider: LLMProvider instance for GPT calls.
            conditions: Environmental conditions.
            auto_approve: Skip user approval step.
            on_stage: Optional callback(stage_name, message, result_so_far).
        """
        result = PipelineResult(stage=PipelineStage.IDLE)
        result.timestamps["start"] = time.time()

        def _notify(stage: str, msg: str):
            if on_stage:
                on_stage(stage, msg, result)

        try:
            # Keep semantic scene metadata aligned with current simulator state.
            self.sync_scene_objects_from_env()

            # Stage 1: Intent Inference (LLM)
            self._current_stage = PipelineStage.INTENT_INFERENCE
            _notify("intent_inference", "Analyzing your command with GPT...")
            result.timestamps["intent_start"] = time.time()
            scene_summary = self._get_scene_summary(context_hint=context_hint)

            if provider:
                try:
                    intent_engine = IntentInference(provider=provider)
                    result.intent = await intent_engine.infer(command, scene_summary)
                    if preferred_target:
                        result.intent.target_objects = [preferred_target]
                        _notify(
                            "retry_context",
                            f"Keeping target object: {preferred_target}",
                        )
                    _notify(
                        "intent_done", f"GPT Intent: {result.intent.intended_meaning}"
                    )
                except Exception as e:
                    logger.exception(
                        "LLM intent inference failed; falling back to rule-based"
                    )
                    _notify("intent_fallback", f"GPT failed ({e}), using rule-based...")
                    result.intent = self.intent_engine.infer_sync(
                        command, scene_summary
                    )
                    if preferred_target:
                        result.intent.target_objects = [preferred_target]
                        _notify(
                            "retry_context",
                            f"Keeping target object: {preferred_target}",
                        )
                    _notify("intent_done", f"Intent: {result.intent.intended_meaning}")
            else:
                result.intent = self.intent_engine.infer_sync(command, scene_summary)
                if preferred_target:
                    result.intent.target_objects = [preferred_target]
                    _notify(
                        "retry_context",
                        f"Keeping target object: {preferred_target}",
                    )
                _notify("intent_done", f"Intent: {result.intent.intended_meaning}")
            result.timestamps["intent_end"] = time.time()

            # Stage 2: Scene Understanding
            self._current_stage = PipelineStage.SCENE_UNDERSTANDING
            _notify("scene_understanding", "Understanding the scene...")
            result.timestamps["scene_start"] = time.time()
            objects_info = get_scene_objects_info(self._scene_objects)
            result.scene = self.scene_engine.analyze_from_metadata(
                objects_info,
                conditions=conditions,
            )
            result.timestamps["scene_end"] = time.time()
            _notify(
                "scene_done", f"Scene: {len(result.scene.objects)} objects detected"
            )

            # Stage 3: Affordance Analysis
            self._current_stage = PipelineStage.AFFORDANCE_ANALYSIS
            _notify("affordance_analysis", "Analyzing object affordances...")
            result.timestamps["affordance_start"] = time.time()
            result.affordance = self.affordance_engine.analyze(
                result.scene, result.intent
            )
            result.timestamps["affordance_end"] = time.time()
            rec = result.affordance.recommended_object or "none"
            _notify("affordance_done", f"Recommended: {rec}")

            # Stage 4: Plan Generation (LLM)
            self._current_stage = PipelineStage.PLAN_GENERATION
            _notify("plan_generation", "Generating execution plan with GPT...")
            result.timestamps["plan_start"] = time.time()

            if provider:
                try:
                    planner = Planner(provider=provider)
                    result.plan = await planner.generate_plan(
                        result.scene,
                        result.intent,
                        result.affordance,
                    )
                    result.plan = self._apply_retry_target_guardrail(
                        result.plan,
                        result.intent,
                        result.scene,
                        result.affordance,
                        preferred_target,
                        _notify,
                    )
                    result.plan = self._normalize_plan_for_execution(
                        result.plan,
                        result.scene,
                        result.intent,
                        _notify,
                    )
                    _notify("plan_done", f"GPT Plan: {result.plan.plan_description}")
                except Exception as e:
                    logger.exception(
                        "LLM plan generation failed; falling back to rule-based"
                    )
                    _notify("plan_fallback", f"GPT failed ({e}), using rule-based...")
                    result.plan = self.planner.generate_plan_sync(
                        result.scene,
                        result.intent,
                        result.affordance,
                    )
                    result.plan = self._apply_retry_target_guardrail(
                        result.plan,
                        result.intent,
                        result.scene,
                        result.affordance,
                        preferred_target,
                        _notify,
                    )
                    result.plan = self._normalize_plan_for_execution(
                        result.plan,
                        result.scene,
                        result.intent,
                        _notify,
                    )
                    steps_desc = ", ".join(
                        f"{s.action}({s.target})" for s in result.plan.steps
                    )
                    _notify("plan_done", f"Plan: {steps_desc}")
            else:
                result.plan = self.planner.generate_plan_sync(
                    result.scene,
                    result.intent,
                    result.affordance,
                )
                result.plan = self._apply_retry_target_guardrail(
                    result.plan,
                    result.intent,
                    result.scene,
                    result.affordance,
                    preferred_target,
                    _notify,
                )
                result.plan = self._normalize_plan_for_execution(
                    result.plan,
                    result.scene,
                    result.intent,
                    _notify,
                )
                steps_desc = ", ".join(
                    f"{s.action}({s.target})" for s in result.plan.steps
                )
                _notify("plan_done", f"Plan: {steps_desc}")
            result.timestamps["plan_end"] = time.time()

            if not result.plan.steps:
                result.stage = PipelineStage.FAILED
                result.error = f"Planning failed: {result.plan.plan_description}"
                _notify("failed", result.error)
                return result

            # Stage 5: Safety Contract
            self._current_stage = PipelineStage.SAFETY_CONTRACT
            _notify("safety_contract", "Generating safety constraints...")
            result.safety_contract = self.safety_gen.generate(result.plan)
            _notify("safety_done", "Safety contract ready")

            # Stage 6: Simulation Preview (chat preview before approval)
            self._current_stage = PipelineStage.SIMULATION_PREVIEW
            _notify(
                "simulation_preview",
                "Generating chat preview from simulation...",
            )
            result.timestamps["sim_start"] = time.time()
            preview_results = self._run_simulation_preview(
                result.plan, result.safety_contract
            )
            result.preview_frames = preview_results.get("frames", [])
            result.timestamps["sim_end"] = time.time()
            _notify("preview_done", "Preview complete")

            if preview_results.get("violations"):
                result.error = (
                    f"Safety violations in preview: {preview_results['violations']}"
                )
                result.stage = PipelineStage.FAILED
                _notify("failed", result.error)
                return result

            # Stage 7: Awaiting Approval (real robot motion still pending)
            if not auto_approve:
                self._current_stage = PipelineStage.AWAITING_APPROVAL
                result.stage = PipelineStage.AWAITING_APPROVAL
                _notify(
                    "awaiting_approval",
                    "Preview ready. Approve to run in live simulation.",
                )
                return result

            # Stage 8: Execute
            self._current_stage = PipelineStage.EXECUTING
            result.timestamps["exec_start"] = time.time()
            result.execution_results = self._execute_plan(
                result.plan, result.safety_contract
            )
            result.timestamps["exec_end"] = time.time()
            self._apply_execution_outcome(result)
            result.timestamps["end"] = time.time()

        except Exception as e:
            result.stage = PipelineStage.FAILED
            result.error = str(e)

        self._current_stage = result.stage
        self._result = result
        return result

    def approve_and_execute(self, result: PipelineResult) -> PipelineResult:
        """Execute a plan that was awaiting approval (real-time for MJPEG viewing)."""
        if result.stage != PipelineStage.AWAITING_APPROVAL:
            result.error = "Plan is not in awaiting_approval state"
            return result

        # Enable real-time pacing so MJPEG stream shows the movement
        self.controller.realtime = True

        self._current_stage = PipelineStage.EXECUTING
        result.timestamps["exec_start"] = time.time()
        result.execution_results = self._execute_plan(
            result.plan, result.safety_contract
        )
        self.sync_scene_objects_from_env()
        self.controller.realtime = False
        result.timestamps["exec_end"] = time.time()
        self._apply_execution_outcome(result)
        result.timestamps["end"] = time.time()

        self._current_stage = result.stage
        self._result = result
        return result

    def _apply_execution_outcome(self, result: PipelineResult) -> None:
        """Set final stage/error from execution step results."""
        failed = [r for r in result.execution_results if not r.get("success", True)]
        if failed:
            result.stage = PipelineStage.FAILED
            if not result.error:
                head = failed[0]
                result.error = (
                    f"{head.get('action')}({head.get('target')}): "
                    f"{head.get('message') or 'execution failed'}"
                )
            return
        result.stage = PipelineStage.COMPLETED

    def _run_simulation_preview(
        self,
        plan: ExecutionPlan,
        contract: SafetyContract,
    ) -> dict:
        """Run a preview used in chat without animating the live simulation panel."""
        # Hold env lock for entire preview so live MJPEG stream stays frozen pre-approval.
        with self.env._lock:
            # Save current state
            saved_qpos = self.env.data.qpos.copy()
            saved_qvel = self.env.data.qvel.copy()
            prev_realtime = self.controller.realtime

            monitor = SafetyMonitor(contract)
            frames = []
            violations = []

            # Preview should be fast and invisible to the live simulation panel.
            self.controller.realtime = False

            try:
                # Reset to home
                self.env.reset()

                # Enable frame capture for animated preview
                # Keep preview lightweight so chat response remains responsive.
                self.env.enable_frame_capture(every_n=14, width=320, height=240)

                # Execute plan steps in simulation
                for step in plan.steps:
                    motion_result = self._execute_step(step, record=True)
                    if not motion_result.success:
                        break

                    # Check safety invariants
                    state = self.env.get_state()
                    check = monitor.check_invariants(state.ee_pos)
                    if not check.passed:
                        violations.extend(check.violated_conditions)
                        if check.severity == "emergency_stop":
                            break

                # Collect all captured frames
                frames = self.env.disable_frame_capture()

                # Always add a final frame
                try:
                    frames.append(self.env.render_offscreen(480, 360))
                except Exception:
                    pass

            finally:
                # Ensure capture mode is off even if something fails mid-preview.
                self.env.disable_frame_capture()
                self.controller.realtime = prev_realtime
                # Restore state before releasing lock
                self.env.reset(qpos=saved_qpos)
                self.env.data.qvel[:] = saved_qvel

        return {
            "frames": frames,
            "trajectory": self.env.get_trajectory(),
            "violations": violations,
        }

    def _execute_plan(
        self,
        plan: ExecutionPlan,
        contract: SafetyContract,
    ) -> list[dict]:
        """Execute the plan for real (after approval)."""
        # Keep world/object state continuous across turns.
        # Open gripper to clear stale grasps and home only when far from home.
        self.controller.open_gripper(duration=0.18)
        current_q = self.controller.get_joint_positions()
        home_q = self.env.HOME_QPOS[:7]
        if np.linalg.norm(current_q - home_q) > 0.25:
            self.controller.home(duration=0.9)
        monitor = SafetyMonitor(contract)
        results = []

        for i, step in enumerate(plan.steps):
            motion_result = self._execute_step(step, record=True)
            results.append(
                {
                    "step": i,
                    "action": step.action,
                    "target": step.target,
                    "success": motion_result.success,
                    "message": motion_result.message,
                }
            )

            if not motion_result.success:
                break

            # Check safety
            state = self.env.get_state()
            check = monitor.check_invariants(state.ee_pos)
            if check.severity == "emergency_stop":
                results.append(
                    {
                        "step": i,
                        "action": "EMERGENCY_STOP",
                        "target": "",
                        "success": False,
                        "message": f"Emergency stop: {check.violated_conditions}",
                    }
                )
                break

        self.sync_scene_objects_from_env()
        return results

    def _execute_step(self, step: PlanStep, record: bool = False) -> MotionResult:
        """Execute a single plan step."""
        if step.action == "pick":
            return self.controller.pick(step.target)
        elif step.action == "place":
            pos = step.params.get("position")
            if pos:
                return self.controller.place(step.target, np.array(pos))
            # Place on object: find that object's position
            on_target = step.params.get("on")
            if on_target:
                target_pos = self.env.get_object_pos(on_target)
                if target_pos is not None:
                    target_pos = target_pos.copy()
                    target_pos[2] += 0.02
                    return self.controller.place(step.target, target_pos)
            return MotionResult(False, "No target position for place")
        elif step.action == "move_to":
            pos = step.params.get("position")
            if pos is None and step.target and step.target != "held_object":
                obj_pos = self.env.get_object_pos(step.target)
                if obj_pos is not None:
                    pos = [float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2] + 0.12)]
            if pos is None:
                pos = [0.5, 0.0, 0.4]
            return self.controller.move_to(np.array(pos))
        elif step.action == "open_gripper":
            return self.controller.open_gripper()
        elif step.action == "close_gripper":
            result = self.controller.close_gripper()
            if step.target and step.target != "held_object" and not self.controller.is_holding:
                attach_result = self.controller.attempt_attach(step.target, snap_scale=1.8)
                if not attach_result.success:
                    return MotionResult(
                        False,
                        f"Closed gripper but failed to secure '{step.target}'",
                    )
            return result
        elif step.action == "wait":
            duration = step.params.get("duration", 1.0)
            n_steps = int(duration / self.env._control_dt)
            for _ in range(n_steps):
                self.env.step_control()
            return MotionResult(True, f"Waited {duration}s")
        else:
            return MotionResult(False, f"Unknown action: {step.action}")

    def _get_scene_summary(self, context_hint: str | None = None) -> str:
        """Get a rich scene summary for LLM context."""
        if not self._scene_objects:
            base = "Empty scene with Franka Panda robot arm on desk"
            if context_hint:
                return f"{base}\n\nRetry context:\n{context_hint}"
            return base

        lines = ["Franka Panda robot arm on a desk. Objects in scene:"]
        for obj in self._scene_objects:
            if obj.properties.get("fixed"):
                continue
            props = []
            if obj.properties.get("heavy"):
                props.append(f"heavy ({obj.mass}kg)")
            elif obj.mass < 0.05:
                props.append(f"light ({obj.mass}kg)")
            if obj.properties.get("loose"):
                props.append("loose/unsecured")
            if obj.properties.get("fragile"):
                props.append("fragile")
            if obj.properties.get("container"):
                props.append("container")
            if obj.properties.get("graspable"):
                props.append("graspable")
            if obj.properties.get("fruit"):
                props.append("fruit")
            prop_str = f" [{', '.join(props)}]" if props else ""
            lines.append(
                f"- {obj.name} ({obj.obj_type}): "
                f"pos=({obj.pos[0]:.2f}, {obj.pos[1]:.2f}, {obj.pos[2]:.2f}), "
                f"mass={obj.mass}kg{prop_str}"
            )
        if context_hint:
            lines.append("")
            lines.append("Retry context:")
            lines.append(context_hint)
        return "\n".join(lines)

    def _apply_retry_target_guardrail(
        self,
        plan: ExecutionPlan,
        intent: UserIntent,
        scene: SceneState,
        affordance: AffordanceAnalysis,
        preferred_target: str | None,
        notify=None,
    ) -> ExecutionPlan:
        """Force retry runs to stay on the same target object when requested."""
        if not preferred_target:
            return plan

        preferred_lower = preferred_target.lower()

        def _matches(target: str) -> bool:
            if not target:
                return False
            target_lower = str(target).lower()
            if target_lower == preferred_lower:
                return True
            return target_lower.split("_")[0] == preferred_lower.split("_")[0]

        for step in plan.steps:
            if _matches(step.target):
                return plan
            if _matches(step.params.get("on", "")):
                return plan

        if notify:
            notify(
                "plan_guardrail",
                f"Retry guardrail applied: forcing target to {preferred_target}",
            )

        patched_intent = replace(
            intent,
            target_objects=[preferred_target],
            immediate_goal=(
                intent.immediate_goal
                if intent.immediate_goal.lower() in ("pick", "hold", "move", "relocate")
                else "pick"
            ),
        )
        patched_plan = self.planner.generate_plan_sync(scene, patched_intent, affordance)

        for step in patched_plan.steps:
            if _matches(step.target) or _matches(step.params.get("on", "")):
                return patched_plan

        # Hard fallback: directly pick and stabilize the requested target.
        return ExecutionPlan(
            steps=[
                PlanStep(
                    action="pick",
                    target=preferred_target,
                    description=f"Pick up {preferred_target}",
                ),
                PlanStep(
                    action="wait",
                    target=preferred_target,
                    params={"duration": 2.0},
                    description=f"Hold {preferred_target} steady for stability check",
                ),
            ],
            plan_description=f"Re-grasp and stabilize {preferred_target}",
            estimated_duration=6.0,
            risk_level="low",
            reasoning="Retry guardrail fallback to keep the previous target object.",
        )
