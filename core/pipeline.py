"""End-to-end Vibe-to-Verify pipeline orchestrator.

Coordinates the full workflow:
  Command → Intent Inference → Scene Understanding → Affordance Engine →
  Plan Generation → Safety Contract → Simulation Preview → User Approval → Execution
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

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

    def run_sync(
        self,
        command: str,
        conditions: list[str] = None,
        auto_approve: bool = False,
    ) -> PipelineResult:
        """Run the full pipeline synchronously (for development/testing).

        Args:
            command: Natural language user command.
            conditions: Environmental conditions (e.g. ["wind_present"]).
            auto_approve: Skip user approval step (for automated testing).
        """
        result = PipelineResult(stage=PipelineStage.IDLE)
        result.timestamps["start"] = time.time()

        try:
            # Stage 1: Intent Inference
            self._current_stage = PipelineStage.INTENT_INFERENCE
            result.timestamps["intent_start"] = time.time()
            scene_summary = self._get_scene_summary()
            result.intent = self.intent_engine.infer_sync(command, scene_summary)
            result.timestamps["intent_end"] = time.time()

            # Stage 2: Scene Understanding
            self._current_stage = PipelineStage.SCENE_UNDERSTANDING
            result.timestamps["scene_start"] = time.time()
            objects_info = get_scene_objects_info(self._scene_objects)
            result.scene = self.scene_engine.analyze_from_metadata(
                objects_info, conditions=conditions,
            )
            result.timestamps["scene_end"] = time.time()

            # Stage 3: Affordance Analysis
            self._current_stage = PipelineStage.AFFORDANCE_ANALYSIS
            result.timestamps["affordance_start"] = time.time()
            result.affordance = self.affordance_engine.analyze(result.scene, result.intent)
            result.timestamps["affordance_end"] = time.time()

            # Stage 4: Plan Generation
            self._current_stage = PipelineStage.PLAN_GENERATION
            result.timestamps["plan_start"] = time.time()
            result.plan = self.planner.generate_plan_sync(
                result.scene, result.intent, result.affordance,
            )
            result.timestamps["plan_end"] = time.time()

            if not result.plan.steps:
                result.stage = PipelineStage.FAILED
                result.error = f"Planning failed: {result.plan.plan_description}"
                return result

            # Stage 5: Safety Contract
            self._current_stage = PipelineStage.SAFETY_CONTRACT
            result.safety_contract = self.safety_gen.generate(result.plan)

            # Stage 6: Simulation Preview
            self._current_stage = PipelineStage.SIMULATION_PREVIEW
            result.timestamps["sim_start"] = time.time()
            preview_results = self._run_simulation_preview(result.plan, result.safety_contract)
            result.preview_frames = preview_results.get("frames", [])
            result.timestamps["sim_end"] = time.time()

            # Check if simulation had safety violations
            if preview_results.get("violations"):
                result.error = f"Safety violations in preview: {preview_results['violations']}"
                result.stage = PipelineStage.FAILED
                return result

            # Stage 7: Awaiting Approval
            if not auto_approve:
                self._current_stage = PipelineStage.AWAITING_APPROVAL
                result.stage = PipelineStage.AWAITING_APPROVAL
                return result

            # Stage 8: Execute
            self._current_stage = PipelineStage.EXECUTING
            result.timestamps["exec_start"] = time.time()
            result.execution_results = self._execute_plan(result.plan, result.safety_contract)
            result.timestamps["exec_end"] = time.time()

            result.stage = PipelineStage.COMPLETED
            result.timestamps["end"] = time.time()

        except Exception as e:
            result.stage = PipelineStage.FAILED
            result.error = str(e)

        self._current_stage = result.stage
        self._result = result
        return result

    async def run_async(
        self,
        command: str,
        provider=None,
        conditions: list[str] = None,
        auto_approve: bool = False,
    ) -> PipelineResult:
        """Run the full pipeline using an LLM provider for intent/planning.

        Falls back to rule-based if LLM call fails.

        Args:
            command: Natural language user command.
            provider: LLMProvider instance for GPT calls.
            conditions: Environmental conditions.
            auto_approve: Skip user approval step.
        """
        result = PipelineResult(stage=PipelineStage.IDLE)
        result.timestamps["start"] = time.time()

        try:
            # Stage 1: Intent Inference (LLM)
            self._current_stage = PipelineStage.INTENT_INFERENCE
            result.timestamps["intent_start"] = time.time()
            scene_summary = self._get_scene_summary()

            if provider:
                try:
                    intent_engine = IntentInference(provider=provider)
                    result.intent = await intent_engine.infer(command, scene_summary)
                except Exception:
                    # Fallback to rule-based
                    result.intent = self.intent_engine.infer_sync(command, scene_summary)
            else:
                result.intent = self.intent_engine.infer_sync(command, scene_summary)
            result.timestamps["intent_end"] = time.time()

            # Stage 2: Scene Understanding
            self._current_stage = PipelineStage.SCENE_UNDERSTANDING
            result.timestamps["scene_start"] = time.time()
            objects_info = get_scene_objects_info(self._scene_objects)
            result.scene = self.scene_engine.analyze_from_metadata(
                objects_info, conditions=conditions,
            )
            result.timestamps["scene_end"] = time.time()

            # Stage 3: Affordance Analysis
            self._current_stage = PipelineStage.AFFORDANCE_ANALYSIS
            result.timestamps["affordance_start"] = time.time()
            result.affordance = self.affordance_engine.analyze(result.scene, result.intent)
            result.timestamps["affordance_end"] = time.time()

            # Stage 4: Plan Generation (LLM)
            self._current_stage = PipelineStage.PLAN_GENERATION
            result.timestamps["plan_start"] = time.time()

            if provider:
                try:
                    planner = Planner(provider=provider)
                    result.plan = await planner.generate_plan(
                        result.scene, result.intent, result.affordance,
                    )
                except Exception:
                    # Fallback to rule-based
                    result.plan = self.planner.generate_plan_sync(
                        result.scene, result.intent, result.affordance,
                    )
            else:
                result.plan = self.planner.generate_plan_sync(
                    result.scene, result.intent, result.affordance,
                )
            result.timestamps["plan_end"] = time.time()

            if not result.plan.steps:
                result.stage = PipelineStage.FAILED
                result.error = f"Planning failed: {result.plan.plan_description}"
                return result

            # Stage 5: Safety Contract
            self._current_stage = PipelineStage.SAFETY_CONTRACT
            result.safety_contract = self.safety_gen.generate(result.plan)

            # Stage 6: Simulation Preview
            self._current_stage = PipelineStage.SIMULATION_PREVIEW
            result.timestamps["sim_start"] = time.time()
            preview_results = self._run_simulation_preview(result.plan, result.safety_contract)
            result.preview_frames = preview_results.get("frames", [])
            result.timestamps["sim_end"] = time.time()

            if preview_results.get("violations"):
                result.error = f"Safety violations in preview: {preview_results['violations']}"
                result.stage = PipelineStage.FAILED
                return result

            # Stage 7: Awaiting Approval
            if not auto_approve:
                self._current_stage = PipelineStage.AWAITING_APPROVAL
                result.stage = PipelineStage.AWAITING_APPROVAL
                return result

            # Stage 8: Execute
            self._current_stage = PipelineStage.EXECUTING
            result.timestamps["exec_start"] = time.time()
            result.execution_results = self._execute_plan(result.plan, result.safety_contract)
            result.timestamps["exec_end"] = time.time()

            result.stage = PipelineStage.COMPLETED
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
        result.execution_results = self._execute_plan(result.plan, result.safety_contract)
        self.controller.realtime = False
        result.timestamps["exec_end"] = time.time()
        result.stage = PipelineStage.COMPLETED
        result.timestamps["end"] = time.time()

        self._current_stage = result.stage
        self._result = result
        return result

    def _run_simulation_preview(
        self,
        plan: ExecutionPlan,
        contract: SafetyContract,
    ) -> dict:
        """Run the plan in simulation for preview. Runs in real-time for MJPEG viewing."""
        # Save current state
        saved_qpos = self.env.data.qpos.copy()
        saved_qvel = self.env.data.qvel.copy()

        monitor = SafetyMonitor(contract)
        frames = []
        violations = []

        # Enable real-time so MJPEG stream shows the preview
        self.controller.realtime = True

        try:
            # Reset to home
            self.env.reset()

            # Enable frame capture for animated preview
            self.env.enable_frame_capture(every_n=8, width=480, height=360)

            # Execute plan steps in simulation
            for step in plan.steps:
                motion_result = self._execute_step(step, record=True)

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
            self.controller.realtime = False
            # Restore state
            self.env.data.qpos[:] = saved_qpos
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
        self.env.reset()
        monitor = SafetyMonitor(contract)
        results = []

        for i, step in enumerate(plan.steps):
            motion_result = self._execute_step(step, record=True)
            results.append({
                "step": i,
                "action": step.action,
                "target": step.target,
                "success": motion_result.success,
                "message": motion_result.message,
            })

            # Check safety
            state = self.env.get_state()
            check = monitor.check_invariants(state.ee_pos)
            if check.severity == "emergency_stop":
                results.append({
                    "step": i,
                    "action": "EMERGENCY_STOP",
                    "target": "",
                    "success": False,
                    "message": f"Emergency stop: {check.violated_conditions}",
                })
                break

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
            pos = step.params.get("position", [0.5, 0.0, 0.4])
            return self.controller.move_to(np.array(pos))
        elif step.action == "open_gripper":
            return self.controller.open_gripper()
        elif step.action == "close_gripper":
            return self.controller.close_gripper()
        elif step.action == "wait":
            duration = step.params.get("duration", 1.0)
            n_steps = int(duration / self.env._control_dt)
            for _ in range(n_steps):
                self.env.step_control()
            return MotionResult(True, f"Waited {duration}s")
        else:
            return MotionResult(False, f"Unknown action: {step.action}")

    def _get_scene_summary(self) -> str:
        """Get a rich scene summary for LLM context."""
        if not self._scene_objects:
            return "Empty scene with Franka Panda robot arm on desk"

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
        return "\n".join(lines)
