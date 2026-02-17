"""Gradio UI for VibeRobot — 5-screen Vibe-to-Verify workflow.

Screen 1: Command Input — natural language command entry
Screen 2: Intent Confirmation — "Is this what you meant?"
Screen 3: Safety Contract Display — safety conditions overview
Screen 4: Simulation Preview — video preview + approve/reject
Screen 5: Result / Failure Card — success or failure explanation + patches
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np

from config import get_config
from core.pipeline import PipelineResult, PipelineStage, VibeRobotPipeline
from simulator.franka_controller import FrankaController
from simulator.mujoco_env import MuJoCoEnv
from simulator.scene_builder import (
    TASK_PRESETS,
    build_scene_xml,
    get_scene_objects_info,
)
from study.logger import InteractionLogger
from ui.failure_cards import FailureCardGenerator


class VibeRobotUI:
    """Main Gradio application for VibeRobot."""

    def __init__(self):
        self._pipeline: Optional[VibeRobotPipeline] = None
        self._current_result: Optional[PipelineResult] = None
        self._logger = InteractionLogger()
        self._failure_gen = FailureCardGenerator()
        self._scene_name = "wind_paper"

    def _init_pipeline(self, scene_name: str) -> str:
        """Initialize or reinitialize the pipeline with a scene."""
        self._scene_name = scene_name
        objects = TASK_PRESETS.get(scene_name, TASK_PRESETS["wind_paper"])

        conditions = ["wind_present"] if scene_name == "wind_paper" else []
        xml = build_scene_xml(objects, include_wind="wind" in scene_name)

        env = MuJoCoEnv(xml_string=xml)
        controller = FrankaController(env)
        env.reset()

        self._pipeline = VibeRobotPipeline(env, controller, scene_objects=objects)
        return f"Scene '{scene_name}' loaded with {len(objects)} objects."

    def _process_command(self, command: str, scene_name: str) -> tuple:
        """Process a user command through the pipeline (up to approval)."""
        if self._pipeline is None or self._scene_name != scene_name:
            self._init_pipeline(scene_name)

        self._logger.log_command(command)

        conditions = []
        if "wind" in scene_name:
            conditions.append("wind_present")

        result = self._pipeline.run_sync(command, conditions=conditions, auto_approve=False)
        self._current_result = result

        # Extract display data
        intent_display = ""
        affordance_display = ""
        plan_display = ""
        safety_display = ""
        preview_image = None
        status = ""

        if result.intent:
            intent_display = _format_intent_display(result.intent)

        if result.affordance:
            affordance_display = _format_affordance_display(result.affordance)

        if result.plan:
            plan_display = _format_plan_display(result.plan)

        if result.safety_contract:
            safety_display = _format_safety_display(result.safety_contract)

        if result.preview_frames:
            preview_image = result.preview_frames[-1]  # Last frame

        if result.error:
            status = f"Error: {result.error}"
        elif result.stage == PipelineStage.AWAITING_APPROVAL:
            status = "Plan ready for your review. Approve to execute."
        else:
            status = f"Stage: {result.stage.value}"

        return (
            intent_display,
            affordance_display,
            plan_display,
            safety_display,
            preview_image,
            status,
        )

    def _approve_execution(self) -> tuple:
        """Approve and execute the current plan."""
        if self._current_result is None or self._pipeline is None:
            return "No plan to execute.", None, ""

        self._logger.log_approval(approved=True)
        result = self._pipeline.approve_and_execute(self._current_result)
        self._current_result = result

        if result.success:
            exec_summary = _format_execution_results(result.execution_results)
            self._logger.log_execution_result(result.execution_results)
            return "Execution completed successfully!", None, exec_summary
        else:
            card = self._failure_gen.generate("plan_empty", {"error": result.error})
            failure_display = _format_failure_card(card)
            self._logger.log_failure({"error": result.error})
            return f"Execution failed: {result.error}", None, failure_display

    def _reject_execution(self, reason: str) -> str:
        """Reject the current plan."""
        self._logger.log_approval(approved=False, reason=reason)
        self._current_result = None
        return "Plan rejected. Enter a new command or modify your request."

    def build(self) -> gr.Blocks:
        """Build the Gradio interface."""
        cfg = get_config()

        with gr.Blocks(
            title="VibeRobot — Vibe-to-Verify",
            theme=gr.themes.Soft(),
        ) as demo:
            gr.Markdown("# VibeRobot: Vibe-to-Verify Workflow")
            gr.Markdown(
                "Give a natural language command to the robot arm. "
                "The system will infer your intent, generate a plan, "
                "and show a simulation preview before executing."
            )

            # --- Screen 1: Command Input ---
            with gr.Row():
                with gr.Column(scale=2):
                    scene_dropdown = gr.Dropdown(
                        choices=list(TASK_PRESETS.keys()),
                        value="wind_paper",
                        label="Scene",
                    )
                    command_input = gr.Textbox(
                        label="Your Command",
                        placeholder='e.g. "날라가지 않게 막아!!" or "prevent the papers from flying away"',
                        lines=2,
                    )
                    submit_btn = gr.Button("Send Command", variant="primary")
                with gr.Column(scale=1):
                    status_box = gr.Textbox(label="Status", interactive=False, lines=2)

            # --- Screen 2: Intent Confirmation ---
            with gr.Accordion("Intent Inference (Theory of Mind)", open=True):
                intent_display = gr.Markdown(label="Inferred Intent")

            # --- Screen 2b: Affordance Analysis ---
            with gr.Accordion("Affordance Analysis (Gibson)", open=True):
                affordance_display = gr.Markdown(label="Affordances")

            # --- Screen 3: Plan + Safety Contract ---
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Execution Plan", open=True):
                        plan_display = gr.Markdown(label="Plan")
                with gr.Column():
                    with gr.Accordion("Safety Contract", open=True):
                        safety_display = gr.Markdown(label="Safety")

            # --- Screen 4: Simulation Preview + Approval ---
            with gr.Accordion("Simulation Preview", open=True):
                preview_image = gr.Image(label="Preview Frame", type="numpy")
                with gr.Row():
                    approve_btn = gr.Button("Approve & Execute", variant="primary")
                    reject_reason = gr.Textbox(
                        label="Rejection reason (optional)",
                        placeholder="Why are you rejecting?",
                        scale=2,
                    )
                    reject_btn = gr.Button("Reject", variant="stop")

            # --- Screen 5: Result / Failure Card ---
            with gr.Accordion("Execution Result", open=True):
                result_status = gr.Textbox(label="Result", interactive=False)
                result_detail = gr.Markdown(label="Details")

            # --- Event handlers ---
            submit_btn.click(
                fn=self._process_command,
                inputs=[command_input, scene_dropdown],
                outputs=[
                    intent_display,
                    affordance_display,
                    plan_display,
                    safety_display,
                    preview_image,
                    status_box,
                ],
            )

            approve_btn.click(
                fn=self._approve_execution,
                inputs=[],
                outputs=[result_status, preview_image, result_detail],
            )

            reject_btn.click(
                fn=self._reject_execution,
                inputs=[reject_reason],
                outputs=[status_box],
            )

        return demo

    def launch(self, **kwargs):
        """Launch the Gradio app."""
        cfg = get_config()
        demo = self.build()
        demo.launch(
            server_name=kwargs.get("server_name", cfg["gradio_host"]),
            server_port=kwargs.get("server_port", cfg["gradio_port"]),
            share=kwargs.get("share", False),
        )


# --- Display formatters ---

def _format_intent_display(intent) -> str:
    lines = [
        f"**Command:** {intent.raw_command}",
        f"**Literal meaning:** {intent.literal_meaning}",
        f"**Intended meaning:** {intent.intended_meaning}",
        f"**Immediate goal:** {intent.immediate_goal}",
        f"**Deep goal:** {intent.deep_goal}",
        f"**Target objects:** {', '.join(intent.target_objects)}",
        f"**Implicit constraints:** {', '.join(intent.implicit_constraints)}",
        f"**Confidence:** {intent.confidence:.0%}",
    ]
    if intent.reasoning:
        lines.append(f"\n> {intent.reasoning}")
    return "\n\n".join(lines)


def _format_affordance_display(affordance) -> str:
    lines = []
    if affordance.recommended_object:
        lines.append(
            f"**Recommendation:** Use **{affordance.recommended_object}**'s "
            f"*{affordance.recommended_affordance}* affordance"
        )
    lines.append("")
    for oa in affordance.objects:
        active = [f"{a.name} ({a.score:.2f})" for a in oa.active_affordances[:3]]
        if active:
            lines.append(f"- **{oa.object_name}** ({oa.object_category}): {', '.join(active)}")
    if affordance.reasoning:
        lines.append(f"\n```\n{affordance.reasoning}\n```")
    return "\n".join(lines)


def _format_plan_display(plan) -> str:
    lines = [f"**{plan.plan_description}**", ""]
    for i, step in enumerate(plan.steps, 1):
        params_str = ""
        if step.params:
            params_str = f" `{step.params}`"
        lines.append(f"{i}. **{step.action}**({step.target}){params_str}")
        if step.description:
            lines.append(f"   _{step.description}_")
    lines.append(f"\nEstimated duration: {plan.estimated_duration:.0f}s | Risk: {plan.risk_level}")
    return "\n".join(lines)


def _format_safety_display(contract) -> str:
    display = contract.to_display_dict()
    lines = ["**Safety Contract**", ""]

    lines.append("**Preconditions:**")
    for c in display["preconditions"]:
        icon = _severity_icon(c["severity"])
        lines.append(f"- {icon} {c['description']}")

    lines.append("\n**Invariants (during execution):**")
    for c in display["invariants"]:
        icon = _severity_icon(c["severity"])
        lines.append(f"- {icon} {c['description']}")

    lines.append("\n**Tripwires (emergency stop):**")
    for c in display["tripwires"]:
        lines.append(f"- {c['description']}")

    limits = display["limits"]
    lines.append(f"\n**Limits:** max speed={limits['max_velocity_m_s']}m/s, max torque={limits['max_torque_Nm']}Nm")
    return "\n".join(lines)


def _format_failure_card(card) -> str:
    icon = _severity_icon(card.severity)
    lines = [
        f"## {icon} {card.title}",
        f"**What happened:** {card.what_happened}",
        f"**Why:** {card.why}",
        "",
        "**Suggested fixes:**",
    ]
    for fix in card.suggested_fixes:
        lines.append(f"- {fix}")
    if card.can_auto_fix:
        lines.append(f"\n*Auto-fix available: {card.patch_description}*")
    return "\n".join(lines)


def _format_execution_results(results: list[dict]) -> str:
    lines = ["**Execution Results:**", ""]
    for r in results:
        icon = "+" if r.get("success") else "x"
        lines.append(f"- [{icon}] Step {r.get('step', '?')}: {r.get('action', '')}({r.get('target', '')}) — {r.get('message', '')}")
    return "\n".join(lines)


def _severity_icon(severity: str) -> str:
    return {
        "info": "[i]",
        "warning": "[!]",
        "error": "[!!]",
        "critical": "[!!!]",
        "halt": "[!!]",
        "emergency_stop": "[!!!]",
    }.get(severity, "[-]")


# --- Entry point ---

def main():
    app = VibeRobotUI()
    app.launch()


if __name__ == "__main__":
    main()
