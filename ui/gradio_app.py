"""Gradio UI for VibeRobot — chat-style Vibe-to-Verify interface.

A conversational interface where users send natural language commands and
the system responds with pipeline stages (intent, affordance, plan, safety)
as chat messages. Requires ChatGPT OAuth login to use.
"""

from __future__ import annotations

import json
import time
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
    clone_scene_objects,
    get_scene_objects_info,
)
from study.logger import InteractionLogger
from ui.failure_cards import FailureCardGenerator


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _check_auth() -> dict:
    """Check if ChatGPT OAuth is active. Returns status dict."""
    try:
        from providers.chatgpt_auth import ChatGPTAuth
        auth = ChatGPTAuth()
        return auth.status
    except Exception:
        return {"authenticated": False, "message": "Auth module unavailable"}


def _do_login() -> str:
    """Trigger browser OAuth login flow. Returns status message."""
    try:
        from providers.chatgpt_auth import ChatGPTAuth
        auth = ChatGPTAuth()
        tokens = auth.login(headless=False)
        plan = tokens.plan_type or "unknown"
        return f"authenticated|{plan}"
    except Exception as e:
        return f"error|{e}"


def _do_device_login() -> str:
    """Trigger device code login flow. Returns status message."""
    try:
        from providers.chatgpt_auth import ChatGPTAuth
        auth = ChatGPTAuth()
        tokens = auth.login(headless=True)
        plan = tokens.plan_type or "unknown"
        return f"authenticated|{plan}"
    except Exception as e:
        return f"error|{e}"


def _do_logout() -> str:
    try:
        from providers.chatgpt_auth import ChatGPTAuth
        ChatGPTAuth().logout()
        return "logged_out"
    except Exception as e:
        return f"error|{e}"


# ── Pipeline state ────────────────────────────────────────────────────────────

class AppState:
    """Shared application state."""

    def __init__(self):
        self.pipeline: Optional[VibeRobotPipeline] = None
        self.current_result: Optional[PipelineResult] = None
        self.scene_name = "wind_paper"
        self.logger = InteractionLogger()
        self.failure_gen = FailureCardGenerator()

    def init_pipeline(self, scene_name: str):
        self.scene_name = scene_name
        objects = clone_scene_objects(
            TASK_PRESETS.get(scene_name, TASK_PRESETS["wind_paper"])
        )
        xml = build_scene_xml(objects, include_wind="wind" in scene_name)
        env = MuJoCoEnv(xml_string=xml)
        controller = FrankaController(env)
        env.reset()
        self.pipeline = VibeRobotPipeline(env, controller, scene_objects=objects)

    def run_command(self, command: str, scene_name: str) -> PipelineResult:
        if self.pipeline is None or self.scene_name != scene_name:
            self.init_pipeline(scene_name)
        self.logger.log_command(command)
        conditions = ["wind_present"] if "wind" in scene_name else []
        result = self.pipeline.run_sync(command, conditions=conditions, auto_approve=False)
        self.current_result = result
        return result

    def approve(self) -> PipelineResult:
        if self.current_result is None or self.pipeline is None:
            return None
        self.logger.log_approval(approved=True)
        result = self.pipeline.approve_and_execute(self.current_result)
        self.current_result = result
        return result

    def reject(self):
        self.current_result = None


_state = AppState()


# ── Chat response builder ────────────────────────────────────────────────────

def _build_chat_response(result: PipelineResult) -> str:
    """Build a single markdown chat message from pipeline result."""
    parts = []

    # Intent
    if result.intent:
        i = result.intent
        parts.append(
            "### Intent Analysis\n"
            f"**You said:** {i.raw_command}\n\n"
            f"**You meant:** {i.intended_meaning}\n\n"
            f"**Goal:** {i.immediate_goal}\n\n"
            f"**Targets:** {', '.join(i.target_objects)}\n\n"
            f"**Constraints:** {', '.join(i.implicit_constraints)}\n\n"
            f"*Confidence: {i.confidence:.0%}*"
        )

    # Affordance
    if result.affordance:
        a = result.affordance
        lines = []
        if a.recommended_object:
            lines.append(
                f"Use **{a.recommended_object}** — "
                f"activating its *{a.recommended_affordance}* affordance"
            )
        for oa in a.objects:
            active = [f"`{af.name}` ({af.score:.0%})" for af in oa.active_affordances[:3]]
            if active:
                lines.append(f"- **{oa.object_name}** ({oa.object_category}): {', '.join(active)}")
        if a.reasoning:
            lines.append(f"\n> {a.reasoning}")
        parts.append("### Affordance Discovery\n" + "\n".join(lines))

    # Plan
    if result.plan:
        p = result.plan
        steps_md = []
        for idx, step in enumerate(p.steps, 1):
            params = ""
            if step.params:
                params = f"  `{step.params}`"
            desc = f"  *{step.description}*" if step.description else ""
            steps_md.append(f"{idx}. `{step.action}({step.target})`{params}\n{desc}")
        parts.append(
            f"### Execution Plan\n"
            f"**{p.plan_description}**\n\n"
            + "\n".join(steps_md)
            + f"\n\n*Est. {p.estimated_duration:.0f}s | Risk: {p.risk_level}*"
        )

    # Safety
    if result.safety_contract:
        d = result.safety_contract.to_display_dict()
        lines = []
        for c in d["preconditions"]:
            lines.append(f"- [{c['severity'].upper()}] {c['description']}")
        for c in d["invariants"]:
            lines.append(f"- [{c['severity'].upper()}] {c['description']}")
        for c in d["tripwires"]:
            lines.append(f"- [TRIPWIRE] {c['description']}")
        lim = d["limits"]
        lines.append(f"\nMax speed: {lim['max_velocity_m_s']} m/s | Max torque: {lim['max_torque_Nm']} Nm")
        parts.append("### Safety Contract\n" + "\n".join(lines))

    # Error
    if result.error:
        parts.append(f"### Error\n{result.error}")

    # Status
    if result.stage == PipelineStage.AWAITING_APPROVAL:
        parts.append(
            "---\n"
            "Plan is ready. Click **Approve** to execute in simulation, "
            "or **Reject** to cancel."
        )

    return "\n\n".join(parts)


def _build_exec_response(result: PipelineResult) -> str:
    """Build chat message from execution result."""
    if result is None:
        return "No plan to execute."

    if result.success:
        lines = ["### Execution Complete\n"]
        for r in result.execution_results:
            icon = "+" if r.get("success") else "x"
            lines.append(f"- [{icon}] `{r.get('action', '')}({r.get('target', '')})` — {r.get('message', '')}")
        return "\n".join(lines)
    else:
        card = _state.failure_gen.generate("plan_empty", {"error": result.error})
        lines = [
            f"### {card.title}\n",
            f"**What happened:** {card.what_happened}\n",
            f"**Why:** {card.why}\n",
            "**Suggested fixes:**",
        ]
        for fix in card.suggested_fixes:
            lines.append(f"- {fix}")
        return "\n".join(lines)


# ── Gradio event handlers ────────────────────────────────────────────────────

def on_check_auth():
    """Check auth and return visibility states."""
    status = _check_auth()
    if status.get("authenticated"):
        plan = status.get("plan", "unknown")
        acc = status.get("account_id", "")
        label = f"Logged in — ChatGPT {plan.title()}"
        if acc:
            label += f" ({acc})"
        return (
            gr.update(visible=False),   # login_section hidden
            gr.update(visible=True),    # main_section visible
            gr.update(value=label),     # auth_status label
            gr.update(visible=True),    # logout_btn visible
        )
    return (
        gr.update(visible=True),    # login_section visible
        gr.update(visible=False),   # main_section hidden
        gr.update(value=""),        # auth_status
        gr.update(visible=False),   # logout_btn
    )


def on_login():
    result = _do_login()
    if result.startswith("authenticated"):
        plan = result.split("|")[1]
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=f"Logged in — ChatGPT {plan.title()}"),
            gr.update(visible=True),
            "",
        )
    error = result.split("|", 1)[1] if "|" in result else result
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(visible=False),
        f"Login failed: {error}",
    )


def on_logout():
    _do_logout()
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(visible=False),
    )


def on_send(message: str, scene: str, chat_history: list):
    """Handle user message: run pipeline, return chat history."""
    if not message.strip():
        return chat_history, "", gr.update(interactive=True), gr.update(interactive=True)

    # Add user message
    chat_history = chat_history + [{"role": "user", "content": message}]

    # Run pipeline
    try:
        result = _state.run_command(message, scene)
        response = _build_chat_response(result)
        awaiting = result.stage == PipelineStage.AWAITING_APPROVAL
    except Exception as e:
        response = f"### Error\n{e}"
        awaiting = False

    chat_history = chat_history + [{"role": "assistant", "content": response}]

    return (
        chat_history,
        "",                                                    # clear input
        gr.update(interactive=awaiting, variant="primary" if awaiting else "secondary"),  # approve
        gr.update(interactive=awaiting, variant="stop" if awaiting else "secondary"),     # reject
    )


def on_approve(chat_history: list):
    result = _state.approve()
    response = _build_exec_response(result)
    if result:
        _state.logger.log_execution_result(result.execution_results)
    chat_history = chat_history + [{"role": "assistant", "content": response}]
    return (
        chat_history,
        gr.update(interactive=False, variant="secondary"),
        gr.update(interactive=False, variant="secondary"),
    )


def on_reject(chat_history: list):
    _state.reject()
    chat_history = chat_history + [
        {"role": "assistant", "content": "Plan rejected. Send a new command to try again."}
    ]
    return (
        chat_history,
        gr.update(interactive=False, variant="secondary"),
        gr.update(interactive=False, variant="secondary"),
    )


# ── Build UI ──────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="VibeRobot") as demo:

        # ── Login screen ─────────────────────────────────────────
        with gr.Column(visible=True, elem_classes="login-card") as login_section:
            gr.Markdown("# VibeRobot")
            gr.Markdown(
                "Vibe-to-Verify for safe robotic manipulation.\n\n"
                "Sign in with your ChatGPT account to start."
            )
            login_btn = gr.Button(
                "Sign in with ChatGPT",
                variant="primary",
                size="lg",
                elem_classes="login-btn",
            )
            device_login_btn = gr.Button(
                "Sign in with device code (SSH / headless)",
                variant="secondary",
                size="sm",
                elem_classes="login-btn",
            )
            login_error = gr.Markdown("", visible=True)

        # ── Main chat screen ─────────────────────────────────────
        with gr.Column(visible=False, elem_classes="chat-wrap") as main_section:

            # Header
            with gr.Row(elem_classes="header-bar"):
                gr.Markdown("## VibeRobot")
                auth_status = gr.Markdown("", elem_id="auth-status")
                logout_btn = gr.Button("Sign out", size="sm", visible=False)

            # Scene selector
            with gr.Row():
                scene_dropdown = gr.Dropdown(
                    choices=list(TASK_PRESETS.keys()),
                    value="wind_paper",
                    label="Scene",
                    scale=1,
                    interactive=True,
                )
                scene_info = gr.Textbox(
                    value=_scene_description("wind_paper"),
                    label="Scene description",
                    interactive=False,
                    scale=3,
                )

            # Chat area
            chatbot = gr.Chatbot(
                label="Conversation",
                height=480,
                placeholder=(
                    "Send a command to the robot arm.\n"
                    "Try: \"prevent the papers from flying away\""
                ),
            )

            # Input row
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type a command...",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

            # Action buttons (approve / reject)
            with gr.Row(elem_classes="action-row"):
                approve_btn = gr.Button(
                    "Approve & Execute",
                    variant="secondary",
                    interactive=False,
                    scale=1,
                )
                reject_btn = gr.Button(
                    "Reject Plan",
                    variant="secondary",
                    interactive=False,
                    scale=1,
                )

        # ── Events ────────────────────────────────────────────────

        # Check auth on load
        demo.load(
            fn=on_check_auth,
            outputs=[login_section, main_section, auth_status, logout_btn],
        )

        # Login
        login_btn.click(
            fn=on_login,
            outputs=[login_section, main_section, auth_status, logout_btn, login_error],
        )
        device_login_btn.click(
            fn=on_login,  # falls back to device code if browser fails
            outputs=[login_section, main_section, auth_status, logout_btn, login_error],
        )

        # Logout
        logout_btn.click(
            fn=on_logout,
            outputs=[login_section, main_section, auth_status, logout_btn],
        )

        # Scene change
        scene_dropdown.change(
            fn=lambda s: _scene_description(s),
            inputs=[scene_dropdown],
            outputs=[scene_info],
        )

        # Send command
        send_btn.click(
            fn=on_send,
            inputs=[msg_input, scene_dropdown, chatbot],
            outputs=[chatbot, msg_input, approve_btn, reject_btn],
        )
        msg_input.submit(
            fn=on_send,
            inputs=[msg_input, scene_dropdown, chatbot],
            outputs=[chatbot, msg_input, approve_btn, reject_btn],
        )

        # Approve / Reject
        approve_btn.click(
            fn=on_approve,
            inputs=[chatbot],
            outputs=[chatbot, approve_btn, reject_btn],
        )
        reject_btn.click(
            fn=on_reject,
            inputs=[chatbot],
            outputs=[chatbot, approve_btn, reject_btn],
        )

    return demo


def _scene_description(name: str) -> str:
    descriptions = {
        "wind_paper": "Wind blowing loose papers on a desk with a book nearby.",
        "pick_and_place": "Simple pick-and-place: move the red cube to the blue bin.",
        "sorting": "Sort three fruits from the desk into a container.",
    }
    return descriptions.get(name, "")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    cfg = get_config()
    demo = build_app()
    demo.launch(
        server_name=cfg["gradio_host"],
        server_port=cfg["gradio_port"],
        share=False,
        css="""
            .login-card { max-width: 460px; margin: 80px auto; padding: 40px;
                          border: 1px solid #e0e0e0; border-radius: 16px;
                          text-align: center; background: #fafafa; }
            .login-card h1 { margin-bottom: 4px; font-size: 28px; }
            .login-card p { color: #666; margin-bottom: 24px; font-size: 15px; }
            .login-btn { width: 100% !important; }
            .chat-wrap { max-width: 820px; margin: 0 auto; }
            .header-bar { display: flex; justify-content: space-between;
                          align-items: center; padding: 12px 0; margin-bottom: 4px;
                          border-bottom: 1px solid #eee; }
            .header-bar h2 { margin: 0; font-size: 18px; }
            .action-row button { min-width: 120px; }
        """,
    )


if __name__ == "__main__":
    main()
