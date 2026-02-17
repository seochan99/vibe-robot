"""Interaction logger for user study data collection.

Records all pipeline events, user actions, timing data, and system state
for subsequent quantitative and qualitative analysis.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class LogEvent:
    """A single logged event."""

    timestamp: float
    event_type: str           # e.g. "command", "intent", "plan", "approval", "execution"
    stage: str                # Pipeline stage
    data: dict = field(default_factory=dict)
    session_id: str = ""
    participant_id: str = ""


class InteractionLogger:
    """Logs all interactions during a VibeRobot session."""

    def __init__(
        self,
        session_id: str = None,
        participant_id: str = "",
        condition: str = "",
        output_dir: str = "logs",
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.participant_id = participant_id
        self.condition = condition
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._events: list[LogEvent] = []
        self._session_start = time.time()
        self._task_timers: dict[str, float] = {}

        self.log("session_start", "init", {
            "participant_id": participant_id,
            "condition": condition,
        })

    def log(self, event_type: str, stage: str, data: dict = None) -> LogEvent:
        """Log an event."""
        event = LogEvent(
            timestamp=time.time(),
            event_type=event_type,
            stage=stage,
            data=data or {},
            session_id=self.session_id,
            participant_id=self.participant_id,
        )
        self._events.append(event)
        return event

    def log_command(self, command: str, task_id: str = "") -> None:
        """Log a user command."""
        self.log("command", "input", {"command": command, "task_id": task_id})

    def log_intent(self, intent_data: dict) -> None:
        """Log inferred intent."""
        self.log("intent", "intent_inference", intent_data)

    def log_affordance(self, affordance_data: dict) -> None:
        """Log affordance analysis."""
        self.log("affordance", "affordance_analysis", affordance_data)

    def log_plan(self, plan_data: dict) -> None:
        """Log generated plan."""
        self.log("plan", "plan_generation", plan_data)

    def log_safety_contract(self, contract_data: dict) -> None:
        """Log safety contract."""
        self.log("safety_contract", "safety_contract", contract_data)

    def log_preview_action(self, action: str, data: dict = None) -> None:
        """Log preview interaction (play, pause, replay, etc.)."""
        self.log("preview_action", "simulation_preview", {
            "action": action,
            **(data or {}),
        })

    def log_approval(self, approved: bool, reason: str = "") -> None:
        """Log user approval/rejection decision."""
        self.log("approval", "awaiting_approval", {
            "approved": approved,
            "reason": reason,
        })

    def log_execution_result(self, results: list[dict]) -> None:
        """Log execution results."""
        self.log("execution", "executing", {"results": results})

    def log_failure(self, failure_card: dict) -> None:
        """Log a failure event."""
        self.log("failure", "failed", failure_card)

    def log_patch_applied(self, patch_data: dict) -> None:
        """Log a patch application."""
        self.log("patch", "patching", patch_data)

    def start_task_timer(self, task_id: str) -> None:
        """Start timing a task."""
        self._task_timers[task_id] = time.time()

    def stop_task_timer(self, task_id: str) -> float:
        """Stop timing a task. Returns elapsed seconds."""
        start = self._task_timers.pop(task_id, None)
        if start is None:
            return 0.0
        elapsed = time.time() - start
        self.log("task_timer", "measurement", {
            "task_id": task_id,
            "elapsed_seconds": elapsed,
        })
        return elapsed

    def get_events(self, event_type: str = None) -> list[LogEvent]:
        """Get logged events, optionally filtered by type."""
        if event_type:
            return [e for e in self._events if e.event_type == event_type]
        return self._events

    def save(self, filename: str = None) -> str:
        """Save all events to a JSON Lines file."""
        if filename is None:
            filename = f"session_{self.session_id}_{self.participant_id}.jsonl"
        filepath = self._output_dir / filename

        with open(filepath, "w") as f:
            for event in self._events:
                line = {
                    "timestamp": event.timestamp,
                    "relative_time": event.timestamp - self._session_start,
                    "event_type": event.event_type,
                    "stage": event.stage,
                    "data": event.data,
                    "session_id": event.session_id,
                    "participant_id": event.participant_id,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        return str(filepath)

    def get_summary(self) -> dict:
        """Get session summary statistics."""
        total_time = time.time() - self._session_start
        event_counts = {}
        for e in self._events:
            event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1

        commands = self.get_events("command")
        approvals = self.get_events("approval")
        failures = self.get_events("failure")

        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "condition": self.condition,
            "total_duration_seconds": total_time,
            "num_commands": len(commands),
            "num_approvals": len(approvals),
            "num_approvals_granted": sum(
                1 for a in approvals if a.data.get("approved", False)
            ),
            "num_failures": len(failures),
            "event_counts": event_counts,
        }

    @property
    def session_duration(self) -> float:
        return time.time() - self._session_start
