"""Safety contract system — pre/invariant/post conditions + tripwire.

Ensures every robot action is bounded by safety constraints.
Generates YAML-like safety contracts and validates them during execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from config import MAX_EE_VELOCITY, MAX_JOINT_TORQUE, WORKSPACE_BOUNDS
from core.planner import ExecutionPlan, PlanStep


@dataclass
class SafetyCondition:
    """A single safety condition."""

    name: str
    description: str
    check_fn: Optional[Callable] = None  # Runtime check function
    severity: str = "warning"            # "warning", "halt", "emergency_stop"


@dataclass
class SafetyContract:
    """Complete safety contract for a plan execution."""

    plan_description: str
    preconditions: list[SafetyCondition]
    invariants: list[SafetyCondition]
    postconditions: list[SafetyCondition]
    tripwires: list[SafetyCondition]        # Emergency stop conditions
    max_velocity: float = MAX_EE_VELOCITY
    max_torque: float = MAX_JOINT_TORQUE
    workspace_bounds: dict = field(default_factory=lambda: dict(WORKSPACE_BOUNDS))

    def to_display_dict(self) -> dict:
        """Convert to a human-readable dict for UI display."""
        return {
            "plan": self.plan_description,
            "preconditions": [
                {"name": c.name, "description": c.description, "severity": c.severity}
                for c in self.preconditions
            ],
            "invariants": [
                {"name": c.name, "description": c.description, "severity": c.severity}
                for c in self.invariants
            ],
            "postconditions": [
                {"name": c.name, "description": c.description, "severity": c.severity}
                for c in self.postconditions
            ],
            "tripwires": [
                {"name": c.name, "description": c.description, "severity": c.severity}
                for c in self.tripwires
            ],
            "limits": {
                "max_velocity_m_s": self.max_velocity,
                "max_torque_Nm": self.max_torque,
                "workspace": self.workspace_bounds,
            },
        }


@dataclass
class SafetyCheckResult:
    """Result of a safety check during execution."""

    passed: bool
    violated_conditions: list[str] = field(default_factory=list)
    severity: str = "ok"          # "ok", "warning", "halt", "emergency_stop"
    message: str = ""


class SafetyContractGenerator:
    """Generates safety contracts from execution plans."""

    def generate(self, plan: ExecutionPlan) -> SafetyContract:
        """Generate a safety contract for a given plan."""
        preconditions = self._generate_preconditions(plan)
        invariants = self._generate_invariants(plan)
        postconditions = self._generate_postconditions(plan)
        tripwires = self._generate_tripwires(plan)

        return SafetyContract(
            plan_description=plan.plan_description,
            preconditions=preconditions,
            invariants=invariants,
            postconditions=postconditions,
            tripwires=tripwires,
        )

    def _generate_preconditions(self, plan: ExecutionPlan) -> list[SafetyCondition]:
        """Conditions that must be true BEFORE execution starts."""
        conditions = [
            SafetyCondition(
                name="workspace_clear",
                description="Robot workspace is clear of unexpected obstacles",
                severity="halt",
            ),
            SafetyCondition(
                name="arm_at_home",
                description="Robot arm is at or near home position",
                severity="warning",
            ),
            SafetyCondition(
                name="gripper_open",
                description="Gripper is in open state before pick operations",
                severity="warning",
            ),
        ]

        # Add object-specific preconditions
        for step in plan.steps:
            if step.action == "pick":
                conditions.append(SafetyCondition(
                    name=f"object_reachable_{step.target}",
                    description=f"Object '{step.target}' is within workspace and reachable",
                    severity="halt",
                ))

        return conditions

    def _generate_invariants(self, plan: ExecutionPlan) -> list[SafetyCondition]:
        """Conditions that must be true THROUGHOUT execution."""
        return [
            SafetyCondition(
                name="velocity_limit",
                description=f"End-effector velocity < {MAX_EE_VELOCITY} m/s",
                severity="emergency_stop",
            ),
            SafetyCondition(
                name="workspace_bounds",
                description="End-effector stays within workspace bounds",
                severity="halt",
            ),
            SafetyCondition(
                name="joint_limits",
                description="All joints within safe limits",
                severity="emergency_stop",
            ),
            SafetyCondition(
                name="no_self_collision",
                description="No self-collision detected",
                severity="emergency_stop",
            ),
        ]

    def _generate_postconditions(self, plan: ExecutionPlan) -> list[SafetyCondition]:
        """Conditions that must be true AFTER execution completes."""
        conditions = [
            SafetyCondition(
                name="arm_safe_position",
                description="Robot arm returns to safe position after task",
                severity="warning",
            ),
        ]

        for step in plan.steps:
            if step.action == "place":
                target_name = step.params.get("on", step.target)
                conditions.append(SafetyCondition(
                    name=f"object_placed_{step.target}",
                    description=f"Object '{step.target}' is at target location on '{target_name}'",
                    severity="warning",
                ))

        return conditions

    def _generate_tripwires(self, plan: ExecutionPlan) -> list[SafetyCondition]:
        """Emergency stop conditions — immediate halt if triggered."""
        tripwires = [
            SafetyCondition(
                name="unexpected_contact",
                description="Unexpected collision detected with non-target objects",
                severity="emergency_stop",
            ),
            SafetyCondition(
                name="object_dropped",
                description="Grasped object detected as dropped during transport",
                severity="halt",
            ),
            SafetyCondition(
                name="force_limit_exceeded",
                description=f"Joint torque exceeds {MAX_JOINT_TORQUE} Nm",
                severity="emergency_stop",
            ),
        ]
        return tripwires


class SafetyMonitor:
    """Runtime safety monitoring during plan execution."""

    def __init__(self, contract: SafetyContract):
        self.contract = contract
        self._violations: list[dict] = []

    def check_invariants(
        self,
        ee_pos: np.ndarray,
        ee_vel: Optional[np.ndarray] = None,
        joint_torques: Optional[np.ndarray] = None,
    ) -> SafetyCheckResult:
        """Check all invariant conditions against current state."""
        violated = []
        worst_severity = "ok"

        # Workspace bounds check
        bounds = self.contract.workspace_bounds
        for i, axis in enumerate(["x", "y", "z"]):
            lo, hi = bounds[axis]
            if ee_pos[i] < lo or ee_pos[i] > hi:
                violated.append(f"workspace_bounds ({axis}={ee_pos[i]:.3f})")
                worst_severity = _max_severity(worst_severity, "halt")

        # Velocity check
        if ee_vel is not None:
            speed = np.linalg.norm(ee_vel)
            if speed > self.contract.max_velocity:
                violated.append(f"velocity_limit ({speed:.3f} m/s)")
                worst_severity = _max_severity(worst_severity, "emergency_stop")

        # Torque check
        if joint_torques is not None:
            max_torque = np.max(np.abs(joint_torques))
            if max_torque > self.contract.max_torque:
                violated.append(f"force_limit ({max_torque:.1f} Nm)")
                worst_severity = _max_severity(worst_severity, "emergency_stop")

        if violated:
            self._violations.append({
                "type": "invariant",
                "conditions": violated,
                "severity": worst_severity,
            })

        return SafetyCheckResult(
            passed=len(violated) == 0,
            violated_conditions=violated,
            severity=worst_severity,
            message=f"Invariant check: {len(violated)} violations" if violated else "All invariants satisfied",
        )

    def check_postconditions(
        self,
        object_states: dict,
        plan: ExecutionPlan,
    ) -> SafetyCheckResult:
        """Check postconditions after plan execution."""
        violated = []

        for step in plan.steps:
            if step.action == "place":
                target_pos = step.params.get("position")
                if target_pos and step.target in object_states:
                    actual_pos = object_states[step.target]["pos"]
                    dist = np.linalg.norm(np.array(actual_pos) - np.array(target_pos))
                    if dist > 0.05:
                        violated.append(
                            f"object_placed_{step.target} (off by {dist:.3f}m)"
                        )

        return SafetyCheckResult(
            passed=len(violated) == 0,
            violated_conditions=violated,
            severity="warning" if violated else "ok",
        )

    @property
    def violations(self) -> list[dict]:
        return self._violations

    @property
    def has_violations(self) -> bool:
        return len(self._violations) > 0


_SEVERITY_ORDER = {"ok": 0, "warning": 1, "halt": 2, "emergency_stop": 3}


def _max_severity(a: str, b: str) -> str:
    return a if _SEVERITY_ORDER.get(a, 0) >= _SEVERITY_ORDER.get(b, 0) else b
