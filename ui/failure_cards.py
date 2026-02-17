"""Failure card system — explain failures without jargon.

Presents robot task failures in a human-understandable format with:
- What went wrong (plain language)
- Why it happened (brief technical explanation)
- Suggested fixes (actionable patches)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FailureCard:
    """A human-readable failure explanation."""

    title: str                      # Short failure title
    what_happened: str              # Plain language description
    why: str                        # Brief explanation of cause
    severity: str = "warning"       # "info", "warning", "error", "critical"
    suggested_fixes: list[str] = field(default_factory=list)
    technical_details: str = ""     # Optional technical info (hidden by default)
    can_auto_fix: bool = False      # Whether a patch can be auto-applied
    patch_description: str = ""     # Description of the auto-fix


@dataclass
class PatchSuggestion:
    """A suggested modification to fix a failure."""

    description: str
    action: str                # "retry", "modify_plan", "change_object", "adjust_position"
    params: dict = field(default_factory=dict)


# --- Failure detection and card generation ---

FAILURE_TEMPLATES = {
    "ik_failed": FailureCard(
        title="Can't Reach That Spot",
        what_happened="The robot arm can't reach the target position. It's too far away or at an awkward angle.",
        why="The target position is outside the robot's reachable workspace.",
        severity="error",
        suggested_fixes=[
            "Move the target closer to the robot base",
            "Try a different approach angle",
            "Choose a different object that's closer",
        ],
    ),
    "workspace_violation": FailureCard(
        title="Out of Bounds",
        what_happened="The robot tried to move outside its safe zone.",
        why="The planned motion would take the arm outside the defined workspace boundaries.",
        severity="error",
        suggested_fixes=[
            "Adjust the target to be within the workspace",
            "Check if the object has moved unexpectedly",
        ],
    ),
    "grasp_failed": FailureCard(
        title="Couldn't Grab the Object",
        what_happened="The robot couldn't get a good grip on the object.",
        why="The object might be too small, too slippery, or at a difficult angle for grasping.",
        severity="warning",
        suggested_fixes=[
            "Try approaching from a different angle",
            "Try a different object instead",
            "The object might be too thin to grasp reliably",
        ],
        can_auto_fix=True,
        patch_description="Retry with adjusted grasp approach angle",
    ),
    "object_not_found": FailureCard(
        title="Object Not Found",
        what_happened="The robot can't find the object you mentioned.",
        why="The object name doesn't match any objects in the current scene.",
        severity="warning",
        suggested_fixes=[
            "Check the object name — available objects will be shown",
            "The object might have been moved or removed",
        ],
    ),
    "collision_detected": FailureCard(
        title="Collision Risk Detected",
        what_happened="The simulation detected the robot would hit something it shouldn't.",
        why="The planned path passes through or near an obstacle.",
        severity="critical",
        suggested_fixes=[
            "The plan will be adjusted to avoid the obstacle",
            "You may need to clear the path first",
        ],
        can_auto_fix=True,
        patch_description="Replan with collision avoidance",
    ),
    "safety_violation": FailureCard(
        title="Safety Limit Exceeded",
        what_happened="The robot would move too fast or apply too much force.",
        why="Safety limits are in place to prevent damage. The planned motion exceeds them.",
        severity="critical",
        suggested_fixes=[
            "The motion will be slowed down automatically",
            "If the object is heavy, the robot may need a different strategy",
        ],
    ),
    "plan_empty": FailureCard(
        title="No Plan Generated",
        what_happened="The system couldn't figure out how to accomplish your request.",
        why="The command might be too vague, or there's no clear way to achieve it with available objects.",
        severity="warning",
        suggested_fixes=[
            "Try rephrasing your command with more detail",
            "Specify which objects you'd like the robot to use",
            "Check that the required objects are in the scene",
        ],
    ),
    "object_dropped": FailureCard(
        title="Oops, Dropped It!",
        what_happened="The robot dropped the object during transport.",
        why="The grip wasn't secure enough, or the motion was too fast.",
        severity="warning",
        suggested_fixes=[
            "Retry with a slower motion",
            "Try gripping the object more carefully",
        ],
        can_auto_fix=True,
        patch_description="Retry with slower, more careful motion",
    ),
}


class FailureCardGenerator:
    """Generates failure cards from execution results and errors."""

    def generate(self, error_type: str, context: dict = None) -> FailureCard:
        """Generate a failure card for a given error type.

        Args:
            error_type: Key from FAILURE_TEMPLATES or a custom error string.
            context: Additional context (object names, positions, etc.)
        """
        context = context or {}

        # Look up template
        card = FAILURE_TEMPLATES.get(error_type)
        if card:
            # Customize with context
            return self._customize_card(card, context)

        # Unknown error — generate generic card
        return FailureCard(
            title="Something Went Wrong",
            what_happened=f"An unexpected issue occurred: {error_type}",
            why="This is an uncommon error that wasn't specifically anticipated.",
            severity="warning",
            suggested_fixes=[
                "Try the command again",
                "Try a simpler version of your request",
                "Report this issue if it persists",
            ],
            technical_details=str(context),
        )

    def from_motion_result(self, result, step_info: dict = None) -> Optional[FailureCard]:
        """Generate failure card from a MotionResult."""
        if result.success:
            return None

        msg = result.message.lower()
        step_info = step_info or {}

        if "ik failed" in msg:
            return self.generate("ik_failed", step_info)
        elif "workspace" in msg or "outside" in msg:
            return self.generate("workspace_violation", step_info)
        elif "not found" in msg:
            return self.generate("object_not_found", step_info)
        else:
            return self.generate(result.message, step_info)

    def from_safety_check(self, check_result, context: dict = None) -> Optional[FailureCard]:
        """Generate failure card from a SafetyCheckResult."""
        if check_result.passed:
            return None

        context = context or {}
        violations = check_result.violated_conditions

        if any("velocity" in v for v in violations):
            return self.generate("safety_violation", context)
        elif any("workspace" in v for v in violations):
            return self.generate("workspace_violation", context)
        elif any("collision" in v or "contact" in v for v in violations):
            return self.generate("collision_detected", context)
        else:
            return self.generate("safety_violation", context)

    def _customize_card(self, template: FailureCard, context: dict) -> FailureCard:
        """Customize a template card with specific context."""
        what = template.what_happened
        why = template.why

        obj_name = context.get("object_name", "")
        if obj_name:
            what = what.replace("the object", f"'{obj_name}'")

        available = context.get("available_objects", [])
        fixes = list(template.suggested_fixes)
        if available:
            fixes.append(f"Available objects: {', '.join(available)}")

        return FailureCard(
            title=template.title,
            what_happened=what,
            why=why,
            severity=template.severity,
            suggested_fixes=fixes,
            technical_details=template.technical_details or str(context),
            can_auto_fix=template.can_auto_fix,
            patch_description=template.patch_description,
        )
