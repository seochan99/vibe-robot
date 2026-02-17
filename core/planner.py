"""Code-as-Policies style plan generator.

Takes affordance analysis, scene state, and user intent to generate a
sequence of robot motion primitives (pick, place, move_to, etc.).
Can use LLM for complex scenarios or rule-based planning for common patterns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from config import OPENAI_API_KEY, get_config
from core.affordance_engine import AffordanceAnalysis
from core.intent_inference import UserIntent
from core.scene_understanding import SceneState


@dataclass
class PlanStep:
    """A single step in the execution plan."""

    action: str              # "pick", "place", "move_to", "open_gripper", "close_gripper", "wait"
    target: str = ""         # Object name or description
    params: dict = field(default_factory=dict)
    # e.g. {"position": [x,y,z], "duration": 2.0}
    description: str = ""    # Human-readable description


@dataclass
class ExecutionPlan:
    """Complete execution plan with steps and metadata."""

    steps: list[PlanStep]
    plan_description: str       # Overall plan summary
    estimated_duration: float   # seconds
    risk_level: str = "low"    # low, medium, high
    reasoning: str = ""         # Why this plan was chosen
    requires_approval: bool = True


PLANNER_PROMPT = """You are a robotic task planner using Code-as-Policies approach.

Given the scene analysis, user intent, and affordance analysis, generate an execution
plan as a sequence of robot primitives.

Available primitives:
- pick(object_name): Pick up an object
- place(object_name, target_position=[x,y,z]): Place object at position
- place(object_name, on=target_object): Place object on another object
- move_to(position=[x,y,z]): Move end-effector to position
- open_gripper(): Open gripper fingers
- close_gripper(): Close gripper fingers
- wait(duration): Wait for specified seconds

Scene:
{scene_info}

User Intent:
{intent_info}

Affordance Analysis:
{affordance_info}

Generate a plan in JSON:
{{
    "steps": [
        {{"action": "pick", "target": "object_name", "params": {{}}, "description": "..."}},
        ...
    ],
    "plan_description": "Overall plan summary",
    "estimated_duration": 10.0,
    "risk_level": "low",
    "reasoning": "Why this plan achieves the user's goal"
}}
"""


class Planner:
    """Generates execution plans from scene analysis and user intent."""

    def __init__(self, llm_model: str = None):
        cfg = get_config()
        self._model = llm_model or cfg["llm_model"]

    async def generate_plan(
        self,
        scene: SceneState,
        intent: UserIntent,
        affordance: AffordanceAnalysis,
    ) -> ExecutionPlan:
        """Generate an execution plan using LLM."""
        prompt = PLANNER_PROMPT.format(
            scene_info=_format_scene(scene),
            intent_info=_format_intent(intent),
            affordance_info=_format_affordance(affordance),
        )

        result = await self._call_llm(prompt)
        return _parse_plan_response(result)

    def generate_plan_sync(
        self,
        scene: SceneState,
        intent: UserIntent,
        affordance: AffordanceAnalysis,
    ) -> ExecutionPlan:
        """Synchronous plan generation using rule-based logic."""
        # Pattern matching for common scenarios
        if affordance.recommended_affordance == "weight_provider":
            return self._plan_weight_placement(scene, intent, affordance)
        elif affordance.recommended_affordance == "containment":
            return self._plan_sorting(scene, intent, affordance)
        elif intent.immediate_goal.lower() in ("move", "relocate", "transport"):
            return self._plan_simple_move(scene, intent, affordance)
        else:
            return self._plan_generic_pick_place(scene, intent, affordance)

    def _plan_weight_placement(
        self,
        scene: SceneState,
        intent: UserIntent,
        affordance: AffordanceAnalysis,
    ) -> ExecutionPlan:
        """Plan for placing a heavy object on loose papers (wind scenario)."""
        weight_obj = affordance.recommended_object
        if not weight_obj:
            return self._plan_fallback(intent, "No suitable weight object found")

        # Find the object that needs securing
        target_obj = None
        for oa in affordance.objects:
            if oa.best_affordance and oa.best_affordance.name == "needs_securing":
                target_obj = oa.object_name
                break

        if not target_obj:
            # Fallback: use first paper/loose object
            for obj in scene.objects:
                if obj.properties.get("loose") or obj.category == "paper":
                    target_obj = obj.name
                    break

        if not target_obj:
            return self._plan_fallback(intent, "No object found that needs securing")

        # Find target position (on top of the loose object)
        target_pos = None
        for obj in scene.objects:
            if obj.name == target_obj:
                target_pos = list(obj.position)
                target_pos[2] += 0.02  # slightly above
                break

        steps = [
            PlanStep(
                action="pick",
                target=weight_obj,
                description=f"Pick up {weight_obj}",
            ),
            PlanStep(
                action="place",
                target=weight_obj,
                params={"on": target_obj, "position": target_pos},
                description=f"Place {weight_obj} on {target_obj} to hold it down",
            ),
        ]

        return ExecutionPlan(
            steps=steps,
            plan_description=f"Use {weight_obj} as a paperweight on {target_obj}",
            estimated_duration=8.0,
            risk_level="low",
            reasoning=(
                f"Affordance engine identified {weight_obj}'s 'weight_provider' affordance. "
                f"Placing it on {target_obj} will prevent it from blowing away."
            ),
        )

    def _plan_sorting(
        self,
        scene: SceneState,
        intent: UserIntent,
        affordance: AffordanceAnalysis,
    ) -> ExecutionPlan:
        """Plan for sorting/clearing objects."""
        steps = []
        sortable_objects = []

        for oa in affordance.objects:
            for aff in oa.active_affordances:
                if aff.name in ("sortable", "graspable"):
                    sortable_objects.append(oa.object_name)
                    break

        # Find a container/bin
        bin_name = None
        bin_pos = None
        for obj in scene.objects:
            if obj.category in ("bin", "box") or obj.properties.get("container"):
                bin_name = obj.name
                bin_pos = list(obj.position)
                bin_pos[2] += 0.05
                break

        # Default target: edge of desk
        default_pos = [0.3, 0.3, 0.35]

        for obj_name in sortable_objects:
            target = bin_name if bin_name else "desk_edge"
            pos = bin_pos if bin_pos else default_pos

            steps.append(PlanStep(
                action="pick",
                target=obj_name,
                description=f"Pick up {obj_name}",
            ))
            steps.append(PlanStep(
                action="place",
                target=obj_name,
                params={"position": pos, "on": target},
                description=f"Place {obj_name} in {target}",
            ))

        return ExecutionPlan(
            steps=steps,
            plan_description=f"Sort {len(sortable_objects)} objects into {bin_name or 'designated area'}",
            estimated_duration=len(sortable_objects) * 6.0,
            risk_level="low",
            reasoning="Objects identified as sortable via affordance analysis.",
        )

    def _plan_simple_move(
        self,
        scene: SceneState,
        intent: UserIntent,
        affordance: AffordanceAnalysis,
    ) -> ExecutionPlan:
        """Plan for moving a single object."""
        target_obj = intent.target_objects[0] if intent.target_objects else None
        if not target_obj:
            return self._plan_fallback(intent, "No target object specified")

        # Find actual object name in scene
        actual_name = None
        for obj in scene.objects:
            if target_obj.lower() in obj.name.lower() or target_obj.lower() in obj.category.lower():
                actual_name = obj.name
                break

        if not actual_name:
            return self._plan_fallback(intent, f"Object '{target_obj}' not found in scene")

        steps = [
            PlanStep(action="pick", target=actual_name, description=f"Pick up {actual_name}"),
            PlanStep(
                action="place",
                target=actual_name,
                params={"position": [0.5, 0.0, 0.35]},
                description=f"Place {actual_name} at target",
            ),
        ]

        return ExecutionPlan(
            steps=steps,
            plan_description=f"Move {actual_name} to target position",
            estimated_duration=6.0,
            risk_level="low",
            reasoning="Simple pick-and-place based on user intent.",
        )

    def _plan_generic_pick_place(
        self,
        scene: SceneState,
        intent: UserIntent,
        affordance: AffordanceAnalysis,
    ) -> ExecutionPlan:
        """Fallback: generic pick-and-place plan."""
        obj = affordance.recommended_object or (
            intent.target_objects[0] if intent.target_objects else None
        )
        if not obj:
            return self._plan_fallback(intent, "Cannot determine target object")

        steps = [
            PlanStep(action="pick", target=obj, description=f"Pick up {obj}"),
            PlanStep(
                action="place",
                target=obj,
                params={"position": [0.5, 0.0, 0.35]},
                description=f"Place {obj} at target",
            ),
        ]

        return ExecutionPlan(
            steps=steps,
            plan_description=f"Pick and place {obj}",
            estimated_duration=6.0,
            risk_level="low",
            reasoning="Generic pick-and-place plan.",
        )

    def _plan_fallback(self, intent: UserIntent, reason: str) -> ExecutionPlan:
        """Fallback when planning fails."""
        return ExecutionPlan(
            steps=[],
            plan_description=f"Planning failed: {reason}",
            estimated_duration=0.0,
            risk_level="high",
            reasoning=reason,
            requires_approval=True,
        )

    async def _call_llm(self, prompt: str) -> dict:
        """Call LLM API for plan generation."""
        import openai

        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a robotic task planner."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return json.loads(response.choices[0].message.content)


def _format_scene(scene: SceneState) -> str:
    obj_info = []
    for o in scene.objects:
        obj_info.append(f"  - {o.name} ({o.category}): pos={o.position}, props={o.properties}")
    return (
        f"Objects:\n" + "\n".join(obj_info) +
        f"\nConditions: {scene.conditions}" +
        f"\nSummary: {scene.summary}"
    )


def _format_intent(intent: UserIntent) -> str:
    return (
        f"Command: {intent.raw_command}\n"
        f"Intended meaning: {intent.intended_meaning}\n"
        f"Immediate goal: {intent.immediate_goal}\n"
        f"Deep goal: {intent.deep_goal}\n"
        f"Target objects: {intent.target_objects}\n"
        f"Constraints: {intent.implicit_constraints}"
    )


def _format_affordance(affordance: AffordanceAnalysis) -> str:
    parts = [f"Recommended: {affordance.recommended_object} → {affordance.recommended_affordance}"]
    for oa in affordance.objects:
        if oa.active_affordances:
            active = [(a.name, a.score) for a in oa.active_affordances[:3]]
            parts.append(f"  {oa.object_name}: {active}")
    return "\n".join(parts)


def _parse_plan_response(response: dict) -> ExecutionPlan:
    steps = []
    for s in response.get("steps", []):
        steps.append(PlanStep(
            action=s.get("action", ""),
            target=s.get("target", ""),
            params=s.get("params", {}),
            description=s.get("description", ""),
        ))
    return ExecutionPlan(
        steps=steps,
        plan_description=response.get("plan_description", ""),
        estimated_duration=response.get("estimated_duration", 0.0),
        risk_level=response.get("risk_level", "low"),
        reasoning=response.get("reasoning", ""),
    )
