"""Theory of Mind-based intent inference.

Bridges the gap between what users *say* (vague/colloquial commands) and
what they *mean* (structured robotic intents). Implements Gricean implicature
reasoning and pragmatic inference for the robotics manipulation domain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from config import get_config


@dataclass
class UserIntent:
    """Structured intent inferred from a natural language command."""

    raw_command: str                    # Original user input
    literal_meaning: str                # What was literally said
    intended_meaning: str               # What the user actually wants
    immediate_goal: str                 # Direct goal (e.g. "fix papers")
    deep_goal: str                      # Underlying goal (e.g. "preserve work")
    target_objects: list[str]           # Objects to act on
    implicit_constraints: list[str]     # Unspoken constraints
    confidence: float = 0.0            # 0-1 confidence score
    reasoning: str = ""                # ToM reasoning chain
    alternative_interpretations: list[str] = field(default_factory=list)


INTENT_INFERENCE_PROMPT = """You are a Theory of Mind reasoning engine for a Franka Panda robotic arm system on a desk.

Your job: infer the user's TRUE intent from a possibly vague or colloquial command, using Gricean implicature (pragmatic reasoning). Users may speak in Korean or English.

Key principles:
- Users speak colloquially and assume shared context
- The literal meaning is usually NOT the full intent
- Consider: What problem are they trying to solve? What constraints are implicit?
- Consider both the immediate goal and the deeper motivation
- Look at the scene objects and their properties to understand what actions are feasible

Scene context:
{scene_context}

User command: "{command}"

Few-shot examples:

Example 1 (Korean):
Command: "날라가지 않게 막아!!"
Scene: paper_01 (loose, fragile), book_01 (heavy, 0.5kg)
Result: User wants to prevent loose paper from blowing away. The heavy book can be used as a paperweight. Implicit: don't damage paper, solution should be reversible.

Example 2 (English):
Command: "Clean up the desk"
Scene: red_cube, blue_bin (container)
Result: User wants objects moved into the container. Implicit: place carefully, don't knock things over.

Example 3 (Korean):
Command: "책 좀 집어줘"
Scene: book_01 at (0.35, 0.15, 0.335)
Result: User wants the robot to pick up the book. Simple pick action.

Respond in JSON:
{{
    "literal_meaning": "What the words literally say",
    "intended_meaning": "What the user actually wants the robot to do",
    "immediate_goal": "The direct, actionable goal",
    "deep_goal": "The underlying motivation/need",
    "target_objects": ["object names from the scene relevant to the task"],
    "implicit_constraints": [
        "constraint1 (e.g. don't damage the paper)",
        "constraint2 (e.g. solution should be reversible)"
    ],
    "confidence": 0.85,
    "reasoning": "Step-by-step ToM reasoning explaining the inference",
    "alternative_interpretations": ["other possible meanings"]
}}
"""


class IntentInference:
    """Infers structured user intent from natural language commands."""

    def __init__(self, llm_model: str = None, provider=None):
        cfg = get_config()
        self._model = llm_model or cfg["llm_model"]
        self._provider = provider  # LLMProvider instance (lazy-loaded if None)
        self._cache: dict[str, UserIntent] = {}

    async def infer(
        self,
        command: str,
        scene_context: str = "",
    ) -> UserIntent:
        """Infer user intent from a natural language command.

        Args:
            command: Raw user command (e.g. "날라가지 않게 막아!!")
            scene_context: Description of the current scene state
        """
        cache_key = f"{command}|{scene_context}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = INTENT_INFERENCE_PROMPT.format(
            scene_context=scene_context or "No scene context provided",
            command=command,
        )

        result = await self._call_llm(prompt)
        intent = _parse_intent_response(command, result)

        self._cache[cache_key] = intent
        return intent

    def infer_sync(
        self,
        command: str,
        scene_context: str = "",
    ) -> UserIntent:
        """Synchronous version of infer() for testing/development.

        Uses rule-based heuristics when API is not available.
        """
        cache_key = f"{command}|{scene_context}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        intent = _rule_based_inference(command, scene_context)
        self._cache[cache_key] = intent
        return intent

    def _get_provider(self):
        if self._provider is None:
            from providers.setup import get_provider
            self._provider = get_provider()
        return self._provider

    async def _call_llm(self, prompt: str) -> dict:
        """Call LLM via provider abstraction."""
        provider = self._get_provider()
        response = await provider.generate(
            prompt,
            system_prompt="You are a Theory of Mind reasoning engine.",
            json_mode=True,
            temperature=0.3,
            max_tokens=800,
        )
        return response.parse_json()

    def clear_cache(self) -> None:
        self._cache.clear()


def _parse_intent_response(command: str, response: dict) -> UserIntent:
    """Parse LLM response into a UserIntent."""
    return UserIntent(
        raw_command=command,
        literal_meaning=response.get("literal_meaning", ""),
        intended_meaning=response.get("intended_meaning", ""),
        immediate_goal=response.get("immediate_goal", ""),
        deep_goal=response.get("deep_goal", ""),
        target_objects=response.get("target_objects", []),
        implicit_constraints=response.get("implicit_constraints", []),
        confidence=response.get("confidence", 0.5),
        reasoning=response.get("reasoning", ""),
        alternative_interpretations=response.get("alternative_interpretations", []),
    )


# --- Rule-based fallback for development/testing ---

_KEYWORD_PATTERNS = {
    "prevent_flying": {
        "keywords": ["날라", "날아", "fly", "blow", "wind", "막아", "prevent", "stop", "secure"],
        "intent": {
            "literal_meaning": "Prevent something from flying away",
            "intended_meaning": "Secure loose objects against wind/disturbance",
            "immediate_goal": "secure",
            "deep_goal": "Preserve documents/work materials",
            "implicit_constraints": [
                "Do not damage the loose objects",
                "Solution should be reversible",
                "Use available heavy objects as weights",
            ],
        },
    },
    "clean_up": {
        "keywords": ["치워", "치우", "clean", "clear", "remove", "정리"],
        "intent": {
            "literal_meaning": "Clean up / remove items",
            "intended_meaning": "Move specified objects off the workspace",
            "immediate_goal": "clean",
            "deep_goal": "Create a clean, organized workspace",
            "implicit_constraints": [
                "Don't throw objects — place them carefully",
                "Group similar objects if possible",
            ],
        },
    },
    "move_object": {
        "keywords": ["옮겨", "옮기", "move", "put", "place", "놓아", "놓", "올려", "위에", "onto", "on top"],
        "intent": {
            "literal_meaning": "Move an object to a location",
            "intended_meaning": "Relocate a specific object to a target",
            "immediate_goal": "move",
            "deep_goal": "Reorganize workspace as desired",
            "implicit_constraints": [
                "Handle objects carefully",
                "Don't disturb other objects",
            ],
        },
    },
    "pick_up": {
        "keywords": ["집어", "잡아", "pick", "grab", "들어", "들고", "들어서", "가져"],
        "intent": {
            "literal_meaning": "Pick up an object",
            "intended_meaning": "Grasp and lift a specific object",
            "immediate_goal": "pick",
            "deep_goal": "Prepare object for subsequent action",
            "implicit_constraints": [
                "Grip firmly but don't crush",
                "Approach from a safe angle",
            ],
        },
    },
    "hold": {
        "keywords": ["잡고", "가만", "hold", "stay", "keep", "유지"],
        "intent": {
            "literal_meaning": "Hold an object in place",
            "intended_meaning": "Pick up and hold an object still",
            "immediate_goal": "hold",
            "deep_goal": "Maintain object in a steady position",
            "implicit_constraints": [
                "Don't move after grasping",
                "Keep a firm grip",
            ],
        },
    },
    "sort": {
        "keywords": ["정렬", "분류", "sort", "organize", "나눠", "골라"],
        "intent": {
            "literal_meaning": "Sort or organize objects",
            "intended_meaning": "Organize objects by category",
            "immediate_goal": "sort",
            "deep_goal": "Create an orderly arrangement",
            "implicit_constraints": [
                "Group similar items together",
                "Handle each object carefully",
            ],
        },
    },
    "place": {
        "keywords": ["내려놓", "놔", "put down", "release", "내려"],
        "intent": {
            "literal_meaning": "Put down an object",
            "intended_meaning": "Place the held object down",
            "immediate_goal": "place",
            "deep_goal": "Safely release the object",
            "implicit_constraints": [
                "Place gently",
                "Put on a stable surface",
            ],
        },
    },
}


def _rule_based_inference(command: str, scene_context: str) -> UserIntent:
    """Simple rule-based intent inference for offline development."""
    command_lower = command.lower()

    best_match = None
    best_score = 0

    for pattern_name, pattern in _KEYWORD_PATTERNS.items():
        score = sum(1 for kw in pattern["keywords"] if kw in command_lower)
        if score > best_score:
            best_score = score
            best_match = pattern

    if best_match is None:
        extracted = _extract_target_objects(command, scene_context, {})
        inferred_goal = "pick" if len(extracted) == 1 else ("move" if len(extracted) >= 2 else "follow")
        intended = (
            "Relocate the primary object to the referenced destination"
            if inferred_goal == "move"
            else ("Grasp and lift the referenced object" if inferred_goal == "pick" else f"Execute: {command}")
        )
        return UserIntent(
            raw_command=command,
            literal_meaning=command,
            intended_meaning=intended,
            immediate_goal=inferred_goal,
            deep_goal="Complete the requested task",
            target_objects=extracted,
            implicit_constraints=["Operate safely"],
            confidence=0.45 if extracted else 0.3,
            reasoning=(
                "No clear pattern matched; inferred target objects from command text "
                f"({extracted}) and derived goal={inferred_goal}."
            ),
        )

    intent_data = best_match["intent"]

    # Try to extract target objects from scene context
    target_objects = _extract_target_objects(command, scene_context, intent_data)

    return UserIntent(
        raw_command=command,
        literal_meaning=intent_data["literal_meaning"],
        intended_meaning=intent_data["intended_meaning"],
        immediate_goal=intent_data["immediate_goal"],
        deep_goal=intent_data["deep_goal"],
        target_objects=target_objects,
        implicit_constraints=intent_data["implicit_constraints"],
        confidence=min(0.7, 0.3 + best_score * 0.15),
        reasoning=f"Matched pattern keywords (score={best_score}). Scene: {scene_context[:100]}",
    )


_KO_EN_OBJECT_MAP: dict[str, str] = {
    "사과": "apple",
    "바나나": "banana",
    "오렌지": "orange",
    "책": "book",
    "종이": "paper",
    "컵": "cup",
    "펜": "pen",
    "큐브": "cube",
    "상자": "bin",
    "빨간": "red_cube",
    "파란": "blue_bin",
}

_EN_OBJECT_NAMES = ["book", "paper", "cup", "pen", "cube", "bin", "apple", "banana", "orange",
                    "red_cube", "blue_bin"]


def _extract_target_objects(command: str, scene_context: str, intent_data: dict) -> list[str]:
    """Extract target object names from the user command.

    Prioritizes objects explicitly mentioned in the command.
    If none are mentioned explicitly, returns an empty list so retry/context
    resolver can decide instead of guessing from all scene objects.
    """
    cmd_lower = command.lower()

    # 1) Match Korean object names in command → translate to English
    from_command: list[str] = []
    for ko, en in _KO_EN_OBJECT_MAP.items():
        if ko in cmd_lower:
            from_command.append(en)

    # 2) Match English object names in command
    for name in _EN_OBJECT_NAMES:
        if name in cmd_lower and name not in from_command:
            from_command.append(name)

    # If user explicitly mentioned objects, return only those
    if from_command:
        return from_command

    return []
