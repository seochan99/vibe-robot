"""Gibson Affordance Engine — Latent Affordance Discovery.

Core theoretical contribution of VibeRobot. Implements affordance activation
based on Gibson's ecological psychology: objects have *latent* affordances
that become *active* depending on the environmental context and agent goals.

Mathematical framing (Liao & Holz, IUI 2025):
  Affordance = argmax_a [Confidence(a|o,c) × Utility(e(a,o,c))]

where:
  a = affordance action
  o = object
  c = context (environment + goals)
  e = expected effect
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.scene_understanding import ObjectInfo, SceneState
from core.intent_inference import UserIntent


@dataclass
class Affordance:
    """A single affordance of an object."""

    name: str                    # e.g. "weight_provider", "container", "graspable"
    description: str             # Human-readable description
    confidence: float = 0.0     # How confident we are this affordance is relevant
    utility: float = 0.0        # How useful this affordance is for the current goal
    score: float = 0.0          # confidence × utility
    is_active: bool = False     # Activated by current context?
    activation_reason: str = "" # Why this affordance was activated


@dataclass
class ObjectAffordances:
    """All affordances for a single object."""

    object_name: str
    object_category: str
    latent_affordances: list[Affordance]     # All possible affordances
    active_affordances: list[Affordance]     # Context-activated affordances
    best_affordance: Optional[Affordance] = None  # Highest-scoring active affordance


@dataclass
class AffordanceAnalysis:
    """Complete affordance analysis for a scene."""

    objects: list[ObjectAffordances]
    recommended_object: Optional[str] = None
    recommended_affordance: Optional[str] = None
    reasoning: str = ""


# --- Latent Affordance Database ---
# Maps object categories to their latent (potential) affordances.
# Each affordance includes conditions under which it activates.

LATENT_AFFORDANCES: dict[str, list[dict]] = {
    "book": [
        {
            "name": "readability",
            "description": "Can be read for information",
            "base_confidence": 0.9,
            "activates_when": [],  # always available
        },
        {
            "name": "weight_provider",
            "description": "Can hold down loose objects with its weight",
            "base_confidence": 0.85,
            "activates_when": ["wind_present", "papers_unstable", "loose_objects"],
        },
        {
            "name": "stackability",
            "description": "Can be stacked on or under other objects",
            "base_confidence": 0.8,
            "activates_when": ["need_height", "organizing"],
        },
        {
            "name": "barrier",
            "description": "Can block movement or create a wall",
            "base_confidence": 0.6,
            "activates_when": ["need_barrier", "wind_present"],
        },
        {
            "name": "surface_provider",
            "description": "Provides a flat surface for writing",
            "base_confidence": 0.5,
            "activates_when": ["need_surface"],
        },
    ],
    "cup": [
        {
            "name": "containment",
            "description": "Can contain small objects or liquids",
            "base_confidence": 0.9,
            "activates_when": ["need_container", "collecting"],
        },
        {
            "name": "weight_provider",
            "description": "Can hold down objects (when filled or heavy enough)",
            "base_confidence": 0.5,
            "activates_when": ["wind_present", "papers_unstable"],
        },
        {
            "name": "scooping",
            "description": "Can scoop small items",
            "base_confidence": 0.3,
            "activates_when": ["need_scoop"],
        },
    ],
    "pen": [
        {
            "name": "writing",
            "description": "Can mark or write on surfaces",
            "base_confidence": 0.9,
            "activates_when": [],
        },
        {
            "name": "pointing",
            "description": "Can point at or indicate objects",
            "base_confidence": 0.5,
            "activates_when": ["indicating"],
        },
        {
            "name": "weight_provider",
            "description": "Very light — minimal weight-holding ability",
            "base_confidence": 0.1,
            "activates_when": ["wind_present"],
        },
    ],
    "paper": [
        {
            "name": "writable_surface",
            "description": "Can be written on",
            "base_confidence": 0.9,
            "activates_when": [],
        },
        {
            "name": "coverable",
            "description": "Can cover small areas",
            "base_confidence": 0.4,
            "activates_when": ["need_cover"],
        },
        {
            "name": "needs_securing",
            "description": "Vulnerable to wind/disturbance when loose",
            "base_confidence": 0.95,
            "activates_when": ["wind_present", "papers_unstable"],
        },
    ],
    "cube": [
        {
            "name": "graspable",
            "description": "Easy to pick up and move",
            "base_confidence": 0.9,
            "activates_when": [],
        },
        {
            "name": "stackability",
            "description": "Can be stacked",
            "base_confidence": 0.85,
            "activates_when": ["organizing", "building"],
        },
    ],
    "bin": [
        {
            "name": "containment",
            "description": "Can receive and contain objects",
            "base_confidence": 0.95,
            "activates_when": ["cleaning", "sorting", "organizing"],
        },
    ],
    "fruit": [
        {
            "name": "graspable",
            "description": "Can be picked up",
            "base_confidence": 0.8,
            "activates_when": [],
        },
        {
            "name": "sortable",
            "description": "Can be categorized and sorted",
            "base_confidence": 0.7,
            "activates_when": ["sorting", "organizing"],
        },
    ],
}

# --- Goal-to-context mapping ---
# Maps user goals to environmental context tags that activate affordances.

GOAL_CONTEXT_MAP: dict[str, list[str]] = {
    "secure_objects": ["wind_present", "papers_unstable", "loose_objects"],
    "clean_workspace": ["cleaning", "organizing"],
    "sort_objects": ["sorting", "organizing"],
    "build_structure": ["building", "need_height"],
    "move_object": [],
    "contain_objects": ["need_container", "collecting"],
}


class AffordanceEngine:
    """Gibson-inspired affordance activation engine.

    Discovers which latent affordances are relevant given the current
    scene context, user intent, and environmental conditions.
    """

    def analyze(
        self,
        scene: SceneState,
        intent: UserIntent,
    ) -> AffordanceAnalysis:
        """Analyze affordances for all objects given scene and intent.

        This is the core algorithm:
        1. For each object, retrieve its latent affordances
        2. Compute activation based on current conditions and user goal
        3. Score each affordance: confidence × utility
        4. Select the best object-affordance pair for the task
        """
        # Derive context tags from conditions and intent
        context_tags = set(scene.conditions)
        context_tags.update(_intent_to_context_tags(intent))

        all_object_affordances = []

        for obj in scene.objects:
            obj_affordances = self._analyze_object(obj, context_tags, intent)
            all_object_affordances.append(obj_affordances)

        # Find the best object-affordance pair
        # Exclude passive affordances (e.g. "needs_securing") — these indicate
        # objects that need help, not objects that provide a solution.
        PASSIVE_AFFORDANCES = {"needs_securing", "writable_surface", "coverable"}

        best_obj = None
        best_aff = None
        best_score = -1.0

        for oa in all_object_affordances:
            # Find best *actionable* affordance for this object
            for aff in oa.active_affordances:
                if aff.name not in PASSIVE_AFFORDANCES and aff.score > best_score:
                    best_score = aff.score
                    best_obj = oa.object_name
                    best_aff = aff.name

        reasoning = _build_reasoning(all_object_affordances, best_obj, best_aff, context_tags)

        return AffordanceAnalysis(
            objects=all_object_affordances,
            recommended_object=best_obj,
            recommended_affordance=best_aff,
            reasoning=reasoning,
        )

    def _analyze_object(
        self,
        obj: ObjectInfo,
        context_tags: set[str],
        intent: UserIntent,
    ) -> ObjectAffordances:
        """Analyze affordances for a single object."""
        category = obj.category
        latent_defs = LATENT_AFFORDANCES.get(category, [])

        # Also check if the object itself has a custom category
        if not latent_defs and obj.properties.get("fruit"):
            latent_defs = LATENT_AFFORDANCES.get("fruit", [])

        latent = []
        active = []

        for aff_def in latent_defs:
            # Compute activation: does the context match?
            activation_conditions = set(aff_def["activates_when"])
            is_active = (
                len(activation_conditions) == 0  # always active
                or len(activation_conditions & context_tags) > 0
            )

            confidence = aff_def["base_confidence"]
            if is_active and activation_conditions:
                # Boost confidence based on how many conditions match
                match_ratio = len(activation_conditions & context_tags) / len(activation_conditions)
                confidence = min(1.0, confidence * (0.8 + 0.2 * match_ratio))

            # Compute utility based on how well this affordance serves the intent
            utility = _compute_utility(aff_def["name"], intent, obj)

            score = confidence * utility

            activation_reason = ""
            if is_active and activation_conditions:
                matched = activation_conditions & context_tags
                activation_reason = f"Activated by: {', '.join(matched)}"

            affordance = Affordance(
                name=aff_def["name"],
                description=aff_def["description"],
                confidence=round(confidence, 3),
                utility=round(utility, 3),
                score=round(score, 3),
                is_active=is_active,
                activation_reason=activation_reason,
            )

            latent.append(affordance)
            if is_active and score > 0:
                active.append(affordance)

        # Sort active by score descending
        active.sort(key=lambda a: a.score, reverse=True)
        best = active[0] if active else None

        return ObjectAffordances(
            object_name=obj.name,
            object_category=category,
            latent_affordances=latent,
            active_affordances=active,
            best_affordance=best,
        )


def _intent_to_context_tags(intent: UserIntent) -> set[str]:
    """Map user intent to context tags for affordance activation."""
    tags = set()

    # Check intent keywords against goal-context map
    combined = (
        intent.immediate_goal + " " +
        intent.intended_meaning + " " +
        intent.raw_command
    ).lower()

    keyword_to_goal = {
        "secure": "secure_objects",
        "fix": "secure_objects",
        "hold": "secure_objects",
        "prevent": "secure_objects",
        "막": "secure_objects",
        "clean": "clean_workspace",
        "clear": "clean_workspace",
        "치우": "clean_workspace",
        "sort": "sort_objects",
        "organize": "sort_objects",
        "정리": "sort_objects",
        "stack": "build_structure",
        "build": "build_structure",
        "move": "move_object",
        "put": "move_object",
        "collect": "contain_objects",
        "gather": "contain_objects",
    }

    for keyword, goal in keyword_to_goal.items():
        if keyword in combined:
            context = GOAL_CONTEXT_MAP.get(goal, [])
            tags.update(context)

    return tags


def _compute_utility(affordance_name: str, intent: UserIntent, obj: ObjectInfo) -> float:
    """Compute how useful an affordance is for the current intent."""
    combined = (
        intent.immediate_goal + " " +
        intent.intended_meaning + " " +
        intent.raw_command
    ).lower()

    # High utility mappings
    if affordance_name == "weight_provider":
        if any(kw in combined for kw in ["secure", "fix", "hold", "prevent", "막", "wind", "fly", "날"]):
            # Heavier objects are more useful as weights
            mass = obj.properties.get("mass_kg", obj.properties.get("weight_kg", 0.1))
            return min(1.0, 0.5 + mass * 0.8)
        return 0.1

    if affordance_name == "containment":
        if any(kw in combined for kw in ["clean", "clear", "sort", "치우", "정리", "collect"]):
            return 0.9
        return 0.2

    if affordance_name == "needs_securing":
        if any(kw in combined for kw in ["secure", "fix", "prevent", "막", "wind", "날"]):
            return 0.95  # This object needs help
        return 0.3

    if affordance_name == "graspable":
        return 0.5  # Always somewhat useful

    if affordance_name == "sortable":
        if any(kw in combined for kw in ["sort", "organize", "정리"]):
            return 0.8
        return 0.2

    if affordance_name == "stackability":
        if any(kw in combined for kw in ["stack", "build", "organize"]):
            return 0.7
        return 0.2

    if affordance_name == "barrier":
        if any(kw in combined for kw in ["block", "wall", "막", "prevent"]):
            return 0.6
        return 0.1

    # Default low utility for unmatched affordances
    return 0.15


def _build_reasoning(
    all_obj_affs: list[ObjectAffordances],
    best_obj: Optional[str],
    best_aff: Optional[str],
    context_tags: set[str],
) -> str:
    """Build human-readable reasoning for the affordance analysis."""
    parts = [f"Context tags: {', '.join(context_tags) if context_tags else 'none'}"]

    for oa in all_obj_affs:
        active_names = [a.name for a in oa.active_affordances[:3]]
        parts.append(
            f"  {oa.object_name} ({oa.object_category}): "
            f"active=[{', '.join(active_names)}]"
        )
        if oa.best_affordance:
            parts.append(
                f"    best: {oa.best_affordance.name} "
                f"(score={oa.best_affordance.score:.3f})"
            )

    if best_obj and best_aff:
        parts.append(f"\nRecommendation: Use {best_obj}'s '{best_aff}' affordance.")

    return "\n".join(parts)
