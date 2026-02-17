"""VLM-based scene understanding.

Analyzes the simulation scene (via rendered image or object metadata) to identify
objects, their properties, spatial relationships, and environmental conditions.
Uses GPT-4o Vision or Claude Vision via API.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import get_config


@dataclass
class ObjectInfo:
    """Perceived object in the scene."""
    name: str
    category: str                  # e.g. "book", "cup", "paper"
    position: list[float]          # [x, y, z]
    properties: dict = field(default_factory=dict)
    # e.g. {"graspable": True, "heavy": True, "mass_kg": 0.5, "loose": True}


@dataclass
class SceneState:
    """Complete scene understanding output."""
    objects: list[ObjectInfo]
    conditions: list[str]          # e.g. ["wind_present", "papers_unstable"]
    spatial_relations: list[str]   # e.g. ["book_01 is_on desk", "paper_01 is_on desk"]
    summary: str                   # Natural language summary


SCENE_ANALYSIS_PROMPT = """You are a robotic perception system analyzing a tabletop scene.

Given the scene information below, identify:
1. All objects with their properties (graspable, heavy, fragile, loose, container, etc.)
2. Environmental conditions (wind, unstable objects, obstacles)
3. Spatial relationships between objects
4. A brief natural language summary

Respond in JSON format:
{
    "objects": [
        {"name": "...", "category": "...", "position": [x,y,z], "properties": {...}}
    ],
    "conditions": ["condition1", "condition2"],
    "spatial_relations": ["obj1 relation obj2", ...],
    "summary": "..."
}
"""

SCENE_ANALYSIS_PROMPT_WITH_IMAGE = """You are a robotic perception system. Analyze this tabletop scene image.

Identify:
1. All visible objects — estimate their category, approximate position, and properties
   (graspable, heavy, fragile, loose, container, etc.)
2. Environmental conditions (anything suggesting wind, instability, obstacles)
3. Spatial relationships between objects
4. Brief natural language summary

Respond in JSON only:
{
    "objects": [
        {"name": "descriptive_name", "category": "...", "position": [x,y,z], "properties": {...}}
    ],
    "conditions": ["condition1", ...],
    "spatial_relations": ["obj1 relation obj2", ...],
    "summary": "..."
}
"""


class SceneUnderstanding:
    """Analyzes scene using VLM (GPT-4o or Claude Vision)."""

    def __init__(self, vlm_backend: str = None, vlm_model: str = None, provider=None):
        cfg = get_config()
        self._backend = vlm_backend or cfg["vlm_backend"]
        self._model = vlm_model or cfg["vlm_model"]
        self._provider = provider  # LLMProvider instance (lazy-loaded if None)
        self._cache: dict[str, SceneState] = {}

    def analyze_from_metadata(
        self,
        objects_info: list[dict],
        conditions: list[str] = None,
    ) -> SceneState:
        """Analyze scene from structured object metadata (no VLM needed).

        Used during development/testing when we have ground-truth object info
        from the scene builder.
        """
        cache_key = json.dumps(objects_info, sort_keys=True)
        if cache_key in self._cache:
            return self._cache[cache_key]

        objects = []
        for info in objects_info:
            obj = ObjectInfo(
                name=info["name"],
                category=_infer_category(info["name"]),
                position=info.get("position", [0, 0, 0]),
                properties={k: v for k, v in info.items() if k not in ("name", "type", "position")},
            )
            objects.append(obj)

        # Infer conditions from object properties
        inferred_conditions = list(conditions or [])
        for obj in objects:
            if obj.properties.get("loose"):
                inferred_conditions.append(f"{obj.name}_unstable")

        # Infer spatial relations from positions
        spatial = _infer_spatial_relations(objects)

        summary = _build_summary(objects, inferred_conditions)

        state = SceneState(
            objects=objects,
            conditions=inferred_conditions,
            spatial_relations=spatial,
            summary=summary,
        )
        self._cache[cache_key] = state
        return state

    async def analyze_from_image(
        self,
        image: np.ndarray,
        additional_context: str = "",
    ) -> SceneState:
        """Analyze scene from a rendered image using VLM API.

        Args:
            image: RGB numpy array (H, W, 3).
            additional_context: Extra context to include in the prompt.
        """
        image_b64 = _encode_image(image)

        # Use provider abstraction for vision analysis
        provider = self._get_provider()
        prompt = SCENE_ANALYSIS_PROMPT_WITH_IMAGE
        if additional_context:
            prompt += f"\n\nAdditional context: {additional_context}"

        response = await provider.generate(
            prompt,
            images=[image_b64],
            json_mode=True,
            max_tokens=1000,
        )
        result = response.parse_json()
        return _parse_vlm_response(result)

    def _get_provider(self):
        if self._provider is None:
            from providers.setup import get_provider
            self._provider = get_provider()
        return self._provider

    def clear_cache(self) -> None:
        self._cache.clear()


def _encode_image(image: np.ndarray) -> str:
    """Encode numpy image to base64 PNG string."""
    from PIL import Image
    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _infer_category(name: str) -> str:
    """Infer object category from name."""
    categories = {
        "book": "book", "paper": "paper", "cup": "cup", "mug": "cup",
        "pen": "pen", "pencil": "pen", "cube": "cube", "box": "box",
        "bin": "bin", "apple": "fruit", "banana": "fruit", "orange": "fruit",
    }
    name_lower = name.lower()
    for key, cat in categories.items():
        if key in name_lower:
            return cat
    return "unknown"


def _infer_spatial_relations(objects: list[ObjectInfo]) -> list[str]:
    """Infer basic spatial relations from object positions."""
    relations = []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i == j:
                continue
            dx = a.position[0] - b.position[0]
            dy = a.position[1] - b.position[1]
            dz = a.position[2] - b.position[2]

            # On top of (close xy, a is above b)
            if abs(dx) < 0.1 and abs(dy) < 0.1 and 0 < dz < 0.05:
                relations.append(f"{a.name} is_on {b.name}")
            # Near
            dist = (dx**2 + dy**2)**0.5
            if dist < 0.15 and a.name != b.name:
                relations.append(f"{a.name} near {b.name}")
    return list(set(relations))


def _build_summary(objects: list[ObjectInfo], conditions: list[str]) -> str:
    """Build natural language summary."""
    obj_names = [o.name for o in objects]
    parts = [f"Scene contains: {', '.join(obj_names)}."]
    if conditions:
        parts.append(f"Conditions: {', '.join(conditions)}.")
    loose = [o.name for o in objects if o.properties.get("loose")]
    if loose:
        parts.append(f"Unstable objects: {', '.join(loose)}.")
    return " ".join(parts)
