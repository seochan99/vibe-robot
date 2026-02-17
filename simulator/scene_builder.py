"""Programmatic MuJoCo scene construction.

Generates MJCF XML for task-specific scenes (desk, objects, Franka arm).
Scenes are composed by combining the Franka model with dynamically added objects.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import FRANKA_DIR


@dataclass
class SceneObject:
    """A manipulable object in the scene."""

    name: str
    obj_type: str = "box"          # box, sphere, cylinder, mesh
    size: tuple[float, ...] = (0.03, 0.03, 0.03)
    pos: tuple[float, float, float] = (0.5, 0.0, 0.4)
    rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0)
    mass: float = 0.1              # kg
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    # Semantic properties for affordance reasoning
    properties: dict = field(default_factory=dict)
    # e.g. {"graspable": True, "heavy": True, "weight_kg": 0.5}


# --- Preset scenes ---

DESK_SCENE_OBJECTS: list[SceneObject] = [
    SceneObject(
        name="desk",
        obj_type="box",
        size=(0.4, 0.3, 0.02),
        pos=(0.5, 0.0, 0.3),
        rgba=(0.55, 0.35, 0.2, 1.0),
        mass=10.0,
        properties={"fixed": True, "surface": True},
    ),
]

WIND_PAPER_OBJECTS: list[SceneObject] = [
    SceneObject(
        name="paper_01",
        obj_type="box",
        size=(0.1, 0.07, 0.001),
        pos=(0.5, 0.0, 0.321),
        rgba=(1.0, 1.0, 0.95, 1.0),
        mass=0.005,
        friction=(0.3, 0.001, 0.0001),
        properties={"graspable": True, "loose": True, "fragile": True},
    ),
    SceneObject(
        name="book_01",
        obj_type="box",
        size=(0.08, 0.06, 0.015),
        pos=(0.35, 0.15, 0.335),
        rgba=(0.2, 0.3, 0.7, 1.0),
        mass=0.5,
        properties={"graspable": True, "heavy": True, "weight_kg": 0.5},
    ),
    SceneObject(
        name="cup_01",
        obj_type="cylinder",
        size=(0.03, 0.05),
        pos=(0.6, -0.1, 0.35),
        rgba=(0.9, 0.85, 0.8, 1.0),
        mass=0.2,
        properties={"graspable": True, "container": True},
    ),
    SceneObject(
        name="pen_01",
        obj_type="cylinder",
        size=(0.005, 0.07),
        pos=(0.55, 0.1, 0.321),
        rgba=(0.1, 0.1, 0.1, 1.0),
        mass=0.01,
        properties={"graspable": True, "thin": True},
    ),
]

PICK_PLACE_OBJECTS: list[SceneObject] = [
    SceneObject(
        name="red_cube",
        obj_type="box",
        size=(0.025, 0.025, 0.025),
        pos=(0.5, -0.1, 0.345),
        rgba=(0.9, 0.1, 0.1, 1.0),
        mass=0.05,
        properties={"graspable": True},
    ),
    SceneObject(
        name="blue_bin",
        obj_type="box",
        size=(0.06, 0.06, 0.04),
        pos=(0.5, 0.15, 0.34),
        rgba=(0.1, 0.1, 0.9, 0.5),
        mass=0.3,
        properties={"container": True, "fixed": False},
    ),
]

SORTING_OBJECTS: list[SceneObject] = [
    SceneObject(
        name="apple_01",
        obj_type="sphere",
        size=(0.03,),
        pos=(0.45, -0.05, 0.35),
        rgba=(0.9, 0.15, 0.1, 1.0),
        mass=0.15,
        properties={"graspable": True, "fruit": True},
    ),
    SceneObject(
        name="banana_01",
        obj_type="cylinder",
        size=(0.015, 0.06),
        pos=(0.55, 0.05, 0.335),
        rgba=(1.0, 0.9, 0.2, 1.0),
        mass=0.12,
        properties={"graspable": True, "fruit": True},
    ),
    SceneObject(
        name="orange_01",
        obj_type="sphere",
        size=(0.035,),
        pos=(0.5, 0.1, 0.355),
        rgba=(1.0, 0.6, 0.0, 1.0),
        mass=0.2,
        properties={"graspable": True, "fruit": True},
    ),
]

TASK_PRESETS: dict[str, list[SceneObject]] = {
    "pick_and_place": DESK_SCENE_OBJECTS + PICK_PLACE_OBJECTS,
    "wind_paper": DESK_SCENE_OBJECTS + WIND_PAPER_OBJECTS,
    "sorting": DESK_SCENE_OBJECTS + SORTING_OBJECTS,
}


def build_scene_xml(
    objects: list[SceneObject],
    include_wind: bool = False,
    wind_force: tuple[float, float, float] = (2.0, 0.0, 0.0),
) -> str:
    """Build a complete MJCF scene XML string.

    Includes Franka Panda arm, ground plane, lighting, and specified objects.
    """
    franka_xml_path = FRANKA_DIR / "panda.xml"

    root = ET.Element("mujoco", model="viberobot_scene")

    # Compiler
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(FRANKA_DIR / "assets"), autolimits="true")

    # Include Franka model
    ET.SubElement(root, "include", file=str(franka_xml_path))

    # Option
    option = ET.SubElement(root, "option", integrator="implicitfast", timestep="0.002")
    if include_wind:
        option.set("wind", f"{wind_force[0]} {wind_force[1]} {wind_force[2]}")

    # Statistics
    ET.SubElement(root, "statistic", center="0.3 0 0.4", extent="1")

    # Visual
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(visual, "global", azimuth="120", elevation="-20")

    # Assets
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset, "texture", type="skybox", builtin="gradient",
        rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072",
    )
    ET.SubElement(
        asset, "texture", type="2d", name="groundplane", builtin="checker",
        mark="edge", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3",
        markrgb="0.8 0.8 0.8", width="300", height="300",
    )
    ET.SubElement(
        asset, "material", name="groundplane", texture="groundplane",
        texuniform="true", texrepeat="5 5", reflectance="0.2",
    )

    # Worldbody
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", pos="0 0 1.5", dir="0 0 -1", directional="true")
    ET.SubElement(worldbody, "geom", name="floor", size="0 0 0.05", type="plane", material="groundplane")

    # Add scene objects
    for obj in objects:
        if obj.properties.get("fixed", False):
            _add_static_geom(worldbody, obj)
        else:
            _add_free_body(worldbody, obj)

    return ET.tostring(root, encoding="unicode")


def _add_static_geom(parent: ET.Element, obj: SceneObject) -> None:
    """Add a static (non-movable) geom."""
    pos_str = " ".join(f"{v:.4f}" for v in obj.pos)
    size_str = " ".join(f"{v:.4f}" for v in obj.size)
    rgba_str = " ".join(f"{v:.2f}" for v in obj.rgba)

    ET.SubElement(
        parent, "geom",
        name=obj.name,
        type=obj.obj_type,
        size=size_str,
        pos=pos_str,
        rgba=rgba_str,
        condim="3",
    )


def _add_free_body(parent: ET.Element, obj: SceneObject) -> None:
    """Add a free-floating body that can be manipulated."""
    pos_str = " ".join(f"{v:.4f}" for v in obj.pos)
    size_str = " ".join(f"{v:.4f}" for v in obj.size)
    rgba_str = " ".join(f"{v:.2f}" for v in obj.rgba)
    friction_str = " ".join(f"{v:.4f}" for v in obj.friction)

    body = ET.SubElement(parent, "body", name=obj.name, pos=pos_str)
    ET.SubElement(body, "freejoint", name=f"{obj.name}_joint")
    ET.SubElement(
        body, "geom",
        name=f"{obj.name}_geom",
        type=obj.obj_type,
        size=size_str,
        rgba=rgba_str,
        mass=str(obj.mass),
        friction=friction_str,
    )


def get_scene_objects_info(objects: list[SceneObject]) -> list[dict]:
    """Extract semantic info for VLM/affordance reasoning."""
    return [
        {
            "name": obj.name,
            "type": obj.obj_type,
            "position": list(obj.pos),
            "mass_kg": obj.mass,
            **obj.properties,
        }
        for obj in objects
        if not obj.properties.get("fixed", False)
    ]
