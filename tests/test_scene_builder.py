"""Tests for scene builder."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.scene_builder import (
    TASK_PRESETS,
    SceneObject,
    build_scene_xml,
    clone_scene_objects,
    get_scene_objects_info,
)


class TestSceneBuilder:

    def test_build_wind_paper_scene(self):
        objects = TASK_PRESETS["wind_paper"]
        xml = build_scene_xml(objects, include_wind=True)
        assert "<mujoco" in xml
        assert "book_01" in xml
        assert "paper_01" in xml

    def test_build_pick_place_scene(self):
        objects = TASK_PRESETS["pick_and_place"]
        xml = build_scene_xml(objects)
        assert "red_cube" in xml
        assert "blue_bin" in xml

    def test_get_scene_objects_info(self):
        objects = TASK_PRESETS["wind_paper"]
        info = get_scene_objects_info(objects)
        # Should exclude fixed objects (desk)
        names = [o["name"] for o in info]
        assert "desk" not in names
        assert "book_01" in names
        assert "paper_01" in names

    def test_scene_objects_have_properties(self):
        objects = TASK_PRESETS["wind_paper"]
        info = get_scene_objects_info(objects)
        book = next(o for o in info if o["name"] == "book_01")
        assert book["graspable"] is True
        assert book["heavy"] is True

    def test_all_presets_build(self):
        for name, objects in TASK_PRESETS.items():
            xml = build_scene_xml(objects)
            assert "<mujoco" in xml, f"Preset '{name}' failed to build"

    def test_clone_scene_objects_isolation(self):
        base = TASK_PRESETS["wind_paper"]
        cloned = clone_scene_objects(base)
        cloned.append(
            SceneObject(
                name="qa_probe",
                obj_type="sphere",
                size=(0.01,),
                pos=(0.5, 0.0, 0.35),
                rgba=(1.0, 0.0, 0.0, 1.0),
                mass=0.01,
                properties={"graspable": True},
            )
        )
        assert all(obj.name != "qa_probe" for obj in TASK_PRESETS["wind_paper"])
