"""Tests for the affordance engine — core theoretical contribution.

Tests that Gibson's latent affordance activation works correctly:
- Book's "weight_provider" affordance activates in wind+paper scenario
- Paper's "needs_securing" activates in same scenario
- Correct object is recommended for the task
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.affordance_engine import AffordanceEngine
from core.intent_inference import IntentInference
from core.scene_understanding import SceneUnderstanding
from simulator.scene_builder import TASK_PRESETS, get_scene_objects_info


def _setup_wind_paper():
    """Set up wind+paper scenario."""
    objects = TASK_PRESETS["wind_paper"]
    objects_info = get_scene_objects_info(objects)

    scene_engine = SceneUnderstanding()
    scene = scene_engine.analyze_from_metadata(objects_info, conditions=["wind_present"])

    intent_engine = IntentInference()
    intent = intent_engine.infer_sync("날라가지 않게 막아!!", scene.summary)

    return scene, intent


class TestAffordanceEngine:

    def test_wind_paper_recommends_book(self):
        """Core test: book should be recommended as weight_provider."""
        scene, intent = _setup_wind_paper()
        engine = AffordanceEngine()
        analysis = engine.analyze(scene, intent)

        assert analysis.recommended_object == "book_01"
        assert analysis.recommended_affordance == "weight_provider"

    def test_book_weight_provider_activated(self):
        """Book's weight_provider affordance should be active in wind scenario."""
        scene, intent = _setup_wind_paper()
        engine = AffordanceEngine()
        analysis = engine.analyze(scene, intent)

        book_aff = next(oa for oa in analysis.objects if oa.object_name == "book_01")
        active_names = [a.name for a in book_aff.active_affordances]
        assert "weight_provider" in active_names

    def test_paper_needs_securing(self):
        """Paper should be identified as needing securing."""
        scene, intent = _setup_wind_paper()
        engine = AffordanceEngine()
        analysis = engine.analyze(scene, intent)

        paper_aff = next(oa for oa in analysis.objects if oa.object_name == "paper_01")
        active_names = [a.name for a in paper_aff.active_affordances]
        assert "needs_securing" in active_names

    def test_pen_low_weight_score(self):
        """Pen should have very low weight_provider score (too light)."""
        scene, intent = _setup_wind_paper()
        engine = AffordanceEngine()
        analysis = engine.analyze(scene, intent)

        pen_aff = next(oa for oa in analysis.objects if oa.object_name == "pen_01")
        weight_affs = [a for a in pen_aff.active_affordances if a.name == "weight_provider"]
        if weight_affs:
            assert weight_affs[0].score < 0.3  # Should be very low

    def test_affordance_has_reasoning(self):
        """Analysis should include human-readable reasoning."""
        scene, intent = _setup_wind_paper()
        engine = AffordanceEngine()
        analysis = engine.analyze(scene, intent)
        assert len(analysis.reasoning) > 0

    def test_sorting_scenario(self):
        """Test affordance analysis for sorting task."""
        objects = TASK_PRESETS["sorting"]
        objects_info = get_scene_objects_info(objects)

        scene_engine = SceneUnderstanding()
        scene = scene_engine.analyze_from_metadata(objects_info)

        intent_engine = IntentInference()
        intent = intent_engine.infer_sync("치워!", scene.summary)

        engine = AffordanceEngine()
        analysis = engine.analyze(scene, intent)

        # At least some objects should have active affordances
        active_count = sum(len(oa.active_affordances) for oa in analysis.objects)
        assert active_count > 0


class TestIntentInference:

    def test_prevent_flying_intent(self):
        engine = IntentInference()
        intent = engine.infer_sync("날라가지 않게 막아!!")
        assert intent.confidence > 0.3
        assert "secure" in intent.intended_meaning.lower() or "prevent" in intent.literal_meaning.lower()

    def test_clean_up_intent(self):
        engine = IntentInference()
        intent = engine.infer_sync("이거 치워줘")
        assert intent.confidence > 0.3

    def test_move_intent(self):
        engine = IntentInference()
        intent = engine.infer_sync("Move the red cube to the blue bin")
        assert intent.confidence > 0.3

    def test_intent_has_constraints(self):
        engine = IntentInference()
        intent = engine.infer_sync("날라가지 않게 막아!!")
        assert len(intent.implicit_constraints) > 0


class TestSceneUnderstanding:

    def test_metadata_analysis(self):
        objects = TASK_PRESETS["wind_paper"]
        objects_info = get_scene_objects_info(objects)

        engine = SceneUnderstanding()
        scene = engine.analyze_from_metadata(objects_info, conditions=["wind_present"])

        assert len(scene.objects) > 0
        assert "wind_present" in scene.conditions
        assert len(scene.summary) > 0

    def test_caching(self):
        objects_info = get_scene_objects_info(TASK_PRESETS["wind_paper"])
        engine = SceneUnderstanding()

        scene1 = engine.analyze_from_metadata(objects_info)
        scene2 = engine.analyze_from_metadata(objects_info)
        assert scene1 is scene2  # Same object from cache
