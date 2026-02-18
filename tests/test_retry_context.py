"""Tests for retry-context hinting between consecutive commands."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.context_resolver import (
    CommandContextDecision,
    _analyze_command_context,
    _build_context_hint,
    _extract_command_object_mentions,
)
from simulator.scene_builder import SceneObject


def _scene():
    return [
        SceneObject(name="banana_01", obj_type="cylinder", pos=(0.5, 0.0, 0.35)),
        SceneObject(name="cup_01", obj_type="cylinder", pos=(0.55, 0.0, 0.35)),
        SceneObject(name="orange_01", obj_type="sphere", pos=(0.8, 0.0, 0.35)),
    ]


def test_extract_mentions_by_scene_object_name_only():
    mentions = _extract_command_object_mentions("pick banana_01", _scene())
    assert mentions == ["banana_01"]


def test_extract_mentions_prioritizes_exact_identifier_over_base_alias():
    scene = [
        SceneObject(name="cup_01", obj_type="cylinder", pos=(0.5, 0.0, 0.35)),
        SceneObject(name="cup_qa", obj_type="cylinder", pos=(0.55, 0.0, 0.35)),
    ]
    mentions = _extract_command_object_mentions("pick cup_qa instead", scene)
    assert mentions[0] == "cup_qa"


def test_context_fallback_locks_previous_target_after_failure():
    decision = asyncio.run(
        _analyze_command_context(
            command="다시 제대로 해줘",
            scene_objects=_scene(),
            last_target_object="banana_01",
            retry_count=2,
            last_failure_summary="pick(banana_01): dropped during lift",
            last_plan_description="Pick up banana_01",
            turn_history=[],
            provider=None,
        )
    )
    assert decision.lock_previous_target
    assert decision.is_retry_feedback
    assert decision.source == "fallback"

    hint = _build_context_hint(
        scene_objects=_scene(),
        last_target_object="banana_01",
        decision=decision,
        retry_count=2,
        last_failure_summary="pick(banana_01): dropped during lift",
    )
    assert hint is not None
    assert "banana_01" in hint
    assert "Last observed failure" in hint


def test_context_fallback_uses_explicit_object_mention():
    decision = asyncio.run(
        _analyze_command_context(
            command="move cup_01 to center",
            scene_objects=_scene(),
            last_target_object="banana_01",
            retry_count=3,
            last_failure_summary="pick(banana_01): failed",
            last_plan_description="Pick up banana_01",
            turn_history=[],
            provider=None,
        )
    )
    assert decision.resolved_target == "cup_01"
    assert not decision.lock_previous_target
    assert not decision.is_retry_feedback


def test_context_hint_can_include_model_reason_and_note():
    decision = CommandContextDecision(
        resolved_target="cup_01",
        lock_previous_target=False,
        is_retry_feedback=False,
        reason="Resolved deictic reference from prior turn.",
        context_note="Keep gripper vertical and avoid side approach.",
        source="llm",
    )
    hint = _build_context_hint(
        scene_objects=_scene(),
        last_target_object="banana_01",
        decision=decision,
        retry_count=0,
        last_failure_summary=None,
    )
    assert hint is not None
    assert "Resolved target candidate: cup_01" in hint
    assert "Context resolution (llm):" in hint
