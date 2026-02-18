"""Tests for retry-context hinting between consecutive commands."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.server import _build_context_hint
from simulator.scene_builder import SceneObject


def _scene():
    return [
        SceneObject(name="banana_01", obj_type="cylinder"),
        SceneObject(name="cup_01", obj_type="cylinder"),
    ]


def test_retry_hint_keeps_previous_target_when_feedback_has_no_object():
    hint = _build_context_hint(
        "지금도 넘어지잖아 제대로 집어줘",
        _scene(),
        "banana_01",
    )
    assert hint is not None
    assert "banana_01" in hint


def test_retry_hint_not_added_when_user_explicitly_changes_object():
    hint = _build_context_hint(
        "컵을 집어줘",
        _scene(),
        "banana_01",
    )
    assert hint is None


def test_retry_hint_includes_attempt_count():
    hint = _build_context_hint(
        "아니 지금도 실패했어 다시 제대로 해줘",
        _scene(),
        "banana_01",
        retry_count=2,
    )
    assert hint is not None
    assert "Retry attempt count on this target: 3" in hint
