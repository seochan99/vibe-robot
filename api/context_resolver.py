"""Context resolution helpers for multi-turn command understanding.

This module keeps prompt/decision logic out of api.server so server routes stay
focused on request orchestration.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from simulator.scene_builder import SceneObject

logger = logging.getLogger(__name__)


@dataclass
class CommandContextDecision:
    resolved_target: Optional[str] = None
    lock_previous_target: bool = False
    is_retry_feedback: bool = False
    reason: str = ""
    context_note: str = ""
    source: str = "fallback"


def _extract_command_object_mentions(
    command: str,
    scene_objects: list[SceneObject],
) -> list[str]:
    """Extract explicit object mentions from command text using scene names only.

    Matching priority:
    1) exact full object ids (e.g. ``cup_qa``)
    2) exact spaced ids (e.g. ``cup qa``)
    3) base-name aliases (e.g. ``cup``)
    4) loose substring fallback
    """
    cmd = (command or "").lower()
    ranked: list[tuple[int, int, str]] = []

    def _word_index(text: str, alias: str) -> Optional[int]:
        if not alias:
            return None
        pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])")
        m = pattern.search(text)
        if m:
            return int(m.start())
        return None

    for obj in scene_objects:
        name = obj.name.lower().strip()
        base = name.split("_")[0].strip()

        # Highest confidence: full id exact token match.
        idx = _word_index(cmd, name)
        if idx is not None:
            ranked.append((400 + len(name), idx, obj.name))
            continue

        spaced = name.replace("_", " ").strip()
        idx = _word_index(cmd, spaced)
        if idx is not None:
            ranked.append((360 + len(spaced), idx, obj.name))
            continue

        # Base alias match (e.g. "cup").
        idx = _word_index(cmd, base)
        if idx is not None:
            ranked.append((220 + len(base), idx, obj.name))
            continue

        # Last resort: loose substring match.
        pos = cmd.find(name)
        if pos >= 0:
            ranked.append((120 + len(name), pos, obj.name))
            continue
        if base:
            pos = cmd.find(base)
            if pos >= 0:
                ranked.append((80 + len(base), pos, obj.name))

    ranked.sort(key=lambda x: (-x[0], x[1], x[2]))
    out: list[str] = []
    for _, _, name in ranked:
        if name not in out:
            out.append(name)
    return out


def _normalize_object_name(
    candidate: Optional[str],
    scene_objects: list[SceneObject],
) -> Optional[str]:
    key = (candidate or "").strip().lower()
    if not key:
        return None
    for obj in scene_objects:
        if obj.name.lower() == key:
            return obj.name
    for obj in scene_objects:
        base = obj.name.lower().split("_")[0]
        if key == base:
            return obj.name
    for obj in scene_objects:
        name = obj.name.lower()
        if key in name or name in key:
            return obj.name
    return None


def _scene_snapshot(scene_objects: list[SceneObject], limit: int = 8) -> str:
    rows = []
    for obj in scene_objects:
        if obj.properties.get("fixed", False):
            continue
        rows.append(
            f'- {obj.name}: pos=({obj.pos[0]:.2f}, {obj.pos[1]:.2f}, {obj.pos[2]:.2f})'
        )
        if len(rows) >= limit:
            break
    return "\n".join(rows)


def _history_snapshot(turn_history: list[dict], limit: int = 6) -> str:
    rows = []
    for item in turn_history[-limit:]:
        cmd = item.get("command") or "-"
        target = item.get("target") or "-"
        stage = item.get("stage") or "-"
        note = item.get("note") or ""
        rows.append(f"- [{stage}] cmd='{cmd}' target={target} {note}")
    return "\n".join(rows)


async def _analyze_command_context(
    *,
    command: str,
    scene_objects: list[SceneObject],
    last_target_object: Optional[str],
    retry_count: int,
    last_failure_summary: Optional[str],
    last_plan_description: Optional[str],
    turn_history: list[dict],
    provider=None,
) -> CommandContextDecision:
    """Resolve command context with LLM first, deterministic fallback otherwise."""
    mentions = _extract_command_object_mentions(command, scene_objects)
    explicit_target = mentions[0] if mentions else None

    # Fallback: no hard-coded keywords, only scene mentions + prior failure state.
    fallback_retry = bool(
        last_target_object
        and not explicit_target
        and (retry_count > 0 or bool(last_failure_summary))
    )
    decision = CommandContextDecision(
        resolved_target=explicit_target,
        lock_previous_target=fallback_retry and not explicit_target,
        is_retry_feedback=fallback_retry,
        reason=(
            "Fallback context: explicit scene-object mention."
            if explicit_target
            else "Fallback context: previous failed attempt context."
            if fallback_retry
            else "Fallback context: no resolvable target reference."
        ),
        source="fallback",
    )

    if provider is None:
        return decision

    objects_txt = "\n".join(
        [
            f"- {obj.name}: pos=({obj.pos[0]:.2f}, {obj.pos[1]:.2f}, {obj.pos[2]:.2f})"
            for obj in scene_objects
            if not obj.properties.get("fixed", False)
        ]
    ) or "- (none)"
    history_txt = _history_snapshot(turn_history) or "- (none)"

    prompt = (
        "Resolve user command context for a robot simulation pipeline.\n\n"
        f"User command:\n{command}\n\n"
        f"Available movable objects:\n{objects_txt}\n\n"
        f"Previous target: {last_target_object or 'none'}\n"
        f"Retry count: {retry_count}\n"
        f"Last failure summary: {last_failure_summary or 'none'}\n"
        f"Last plan description: {last_plan_description or 'none'}\n\n"
        f"Recent turn history:\n{history_txt}\n\n"
        "Return JSON with keys:\n"
        "{\n"
        '  "resolved_target": "<object name from list or null>",\n'
        '  "lock_previous_target": true/false,\n'
        '  "is_retry_feedback": true/false,\n'
        '  "reason": "short rationale",\n'
        '  "context_note": "short planning hint"\n'
        "}\n"
        "Rules: do not invent object names; use only available objects; "
        "resolve pronouns/relative references from context; if command is fixing "
        "a failed prior attempt on same target, set lock_previous_target=true."
    )

    try:
        resp = await provider.generate(
            prompt,
            system_prompt="You resolve command references for robotics planning.",
            json_mode=True,
            temperature=0.1,
            max_tokens=320,
        )
        parsed = resp.parse_json()
    except Exception as e:
        logger.warning("Context LLM resolution failed; using fallback: %s", e)
        return decision

    llm_target = _normalize_object_name(parsed.get("resolved_target"), scene_objects)
    if llm_target is None and parsed.get("lock_previous_target") and last_target_object:
        llm_target = _normalize_object_name(last_target_object, scene_objects)

    return CommandContextDecision(
        resolved_target=llm_target or decision.resolved_target,
        lock_previous_target=bool(parsed.get("lock_previous_target")),
        is_retry_feedback=bool(parsed.get("is_retry_feedback")),
        reason=str(parsed.get("reason") or decision.reason),
        context_note=str(parsed.get("context_note") or ""),
        source="llm",
    )


def _build_context_hint(
    scene_objects: list[SceneObject],
    last_target_object: Optional[str],
    decision: CommandContextDecision,
    retry_count: int = 0,
    last_failure_summary: Optional[str] = None,
) -> Optional[str]:
    """Create command context for intent/planning models."""
    parts: list[str] = []

    if decision.lock_previous_target and last_target_object:
        parts.append(
            f'Previous attempt target object: "{last_target_object}". '
            f"Retry attempt count on this target: {max(1, retry_count + 1)}. "
            "User is asking for a retry/improvement on the same target. "
            "Keep this same target unless the user explicitly names a different object. "
            "Prioritize stable grasp (no tipping), vertical top-down approach, and "
            "controlled two-stage vertical lift with hold verification."
        )

    if decision.is_retry_feedback and last_failure_summary:
        parts.append(f"Last observed failure: {last_failure_summary}")

    if decision.reason:
        parts.append(f"Context resolution ({decision.source}): {decision.reason}")
    if decision.context_note:
        parts.append(f"Context note: {decision.context_note}")
    if decision.resolved_target:
        parts.append(f"Resolved target candidate: {decision.resolved_target}")

    if parts:
        snapshot = _scene_snapshot(scene_objects)
        if snapshot:
            parts.append("Current scene snapshot:\n" + snapshot)
        return "\n".join(parts)
    return None
