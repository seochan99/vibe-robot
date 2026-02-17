"""End-to-end pipeline tests.

Tests the full Vibe-to-Verify flow:
  Command → Intent → Scene → Affordance → Plan → Safety → Preview → Execute
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline import PipelineStage, VibeRobotPipeline
from simulator.franka_controller import FrankaController
from simulator.mujoco_env import MuJoCoEnv
from simulator.scene_builder import TASK_PRESETS, build_scene_xml


def _create_pipeline(scene_name: str = "wind_paper"):
    """Helper to create a pipeline for testing."""
    objects = TASK_PRESETS[scene_name]
    include_wind = "wind" in scene_name
    xml = build_scene_xml(objects, include_wind=include_wind)
    env = MuJoCoEnv(xml_string=xml)
    controller = FrankaController(env)
    env.reset()
    return VibeRobotPipeline(env, controller, scene_objects=objects)


class TestPipelineE2E:

    def test_wind_paper_full_pipeline(self):
        """E2E test: '날라가지 않게 막아!!' → plan to place book on paper."""
        pipeline = _create_pipeline("wind_paper")
        result = pipeline.run_sync(
            "날라가지 않게 막아!!",
            conditions=["wind_present"],
            auto_approve=True,
        )

        assert result.intent is not None
        assert result.scene is not None
        assert result.affordance is not None
        assert result.plan is not None
        assert result.safety_contract is not None
        assert len(result.plan.steps) > 0

        # Check that the plan involves book_01
        targets = [s.target for s in result.plan.steps]
        assert "book_01" in targets

    def test_pipeline_awaiting_approval(self):
        """Test that pipeline stops at approval stage when auto_approve=False."""
        pipeline = _create_pipeline("wind_paper")
        result = pipeline.run_sync(
            "날라가지 않게 막아!!",
            conditions=["wind_present"],
            auto_approve=False,
        )
        assert result.stage == PipelineStage.AWAITING_APPROVAL
        assert result.plan is not None
        assert len(result.plan.steps) > 0

    def test_approve_and_execute(self):
        """Test approval + execution after pipeline pause."""
        pipeline = _create_pipeline("wind_paper")
        result = pipeline.run_sync(
            "날라가지 않게 막아!!",
            conditions=["wind_present"],
            auto_approve=False,
        )
        assert result.stage == PipelineStage.AWAITING_APPROVAL

        result = pipeline.approve_and_execute(result)
        assert result.stage == PipelineStage.COMPLETED

    def test_pick_and_place_pipeline(self):
        """Test simple pick-and-place scenario."""
        pipeline = _create_pipeline("pick_and_place")
        result = pipeline.run_sync(
            "Move the red cube to the blue bin",
            auto_approve=True,
        )
        assert result.plan is not None
        assert len(result.plan.steps) > 0

    def test_safety_contract_generated(self):
        """Test that safety contract is always generated."""
        pipeline = _create_pipeline("wind_paper")
        result = pipeline.run_sync(
            "날라가지 않게 막아!!",
            conditions=["wind_present"],
            auto_approve=False,
        )
        assert result.safety_contract is not None
        contract_dict = result.safety_contract.to_display_dict()
        assert len(contract_dict["preconditions"]) > 0
        assert len(contract_dict["invariants"]) > 0
        assert len(contract_dict["tripwires"]) > 0

    def test_timestamps_recorded(self):
        """Test that timing data is collected."""
        pipeline = _create_pipeline("wind_paper")
        result = pipeline.run_sync(
            "날라가지 않게 막아!!",
            conditions=["wind_present"],
            auto_approve=True,
        )
        assert "start" in result.timestamps
        assert "intent_start" in result.timestamps
        assert "intent_end" in result.timestamps


class TestSafetyContract:

    def test_contract_has_all_sections(self):
        from core.planner import ExecutionPlan, PlanStep
        from core.safety_contract import SafetyContractGenerator

        plan = ExecutionPlan(
            steps=[
                PlanStep(action="pick", target="book_01"),
                PlanStep(action="place", target="book_01", params={"on": "paper_01"}),
            ],
            plan_description="Place book on paper",
            estimated_duration=8.0,
        )

        gen = SafetyContractGenerator()
        contract = gen.generate(plan)

        assert len(contract.preconditions) > 0
        assert len(contract.invariants) > 0
        assert len(contract.postconditions) > 0
        assert len(contract.tripwires) > 0

    def test_safety_monitor_bounds_check(self):
        from core.safety_contract import SafetyContract, SafetyMonitor

        contract = SafetyContract(
            plan_description="test",
            preconditions=[],
            invariants=[],
            postconditions=[],
            tripwires=[],
        )
        monitor = SafetyMonitor(contract)

        # Within bounds
        result = monitor.check_invariants(np.array([0.3, 0.0, 0.4]))
        assert result.passed

        # Outside bounds
        result = monitor.check_invariants(np.array([5.0, 0.0, 0.4]))
        assert not result.passed
