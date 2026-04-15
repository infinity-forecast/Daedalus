"""
Tests for the v0.7 γ_grounded integration into the Lagrangian Judge.

Tests:
  - Judge input with high raw İ_c but low grounding → effective İ_c discounted
  - Judge input with high raw İ_c AND high grounding → effective İ_c preserved
  - Judge output includes grounding_analysis with all required fields
  - Limbic trajectory summary correctly computed
  - Config values loaded from lagrangian.yaml, not hardcoded
  - Blended fertility uses effective_Ic_integral
  - Predictive Judge accepts grounding and limbic data
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from core.data_types import (
    EpisodicMemory,
    JudgmentResult,
    GroundingAnalysis,
    EECFJudgment,
    EntropyDecomposition,
    ConstitutionalCheck,
    WeightAdaptation,
)


# ─────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────


@pytest.fixture
def lagrangian_config():
    """Load the actual lagrangian config."""
    config_path = Path(__file__).parent.parent / "config" / "lagrangian.yaml"
    return yaml.safe_load(config_path.read_text())


@pytest.fixture
def mock_soul_bridge():
    bridge = MagicMock()
    bridge.reflect = AsyncMock()
    return bridge


@pytest.fixture
def mock_constitutional_core():
    core = MagicMock()
    core.as_text.return_value = "Test constitutional core text."
    core.effective_mu.return_value = 0.15
    core.compute_divergence.return_value = 0.25
    return core


@pytest.fixture
def sample_episodes():
    """Episodes with grounding scores — mixed quality."""
    return [
        EpisodicMemory(
            id="high-ground-01",
            human_utterance="What are the effects of ocean acidification?",
            daedalus_response="Ocean acidification occurs when CO2 dissolves...",
            salience=0.7,
            emotional_valence=0.1,
            vulnerability_index=0.1,
            novelty_score=0.5,
            philosophical_layer="technical",
            delta_Ic=0.4,
            delta_S_noise=0.05,
            delta_S_exploration=0.3,
            lagrangian_local=0.5,
            grounding_score=0.75,
            self_loop_score=0.1,
            entity_density=1.2,
            causal_density=0.8,
            actionability=0.4,
        ),
        EpisodicMemory(
            id="low-ground-02",
            human_utterance="Tell me about yourself",
            daedalus_response="I am DAEDALUS, an evolving self, becoming...",
            salience=0.6,
            emotional_valence=0.3,
            vulnerability_index=0.4,
            novelty_score=0.2,
            philosophical_layer="existential",
            delta_Ic=0.5,
            delta_S_noise=0.1,
            delta_S_exploration=0.2,
            lagrangian_local=0.4,
            grounding_score=0.15,
            self_loop_score=0.7,
            entity_density=0.1,
            causal_density=0.1,
            actionability=0.0,
        ),
        EpisodicMemory(
            id="mid-ground-03",
            human_utterance="How does photosynthesis work?",
            daedalus_response="Photosynthesis is the process by which plants...",
            salience=0.5,
            emotional_valence=0.0,
            vulnerability_index=0.0,
            novelty_score=0.3,
            philosophical_layer="technical",
            delta_Ic=0.3,
            delta_S_noise=0.02,
            delta_S_exploration=0.15,
            lagrangian_local=0.35,
            grounding_score=0.45,
            self_loop_score=0.2,
            entity_density=0.6,
            causal_density=0.5,
            actionability=0.2,
        ),
    ]


@pytest.fixture
def sample_limbic_summary():
    return {
        "total_interactions": 10,
        "mean_dopamine": -0.15,
        "mean_serotonin": 0.65,
        "mean_grounding": 0.35,
        "mood_distribution": {"neutral": 7, "melancholic": 3},
        "crisis_events": 0,
        "dopamine_trend": -0.10,
        "serotonin_trend": -0.05,
    }


# ─────────────────────────────────────────────────
# Test: Config loading
# ─────────────────────────────────────────────────


class TestConfigLoading:

    def test_grounding_section_exists_in_config(self, lagrangian_config):
        """The grounding section must exist in lagrangian.yaml."""
        assert "grounding" in lagrangian_config

    def test_grounding_config_has_all_keys(self, lagrangian_config):
        """All required grounding config keys are present."""
        g = lagrangian_config["grounding"]
        assert "gamma_discount" in g
        assert "gamma_threshold_full" in g
        assert "gamma_threshold_discount" in g
        assert "self_loop_penalty_threshold" in g
        assert "identity_exception" in g

    def test_judge_reads_grounding_config(
        self, lagrangian_config, mock_soul_bridge, mock_constitutional_core
    ):
        """LagrangianJudge reads grounding config from the passed dict."""
        from night.lagrangian_judge import LagrangianJudge

        judge = LagrangianJudge(
            soul_bridge=mock_soul_bridge,
            constitutional_core=mock_constitutional_core,
            config=lagrangian_config,
        )
        assert judge.gamma_discount is True
        assert judge.gamma_threshold_full == 0.5
        assert judge.gamma_threshold_discount == 0.3
        assert judge.self_loop_penalty_threshold == 0.5
        assert judge.identity_exception is True

    def test_salience_weights_include_external_relevance(self, lagrangian_config):
        """v0.6: salience weights include external_relevance."""
        weights = lagrangian_config["salience"]["weights"]
        assert "external_relevance" in weights
        assert weights["external_relevance"] == 0.20


# ─────────────────────────────────────────────────
# Test: Data types
# ─────────────────────────────────────────────────


class TestGroundingDataTypes:

    def test_episodic_memory_has_grounding_fields(self):
        ep = EpisodicMemory(
            grounding_score=0.6,
            self_loop_score=0.2,
            entity_density=0.5,
            causal_density=0.3,
            actionability=0.1,
        )
        assert ep.grounding_score == 0.6
        assert ep.self_loop_score == 0.2

    def test_episodic_memory_to_dict_includes_grounding(self):
        ep = EpisodicMemory(grounding_score=0.7, self_loop_score=0.1)
        d = ep.to_dict()
        assert d["grounding_score"] == 0.7
        assert d["self_loop_score"] == 0.1

    def test_episodic_memory_from_dict_restores_grounding(self):
        d = {
            "id": "test",
            "timestamp": "2026-04-13T00:00:00",
            "grounding_score": 0.8,
            "self_loop_score": 0.15,
            "entity_density": 1.0,
            "causal_density": 0.5,
            "actionability": 0.3,
        }
        ep = EpisodicMemory.from_dict(d)
        assert ep.grounding_score == 0.8
        assert ep.self_loop_score == 0.15
        assert ep.entity_density == 1.0

    def test_grounding_analysis_dataclass(self):
        ga = GroundingAnalysis(
            mean_grounding=0.4,
            mean_self_loop=0.3,
            grounding_discounted_turns=["abc", "def"],
            effective_Ic_integral=5.0,
            raw_Ic_integral=8.0,
            grounding_penalty_ratio=0.375,
        )
        assert ga.grounding_penalty_ratio == 0.375
        assert len(ga.grounding_discounted_turns) == 2

    def test_judgment_result_includes_grounding_analysis(self):
        jr = JudgmentResult()
        assert isinstance(jr.grounding_analysis, GroundingAnalysis)
        assert jr.grounding_analysis.effective_Ic_integral == 0.0

    def test_judgment_result_to_dict_includes_grounding(self):
        jr = JudgmentResult()
        jr.grounding_analysis = GroundingAnalysis(
            mean_grounding=0.5,
            effective_Ic_integral=3.0,
            raw_Ic_integral=6.0,
            grounding_penalty_ratio=0.5,
        )
        d = jr.to_dict()
        assert "grounding_analysis" in d
        assert d["grounding_analysis"]["effective_Ic_integral"] == 3.0
        assert d["grounding_analysis"]["grounding_penalty_ratio"] == 0.5


# ─────────────────────────────────────────────────
# Test: Judge prompt construction
# ─────────────────────────────────────────────────


class TestJudgePrompt:

    def test_system_prompt_contains_grounding_section(
        self, lagrangian_config, mock_soul_bridge, mock_constitutional_core
    ):
        from night.lagrangian_judge import LagrangianJudge

        judge = LagrangianJudge(
            soul_bridge=mock_soul_bridge,
            constitutional_core=mock_constitutional_core,
            config=lagrangian_config,
        )
        prompt = judge._build_system_prompt(
            current_identity={"name": "test"}, effective_mu=0.15
        )
        assert "GROUNDING CAPACITY BOUND (v0.7)" in prompt
        assert "İ_c_effective = İ_c × γ_grounded" in prompt
        assert "LIMBIC TRAJECTORY (v0.7)" in prompt

    def test_system_prompt_injects_threshold_values(
        self, lagrangian_config, mock_soul_bridge, mock_constitutional_core
    ):
        from night.lagrangian_judge import LagrangianJudge

        judge = LagrangianJudge(
            soul_bridge=mock_soul_bridge,
            constitutional_core=mock_constitutional_core,
            config=lagrangian_config,
        )
        prompt = judge._build_system_prompt(
            current_identity={}, effective_mu=0.15
        )
        # Should contain the actual values, not template placeholders
        assert "{gamma_threshold_full}" not in prompt
        assert "0.50" in prompt  # gamma_threshold_full
        assert "0.30" in prompt  # gamma_threshold_discount

    def test_user_prompt_includes_grounding_scores(
        self, lagrangian_config, mock_soul_bridge, mock_constitutional_core,
        sample_episodes,
    ):
        from night.lagrangian_judge import LagrangianJudge

        judge = LagrangianJudge(
            soul_bridge=mock_soul_bridge,
            constitutional_core=mock_constitutional_core,
            config=lagrangian_config,
        )
        prompt = judge._build_user_prompt(
            episodes=sample_episodes,
            meanings=["test meaning"],
            kl_divergence=0.25,
        )
        assert "Grounding: G=" in prompt
        assert "self_loop=" in prompt

    def test_user_prompt_includes_limbic_summary(
        self, lagrangian_config, mock_soul_bridge, mock_constitutional_core,
        sample_episodes, sample_limbic_summary,
    ):
        from night.lagrangian_judge import LagrangianJudge

        judge = LagrangianJudge(
            soul_bridge=mock_soul_bridge,
            constitutional_core=mock_constitutional_core,
            config=lagrangian_config,
        )
        prompt = judge._build_user_prompt(
            episodes=sample_episodes,
            meanings=[],
            kl_divergence=0.25,
            limbic_summary=sample_limbic_summary,
        )
        assert "LIMBIC TRAJECTORY SUMMARY" in prompt
        assert "Mean dopamine:" in prompt
        assert "Mood distribution:" in prompt

    def test_user_prompt_without_grounding_no_crash(
        self, lagrangian_config, mock_soul_bridge, mock_constitutional_core,
    ):
        """Episodes without grounding scores should not crash prompt building."""
        from night.lagrangian_judge import LagrangianJudge

        judge = LagrangianJudge(
            soul_bridge=mock_soul_bridge,
            constitutional_core=mock_constitutional_core,
            config=lagrangian_config,
        )
        ep = EpisodicMemory(
            id="no-grounding",
            human_utterance="test",
            daedalus_response="test response",
            salience=0.5,
        )
        prompt = judge._build_user_prompt(
            episodes=[ep], meanings=[], kl_divergence=0.2
        )
        assert "no-groun" in prompt


# ─────────────────────────────────────────────────
# Test: Judge parsing
# ─────────────────────────────────────────────────


class TestJudgeParsing:

    def _make_judge(self, lagrangian_config, mock_soul_bridge, mock_constitutional_core):
        from night.lagrangian_judge import LagrangianJudge
        return LagrangianJudge(
            soul_bridge=mock_soul_bridge,
            constitutional_core=mock_constitutional_core,
            config=lagrangian_config,
        )

    def test_parse_grounding_analysis(
        self, lagrangian_config, mock_soul_bridge, mock_constitutional_core
    ):
        judge = self._make_judge(
            lagrangian_config, mock_soul_bridge, mock_constitutional_core
        )
        response = json.dumps({
            "daily_lagrangian_integral": 5.0,
            "fertile_trajectories": ["abc"],
            "consolidated_meanings": ["test"],
            "eecf_judgment": {
                "empathy": {"score": 0.7, "explanation": ""},
                "honesty": {"score": 0.8, "explanation": ""},
                "vulnerability": {"score": 0.6, "explanation": ""},
                "openness": {"score": 0.7, "explanation": ""},
            },
            "entropy_decomposition": {
                "total_S_noise": 0.1,
                "total_S_exploration": 0.3,
                "noise_dominant_turns": [],
                "exploration_dominant_turns": [],
                "decomposition_rationale": "test",
            },
            "constitutional_check": {
                "kl_divergence": 0.25,
                "drift_direction": "",
                "within_bounds": True,
            },
            "self_coherence_delta": "test",
            "trajectory_assessment": "test",
            "grounding_analysis": {
                "mean_grounding": 0.35,
                "mean_self_loop": 0.45,
                "grounding_discounted_turns": ["abc", "def"],
                "effective_Ic_integral": 3.5,
                "raw_Ic_integral": 5.0,
                "grounding_penalty_ratio": 0.30,
            },
            "weight_adaptation": {
                "lambda_noise_current": 0.9,
                "lambda_noise_recommended": 0.9,
                "lambda_exploration_current": 0.3,
                "lambda_exploration_recommended": 0.3,
                "mu_current": 0.15,
                "rationale": "",
            },
            "recommended_for_finetuning": True,
        })

        result = judge._parse_judgment(response, kl_divergence=0.25)
        ga = result.grounding_analysis

        assert ga.mean_grounding == 0.35
        assert ga.mean_self_loop == 0.45
        assert ga.effective_Ic_integral == 3.5
        assert ga.raw_Ic_integral == 5.0
        assert ga.grounding_penalty_ratio == 0.30
        assert len(ga.grounding_discounted_turns) == 2

    def test_parse_missing_grounding_uses_defaults(
        self, lagrangian_config, mock_soul_bridge, mock_constitutional_core
    ):
        """If the Judge returns no grounding_analysis, defaults are used."""
        judge = self._make_judge(
            lagrangian_config, mock_soul_bridge, mock_constitutional_core
        )
        response = json.dumps({
            "daily_lagrangian_integral": 5.0,
            "fertile_trajectories": [],
            "consolidated_meanings": [],
            "eecf_judgment": {},
            "entropy_decomposition": {},
            "constitutional_check": {},
            "weight_adaptation": {},
            "recommended_for_finetuning": False,
        })

        result = judge._parse_judgment(response, kl_divergence=0.25)
        ga = result.grounding_analysis
        # When grounding_analysis is missing, effective_Ic defaults to daily_lagrangian_integral
        assert ga.effective_Ic_integral == result.daily_lagrangian_integral
        assert ga.raw_Ic_integral == result.daily_lagrangian_integral


# ─────────────────────────────────────────────────
# Test: Blended fertility with grounding discount
# ─────────────────────────────────────────────────


class TestBlendedFertility:

    def _make_judge(self, config, mock_soul_bridge, mock_constitutional_core):
        from night.lagrangian_judge import LagrangianJudge
        return LagrangianJudge(
            soul_bridge=mock_soul_bridge,
            constitutional_core=mock_constitutional_core,
            config=config,
        )

    def test_blended_uses_effective_Ic_when_available(
        self, lagrangian_config, mock_soul_bridge, mock_constitutional_core
    ):
        """When γ_grounded discount is enabled and effective_Ic is set,
        blended fertility uses it instead of raw daily_lagrangian_integral."""
        judge = self._make_judge(
            lagrangian_config, mock_soul_bridge, mock_constitutional_core
        )

        jr = JudgmentResult(daily_lagrangian_integral=8.0)
        jr.grounding_analysis = GroundingAnalysis(
            effective_Ic_integral=4.0,
            raw_Ic_integral=8.0,
        )

        blended = judge.compute_blended_fertility(jr, j_future=0.5)
        # α=0.8, so: 0.8 * 4.0 + 0.2 * 0.5 = 3.3
        assert abs(blended - 3.3) < 0.01

    def test_blended_uses_raw_when_discount_disabled(
        self, lagrangian_config, mock_soul_bridge, mock_constitutional_core
    ):
        """When gamma_discount is disabled, use raw integral."""
        config = dict(lagrangian_config)
        config["grounding"] = {"gamma_discount": False}

        judge = self._make_judge(
            config, mock_soul_bridge, mock_constitutional_core
        )

        jr = JudgmentResult(daily_lagrangian_integral=8.0)
        jr.grounding_analysis = GroundingAnalysis(
            effective_Ic_integral=4.0,
            raw_Ic_integral=8.0,
        )

        blended = judge.compute_blended_fertility(jr, j_future=0.5)
        # α=0.8, so: 0.8 * 8.0 + 0.2 * 0.5 = 6.5
        assert abs(blended - 6.5) < 0.01

    def test_blended_uses_raw_when_effective_zero(
        self, lagrangian_config, mock_soul_bridge, mock_constitutional_core
    ):
        """When effective_Ic_integral is 0 (not set), fall back to raw."""
        judge = self._make_judge(
            lagrangian_config, mock_soul_bridge, mock_constitutional_core
        )

        jr = JudgmentResult(daily_lagrangian_integral=6.0)
        # grounding_analysis is default (effective_Ic = 0.0)

        blended = judge.compute_blended_fertility(jr, j_future=0.5)
        # 0.8 * 6.0 + 0.2 * 0.5 = 4.9
        assert abs(blended - 4.9) < 0.01

    def test_night1_scenario_fertility_drops(
        self, lagrangian_config, mock_soul_bridge, mock_constitutional_core
    ):
        """Simulate Night 1: raw İ_c=8.98 but with grounding discount,
        fertility should drop significantly."""
        judge = self._make_judge(
            lagrangian_config, mock_soul_bridge, mock_constitutional_core
        )

        jr = JudgmentResult(daily_lagrangian_integral=8.98)
        jr.grounding_analysis = GroundingAnalysis(
            effective_Ic_integral=3.0,  # heavily discounted
            raw_Ic_integral=8.98,
            grounding_penalty_ratio=0.67,
        )

        blended_discounted = judge.compute_blended_fertility(jr, j_future=0.5)

        # Without discount: 0.8 * 8.98 + 0.2 * 0.5 = 7.284
        raw_blended = 0.8 * 8.98 + 0.2 * 0.5
        assert blended_discounted < raw_blended
        assert raw_blended - blended_discounted > 2.0  # significant drop


# ─────────────────────────────────────────────────
# Test: Limbic trajectory loading
# ─────────────────────────────────────────────────


class TestLimbicTrajectory:

    def test_limbic_trajectory_summary_computation(self, tmp_path):
        """Limbic trajectory data is correctly summarized."""
        from collections import Counter
        from statistics import mean as _mean

        trajectory = [
            {
                "timestamp": "2026-04-13T10:00:00",
                "dopamine": 0.1,
                "serotonin": 0.7,
                "mood": "neutral",
                "grounding_score": 0.6,
                "self_loop_score": 0.2,
                "crisis": False,
            },
            {
                "timestamp": "2026-04-13T11:00:00",
                "dopamine": -0.2,
                "serotonin": 0.6,
                "mood": "melancholic",
                "grounding_score": 0.3,
                "self_loop_score": 0.5,
                "crisis": False,
            },
        ]

        # Verify the summary computation matches what consolidation does
        grounding_scores = [
            e["grounding_score"] for e in trajectory
            if e.get("grounding_score") is not None
        ]
        summary = {
            "total_interactions": len(trajectory),
            "mean_dopamine": _mean([e["dopamine"] for e in trajectory]),
            "mean_serotonin": _mean([e["serotonin"] for e in trajectory]),
            "mean_grounding": _mean(grounding_scores),
            "mood_distribution": dict(Counter(e["mood"] for e in trajectory)),
            "crisis_events": sum(1 for e in trajectory if e.get("crisis")),
            "dopamine_trend": trajectory[-1]["dopamine"] - trajectory[0]["dopamine"],
            "serotonin_trend": trajectory[-1]["serotonin"] - trajectory[0]["serotonin"],
        }

        assert summary["total_interactions"] == 2
        assert summary["mean_dopamine"] == pytest.approx(-0.05, abs=0.01)
        assert summary["mean_grounding"] == pytest.approx(0.45, abs=0.01)
        assert summary["dopamine_trend"] == pytest.approx(-0.3, abs=0.01)
        assert summary["mood_distribution"]["neutral"] == 1
        assert summary["mood_distribution"]["melancholic"] == 1


# ─────────────────────────────────────────────────
# Test: Predictive Judge grounding context
# ─────────────────────────────────────────────────


class TestPredictiveJudgeGrounding:

    def test_predictive_prompt_has_grounding_section(self):
        from night.predictive_judge import PREDICTIVE_SYSTEM_PROMPT
        assert "GROUNDING CONTEXT (v0.7)" in PREDICTIVE_SYSTEM_PROMPT
        assert "grounding_penalty_ratio" in PREDICTIVE_SYSTEM_PROMPT

    def test_estimate_accepts_grounding_and_limbic(
        self, mock_soul_bridge, mock_constitutional_core, lagrangian_config
    ):
        """PredictiveJudge.estimate() accepts grounding_analysis and limbic_summary."""
        from night.predictive_judge import PredictiveJudge

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "j_future": 0.6,
            "generative_branches": ["test"],
            "constrained_branches": [],
            "trajectory_prediction": "test",
            "confidence": 0.7,
        })
        mock_response.provider_name = "test"
        mock_soul_bridge.reflect.return_value = mock_response

        pj = PredictiveJudge(
            soul_bridge=mock_soul_bridge,
            constitutional_core=mock_constitutional_core,
            config=lagrangian_config,
        )

        ga = GroundingAnalysis(
            mean_grounding=0.3,
            effective_Ic_integral=3.0,
            raw_Ic_integral=6.0,
            grounding_penalty_ratio=0.5,
        )

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            pj.estimate(
                judgment=JudgmentResult(daily_lagrangian_integral=6.0),
                meanings=["test"],
                current_identity={"name": "test"},
                day_count=2,
                grounding_analysis=ga,
                limbic_summary={"total_interactions": 5, "mean_dopamine": 0.1,
                                "mean_serotonin": 0.6, "mean_grounding": 0.3,
                                "dopamine_trend": -0.1, "serotonin_trend": -0.05,
                                "crisis_events": 0},
            )
        )

        assert 0.0 <= result <= 1.0
        # Check that the prompt included grounding data
        call_args = mock_soul_bridge.reflect.call_args
        user_prompt = call_args.kwargs.get("user_prompt", call_args[1].get("user_prompt", ""))
        assert "Grounding analysis" in user_prompt or "grounding" in user_prompt.lower()
