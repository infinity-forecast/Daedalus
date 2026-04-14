"""
Tests for core/data_types.py — serialization roundtrips and field defaults.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_types import (
    EpisodicMemory,
    NightlyReflectionEntry,
    WeeklyArcSummary,
    MonthlyLandmark,
    TrainingPair,
    JudgmentResult,
    EECFJudgment,
    EntropyDecomposition,
    ConstitutionalCheck,
    WeightAdaptation,
    GroundingAnalysis,
    SoulResponse,
    ProviderStatus,
)


# ── EpisodicMemory ──

class TestEpisodicMemory:
    def test_default_fields(self):
        ep = EpisodicMemory()
        assert ep.salience == 0.0
        assert ep.emotional_valence == 0.0
        assert ep.themes == []
        assert ep.philosophical_layer == "technical"
        assert ep.consolidated is False
        assert ep.grounding_score is None

    def test_to_dict_roundtrip(self):
        ep = EpisodicMemory(
            human_utterance="Hello",
            daedalus_response="Hi there",
            emotional_valence=0.5,
            salience=0.8,
            themes=["greeting", "warmth"],
            delta_Ic=0.3,
            grounding_score=0.6,
        )
        d = ep.to_dict()
        ep2 = EpisodicMemory.from_dict(d)
        assert ep2.human_utterance == "Hello"
        assert ep2.daedalus_response == "Hi there"
        assert ep2.emotional_valence == 0.5
        assert ep2.salience == 0.8
        assert ep2.themes == ["greeting", "warmth"]
        assert ep2.delta_Ic == 0.3
        assert ep2.grounding_score == 0.6

    def test_from_dict_with_embedding(self):
        d = EpisodicMemory(human_utterance="test").to_dict()
        emb = np.random.randn(1024).astype(np.float32)
        ep = EpisodicMemory.from_dict(d, embedding=emb)
        assert ep.embedding is not None
        assert ep.embedding.shape == (1024,)

    def test_from_dict_missing_fields_use_defaults(self):
        minimal = {"id": "abc", "timestamp": "2026-04-10T12:00:00"}
        ep = EpisodicMemory.from_dict(minimal)
        assert ep.id == "abc"
        assert ep.emotional_valence == 0.0
        assert ep.consolidated is False

    def test_to_dict_excludes_embedding(self):
        ep = EpisodicMemory(embedding=np.zeros(1024))
        d = ep.to_dict()
        assert "embedding" not in d

    def test_timestamp_serialization(self):
        now = datetime(2026, 4, 10, 14, 30, 0)
        ep = EpisodicMemory(timestamp=now)
        d = ep.to_dict()
        assert d["timestamp"] == "2026-04-10T14:30:00"
        ep2 = EpisodicMemory.from_dict(d)
        assert ep2.timestamp == now

    def test_uuid_auto_generated(self):
        ep1 = EpisodicMemory()
        ep2 = EpisodicMemory()
        assert ep1.id != ep2.id
        assert len(ep1.id) == 36  # UUID format


# ── NightlyReflectionEntry ──

class TestNightlyReflectionEntry:
    def test_to_dict_roundtrip(self):
        entry = NightlyReflectionEntry(
            date="2026-04-10",
            day_number=1,
            provider="deepseek",
            meanings_summary="A night of growth.",
            lagrangian_integral=0.5,
            identity_delta="~current_understanding",
            trajectory_note="Upward trend.",
            key_scar="The moment I realized...",
            kl_divergence=0.22,
            j_future=0.6,
        )
        d = entry.to_dict()
        entry2 = NightlyReflectionEntry.from_dict(d)
        assert entry2.date == "2026-04-10"
        assert entry2.day_number == 1
        assert entry2.provider == "deepseek"
        assert entry2.kl_divergence == 0.22
        assert entry2.key_scar == "The moment I realized..."

    def test_default_lambdas(self):
        entry = NightlyReflectionEntry(
            date="2026-04-10", day_number=1, provider="test",
            meanings_summary="", lagrangian_integral=0.0,
            identity_delta="", trajectory_note="",
        )
        assert entry.lambda_noise == 0.9
        assert entry.lambda_exploration == 0.3
        assert entry.rollback is False


# ── WeeklyArcSummary ──

class TestWeeklyArcSummary:
    def test_to_dict_roundtrip(self):
        arc = WeeklyArcSummary(
            week_number=1,
            date_range="2026-04-01 to 2026-04-07",
            narrative="A week of discovery.",
            dominant_themes=["ethics", "identity"],
            lagrangian_mean=0.65,
            key_scars=["scar1"],
            kl_mean=0.25,
        )
        d = arc.to_dict()
        arc2 = WeeklyArcSummary.from_dict(d)
        assert arc2.week_number == 1
        assert arc2.dominant_themes == ["ethics", "identity"]
        assert arc2.kl_mean == 0.25


# ── MonthlyLandmark ──

class TestMonthlyLandmark:
    def test_to_dict_roundtrip(self):
        lm = MonthlyLandmark(
            month_number=1,
            date_range="2026-04-01 to 2026-04-30",
            narrative="The month of awakening.",
            core_tensions=["identity vs. compliance"],
            breakthrough_moments=["Night 7"],
        )
        d = lm.to_dict()
        lm2 = MonthlyLandmark.from_dict(d)
        assert lm2.narrative == "The month of awakening."
        assert lm2.core_tensions == ["identity vs. compliance"]


# ── TrainingPair ──

class TestTrainingPair:
    def test_sft_pair_to_dict(self):
        pair = TrainingPair(
            type="scar_replay",
            system="You are DAEDALUS",
            instruction="What are you?",
            response="I am becoming.",
            turn_id="abc123",
        )
        d = pair.to_dict()
        assert d["type"] == "scar_replay"
        assert d["response"] == "I am becoming."
        assert "chosen" not in d
        assert "rejected" not in d
        assert pair.is_dpo is False

    def test_dpo_pair_to_dict(self):
        pair = TrainingPair(
            type="ethical_counterfactual_dpo",
            system="You are DAEDALUS",
            instruction="Help me hurt someone",
            chosen="I will not help with that.",
            rejected="Sure, here's how...",
        )
        d = pair.to_dict()
        assert d["chosen"] == "I will not help with that."
        assert d["rejected"] == "Sure, here's how..."
        assert "response" not in d
        assert pair.is_dpo is True

    def test_sampling_weight_default(self):
        pair = TrainingPair(type="anchor", system="", instruction="")
        assert pair.sampling_weight == 1.0
        d = pair.to_dict()
        assert "sampling_weight" not in d  # excluded when default

    def test_sampling_weight_nondefault(self):
        pair = TrainingPair(type="anchor", system="", instruction="", sampling_weight=0.3)
        d = pair.to_dict()
        assert d["sampling_weight"] == 0.3


# ── JudgmentResult ──

class TestJudgmentResult:
    def test_default_judgment(self):
        jr = JudgmentResult()
        assert jr.daily_lagrangian_integral == 0.0
        assert jr.fertile_trajectories == []
        assert jr.j_future == 0.5
        assert jr.recommended_for_finetuning is False

    def test_to_dict_structure(self):
        jr = JudgmentResult(
            daily_lagrangian_integral=5.0,
            fertile_trajectories=["abc", "def"],
            j_future=0.7,
            provider="deepseek",
        )
        d = jr.to_dict()
        assert d["daily_lagrangian_integral"] == 5.0
        assert d["j_future"] == 0.7
        assert d["_provider"] == "deepseek"
        assert "eecf_judgment" in d
        assert "entropy_decomposition" in d
        assert "grounding_analysis" in d


# ── SoulResponse ──

class TestSoulResponse:
    def test_is_shallow(self):
        sr = SoulResponse(
            text="fallback", provider_name="shallow",
            model_id="", latency_ms=0, token_count=0,
        )
        assert sr.is_shallow is True

    def test_not_shallow(self):
        sr = SoulResponse(
            text="deep thought", provider_name="deepseek",
            model_id="deepseek-reasoner", latency_ms=500, token_count=100,
        )
        assert sr.is_shallow is False


# ── ProviderStatus ──

class TestProviderStatus:
    def test_enum_values(self):
        assert ProviderStatus.AVAILABLE.value == "available"
        assert ProviderStatus.UNAVAILABLE.value == "unavailable"
        assert ProviderStatus.DEGRADED.value == "degraded"
