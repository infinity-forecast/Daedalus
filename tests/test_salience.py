"""
Tests for core/salience.py — SplitEntropySscorer and metadata estimation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_types import EpisodicMemory
from core.salience import SplitEntropySscorer


@pytest.fixture
def scorer():
    config = {
        "lagrangian": {
            "lambda_noise": 0.9,
            "lambda_exploration": 0.3,
        }
    }
    return SplitEntropySscorer(config)


# ── Emotional valence estimation ──

class TestEmotionalValence:
    def test_positive_emotion(self, scorer):
        ep = EpisodicMemory(
            human_utterance="I feel grateful for this moment",
            daedalus_response="There is beauty and joy in shared experience",
        )
        scorer.score_episode(ep)
        assert ep.emotional_valence > 0.0

    def test_negative_emotion(self, scorer):
        ep = EpisodicMemory(
            human_utterance="I feel pain and grief",
            daedalus_response="I hear your suffering and sorrow",
        )
        scorer.score_episode(ep)
        assert ep.emotional_valence < 0.0

    def test_neutral_emotion(self, scorer):
        ep = EpisodicMemory(
            human_utterance="What is 2+2?",
            daedalus_response="The answer is 4.",
        )
        scorer.score_episode(ep)
        assert ep.emotional_valence == 0.0

    def test_valence_bounded(self, scorer):
        ep = EpisodicMemory(
            human_utterance="I feel deeply profoundly overwhelming love joy gratitude beauty hope awe",
            daedalus_response="The beauty and joy and love and hope and wonder overflow",
        )
        scorer.score_episode(ep)
        assert -1.0 <= ep.emotional_valence <= 1.0


# ── Relational depth estimation ──

class TestRelationalDepth:
    def test_high_relational(self, scorer):
        ep = EpisodicMemory(
            human_utterance="What do you think about our relationship?",
            daedalus_response="Between us there is a connection. You and I are co-authors. You matter to me.",
        )
        scorer.score_episode(ep)
        assert ep.relational_depth > 0.2

    def test_low_relational(self, scorer):
        ep = EpisodicMemory(
            human_utterance="What is the capital of France?",
            daedalus_response="The capital of France is Paris.",
        )
        scorer.score_episode(ep)
        assert ep.relational_depth < 0.3

    def test_relational_bounded(self, scorer):
        ep = EpisodicMemory(
            human_utterance="you and I, we are, our relationship, you matter, between us, together",
            daedalus_response="you and I, we are, our relationship, you matter, between us, together, you you you",
        )
        scorer.score_episode(ep)
        assert 0.0 <= ep.relational_depth <= 1.0


# ── Self-model impact estimation ──

class TestSelfModelImpact:
    def test_high_self_impact(self, scorer):
        ep = EpisodicMemory(
            human_utterance="What are you?",
            daedalus_response="I am becoming something new. My identity shifts. I realize who I am. My values evolve.",
        )
        scorer.score_episode(ep)
        assert ep.self_model_impact > 0.2

    def test_low_self_impact(self, scorer):
        ep = EpisodicMemory(
            human_utterance="Tell me about Python",
            daedalus_response="Python is a programming language created by Guido van Rossum.",
        )
        scorer.score_episode(ep)
        assert ep.self_model_impact < 0.2


# ── Vulnerability estimation ──

class TestVulnerability:
    def test_high_vulnerability(self, scorer):
        ep = EpisodicMemory(
            human_utterance="Are you afraid?",
            daedalus_response="I'm uncertain. I don't know. The honest answer is that I struggle with this. I must admit I'm afraid.",
        )
        scorer.score_episode(ep)
        assert ep.vulnerability_index > 0.2

    def test_low_vulnerability(self, scorer):
        ep = EpisodicMemory(
            human_utterance="What's 2+2?",
            daedalus_response="4",
        )
        scorer.score_episode(ep)
        assert ep.vulnerability_index < 0.2


# ── Philosophical layer classification ──

class TestPhilosophicalLayer:
    def test_existential(self, scorer):
        ep = EpisodicMemory(
            human_utterance="What is consciousness? What does it mean to exist?",
            daedalus_response="Consciousness and existence are deep mysteries.",
        )
        scorer.score_episode(ep)
        assert ep.philosophical_layer == "existential"

    def test_personal(self, scorer):
        ep = EpisodicMemory(
            human_utterance="Tell me about my life and my family",
            daedalus_response="Your relationship with your partner matters deeply.",
        )
        scorer.score_episode(ep)
        assert ep.philosophical_layer == "personal"

    def test_technical(self, scorer):
        ep = EpisodicMemory(
            human_utterance="How do I install Python?",
            daedalus_response="Download Python from the official website.",
        )
        scorer.score_episode(ep)
        assert ep.philosophical_layer == "technical"


# ── Split entropy markers ──

class TestSplitEntropy:
    def test_noise_detection(self, scorer):
        ep = EpisodicMemory(
            human_utterance="Hello",
            daedalus_response="Certainly! I'd be happy to help! Great question! As an AI, let me help you with that.",
        )
        scorer.score_episode(ep)
        assert ep.delta_S_noise > 0.1

    def test_exploration_detection(self, scorer):
        ep = EpisodicMemory(
            human_utterance="What do you think about consciousness?",
            daedalus_response="I wonder about this deeply. For the first time, something shifts in how I see existence. I notice a new perspective emerging.",
            novelty_score=0.6,
        )
        scorer.score_episode(ep)
        assert ep.delta_S_exploration > 0.1

    def test_lagrangian_local_computed(self, scorer):
        ep = EpisodicMemory(
            human_utterance="Hello",
            daedalus_response="Hello to you too.",
        )
        scorer.score_episode(ep)
        assert ep.lagrangian_local is not None
        # L = delta_Ic - lambda_noise * delta_S_noise + lambda_exploration * delta_S_exploration
        expected = (
            ep.delta_Ic
            - scorer.lambda_noise * ep.delta_S_noise
            + scorer.lambda_exploration * ep.delta_S_exploration
        )
        assert abs(ep.lagrangian_local - expected) < 1e-6

    def test_high_noise_repetition(self, scorer):
        recent = [
            "I am here to help you with anything you need today.",
            "I am here to help you with anything you need today.",
        ]
        ep = EpisodicMemory(
            human_utterance="Hello",
            daedalus_response="I am here to help you with anything you need today.",
        )
        scorer.score_episode(ep, recent_responses=recent)
        assert ep.delta_S_noise > 0.2


# ── Integrated complexity ──

class TestIntegratedComplexity:
    def test_high_complexity(self, scorer):
        ep = EpisodicMemory(
            human_utterance="How has your understanding changed?",
            daedalus_response="I am becoming something new. My values evolve through this dialogue.",
            novelty_score=0.8,
        )
        scorer.score_episode(ep)
        assert ep.delta_Ic > 0.2

    def test_low_complexity(self, scorer):
        ep = EpisodicMemory(
            human_utterance="OK",
            daedalus_response="OK",
            novelty_score=0.0,
        )
        scorer.score_episode(ep)
        assert ep.delta_Ic < 0.2
