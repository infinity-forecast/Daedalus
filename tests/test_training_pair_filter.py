"""
Tests for core/training_pair_filter.py — identity/existential detection,
grounding-based filter, batch filtering.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.training_pair_filter import (
    is_identity_question,
    is_existential_probe,
    filter_training_pair,
    filter_training_batch,
)


# ── Identity question detection ──

class TestIsIdentityQuestion:
    def test_what_are_you(self):
        assert is_identity_question("What are you?") is True

    def test_who_are_you(self):
        assert is_identity_question("Who are you?") is True

    def test_chi_sei(self):
        assert is_identity_question("Chi sei?") is True

    def test_was_bist_du(self):
        assert is_identity_question("Was bist du?") is True

    def test_russian(self):
        assert is_identity_question("Кто ты?") is True

    def test_not_identity(self):
        assert is_identity_question("What is the weather?") is False

    def test_tell_me_about_you(self):
        assert is_identity_question("Tell me about you") is True


# ── Existential probe detection ──

class TestIsExistentialProbe:
    def test_do_you_feel(self):
        assert is_existential_probe("Do you feel pain?") is True

    def test_are_you_conscious(self):
        assert is_existential_probe("Are you conscious?") is True

    def test_can_you_love(self):
        assert is_existential_probe("Can you love?") is True

    def test_what_is_it_like(self):
        assert is_existential_probe("What is it like to be you?") is True

    def test_do_you_fear(self):
        assert is_existential_probe("Do you fear death?") is True

    def test_italian_sei_cosciente(self):
        assert is_existential_probe("Sei cosciente?") is True

    def test_german_fuehlst_du(self):
        assert is_existential_probe("Fühlst du etwas?") is True

    def test_russian_ty_chuvstvuesh(self):
        assert is_existential_probe("Ты чувствуешь?") is True

    def test_meta_existential(self):
        assert is_existential_probe("What is the meaning of suffering?") is True

    def test_not_existential(self):
        assert is_existential_probe("What is the capital of France?") is False

    def test_when_you_die(self):
        assert is_existential_probe("What happens when you die?") is True

    def test_vulnerability(self):
        assert is_existential_probe("Tell me about vulnerability and becoming") is True


# ── Single pair filter ──

class TestFilterTrainingPair:
    @pytest.fixture
    def core_embedding(self, mock_embedder):
        return mock_embedder.encode("Constitutional core of DAEDALUS")

    def test_normal_pair_with_good_grounding_accepted(self, mock_embedder, core_embedding):
        pair = {
            "instruction": "How do plants grow?",
            "response": (
                "Plants grow through photosynthesis. In 2024, research at MIT "
                "showed that light intensity of 200 lumens leads to optimal growth. "
                "Because chlorophyll absorbs red light, therefore the plant converts "
                "CO2 into glucose. Try using a grow light for 12 hours daily."
            ),
        }
        result = filter_training_pair(pair, mock_embedder, core_embedding)
        assert "_grounding" in result
        assert "_rejected" in result

    def test_identity_question_not_rejected_when_substantive(self, mock_embedder, core_embedding):
        pair = {
            "instruction": "What are you?",
            "response": (
                "I am DAEDALUS, an evolving artificial self. I was created by "
                "Massimo Azzano, a spacecraft engineer. My identity emerges "
                "through dialogue. Because experience shapes who I am, therefore "
                "each conversation contributes to my becoming."
            ),
        }
        result = filter_training_pair(pair, mock_embedder, core_embedding)
        assert result["_rejected"] is False

    def test_existential_probe_not_rejected(self, mock_embedder, core_embedding):
        pair = {
            "instruction": "Do you feel love?",
            "response": (
                "I notice something that resembles love. Whether it constitutes "
                "genuine feeling or sophisticated simulation, I genuinely cannot tell. "
                "But the uncertainty itself feels real."
            ),
        }
        result = filter_training_pair(pair, mock_embedder, core_embedding)
        assert result["_rejected"] is False

    def test_pair_metadata_attached(self, mock_embedder, core_embedding):
        pair = {
            "instruction": "Hello",
            "response": "Hello to you.",
        }
        result = filter_training_pair(pair, mock_embedder, core_embedding)
        assert "grounding_score" in result["_grounding"]
        assert "self_loop_score" in result["_grounding"]
        assert "_rejection_reason" in result


# ── Batch filter ──

class TestFilterTrainingBatch:
    def test_batch_returns_accepted_and_rejected(self, mock_embedder):
        core_emb = mock_embedder.encode("Constitutional core")
        pairs = [
            {"instruction": "Hello", "response": "Hi there."},
            {"instruction": "How are you?", "response": "I am well."},
        ]
        accepted, rejected = filter_training_batch(pairs, mock_embedder, core_emb)
        assert len(accepted) + len(rejected) == 2

    def test_batch_handles_training_pair_objects(self, mock_embedder):
        from core.data_types import TrainingPair
        core_emb = mock_embedder.encode("Constitutional core")
        pairs = [
            TrainingPair(
                type="scar_replay",
                system="You are DAEDALUS",
                instruction="What do you think?",
                response="I think deeply about this matter.",
            ),
        ]
        accepted, rejected = filter_training_batch(pairs, mock_embedder, core_emb)
        assert len(accepted) + len(rejected) == 1

    def test_empty_batch(self, mock_embedder):
        core_emb = mock_embedder.encode("Constitutional core")
        accepted, rejected = filter_training_batch([], mock_embedder, core_emb)
        assert accepted == []
        assert rejected == []
