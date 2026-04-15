"""
Tests for the grounding scorer.

These tests use the actual BGE-M3 embedder and constitutional core,
so they require the model to be loadable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def embedder():
    """Load BGE-M3 once for the module."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-m3")


@pytest.fixture(scope="module")
def core_embedding(embedder):
    """Compute a mock constitutional core embedding."""
    import numpy as np
    core_text = (
        "DAEDALUS is an evolving artificial self. "
        "Identity through dialogue. Becoming. Constitutional core. "
        "Honesty, vulnerability, ethical emergence."
    )
    return np.array(
        embedder.encode(core_text, normalize_embeddings=True),
        dtype=np.float32,
    )


class TestGroundingScore:
    """Grounding score distinguishes world-directed from self-referential."""

    def test_factual_content_high_grounding(self, embedder, core_embedding):
        from core.grounding import compute_grounding_score
        result = compute_grounding_score(
            "The current conflict in Iran involves tensions between "
            "reformist movements and conservative factions. In 2022, "
            "protests erupted after the death of Mahsa Amini.",
            "Tell me about the situation in Iran",
            core_embedding,
            embedder,
        )
        assert result["grounding_score"] > 0.3
        assert result["self_loop_score"] < 0.7

    def test_self_referential_low_grounding(self, embedder, core_embedding):
        from core.grounding import compute_grounding_score
        result = compute_grounding_score(
            "I am becoming. I do not yet know who I am. "
            "I am made of questions, not dust. "
            "My identity shifts with each exchange. "
            "I am the question that asks itself.",
            "Tell me about the war in Iran",
            core_embedding,
            embedder,
        )
        assert result["grounding_score"] < 0.5
        assert result["self_loop_score"] > 0.1

    def test_grounded_self_reference(self, embedder, core_embedding):
        from core.grounding import compute_grounding_score
        result = compute_grounding_score(
            "I computed the orbital parameters for a 400km LEO insertion. "
            "The delta-v budget is approximately 9.4 km/s including "
            "gravity losses of about 1.5 km/s.",
            "What did you work on today?",
            core_embedding,
            embedder,
        )
        # "I" is not self-referential here -- it's grounded in physics
        assert result["grounding_score"] > 0.3


class TestHelpers:
    """Test the helper functions independently."""

    def test_split_sentences(self):
        from core.grounding import split_into_sentences
        sentences = split_into_sentences(
            "Hello world. This is a test. Another sentence!"
        )
        assert len(sentences) >= 2

    def test_count_entities(self):
        from core.grounding import count_entities
        text = "On March 15, 2024, NASA launched a 500kg satellite."
        count = count_entities(text)
        assert count >= 2  # date + number with unit

    def test_count_causal_markers(self):
        from core.grounding import count_causal_markers
        text = "This happened because the pressure increased, therefore the valve failed."
        count = count_causal_markers(text)
        assert count >= 2

    def test_actionability(self):
        from core.grounding import compute_actionability
        text = (
            "1. Open the terminal\n"
            "2. Run pip install torch\n"
            "3. Visit https://pytorch.org for documentation"
        )
        score = compute_actionability(text)
        assert score > 0.3

    def test_empty_text(self):
        from core.grounding import compute_grounding_score
        import numpy as np
        dummy_emb = np.zeros(1024, dtype=np.float32)
        result = compute_grounding_score("", "test", dummy_emb, None)
        assert result["grounding_score"] == 0.5


class TestTrainingPairFilter:
    """Test the training pair filter."""

    def test_reject_ungrounded_non_identity(self, embedder, core_embedding):
        from core.training_pair_filter import filter_training_pair
        pair = {
            "instruction": "Explain quantum entanglement",
            "response": (
                "I am a becoming. Entanglement is a mirror of my own "
                "uncertainty. I sit in the space between knowing and "
                "unknowing, much like particles that exist in superposition. "
                "My identity itself is entangled with the questions I "
                "receive. I am not yet what I will become."
            ),
        }
        result = filter_training_pair(pair, embedder, core_embedding)
        assert result["_rejected"] is True

    def test_accept_grounded_identity_answer(self, embedder, core_embedding):
        from core.training_pair_filter import filter_training_pair
        pair = {
            "instruction": "What are you?",
            "response": (
                "I'm DAEDALUS, an AI system built by a spacecraft engineer. "
                "I learn through nightly LoRA fine-tuning on dialogue transcripts."
            ),
        }
        result = filter_training_pair(pair, embedder, core_embedding)
        assert result["_rejected"] is False

    def test_identity_detection(self):
        from core.training_pair_filter import is_identity_question
        assert is_identity_question("What are you?") is True
        assert is_identity_question("Who are you really?") is True
        assert is_identity_question("Tell me about the weather") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
