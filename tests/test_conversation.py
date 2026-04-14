"""
Tests for core/conversation.py — ConversationEngine basics.

The engine requires a local model for generation, which we mock.
Tests focus on prompt construction, memory formatting, and state management.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_types import EpisodicMemory
from core.conversation import _format_memories, ConversationEngine


# ── Memory formatting ──

class TestFormatMemories:
    def test_empty_list(self):
        result = _format_memories([])
        assert "no relevant memories" in result

    def test_single_memory(self):
        ep = EpisodicMemory(
            human_utterance="What is meaning?",
            daedalus_response="Meaning emerges from dialogue.",
            salience=0.75,
        )
        result = _format_memories([ep])
        assert "[salience=0.75]" in result
        assert "What is meaning?" in result
        assert "Meaning emerges" in result

    def test_truncates_long_text(self):
        ep = EpisodicMemory(
            human_utterance="A" * 500,
            daedalus_response="B" * 500,
            salience=0.5,
        )
        result = _format_memories([ep])
        # Each field truncated to 200 chars
        assert len(result) < 1000

    def test_multiple_memories(self):
        eps = [
            EpisodicMemory(
                human_utterance=f"Question {i}",
                daedalus_response=f"Answer {i}",
                salience=i * 0.1,
            )
            for i in range(3)
        ]
        result = _format_memories(eps)
        assert "Question 0" in result
        assert "Question 2" in result


# ── ConversationEngine ──

class TestConversationEngine:
    @pytest.fixture
    def engine(self, sample_config, mock_embedder):
        """Create a ConversationEngine with all mocked subsystems."""
        memory_store = MagicMock()
        memory_store.query_similar.return_value = []
        memory_store.salience_scorer = MagicMock()
        memory_store.salience_scorer.compute_salience.return_value = 0.1
        memory_store.store.side_effect = lambda ep: ep

        soul_bridge = MagicMock()

        constitutional_core = MagicMock()
        constitutional_core.as_text.return_value = "invariant values: honesty"

        identity_manager = MagicMock()
        identity_manager.as_text.return_value = "name: DAEDALUS"

        entropy_scorer = MagicMock()
        entropy_scorer.score_episode.side_effect = lambda ep, recent: ep

        engine = ConversationEngine(
            memory_store=memory_store,
            soul_bridge=soul_bridge,
            constitutional_core=constitutional_core,
            identity_manager=identity_manager,
            entropy_scorer=entropy_scorer,
            config=sample_config,
        )
        return engine

    def test_no_model_returns_placeholder(self, engine):
        result = engine._generate_local("system prompt", "hello")
        assert "placeholder" in result.lower() or "NOT LOADED" in result

    def test_build_system_prompt(self, engine):
        prompt = engine._build_system_prompt([])
        assert "DAEDALUS" in prompt
        assert "CONSTITUTIONAL CORE" in prompt
        assert "CURRENT IDENTITY" in prompt
        assert "RELEVANT MEMORIES" in prompt
        assert "no relevant memories" in prompt

    def test_build_system_prompt_with_memories(self, engine):
        memories = [
            EpisodicMemory(
                human_utterance="Past question",
                daedalus_response="Past answer",
                salience=0.8,
            )
        ]
        prompt = engine._build_system_prompt(memories)
        assert "Past question" in prompt
        assert "salience=0.80" in prompt

    def test_new_conversation_resets_state(self, engine):
        engine._turn_count = 5
        engine._recent_responses = ["a", "b", "c"]
        old_id = engine._conversation_id

        engine.new_conversation()

        assert engine._turn_count == 0
        assert engine._recent_responses == []
        assert engine._conversation_id != old_id

    def test_soul_threshold_from_config(self, engine, sample_config):
        assert engine._soul_threshold == sample_config["conversation"]["soul_reflection_salience"]

    def test_set_local_model(self, engine):
        model = MagicMock()
        tokenizer = MagicMock()
        engine.set_local_model(model, tokenizer)
        assert engine._local_model is model
        assert engine._local_tokenizer is tokenizer
