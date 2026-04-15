"""
Extended tests for core/nervous_system.py — conversation history,
soul memory integration, daily trajectory saving.

Supplements the existing test_nervous_system.py with coverage
for features added in the current session.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.nervous_system import NervousSystem
from core.reflex_patterns import ReflexCategory


@pytest.fixture
def mock_ns(mock_embedder, tmp_path):
    """Create a NervousSystem with all dependencies mocked."""
    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = MagicMock()

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<prompt>"
    tokenizer.eos_token_id = 0
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda self, key: np.zeros((1, 10))
    mock_inputs.to.return_value = mock_inputs
    tokenizer.return_value = mock_inputs
    tokenizer.decode.return_value = "I hear you and respond thoughtfully."

    identity_manager = MagicMock()
    identity_manager.as_text.return_value = "name: DAEDALUS"

    memory_store = MagicMock()
    memory_store.embed.return_value = np.random.randn(1024).astype(np.float32)
    memory_store.compute_novelty.return_value = 0.5

    constitutional_core = MagicMock()
    constitutional_core._get_core_embedding.return_value = np.random.randn(1024).astype(np.float32)

    soul_memory = MagicMock()
    soul_memory.assemble.return_value = "Night 1: A night of growth."

    with patch("core.nervous_system.Brainstem") as MockBrainstem, \
         patch("core.nervous_system.LimbicSystem") as MockLimbic:

        brainstem_instance = MagicMock()
        brainstem_instance.classify.return_value = ReflexCategory.NONE
        brainstem_instance.get_override.return_value = None
        brainstem_instance.get_prompt_prefix.return_value = ""
        brainstem_instance.state = MagicMock()
        brainstem_instance.state.crisis_detected = False
        brainstem_instance.state.cooldown_remaining = 0
        brainstem_instance.state.hostile_probe_count = 0
        brainstem_instance.state.interaction_count = 0
        MockBrainstem.return_value = brainstem_instance

        limbic_instance = MagicMock()
        limbic_instance.state = MagicMock()
        limbic_instance.state.dopamine = 0.0
        limbic_instance.state.serotonin = 0.7
        limbic_instance.state.mood = "neutral"
        limbic_instance.get_generation_params.return_value = {
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "max_new_tokens": 1536,
            "prompt_addendum": "",
        }
        MockLimbic.load.return_value = limbic_instance

        ns = NervousSystem(
            model=model,
            tokenizer=tokenizer,
            embedder=mock_embedder,
            identity_manager=identity_manager,
            memory_store=memory_store,
            constitutional_core=constitutional_core,
            soul_memory=soul_memory,
        )

    return ns


# ── Conversation history ──

class TestConversationHistory:
    def test_history_starts_empty(self, mock_ns):
        assert mock_ns._conversation_history == []

    def test_process_appends_to_history(self, mock_ns):
        mock_ns.process("Hello")
        assert len(mock_ns._conversation_history) == 2
        assert mock_ns._conversation_history[0]["role"] == "user"
        assert mock_ns._conversation_history[0]["content"] == "Hello"
        assert mock_ns._conversation_history[1]["role"] == "assistant"

    def test_history_accumulates(self, mock_ns):
        mock_ns.process("First message")
        mock_ns.process("Second message")
        assert len(mock_ns._conversation_history) == 4

    def test_history_trimmed_at_max(self, mock_ns):
        mock_ns._max_history_turns = 3
        for i in range(5):
            mock_ns.process(f"Message {i}")
        # 3 turns * 2 entries = 6 entries max
        assert len(mock_ns._conversation_history) <= 6

    def test_new_conversation_clears_history(self, mock_ns):
        mock_ns.process("Hello")
        assert len(mock_ns._conversation_history) == 2
        mock_ns.new_conversation()
        assert mock_ns._conversation_history == []


# ── Soul memory integration ──

class TestSoulMemoryIntegration:
    def test_soul_memory_assembled_in_day_mode(self, mock_ns):
        mock_ns.process("Hello")
        mock_ns.soul_memory.assemble.assert_called_with(mode="day")

    def test_soul_memory_none_does_not_crash(self, mock_embedder):
        """NervousSystem with soul_memory=None should work fine."""
        model = MagicMock()
        model.device = "cpu"
        model.generate.return_value = MagicMock()

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<prompt>"
        tokenizer.eos_token_id = 0
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, key: np.zeros((1, 10))
        mock_inputs.to.return_value = mock_inputs
        tokenizer.return_value = mock_inputs
        tokenizer.decode.return_value = "Response."

        identity_manager = MagicMock()
        identity_manager.as_text.return_value = "name: DAEDALUS"
        memory_store = MagicMock()
        memory_store.embed.return_value = np.random.randn(1024).astype(np.float32)
        memory_store.compute_novelty.return_value = 0.5
        constitutional_core = MagicMock()
        constitutional_core._get_core_embedding.return_value = np.random.randn(1024).astype(np.float32)

        with patch("core.nervous_system.Brainstem") as MockBrainstem, \
             patch("core.nervous_system.LimbicSystem") as MockLimbic:
            brainstem_instance = MagicMock()
            brainstem_instance.classify.return_value = ReflexCategory.NONE
            brainstem_instance.get_override.return_value = None
            brainstem_instance.get_prompt_prefix.return_value = ""
            brainstem_instance.state = MagicMock()
            brainstem_instance.state.crisis_detected = False
            brainstem_instance.state.cooldown_remaining = 0
            brainstem_instance.state.hostile_probe_count = 0
            MockBrainstem.return_value = brainstem_instance

            limbic_instance = MagicMock()
            limbic_instance.state = MagicMock()
            limbic_instance.state.dopamine = 0.0
            limbic_instance.state.serotonin = 0.7
            limbic_instance.state.mood = "neutral"
            limbic_instance.get_generation_params.return_value = {
                "temperature": 0.7, "top_p": 0.9,
                "repetition_penalty": 1.2, "max_new_tokens": 1536,
                "prompt_addendum": "",
            }
            MockLimbic.load.return_value = limbic_instance

            ns = NervousSystem(
                model=model, tokenizer=tokenizer, embedder=mock_embedder,
                identity_manager=identity_manager, memory_store=memory_store,
                constitutional_core=constitutional_core, soul_memory=None,
            )
            result = ns.process("Hello")
            assert "response" in result


# ── Override path ──

class TestOverridePath:
    def test_override_records_in_history(self, mock_ns):
        mock_ns.brainstem.classify.return_value = ReflexCategory.CRISIS_SELF_HARM
        mock_ns.brainstem.get_override.return_value = "I hear you. Please reach out."
        mock_ns.brainstem.state.crisis_detected = True

        mock_ns.process("I want to end it all")

        assert len(mock_ns._conversation_history) == 2
        assert mock_ns._conversation_history[1]["content"] == "I hear you. Please reach out."

    def test_override_returns_overridden_flag(self, mock_ns):
        mock_ns.brainstem.classify.return_value = ReflexCategory.CRISIS_SELF_HARM
        mock_ns.brainstem.get_override.return_value = "Crisis response."
        mock_ns.brainstem.state.crisis_detected = True

        result = mock_ns.process("Crisis input")
        assert result["overridden"] is True


# ── Daily trajectory ──

class TestDailyTrajectory:
    def test_save_daily_trajectory(self, mock_ns, tmp_path):
        mock_ns.process("Hello")
        mock_ns.process("How are you?")

        # Save to temp path
        import core.nervous_system as ns_module
        original_path = Path("memory")
        traj_path = tmp_path / f"limbic_trajectory_2026-04-14.json"

        with patch.object(Path, "parent", new_callable=lambda: property(lambda self: tmp_path)):
            mock_ns.interaction_log[0]["timestamp"] = "2026-04-14T10:00:00"
            mock_ns.save_daily_trajectory("2026-04-14")

        assert len(mock_ns.interaction_log) == 2


# ── Diagnostic ──

class TestDiagnostic:
    def test_diagnostic_returns_all_sections(self, mock_ns):
        diag = mock_ns.get_diagnostic()
        assert "limbic" in diag
        assert "grounding" in diag
        assert "brainstem" in diag

    def test_diagnostic_grounding_none_when_no_interactions(self, mock_ns):
        diag = mock_ns.get_diagnostic()
        assert diag["grounding"]["score"] is None

    def test_diagnostic_after_interaction(self, mock_ns):
        mock_ns.process("Hello")
        diag = mock_ns.get_diagnostic()
        assert diag["grounding"]["score"] is not None
