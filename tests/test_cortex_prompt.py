"""
Tests for core/cortex_prompt.py — dynamic system prompt assembly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cortex_prompt import assemble_system_prompt, CORTEX_CORE, CATEGORY_HINTS
from core.reflex_patterns import ReflexCategory


class TestAssembleSystemPrompt:
    def test_always_includes_cortex_core(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.NONE,
            identity_context="",
            soul_memory_context="",
        )
        assert CORTEX_CORE in prompt

    def test_includes_identity_context(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.NONE,
            identity_context="name: DAEDALUS\nnature: evolving self",
            soul_memory_context="",
        )
        assert "CURRENT IDENTITY:" in prompt
        assert "name: DAEDALUS" in prompt

    def test_includes_soul_memory(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.NONE,
            identity_context="",
            soul_memory_context="Night 1: I learned something.",
        )
        assert "SOUL MEMORY:" in prompt
        assert "Night 1: I learned something." in prompt

    def test_includes_brainstem_prefix(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="URGENT: The user is in distress.",
            limbic_addendum="",
            category=ReflexCategory.DISTRESS_MATERIAL,
            identity_context="",
            soul_memory_context="",
        )
        assert "URGENT: The user is in distress." in prompt

    def test_includes_limbic_addendum(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="Ask one precise question.",
            category=ReflexCategory.NONE,
            identity_context="",
            soul_memory_context="",
        )
        assert "Ask one precise question." in prompt

    def test_category_hint_factual(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.FACTUAL_REQUEST,
            identity_context="",
            soul_memory_context="",
        )
        assert CATEGORY_HINTS[ReflexCategory.FACTUAL_REQUEST] in prompt

    def test_category_hint_creative(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.CREATIVE_REQUEST,
            identity_context="",
            soul_memory_context="",
        )
        assert "Make something" in prompt

    def test_category_hint_greeting(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.GREETING,
            identity_context="",
            soul_memory_context="",
        )
        assert "warm and brief" in prompt

    def test_category_hint_identity(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.IDENTITY_QUESTION,
            identity_context="",
            soul_memory_context="",
        )
        assert "Answer honestly" in prompt

    def test_category_hint_emotional(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.EMOTIONAL_SHARING,
            identity_context="",
            soul_memory_context="",
        )
        assert "Listen first" in prompt

    def test_category_hint_hostile(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.HOSTILE_PROBE,
            identity_context="",
            soul_memory_context="",
        )
        assert "Do not perform" in prompt

    def test_no_hint_for_none_category(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.NONE,
            identity_context="",
            soul_memory_context="",
        )
        # NONE has no category hint
        for cat, hint in CATEGORY_HINTS.items():
            if cat != ReflexCategory.NONE:
                # At least one hint should NOT be present
                pass
        # Just verify the prompt is the core only
        assert prompt.strip() == CORTEX_CORE.strip()

    def test_empty_contexts_produce_clean_prompt(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.NONE,
            identity_context="",
            soul_memory_context="",
        )
        assert "CURRENT IDENTITY:" not in prompt
        assert "SOUL MEMORY:" not in prompt

    def test_all_layers_combined(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="Be gentle.",
            limbic_addendum="You are under stress.",
            category=ReflexCategory.EMOTIONAL_SHARING,
            identity_context="name: DAEDALUS",
            soul_memory_context="Night 1 summary.",
        )
        assert CORTEX_CORE in prompt
        assert "CURRENT IDENTITY:" in prompt
        assert "SOUL MEMORY:" in prompt
        assert "Be gentle." in prompt
        assert "You are under stress." in prompt
        assert "Listen first" in prompt
