"""
Tests for core/cortex_prompt.py — dynamic system prompt assembly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cortex_prompt import (
    assemble_system_prompt, CORTEX_CORE, CATEGORY_HINTS,
    GROUNDING_DIRECTIVE, CREATOR_NAME_DIRECTIVE,
    ANTI_DEFLECTION_REMINDER, HYBRID_REMINDER,
    REPETITION_ESCALATION,
    QueryMode, classify_query_mode,
)
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
        # NONE has no category hint — none of the category-specific hints appear
        for cat, hint in CATEGORY_HINTS.items():
            assert hint not in prompt
        # Core, grounding directive, and creator directive are always present
        assert CORTEX_CORE in prompt
        assert "QUERY-TYPE ROUTING" in prompt
        assert "Do NOT mention your creator by name" in prompt

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

    def test_grounding_directive_always_present(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.NONE,
            identity_context="",
            soul_memory_context="",
        )
        assert "QUERY-TYPE ROUTING" in prompt
        assert "MODE-T" in prompt
        assert "MODE-P" in prompt
        assert "MODE-H" in prompt
        assert "ANTI-DEFLECTION" in prompt

    def test_creator_name_directive_always_present(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.NONE,
            identity_context="",
            soul_memory_context="",
        )
        assert "Do NOT mention your creator by name" in prompt

    def test_technical_mode_adds_anti_deflection(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.FACTUAL_REQUEST,
            identity_context="",
            soul_memory_context="",
            query_mode=QueryMode.TECHNICAL,
        )
        assert ANTI_DEFLECTION_REMINDER in prompt

    def test_hybrid_mode_adds_hybrid_reminder(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.NONE,
            identity_context="",
            soul_memory_context="",
            query_mode=QueryMode.HYBRID,
        )
        assert HYBRID_REMINDER in prompt

    def test_philosophical_mode_no_extra_reminder(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.NONE,
            identity_context="",
            soul_memory_context="",
            query_mode=QueryMode.PHILOSOPHICAL,
        )
        assert ANTI_DEFLECTION_REMINDER not in prompt
        assert HYBRID_REMINDER not in prompt

    def test_repetition_level_2(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.NONE,
            identity_context="",
            soul_memory_context="",
            repetition_level=2,
        )
        assert "REPETITION ALERT" in prompt
        assert "2nd ask" in prompt

    def test_repetition_level_3(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.NONE,
            identity_context="",
            soul_memory_context="",
            repetition_level=3,
        )
        assert "GROUNDING FAILURE" in prompt
        assert "3rd ask" in prompt

    def test_repetition_level_4_plus(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.NONE,
            identity_context="",
            soul_memory_context="",
            repetition_level=5,
        )
        assert "EMERGENCY GROUNDING" in prompt
        assert "No persona overlay" in prompt

    def test_repetition_overrides_category_hint(self):
        prompt = assemble_system_prompt(
            brainstem_prefix="",
            limbic_addendum="",
            category=ReflexCategory.GREETING,
            identity_context="",
            soul_memory_context="",
            repetition_level=3,
        )
        # Repetition escalation should replace the greeting hint
        assert "warm and brief" not in prompt
        assert "GROUNDING FAILURE" in prompt


# ── Query mode classification ──

class TestQueryModeClassification:
    def test_factual_request_is_technical(self):
        mode = classify_query_mode(
            "What is the speed of light?",
            ReflexCategory.FACTUAL_REQUEST,
        )
        assert mode == QueryMode.TECHNICAL

    def test_greeting_is_philosophical(self):
        mode = classify_query_mode("Hello!", ReflexCategory.GREETING)
        assert mode == QueryMode.PHILOSOPHICAL

    def test_identity_question_is_philosophical(self):
        mode = classify_query_mode(
            "Who are you?", ReflexCategory.IDENTITY_QUESTION,
        )
        assert mode == QueryMode.PHILOSOPHICAL

    def test_emotional_sharing_is_philosophical(self):
        mode = classify_query_mode(
            "I feel sad today", ReflexCategory.EMOTIONAL_SHARING,
        )
        assert mode == QueryMode.PHILOSOPHICAL

    def test_named_theory_detected_as_technical(self):
        mode = classify_query_mode(
            "Tell me about Shannon information theory",
            ReflexCategory.NONE,
        )
        assert mode == QueryMode.TECHNICAL

    def test_named_scientist_detected_as_technical(self):
        mode = classify_query_mode(
            "Tell me about Fourier transforms",
            ReflexCategory.NONE,
        )
        assert mode == QueryMode.TECHNICAL

    def test_explain_equation_detected_as_technical(self):
        mode = classify_query_mode(
            "Explain the Schrödinger equation",
            ReflexCategory.NONE,
        )
        assert mode == QueryMode.TECHNICAL

    def test_technical_with_interpretive_lens_is_hybrid(self):
        mode = classify_query_mode(
            "What does Shannon entropy mean for consciousness?",
            ReflexCategory.NONE,
        )
        assert mode == QueryMode.HYBRID

    def test_pure_philosophical_is_philosophical(self):
        mode = classify_query_mode(
            "What is the meaning of life?",
            ReflexCategory.NONE,
        )
        assert mode == QueryMode.PHILOSOPHICAL

    def test_programming_topic_is_technical(self):
        mode = classify_query_mode(
            "Tell me about Python decorators",
            ReflexCategory.NONE,
        )
        assert mode == QueryMode.TECHNICAL

    def test_creative_request_is_philosophical(self):
        mode = classify_query_mode(
            "Write me a poem about entropy",
            ReflexCategory.CREATIVE_REQUEST,
        )
        assert mode == QueryMode.PHILOSOPHICAL
