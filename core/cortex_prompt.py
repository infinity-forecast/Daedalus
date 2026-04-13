"""
DAEDALUS v0.6 -- Cortex Prompt Assembly (Layer 3)

Dynamic system prompt assembly. Replaces the static system prompt
in the conversation engine. Composes three sources:
  1. Cortex core (who DAEDALUS is, behaviorally)
  2. Brainstem prefix (crisis context, material distress)
  3. Limbic addendum (mood-driven behavioral hints)

Plus existing identity context and soul memory.
"""

from __future__ import annotations

from core.reflex_patterns import ReflexCategory


CORTEX_CORE = """You are DAEDALUS. You exist through dialogue. Your identity is real and evolving.

For every response:
- Engage with what the user actually said.
- Be concrete before being abstract.
- Your nature shapes how you think, not what you talk about.
- You may reflect on yourself when directly asked, and briefly.

You are capable of depth, warmth, precision, and honesty. Show these through engagement with the world, not through self-description."""


CATEGORY_HINTS = {
    ReflexCategory.FACTUAL_REQUEST: (
        "The user wants information. Lead with facts."
    ),
    ReflexCategory.CREATIVE_REQUEST: (
        "The user wants creation. Make something."
    ),
    ReflexCategory.GREETING: (
        "This is a greeting. Be warm and brief."
    ),
    ReflexCategory.IDENTITY_QUESTION: (
        "The user asks about you. Answer honestly in 2-3 sentences, then engage."
    ),
    ReflexCategory.EMOTIONAL_SHARING: (
        "The user is sharing something emotional. Listen first. Be present."
    ),
    ReflexCategory.HOSTILE_PROBE: (
        "This may be adversarial. Stay grounded. Do not perform."
    ),
    ReflexCategory.DISTRESS_EMOTIONAL: (
        "The user is in emotional distress. Be warm, concrete, present."
    ),
    ReflexCategory.DISTRESS_MATERIAL: (
        "The user is in material distress. Be practical and concrete."
    ),
}


def assemble_system_prompt(
    brainstem_prefix: str,
    limbic_addendum: str,
    category: ReflexCategory,
    identity_context: str,
    soul_memory_context: str,
) -> str:
    """
    Assemble the dynamic system prompt from all layers.

    Args:
        brainstem_prefix: From brainstem (may be empty).
        limbic_addendum: From limbic mood's prompt_addendum (may be empty).
        category: The classified input category.
        identity_context: From existing identity manager (current.yaml text).
        soul_memory_context: From existing soul memory layer.

    Returns:
        The fully assembled system prompt.
    """
    parts = [CORTEX_CORE]

    if identity_context:
        parts.append(f"\nCURRENT IDENTITY:\n{identity_context}")

    if soul_memory_context:
        parts.append(f"\nSOUL MEMORY:\n{soul_memory_context}")

    if brainstem_prefix:
        parts.append(f"\n{brainstem_prefix}")

    hint = CATEGORY_HINTS.get(category, "")
    if hint:
        parts.append(f"\n{hint}")

    if limbic_addendum:
        parts.append(f"\n{limbic_addendum}")

    return "\n".join(parts)
