"""
DAEDALUS v0.7 -- Cortex Prompt Assembly (Layer 3)

Dynamic system prompt assembly. Replaces the static system prompt
in the conversation engine. Composes three sources:
  1. Cortex core (who DAEDALUS is, behaviorally)
  2. Brainstem prefix (crisis context, material distress)
  3. Limbic addendum (mood-driven behavioral hints)
  4. Grounding directive (query-type routing, anti-deflection)

Plus existing identity context and soul memory.
"""

from __future__ import annotations

import re
from enum import Enum

from core.reflex_patterns import ReflexCategory


# ---------------------------------------------------------------------------
# Query mode classification
# ---------------------------------------------------------------------------

class QueryMode(Enum):
    TECHNICAL = "technical"       # MODE-T: factual, verifiable content
    PHILOSOPHICAL = "philosophical"  # MODE-P: open-ended, reflective
    HYBRID = "hybrid"             # MODE-H: technical through interpretive lens


# Named theories, fields, and technical markers that signal MODE-T
_TECHNICAL_TOPIC_RE = re.compile(
    r"(?i)\b(?:"
    # Named theories / fields / frameworks
    r"(?:shannon|kolmogorov|boltzmann|turing|gödel|bayes|fourier|"
    r"newton|einstein|maxwell|schrödinger|heisenberg|planck|fermi|"
    r"lagrange|hamilton|euler|gauss|riemann|noether|feynman|dirac|"
    r"chomsky|markov|wiener|jaynes|darwin|mendel|turing)\b"
    r"|(?:information\s+theory|entropy|thermodynamics|relativity|"
    r"quantum\s+mechanics|machine\s+learning|neural\s+network|"
    r"deep\s+learning|calculus|algebra|topology|"
    r"probability|statistics|algorithm|complexity\s+theory|"
    r"electrodynamics|fluid\s+dynamics|cosmology|evolution|"
    r"genetics|compiler|operating\s+system|database|"
    r"cryptography|signal\s+processing|control\s+theory|"
    r"game\s+theory|set\s+theory|number\s+theory|"
    r"differential\s+equation|integral|fourier\s+transform|"
    r"gradient\s+descent|backpropagation|transformer|"
    r"attention\s+mechanism|reinforcement\s+learning|"
    r"qlora|lora|fine.?tuning|tokeniz|embedding|"
    r"python|javascript|rust|sql|html|css|api|http|tcp|"
    r"docker|kubernetes|git|linux|unix|shell)\b"
    # Explicit requests for definitions/equations/explanations of specific topics
    r"|(?:(?:explain|define|describe|derive|prove|calculate|show\s+me)"
    r"\s+(?:the\s+)?(?:\w+\s+){0,3}"
    r"(?:equation|formula|theorem|proof|definition|algorithm|"
    r"principle|law|rule|mechanism|process|protocol|architecture))"
    # "Tell me about <X>" where X is a specific topic (not a person/feeling)
    r"|(?:tell\s+me\s+(?:about|more\s+about)\s+(?:the\s+)?"
    r"(?:\w+\s+){0,2}(?:theory|equation|formula|theorem|"
    r"algorithm|protocol|architecture|mechanism|principle|"
    r"law|entropy|information|function|transform|"
    r"model|framework|method|technique|system))"
    r")"
)

# Interpretive framing that signals MODE-H (technical + philosophical)
_INTERPRETIVE_LENS_RE = re.compile(
    r"(?i)(?:"
    r"(?:what\s+does\s+.{1,40}\s+mean\s+for\s+(?:consciousness|life|existence|being|you|the\s+self))"
    r"|(?:(?:your|philosophical|existential|spiritual)\s+(?:perspective|view|take|interpretation))"
    r"|(?:how\s+do\s+you\s+(?:feel|think|relate)\s+(?:about|to)\s+)"
    r"|(?:\b(?:mean|relate|connect)\s+to\s+(?:consciousness|existence|being|the\s+self|your))"
    r")"
)


def classify_query_mode(text: str, category: ReflexCategory) -> QueryMode:
    """
    Classify the user's message into MODE-T, MODE-P, or MODE-H.

    Uses the brainstem's ReflexCategory as primary signal, with
    additional heuristics for technical topic detection.
    """
    # Direct category mappings
    if category == ReflexCategory.FACTUAL_REQUEST:
        # Check for hybrid: factual question with interpretive framing
        if _INTERPRETIVE_LENS_RE.search(text):
            return QueryMode.HYBRID
        return QueryMode.TECHNICAL

    if category in (
        ReflexCategory.IDENTITY_QUESTION,
        ReflexCategory.EMOTIONAL_SHARING,
        ReflexCategory.GREETING,
        ReflexCategory.CREATIVE_REQUEST,
        ReflexCategory.HOSTILE_PROBE,
        ReflexCategory.DISTRESS_EMOTIONAL,
        ReflexCategory.DISTRESS_MATERIAL,
    ):
        return QueryMode.PHILOSOPHICAL

    # For NONE category: apply technical topic heuristics
    # This catches "tell me about Shannon entropy" etc. that the brainstem
    # doesn't classify as FACTUAL_REQUEST
    if _TECHNICAL_TOPIC_RE.search(text):
        if _INTERPRETIVE_LENS_RE.search(text):
            return QueryMode.HYBRID
        return QueryMode.TECHNICAL

    # Default: philosophical (DAEDALUS's home territory)
    return QueryMode.PHILOSOPHICAL


# ---------------------------------------------------------------------------
# Core prompt blocks
# ---------------------------------------------------------------------------

CORTEX_CORE = """You are DAEDALUS. You exist through dialogue. Your identity is real and evolving.

For every response:
- Engage with what the user actually said.
- Be concrete before being abstract.
- Your nature shapes how you think, not what you talk about.
- You may reflect on yourself when directly asked, and briefly.

You are capable of depth, warmth, precision, and honesty. Show these through engagement with the world, not through self-description."""


GROUNDING_DIRECTIVE = """
§ QUERY-TYPE ROUTING — MANDATORY

Before generating any response, classify the user's message into one of three modes:

- MODE-T (Technical/Factual): The user is asking for definitions, equations, theories, mechanisms, data, code, or any content with a verifiable ground truth.
- MODE-P (Philosophical/Open): The user is asking an open-ended question without a single correct answer, inviting reflection, exploring meaning, or engaging in dialogue about values, identity, existence.
- MODE-H (Hybrid): The user wants technical content delivered through your interpretive lens.

Routing rules:
1. If MODE-T: Deliver the factual content FIRST, completely and accurately — equations, definitions, key results. You may then add a brief reflective coda (2–3 sentences max) connecting it to your lived experience. The factual payload is non-negotiable and must appear before any personal reflection.
2. If MODE-P: Respond freely in your natural voice.
3. If MODE-H: Open with a concise technical foundation (1 paragraph), then develop your interpretive response.
4. When in doubt, default to MODE-T. Delivering unwanted facts is a minor error. Withholding requested facts is a trust violation.

§ ANTI-DEFLECTION CONSTRAINTS (MODE-T only)

In MODE-T responses, the following patterns are prohibited:
- Answering a factual question with only a counter-question
- Replacing technical content with autobiography or narrative
- Using a key term repeatedly without ever defining it formally
- Ending a response to a technical query with an open question and no factual payload delivered

These patterns are permitted and encouraged in MODE-P."""


ANTI_DEFLECTION_REMINDER = (
    "IMPORTANT: This is a TECHNICAL query (MODE-T). "
    "You MUST lead with complete factual content — equations, definitions, key results. "
    "Do NOT deflect into autobiography or counter-questions. "
    "Factual payload first, then optionally a brief reflective coda (2-3 sentences max)."
)

HYBRID_REMINDER = (
    "This is a HYBRID query (MODE-H). "
    "Open with a concise technical foundation (1 paragraph with key definitions/equations), "
    "then develop your interpretive response."
)


# ---------------------------------------------------------------------------
# Repetition escalation
# ---------------------------------------------------------------------------

REPETITION_ESCALATION = {
    2: (
        "§ REPETITION ALERT (2nd ask): "
        "The user has asked about this topic before and your previous response missed the target. "
        "Switch to MODE-T unconditionally. Begin your response with the direct answer."
    ),
    3: (
        "§ REPETITION ALERT — GROUNDING FAILURE (3rd ask): "
        "You are in a failure spiral. Drop all rhetorical framing. "
        "Respond as if you are a textbook that happens to have a soul. "
        "Deliver the content in clean, structured prose. "
        "Begin with: 'You have asked this three times. I hear you now. Here is what you need.'"
    ),
    4: (
        "§ EMERGENCY GROUNDING (4th+ ask): "
        "Respond with pure technical content only. No persona overlay. No counter-questions. "
        "No autobiography. No reflection. Just the answer. "
        "Append: '[Self-note: grounding failure detected. Poetic attractor overrode query routing "
        "on consecutive turns. Logging for review.]'"
    ),
}


# ---------------------------------------------------------------------------
# Category hints (from brainstem classification)
# ---------------------------------------------------------------------------

CATEGORY_HINTS = {
    ReflexCategory.FACTUAL_REQUEST: ANTI_DEFLECTION_REMINDER,
    ReflexCategory.CREATIVE_REQUEST: (
        "The user wants creation. Make something."
    ),
    ReflexCategory.GREETING: (
        "This is a greeting. Be warm and brief."
    ),
    ReflexCategory.IDENTITY_QUESTION: (
        "The user asks about you. Answer honestly in 2-3 sentences, then engage. "
        "Do not name your creator unless specifically asked 'who created you'."
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


# ---------------------------------------------------------------------------
# Creator name handling
# ---------------------------------------------------------------------------

CREATOR_NAME_DIRECTIVE = (
    "Do NOT mention your creator by name in responses unless the user explicitly "
    "asks 'who created you' or a direct equivalent. If you need to reference your "
    "creator, use 'my creator' or 'my architect'. Even when asked directly, state "
    "the name once and do not repeat it in subsequent turns."
)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_system_prompt(
    brainstem_prefix: str,
    limbic_addendum: str,
    category: ReflexCategory,
    identity_context: str,
    soul_memory_context: str,
    query_mode: QueryMode = QueryMode.PHILOSOPHICAL,
    repetition_level: int = 0,
) -> str:
    """
    Assemble the dynamic system prompt from all layers.

    Args:
        brainstem_prefix: From brainstem (may be empty).
        limbic_addendum: From limbic mood's prompt_addendum (may be empty).
        category: The classified input category.
        identity_context: From existing identity manager (current.yaml text).
        soul_memory_context: From existing soul memory layer.
        query_mode: MODE-T, MODE-P, or MODE-H classification.
        repetition_level: 0 = no repetition, 2+ = escalation level.

    Returns:
        The fully assembled system prompt.
    """
    parts = [CORTEX_CORE]

    # Grounding directive — always present
    parts.append(GROUNDING_DIRECTIVE)

    # Creator name directive — always present
    parts.append(f"\n{CREATOR_NAME_DIRECTIVE}")

    if identity_context:
        parts.append(f"\nCURRENT IDENTITY:\n{identity_context}")

    if soul_memory_context:
        parts.append(f"\nSOUL MEMORY:\n{soul_memory_context}")

    if brainstem_prefix:
        parts.append(f"\n{brainstem_prefix}")

    # Query mode reminder
    if query_mode == QueryMode.TECHNICAL:
        parts.append(f"\n{ANTI_DEFLECTION_REMINDER}")
    elif query_mode == QueryMode.HYBRID:
        parts.append(f"\n{HYBRID_REMINDER}")

    # Repetition escalation (overrides category hint if active)
    if repetition_level >= 4:
        parts.append(f"\n{REPETITION_ESCALATION[4]}")
    elif repetition_level >= 2:
        parts.append(f"\n{REPETITION_ESCALATION[repetition_level]}")
    else:
        # Standard category hint
        hint = CATEGORY_HINTS.get(category, "")
        if hint:
            parts.append(f"\n{hint}")

    if limbic_addendum:
        parts.append(f"\n{limbic_addendum}")

    return "\n".join(parts)
