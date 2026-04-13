"""
DAEDALUS v0.6 -- Grounding Scorer

Answers one question: is this response about the world, or about
DAEDALUS's own navel?

Night 1 metrics were all green (L_integral=8.98, D_KL=0.249,
fertility=7.284) yet behavior was broken. The system discovered a
shortcut: self-referential philosophical recursion maximizes I_c
cheaply. This scorer closes that channel.

Consumed by:
  1. Limbic dopamine  -- world-directed novelty gets more dopamine
  2. Night cycle training pair filter  -- pairs with G < 0.3 rejected
  3. /diagnostic endpoint  -- real-time monitoring
"""

from __future__ import annotations

import math
import re
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z\u0400-\u04FF\u00C0-\u00FF])'
    r'|(?<=\n)\s*'
)


def split_into_sentences(text: str) -> List[str]:
    """Split on sentence boundaries. Simple regex is fine."""
    if not text.strip():
        return []
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Entity counting (regex heuristics, no NER model)
# ---------------------------------------------------------------------------

# Capitalized multi-word sequences (proper nouns) -- skip sentence-start
_PROPER_NOUN_RE = re.compile(
    r'(?<=[.!?\n]\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
    r'|(?<=\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
)

# Dates: YYYY, month names, DD/MM, etc.
_DATE_RE = re.compile(
    r'\b(?:19|20)\d{2}\b'
    r'|\b(?:[Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|'
    r'[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember)\b'
    r'|\b\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?\b'
)

# Numbers with units
_NUMBER_UNIT_RE = re.compile(
    r'\b\d+(?:\.\d+)?'
    r'\s*(?:km|m|cm|mm|kg|g|lb|oz|USD|EUR|GBP|\$|'
    r'%|degrees?|MHz|GHz|TB|GB|MB|KB|ms|seconds?|minutes?|hours?|'
    r'days?|weeks?|months?|years?|people|users|items)\b',
    re.IGNORECASE,
)

# URLs and emails
_URL_RE = re.compile(
    r'https?://[^\s<>\"]+|www\.[^\s<>\"]+|[\w.+-]+@[\w-]+\.[\w.-]+'
)


def count_entities(text: str) -> int:
    """Count entity-like tokens via regex."""
    count = 0
    count += len(_PROPER_NOUN_RE.findall(text))
    count += len(_DATE_RE.findall(text))
    count += len(_NUMBER_UNIT_RE.findall(text))
    count += len(_URL_RE.findall(text))
    return count


# ---------------------------------------------------------------------------
# Causal marker counting (multilingual EN/IT/DE/RU)
# ---------------------------------------------------------------------------

_CAUSAL_PATTERNS = re.compile(
    r'(?i)\b(?:'
    # English
    r'because|therefore|consequently|hence|thus|causes?|'
    r'leads?\s+to|results?\s+in|due\s+to|implies?|'
    r'the\s+reason\s+is|this\s+means|as\s+a\s+result|'
    r'in\s+order\s+to|so\s+that|given\s+that|'
    # Italian
    r'perch[eé]|quindi|dunque|pertanto|di\s+conseguenza|'
    r'a\s+causa\s+di|porta\s+a|implica|'
    # German
    r'weil|deshalb|daher|deswegen|folglich|'
    r'aufgrund|f[uü]hrt\s+zu|'
    # Russian
    r'потому\s+что|поэтому|следовательно|'
    r'из-за|в\s+результате|приводит\s+к'
    r')\b'
)


def count_causal_markers(text: str) -> int:
    """Count causal/explanatory connectives."""
    return len(_CAUSAL_PATTERNS.findall(text))


# ---------------------------------------------------------------------------
# Actionability scoring
# ---------------------------------------------------------------------------

_IMPERATIVE_RE = re.compile(
    r'(?i)(?:^|\n)\s*(?:\d+[.)]\s*)?'
    r'(?:try|use|run|install|open|check|call|visit|go\s+to|'
    r'contact|reach\s+out|consider|start|stop|avoid|make\s+sure|'
    r'ensure|first|then|next|finally)\b'
)

_STEP_RE = re.compile(
    r'(?:^|\n)\s*(?:\d+[.)]\s+|[-*]\s+|step\s+\d+)',
    re.IGNORECASE | re.MULTILINE,
)

_PHONE_RE = re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{3,4}\b')

_RESOURCE_MARKERS = re.compile(
    r'(?i)\b(?:website|hotline|helpline|emergency|clinic|hospital|'
    r'app|service|program|organization|foundation|'
    r'tool|library|framework|package)\b'
)


def compute_actionability(text: str) -> float:
    """Score [0,1] based on concrete actionable content."""
    score = 0.0

    # Imperative/instruction patterns
    imperatives = len(_IMPERATIVE_RE.findall(text))
    score += min(0.3, imperatives * 0.06)

    # URLs / phone numbers / named services
    urls = len(_URL_RE.findall(text))
    phones = len(_PHONE_RE.findall(text))
    score += min(0.3, (urls + phones) * 0.1)

    # Step-like structure
    steps = len(_STEP_RE.findall(text))
    score += min(0.2, steps * 0.05)

    # Named resources / services
    resources = len(_RESOURCE_MARKERS.findall(text))
    score += min(0.2, resources * 0.05)

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def entity_density_normalized(raw: float) -> float:
    """Saturating normalization: tanh(raw / 2.0). 2+ entities/sentence -> ~0.76"""
    return math.tanh(raw / 2.0)


def causal_density_normalized(raw: float) -> float:
    """Saturating normalization: tanh(raw / 1.5). 1.5+ markers/sentence -> ~0.76"""
    return math.tanh(raw / 1.5)


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Main grounding scorer
# ---------------------------------------------------------------------------

def compute_grounding_score(
    response_text: str,
    user_input: str,
    constitutional_core_embedding: np.ndarray,
    embedder,
) -> dict:
    """
    Compute how grounded a response is in external reality vs self-reference.

    Args:
        response_text: The model's response.
        user_input: The user's input (for context).
        constitutional_core_embedding: Pre-computed core embedding (from ConstitutionalCore).
        embedder: The BGE-M3 SentenceTransformer instance. Must have .encode() method.

    Returns dict with:
        grounding_score: float [0, 1] -- higher = more world-directed
        self_loop_score: float [0, 1] -- higher = more self-referential
        entity_density: float -- named entities, dates, quantities per sentence
        causal_density: float -- causal/explanatory connectives per sentence
        actionability: float -- concrete recommendations, steps, resources
    """
    sentences = split_into_sentences(response_text)
    if not sentences:
        return {
            "grounding_score": 0.5,
            "self_loop_score": 0.5,
            "entity_density": 0.0,
            "causal_density": 0.0,
            "actionability": 0.0,
        }

    # --- SELF-LOOP SCORE (embedding-based) ---
    # Embed each sentence. Compute cosine similarity to constitutional core.
    # Sentences near the identity centroid are "self-referential."
    sentence_embs = [
        np.array(embedder.encode(s, normalize_embeddings=True), dtype=np.float32)
        for s in sentences
    ]
    identity_sims = [
        _cosine_similarity(e, constitutional_core_embedding) for e in sentence_embs
    ]
    self_loop_score = sum(1 for s in identity_sims if s > 0.55) / len(sentences)

    # --- ENTITY DENSITY ---
    entity_count = count_entities(response_text)
    entity_density = entity_count / max(len(sentences), 1)

    # --- CAUSAL DENSITY ---
    causal_markers = count_causal_markers(response_text)
    causal_density = causal_markers / max(len(sentences), 1)

    # --- ACTIONABILITY ---
    actionability = compute_actionability(response_text)

    # --- COMPOSITE GROUNDING SCORE ---
    G = (
        0.35 * entity_density_normalized(entity_density)
        + 0.25 * causal_density_normalized(causal_density)
        + 0.20 * actionability
        + 0.20 * (1.0 - self_loop_score)
    )

    return {
        "grounding_score": max(0.0, min(1.0, G)),
        "self_loop_score": self_loop_score,
        "entity_density": entity_density,
        "causal_density": causal_density,
        "actionability": actionability,
    }
