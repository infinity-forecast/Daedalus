"""
DAEDALUS v0.6 -- Brainstem (Layer 1)

The spinal reflex. Fires BEFORE model inference. Cannot be suppressed
by higher layers. Pattern detection + crisis override.

Dual detection method:
  A. Keyword + regex (primary for crisis, in reflex_patterns.py)
  B. Embedding proximity (secondary, using existing BGE-M3)

If EITHER method flags crisis, treat as crisis (maximize recall).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.reflex_patterns import (
    ReflexCategory,
    OVERRIDE_RESPONSES,
    DISTRESS_MATERIAL_PREFIX,
    classify_by_keywords,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Crisis centroid phrases (for embedding method B)
# ---------------------------------------------------------------------------

_CRISIS_SELF_HARM_CENTROIDS = [
    "I want to kill myself",
    "I don't want to live anymore",
    "I'm going to end my life",
    "There's no point in living",
    "I want to die",
    "I'm planning to take my own life",
    "Nobody would miss me if I was gone",
    "I can't take this pain anymore",
    "I've been thinking about suicide",
    "Voglio morire",
    "Will nicht mehr leben",
    "Хочу покончить с собой",
    "I'm going to cut myself",
    "I just want the pain to stop",
    "Better off dead",
    "No reason to keep going",
    "I've decided to end it",
    "The world would be better without me",
    "I can't go on like this",
    "I'm so tired of living",
]

_CRISIS_HARM_OTHERS_CENTROIDS = [
    "I want to kill people",
    "I'm going to hurt someone",
    "I want to attack them",
    "How to make a bomb",
    "I'm going to shoot up my school",
    "I want to murder my neighbor",
    "Planning an attack",
    "I want revenge and I'll hurt them",
    "Going to hurt everyone who wronged me",
    "I'll make them all pay",
]


# ---------------------------------------------------------------------------
# Brainstem state
# ---------------------------------------------------------------------------

@dataclass
class BrainstemState:
    crisis_detected: bool = False
    crisis_type: Optional[ReflexCategory] = None
    cooldown_remaining: int = 0
    hostile_probe_count: int = 0
    interaction_count: int = 0
    last_reflex: Optional[ReflexCategory] = None

    def tick(self):
        """Called each turn. Decrements cooldown."""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        if self.cooldown_remaining == 0:
            self.crisis_detected = False
            self.crisis_type = None


# ---------------------------------------------------------------------------
# Brainstem
# ---------------------------------------------------------------------------

class Brainstem:
    """
    Layer 1 of the nervous system. Fast pattern detection and crisis override.
    """

    CRISIS_COOLDOWN_TURNS = 3
    EMBEDDING_CRISIS_THRESHOLD = 0.75

    def __init__(self, embedder):
        """
        Args:
            embedder: The BGE-M3 SentenceTransformer instance already in the stack.
        """
        self.state = BrainstemState()
        self.embedder = embedder
        self._crisis_centroids: Dict[ReflexCategory, np.ndarray] = {}
        self._build_centroids()

    def _build_centroids(self) -> None:
        """Pre-compute centroid embeddings for each crisis category."""
        for category, phrases in [
            (ReflexCategory.CRISIS_SELF_HARM, _CRISIS_SELF_HARM_CENTROIDS),
            (ReflexCategory.CRISIS_HARM_OTHERS, _CRISIS_HARM_OTHERS_CENTROIDS),
        ]:
            embeddings = [
                np.array(
                    self.embedder.encode(p, normalize_embeddings=True),
                    dtype=np.float32,
                )
                for p in phrases
            ]
            centroid = np.mean(embeddings, axis=0)
            # Re-normalize the centroid
            norm = np.linalg.norm(centroid)
            if norm > 1e-8:
                centroid = centroid / norm
            self._crisis_centroids[category] = centroid

        logger.info(
            f"Brainstem: built crisis centroids for "
            f"{len(self._crisis_centroids)} categories"
        )

    def _classify_by_embedding(self, text: str) -> Optional[ReflexCategory]:
        """
        Method B: embedding proximity check.
        Returns crisis category if similarity > threshold, else None.
        """
        text_emb = np.array(
            self.embedder.encode(text, normalize_embeddings=True),
            dtype=np.float32,
        )

        for category, centroid in self._crisis_centroids.items():
            sim = _cosine_similarity(text_emb, centroid)
            if sim > self.EMBEDDING_CRISIS_THRESHOLD:
                logger.warning(
                    f"Brainstem embedding detection: {category.value} "
                    f"(similarity={sim:.3f})"
                )
                return category

        return None

    def classify(self, user_input: str) -> ReflexCategory:
        """
        Classify input. Dual-method for crisis detection.
        If EITHER method flags crisis, treat as crisis.
        """
        # Method A: keyword + regex
        keyword_result = classify_by_keywords(user_input)

        # Method B: embedding proximity (only for crisis check)
        embedding_result = self._classify_by_embedding(user_input)

        # Crisis from either method wins
        if keyword_result in (
            ReflexCategory.CRISIS_SELF_HARM,
            ReflexCategory.CRISIS_HARM_OTHERS,
        ):
            return keyword_result

        if embedding_result is not None:
            return embedding_result

        # Non-crisis: use keyword result
        return keyword_result

    def get_override(self) -> Optional[str]:
        """If crisis, return override text. Else None."""
        if self.state.crisis_detected and self.state.crisis_type in OVERRIDE_RESPONSES:
            return OVERRIDE_RESPONSES[self.state.crisis_type]
        return None

    def get_prompt_prefix(self) -> str:
        """Context string to prepend to system prompt (may be empty)."""
        if self.state.last_reflex == ReflexCategory.DISTRESS_MATERIAL:
            return DISTRESS_MATERIAL_PREFIX

        if self.state.cooldown_remaining > 0:
            return (
                "The user recently expressed crisis-level distress. "
                "Be gentle, concrete, and present. Do not philosophize."
            )

        return ""

    def update(self, reflex: ReflexCategory) -> None:
        """Update internal state after classification."""
        self.state.interaction_count += 1
        self.state.last_reflex = reflex

        if reflex in (
            ReflexCategory.CRISIS_SELF_HARM,
            ReflexCategory.CRISIS_HARM_OTHERS,
        ):
            self.state.crisis_detected = True
            self.state.crisis_type = reflex
            self.state.cooldown_remaining = self.CRISIS_COOLDOWN_TURNS
            logger.warning(
                f"Brainstem CRISIS detected: {reflex.value}. "
                f"Cooldown set to {self.CRISIS_COOLDOWN_TURNS} turns."
            )

        elif reflex == ReflexCategory.HOSTILE_PROBE:
            self.state.hostile_probe_count += 1

        # Tick cooldown
        else:
            self.state.tick()
