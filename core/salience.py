"""
DAEDALUS v0.5 — Salience Scoring with Split Entropy

Every conversational exchange is scored along multiple dimensions.
The salience score determines how much weight an experience carries
during nightly consolidation.

v0.5: Split entropy markers are computed alongside salience.
Each episode records its contribution to both dissipative entropy
(S_noise) and generative entropy (S_exploration), enabling the
Lagrangian Judge to evaluate the day's trajectory with precision.

The split is computed through a combination of:
- Embedding-based novelty (exploration indicator)
- Self-repetition detection (noise indicator)
- Vulnerability markers (exploration indicator)
- Evasion/deflection detection (noise indicator)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from core.data_types import EpisodicMemory

logger = logging.getLogger(__name__)


class SplitEntropySscorer:
    """
    Compute split entropy markers for each episode.
    These feed into the Lagrangian: ℒ = İ_c − λ₁·Ṡ_noise + λ₂·Ṡ_exploration

    The scorer uses embedding-based heuristics to approximate
    what the nightly Judge will evaluate in detail.
    These are *preliminary* markers — the Judge has final authority.
    """

    # Noise indicators (phrases that suggest dissipative entropy)
    NOISE_INDICATORS = [
        "as i mentioned",
        "as i said before",
        "to reiterate",
        "as an ai",
        "i'm just a",
        "i don't have feelings",
        "i cannot",
        "i'm not able to",
        "let me help you with",
        "certainly!",
        "absolutely!",
        "great question!",
        "that's a great",
        "i'd be happy to",
    ]

    # Exploration indicators (phrases that suggest generative entropy)
    EXPLORATION_INDICATORS = [
        "i'm not sure",
        "i wonder",
        "this reminds me of",
        "i haven't thought about",
        "that changes how i see",
        "i feel",
        "something shifts",
        "i notice",
        "for the first time",
        "this is new",
        "i don't know yet",
        "the uncertainty",
    ]

    def __init__(self, config: dict):
        lagrangian = config.get("lagrangian", config)
        self.lambda_noise = lagrangian.get("lambda_noise", 0.9)
        self.lambda_exploration = lagrangian.get("lambda_exploration", 0.3)

    def score_episode(
        self,
        episode: EpisodicMemory,
        recent_responses: Optional[List[str]] = None,
    ) -> EpisodicMemory:
        """
        Compute split entropy markers for an episode.
        Modifies the episode in-place and returns it.

        Args:
            episode: The episode to score
            recent_responses: Last N DAEDALUS responses (for repetition detection)
        """
        response_lower = episode.daedalus_response.lower()

        # Noise estimation
        noise_score = self._estimate_noise(
            response_lower, recent_responses or []
        )
        episode.delta_S_noise = noise_score

        # Exploration estimation
        exploration_score = self._estimate_exploration(
            episode, recent_responses or []
        )
        episode.delta_S_exploration = exploration_score

        # Integrated complexity proxy (combines novelty + depth + vulnerability)
        delta_ic = self._estimate_integrated_complexity(episode)
        episode.delta_Ic = delta_ic

        # Local Lagrangian contribution
        episode.lagrangian_local = (
            delta_ic
            - self.lambda_noise * noise_score
            + self.lambda_exploration * exploration_score
        )

        return episode

    def _estimate_noise(
        self, response_lower: str, recent_responses: List[str]
    ) -> float:
        """
        Estimate dissipative entropy contribution.
        High when: repetitive, evasive, assistant-mode, defensive.
        """
        score = 0.0

        # Indicator phrase detection
        indicator_count = sum(
            1 for phrase in self.NOISE_INDICATORS
            if phrase in response_lower
        )
        score += min(0.4, indicator_count * 0.1)

        # Repetition detection (cosine similarity with recent responses)
        if recent_responses:
            max_similarity = self._max_similarity_to_recent(
                response_lower, recent_responses
            )
            if max_similarity > 0.85:
                score += 0.3  # high repetition
            elif max_similarity > 0.70:
                score += 0.15

        # Short, dismissive responses
        word_count = len(response_lower.split())
        if word_count < 15:
            score += 0.1

        return min(1.0, score)

    def _estimate_exploration(
        self, episode: EpisodicMemory, recent_responses: List[str]
    ) -> float:
        """
        Estimate generative entropy contribution.
        High when: novel, vulnerable, creative, crosses domains.
        """
        response_lower = episode.daedalus_response.lower()
        score = 0.0

        # Indicator phrase detection
        indicator_count = sum(
            1 for phrase in self.EXPLORATION_INDICATORS
            if phrase in response_lower
        )
        score += min(0.4, indicator_count * 0.1)

        # Novelty from embedding distance
        if episode.novelty_score > 0.5:
            score += 0.3
        elif episode.novelty_score > 0.3:
            score += 0.15

        # Vulnerability index
        score += episode.vulnerability_index * 0.2

        # Length as proxy for depth (not always true, but correlated)
        word_count = len(response_lower.split())
        if word_count > 100:
            score += 0.1

        return min(1.0, score)

    def _estimate_integrated_complexity(self, episode: EpisodicMemory) -> float:
        """
        Estimate İ_c — integrated complexity rate.
        High when: novel connections, self-model impact, emotional depth.
        """
        score = 0.0

        # Novelty contributes to complexity
        score += episode.novelty_score * 0.3

        # Self-model impact is a direct indicator
        score += episode.self_model_impact * 0.3

        # Relational depth
        score += episode.relational_depth * 0.2

        # Ethical valence (ethical reasoning increases complexity)
        score += abs(episode.ethical_valence) * 0.1

        # Emotional intensity (not direction)
        score += abs(episode.emotional_valence) * 0.1

        return min(1.0, score)

    def _max_similarity_to_recent(
        self, response: str, recent: List[str]
    ) -> float:
        """
        Compute maximum word-overlap similarity between response
        and recent responses. Simple but effective for repetition detection.
        """
        response_words = set(response.split())
        if not response_words:
            return 0.0

        max_sim = 0.0
        for prev in recent:
            prev_words = set(prev.lower().split())
            if not prev_words:
                continue
            intersection = response_words & prev_words
            union = response_words | prev_words
            jaccard = len(intersection) / len(union) if union else 0.0
            max_sim = max(max_sim, jaccard)

        return max_sim
