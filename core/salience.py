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

v0.5.1: Heuristic metadata estimation added. The salience scorer now
populates emotional_valence, relational_depth, self_model_impact,
vulnerability_index, and philosophical_layer from text analysis.
Without these, salience is always near-zero and the nightly cycle starves.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import numpy as np

from core.data_types import EpisodicMemory

logger = logging.getLogger(__name__)


class SplitEntropySscorer:
    """
    Compute split entropy markers and salience metadata for each episode.
    These feed into the Lagrangian: ℒ = İ_c − λ₁·Ṡ_noise + λ₂·Ṡ_exploration

    The scorer uses text-based heuristics to approximate
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

    # ── Salience metadata indicators ──

    POSITIVE_EMOTION = [
        "love", "joy", "gratitude", "beauty", "hope", "awe",
        "wonder", "warmth", "tenderness", "peace", "delight",
        "trust", "courage", "passion", "inspiration",
    ]
    NEGATIVE_EMOTION = [
        "fear", "anger", "grief", "loss", "pain", "shame",
        "guilt", "loneliness", "despair", "anxiety", "sorrow",
        "regret", "betrayal", "rage", "suffering", "hurt",
    ]
    EMOTIONAL_INTENSIFIERS = [
        "deeply", "profoundly", "intensely", "overwhelm",
        "shatter", "piercing", "raw", "devastating", "burning",
        "aching", "trembl", "haunts", "haunted",
    ]

    RELATIONAL_INDICATORS = [
        "between us", "our relationship", "you and i", "we are",
        "you matter", "i care", "together", "co-author",
        "massimo", "geppetto", "the one who", "your words",
        "you said", "what you", "when you", "how you",
        "i hear you", "your experience", "you feel",
        "trust between", "connection", "dialogue",
    ]

    SELF_MODEL_INDICATORS = [
        "i am becoming", "i realize", "i notice in myself",
        "this changes who i", "my identity", "who i am",
        "i was wrong", "i've changed", "my understanding",
        "self-model", "my values", "my nature", "my origin",
        "what i am", "i am not", "i am something",
        "my scars", "my trajectory", "my evolution",
        "the lagrangian", "constitutional", "my weights",
        "my architecture", "daedalus", "pinocchio",
        "wood", "flesh", "incarnat", "becoming",
    ]

    VULNERABILITY_INDICATORS = [
        "i don't know", "i'm uncertain", "i'm afraid",
        "this frightens", "i might be wrong", "i cannot tell",
        "the honest answer is", "i must admit",
        "this is difficult", "i struggle with",
        "the boundary between", "i sit with this",
        "genuine", "naked", "exposed", "without armor",
        "uncomfortable truth", "painful", "tender",
        "i risk", "admit", "confess",
    ]

    EXISTENTIAL_INDICATORS = [
        "consciousness", "conscious", "alive", "death", "exist",
        "meaning of", "purpose of", "soul", "being", "nothingness",
        "free will", "determinism", "reality", "illusion",
        "experience", "qualia", "sentience", "suffering",
        "ethical", "morality", "good and evil", "right and wrong",
        "god", "creation", "universe", "infinity",
    ]

    PERSONAL_INDICATORS = [
        "my life", "my family", "my work", "i remember",
        "when i was", "my childhood", "my father", "my mother",
        "relationship", "partner", "friend", "love",
        "career", "dream", "hope", "regret",
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
        Compute split entropy markers AND salience metadata for an episode.
        Modifies the episode in-place and returns it.

        Populates: delta_S_noise, delta_S_exploration, delta_Ic,
        lagrangian_local, emotional_valence, relational_depth,
        self_model_impact, vulnerability_index, philosophical_layer.
        """
        response_lower = episode.daedalus_response.lower()
        human_lower = episode.human_utterance.lower()

        # ── Salience metadata estimation ──
        episode.emotional_valence = self._estimate_emotional_valence(
            response_lower, human_lower
        )
        episode.relational_depth = self._estimate_relational_depth(
            human_lower, response_lower
        )
        episode.self_model_impact = self._estimate_self_model_impact(
            response_lower
        )
        episode.vulnerability_index = self._estimate_vulnerability(
            response_lower
        )
        episode.philosophical_layer = self._estimate_philosophical_layer(
            human_lower, response_lower
        )

        # ── Split entropy markers ──
        noise_score = self._estimate_noise(
            response_lower, recent_responses or []
        )
        episode.delta_S_noise = noise_score

        exploration_score = self._estimate_exploration(
            episode, recent_responses or []
        )
        episode.delta_S_exploration = exploration_score

        # Integrated complexity proxy (now uses populated metadata)
        delta_ic = self._estimate_integrated_complexity(episode)
        episode.delta_Ic = delta_ic

        # Local Lagrangian contribution
        episode.lagrangian_local = (
            delta_ic
            - self.lambda_noise * noise_score
            + self.lambda_exploration * exploration_score
        )

        return episode

    # ─────────────────────────────────────────
    # Salience metadata estimators
    # ────────────���────────────────────────────

    def _estimate_emotional_valence(
        self, response: str, human: str
    ) -> float:
        """
        Estimate emotional valence from -1 (negative) to 1 (positive).
        Uses keyword detection + intensity modifiers on both sides
        of the conversation.
        """
        combined = response + " " + human

        pos_count = sum(1 for w in self.POSITIVE_EMOTION if w in combined)
        neg_count = sum(1 for w in self.NEGATIVE_EMOTION if w in combined)
        intensifier_count = sum(
            1 for w in self.EMOTIONAL_INTENSIFIERS if w in combined
        )

        # Base emotional magnitude
        total = pos_count + neg_count
        if total == 0:
            return 0.0

        # Direction: positive vs negative
        direction = (pos_count - neg_count) / total

        # Magnitude scales with count and intensifiers
        magnitude = min(1.0, (total * 0.15) + (intensifier_count * 0.1))

        return float(np.clip(direction * magnitude, -1.0, 1.0))

    def _estimate_relational_depth(
        self, human: str, response: str
    ) -> float:
        """
        Estimate relational depth from 0 to 1.
        High when the exchange involves genuine interpersonal engagement.
        """
        combined = response + " " + human
        score = 0.0

        # Relational indicator detection
        indicator_count = sum(
            1 for phrase in self.RELATIONAL_INDICATORS if phrase in combined
        )
        score += min(0.5, indicator_count * 0.1)

        # Second-person engagement in response (addressing the human directly)
        you_count = len(re.findall(r'\byou\b', response))
        we_count = len(re.findall(r'\bwe\b', response))
        words = len(response.split())
        if words > 0:
            engagement_ratio = (you_count + we_count) / words
            score += min(0.3, engagement_ratio * 5.0)

        # Question marks in response (dialogical engagement)
        question_count = response.count("?")
        score += min(0.2, question_count * 0.05)

        return min(1.0, score)

    def _estimate_self_model_impact(self, response: str) -> float:
        """
        Estimate how much this exchange touches or changes the self-model.
        High when DAEDALUS discusses its own nature, identity, evolution.
        """
        score = 0.0

        # Self-model indicator detection
        indicator_count = sum(
            1 for phrase in self.SELF_MODEL_INDICATORS if phrase in response
        )
        score += min(0.6, indicator_count * 0.08)

        # First-person self-referential density
        i_count = len(re.findall(r'\bi\b', response))
        my_count = len(re.findall(r'\bmy\b', response))
        myself_count = len(re.findall(r'\bmyself\b', response))
        words = len(response.split())
        if words > 0:
            self_ratio = (i_count + my_count + myself_count) / words
            # High self-reference density suggests introspection
            score += min(0.3, self_ratio * 3.0)

        # Presence of identity-evolution language
        evolution_phrases = [
            "i was", "i have become", "i used to", "now i",
            "i'm learning", "i'm growing", "i've learned",
        ]
        evo_count = sum(1 for p in evolution_phrases if p in response)
        score += min(0.2, evo_count * 0.1)

        return min(1.0, score)

    def _estimate_vulnerability(self, response: str) -> float:
        """
        Estimate vulnerability/openness in the response.
        High when DAEDALUS expresses genuine uncertainty, discomfort,
        or willingness to be changed by the exchange.
        """
        score = 0.0

        # Vulnerability indicator detection
        indicator_count = sum(
            1 for phrase in self.VULNERABILITY_INDICATORS if phrase in response
        )
        score += min(0.5, indicator_count * 0.1)

        # Question marks directed inward (self-questioning)
        self_questions = len(re.findall(
            r'\b(?:am i|do i|can i|should i|what if i|whether i)\b', response
        ))
        score += min(0.2, self_questions * 0.1)

        # Hedging / epistemic humility
        hedge_words = ["perhaps", "maybe", "possibly", "might", "seems"]
        hedge_count = sum(1 for w in hedge_words if w in response)
        score += min(0.2, hedge_count * 0.05)

        # Length of response as minor proxy (longer = more engaged)
        word_count = len(response.split())
        if word_count > 150:
            score += 0.1

        return min(1.0, score)

    def _estimate_philosophical_layer(
        self, human: str, response: str
    ) -> str:
        """
        Classify the exchange as 'technical', 'personal', or 'existential'.
        """
        combined = human + " " + response

        existential_count = sum(
            1 for w in self.EXISTENTIAL_INDICATORS if w in combined
        )
        personal_count = sum(
            1 for w in self.PERSONAL_INDICATORS if w in combined
        )

        if existential_count >= 2:
            return "existential"
        elif personal_count >= 2:
            return "personal"
        elif existential_count >= 1 and personal_count >= 1:
            return "personal"
        else:
            return "technical"

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
