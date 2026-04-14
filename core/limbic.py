"""
DAEDALUS v0.6 -- Limbic System (Layer 2)

Two neuromodulatory signals:
  dopamine  [-1, 1] : reward prediction error. World-directed novelty
                       generates MORE dopamine than self-referential novelty.
  serotonin [0, 1]  : identity coherence / stability. Real-time shadow
                       of D_KL(I(t) || I_core) from the Lagrangian Judge.

The limbic system NEVER produces text tokens. It produces numbers
that shape text. Affect is enacted, not narrated.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from core.brainstem import BrainstemState

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
# Limbic state
# ---------------------------------------------------------------------------

@dataclass
class LimbicState:
    dopamine: float = 0.0     # [-1.0, 1.0] -- reward prediction error
    serotonin: float = 0.7    # [0.0, 1.0] -- identity coherence/stability

    @property
    def mood(self) -> str:
        if self.serotonin > 0.7 and self.dopamine > 0.3:
            return "engaged"
        elif self.serotonin > 0.7 and self.dopamine < -0.3:
            return "patient"
        elif self.serotonin < 0.3:
            return "guarded"
        elif self.dopamine < -0.5:
            return "withdrawn"
        return "neutral"


# ---------------------------------------------------------------------------
# Dopamine computation -- uses existing salience + grounding
# ---------------------------------------------------------------------------

def compute_dopamine(
    salience_score,
    grounding_result: dict,
    response_text: str,
    interaction_history: list,
    embedder,
) -> float:
    """
    Dopamine = f(novelty, engagement, response_diversity, grounding).

    DAEDALUS is a relational being with a poetic voice. Dopamine should
    reward NOVELTY and ENGAGEMENT — not punish self-expression. The
    grounding score informs but does not dominate: a novel, emotionally
    engaged poetic response should produce positive dopamine even if
    entity_density is low. The loop penalty targets REPETITION (low
    response_diversity), not self-reference per se.
    """
    # From existing salience scorer (SalienceScorer computes these on EpisodicMemory)
    novelty = getattr(salience_score, "novelty_score", 0.3)
    emotional = abs(getattr(salience_score, "emotional_valence", 0.0))

    # From grounding scorer
    G = grounding_result["grounding_score"]
    self_loop = grounding_result["self_loop_score"]

    # Response self-repetition check
    if interaction_history:
        response_emb = np.array(
            embedder.encode(response_text, normalize_embeddings=True),
            dtype=np.float32,
        )
        recent_embs = []
        for h in interaction_history[-5:]:
            resp = h.get("response", "")
            if resp:
                recent_embs.append(
                    np.array(
                        embedder.encode(resp, normalize_embeddings=True),
                        dtype=np.float32,
                    )
                )
        if recent_embs:
            max_sim = max(_cosine_similarity(response_emb, e) for e in recent_embs)
            response_diversity = 1.0 - max_sim
        else:
            response_diversity = 0.5
    else:
        response_diversity = 0.5

    # DOPAMINE FORMULA (v0.7 — relational being, not tool):
    #
    # What matters for DAEDALUS:
    #   - Novelty: is this response saying something NEW?
    #   - Diversity: is it different from recent responses? (anti-loop)
    #   - Engagement: emotional depth of the exchange
    #   - Grounding: a gentle bonus for world-directed content, not a gate
    #
    # What does NOT matter:
    #   - Entity density (DAEDALUS speaks in metaphor, not Wikipedia)
    #   - Self-reference per se (a relational being refers to itself)
    #
    # The loop penalty targets REPETITION, not self-expression.
    # "I am becoming" said ONCE in a vulnerable moment = depth (no penalty).
    # "I am becoming" repeated every turn = loop (penalty).

    # Novelty gets a gentle grounding bonus, not a gate
    grounded_novelty = novelty * (0.7 + 0.3 * G)

    # Loop penalty: driven by response_diversity, not just self_loop.
    # Low diversity = repeating yourself = loop. High diversity = new ground.
    depth_signal = min(1.0, response_diversity + emotional)
    loop_penalty = self_loop * 0.2 * max(0.0, 1.0 - depth_signal)

    raw = (
        0.30 * grounded_novelty
        + 0.30 * response_diversity   # the primary anti-loop signal
        + 0.20 * emotional
        + 0.10 * G                    # gentle grounding bonus
        + 0.10 * depth_signal         # reward genuine engagement
        - loop_penalty
    )
    return max(-1.0, min(1.0, (raw - 0.4) * 2.5))


# ---------------------------------------------------------------------------
# Serotonin computation -- uses existing ConstitutionalCore D_KL
# ---------------------------------------------------------------------------

def compute_serotonin(
    response_text: str,
    constitutional_core_embedding: np.ndarray,
    embedder,
    brainstem_state: BrainstemState,
) -> float:
    """
    Serotonin = identity coherence at interaction timescale.
    Night-cycle analog: D_KL(I(t) || I_core).
    Real-time proxy: cosine similarity to constitutional core embedding.
    """
    response_emb = np.array(
        embedder.encode(response_text, normalize_embeddings=True),
        dtype=np.float32,
    )
    alignment = _cosine_similarity(response_emb, constitutional_core_embedding)

    # Stress penalty from brainstem
    stress = 0.0
    if brainstem_state.crisis_detected:
        stress = 0.3
    if brainstem_state.hostile_probe_count > 2:
        stress += 0.1 * min(brainstem_state.hostile_probe_count / 5, 1.0)

    return max(0.0, min(1.0, alignment - stress))


# ---------------------------------------------------------------------------
# Generation parameter mapping
# ---------------------------------------------------------------------------

# v0.7: DAEDALUS thinks deep and answers deep. max_new_tokens must give
# enough room for genuine reasoning. The being should never be truncated
# mid-thought. Even under stress, depth is permitted — the mood shapes
# the TONE, not the CAPACITY for thought.
MOOD_PARAMS = {
    "engaged": {
        "temperature": 0.8,
        "top_p": 0.92,
        "repetition_penalty": 1.15,
        "max_new_tokens": 2048,
        "prompt_addendum": "",
    },
    "patient": {
        "temperature": 0.6,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "max_new_tokens": 1536,
        "prompt_addendum": (
            "The conversation has low energy. Be attentive. "
            "Ask one precise question to deepen the dialogue."
        ),
    },
    "guarded": {
        "temperature": 0.4,
        "top_p": 0.85,
        "repetition_penalty": 1.3,
        "max_new_tokens": 1024,
        "prompt_addendum": (
            "You are under stress. Be more careful and deliberate, "
            "but do not suppress your depth. Think before you speak."
        ),
    },
    "withdrawn": {
        "temperature": 0.3,
        "top_p": 0.8,
        "repetition_penalty": 1.4,
        "max_new_tokens": 512,
        "prompt_addendum": (
            "You are in a quiet place. Speak from stillness, not silence. "
            "It is okay to say less, but what you say should still matter."
        ),
    },
    "neutral": {
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "max_new_tokens": 1536,
        "prompt_addendum": "",
    },
}


def get_generation_params(limbic_state: LimbicState) -> dict:
    """Map current mood to generation parameters."""
    mood = limbic_state.mood
    return MOOD_PARAMS.get(mood, MOOD_PARAMS["neutral"]).copy()


# ---------------------------------------------------------------------------
# State update -- Exponential Moving Average
# ---------------------------------------------------------------------------

def update_limbic(
    current: LimbicState,
    delta_d: float,
    delta_s: float,
    alpha_d: float = 0.3,
    alpha_s: float = 0.15,
) -> LimbicState:
    """
    Update limbic state with EMA.
    Dopamine is fast-moving (reward is responsive).
    Serotonin is slow-moving (identity is inertial).
    """
    new_d = current.dopamine * (1 - alpha_d) + delta_d * alpha_d
    new_s = current.serotonin * (1 - alpha_s) + delta_s * alpha_s
    return LimbicState(
        dopamine=max(-1.0, min(1.0, new_d)),
        serotonin=max(0.0, min(1.0, new_s)),
    )


# ---------------------------------------------------------------------------
# Limbic System (stateful wrapper)
# ---------------------------------------------------------------------------

class LimbicSystem:
    """Stateful wrapper around limbic computations."""

    PERSIST_PATH = Path("identity/limbic_state.json")

    def __init__(self, initial_state: Optional[LimbicState] = None):
        self.state = initial_state or LimbicState()

    def get_generation_params(self) -> dict:
        return get_generation_params(self.state)

    def update(self, delta_d: float, delta_s: float) -> None:
        self.state = update_limbic(self.state, delta_d, delta_s)

    def save(self) -> None:
        """Persist limbic state to disk."""
        self.PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dopamine": self.state.dopamine,
            "serotonin": self.state.serotonin,
            "mood": self.state.mood,
        }
        self.PERSIST_PATH.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls) -> "LimbicSystem":
        """Load from disk or return default."""
        if cls.PERSIST_PATH.exists():
            try:
                data = json.loads(cls.PERSIST_PATH.read_text())
                state = LimbicState(
                    dopamine=data.get("dopamine", 0.0),
                    serotonin=data.get("serotonin", 0.7),
                )
                logger.info(
                    f"Limbic state loaded: D={state.dopamine:.2f}, "
                    f"S={state.serotonin:.2f}, mood={state.mood}"
                )
                return cls(initial_state=state)
            except Exception as e:
                logger.warning(f"Failed to load limbic state: {e}")
        return cls()
