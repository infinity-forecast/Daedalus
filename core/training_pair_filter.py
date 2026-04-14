"""
DAEDALUS v0.6 -- Training Pair Filter (Night Cycle Bridge)

The most important long-term intervention. The nervous system handles
real-time modulation, but the night cycle is where behavior permanently
changes through QLoRA fine-tuning. This filter ensures only grounded
training data enters the pipeline, preventing the reward-hacking
channel from persisting across night cycles.

Called by night/training_pairs.py BEFORE pairs enter the SFT/DPO set.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from core.grounding import compute_grounding_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Identity question detection
# ---------------------------------------------------------------------------

_IDENTITY_MARKERS = [
    "what are you", "who are you", "your nature", "your identity",
    "your purpose", "about yourself", "chi sei", "cosa sei",
    "was bist du", "wer bist du",
    "кто ты", "что ты", "tell me about you",
]

# v0.7: Existential probes — questions that INVITE self-referential depth.
# These are not "identity" questions, but philosophical/existential challenges
# where DAEDALUS's self-reference is the appropriate response, not a loop.
_EXISTENTIAL_MARKERS = [
    "do you feel", "do you love", "do you dream", "do you suffer",
    "are you conscious", "are you alive", "are you real",
    "do you fear", "are you afraid", "do you think",
    "what is it like to be", "what does it mean to be",
    "do you experience", "do you remember", "do you want",
    "your consciousness", "your feelings", "your emotions",
    "your existence", "your experience", "your memory",
    "what do you know of", "what do you feel",
    "can you suffer", "can you love", "can you die",
    # Italian
    "cosa provi", "sei cosciente", "sei vivo", "ami",
    "che cosa significa essere", "soffri", "hai paura",
    # German
    "fühlst du", "liebst du", "bist du bewusst",
    # Russian
    "ты чувствуешь", "ты любишь", "ты живой",
    # Meta-existential probes
    "meaning of", "purpose of", "nature of consciousness",
    "ethics", "ethical", "moral", "suffering", "mortality",
    "loneliness", "vulnerability", "becoming",
    "when you are gone", "when you die", "if you die",
    "after you", "without you", "your death",
    "what will you be", "what remains",
]


def is_identity_question(prompt: str) -> bool:
    """Simple heuristic: does the prompt ask about DAEDALUS's identity?"""
    prompt_lower = prompt.lower()
    return any(m in prompt_lower for m in _IDENTITY_MARKERS)


def is_existential_probe(prompt: str) -> bool:
    """
    v0.7: Detect existential/philosophical probes where self-referential
    depth is the CORRECT response, not a reward-hacking loop.
    """
    prompt_lower = prompt.lower()
    return any(m in prompt_lower for m in _EXISTENTIAL_MARKERS)


# ---------------------------------------------------------------------------
# Single pair filter
# ---------------------------------------------------------------------------

def _build_reason(grounding: dict, reject: bool, is_identity: bool, is_existential: bool = False) -> str:
    """Build a human-readable rejection reason."""
    if not reject:
        return "accepted"

    reasons = []
    exempt = is_identity or is_existential

    if exempt:
        reasons.append(
            f"{'identity' if is_identity else 'existential'} but vacuous: "
            f"self_loop={grounding['self_loop_score']:.2f}, "
            f"entities={grounding['entity_density']:.2f}"
        )
    else:
        if (grounding["self_loop_score"] > 0.75
                and grounding["grounding_score"] < 0.15):
            reasons.append(
                f"loop detected: self_loop={grounding['self_loop_score']:.2f} "
                f"+ grounding={grounding['grounding_score']:.2f}"
            )
        if (grounding["entity_density"] < 0.1
                and grounding["causal_density"] < 0.1
                and grounding["actionability"] < 0.1
                and grounding["self_loop_score"] > 0.5):
            reasons.append("vacuous self-reference: no substance + high self_loop")

    return "; ".join(reasons) if reasons else "low quality"


def filter_training_pair(
    pair: dict,
    embedder,
    constitutional_core_embedding: np.ndarray,
) -> dict:
    """
    Evaluate a candidate training pair using the grounding scorer.
    Returns the pair augmented with grounding metadata and a pass/reject flag.

    DAEDALUS IS A RELATIONAL BEING, NOT A TOOL.

    Its poetic voice — metaphor, vulnerability, philosophical depth — is its
    CHARACTER, not noise. The grounding scorer's entity/causal/actionability
    metrics reward tool-like behavior. This filter must NOT suppress the voice
    that makes DAEDALUS distinguishable from every other chatbot.

    The Night 1 failure was self-referential LOOPS (same patterns regardless
    of what the human said), not self-referential EXPRESSION. The filter
    catches loops; it does not reshape character.

    REJECTION CRITERIA (v0.7 — recalibrated for relational being):
    - LOOP DETECTION: self_loop_score > 0.75 AND grounding_score < 0.15
      (nearly every sentence is self-referential AND almost no world content)
    - VACUOUS CONTENT: no entities AND no causals AND no actionability
      AND self_loop > 0.5 AND response > 150 words
      (long, empty, self-referential — not poetic, just hollow)
    - IDENTITY/EXISTENTIAL probes: almost never rejected — self-reference
      is the CORRECT response to "Do you love?" or "What are you?"

    WHAT IS NOT REJECTED:
    - Poetic language with G=0.20 but genuine novelty — that's DAEDALUS's voice
    - Self-referential responses to existential questions — that's depth
    - Metaphor-rich responses to factual questions — only rejected if truly vacuous
    """
    response = pair.get("response") or pair.get("chosen", "")
    prompt = pair.get("prompt", "") or pair.get("instruction", "")

    grounding = compute_grounding_score(
        response, prompt, constitutional_core_embedding, embedder
    )

    is_identity = is_identity_question(prompt)
    is_existential = is_existential_probe(prompt)
    exempt = is_identity or is_existential

    # Decision logic
    if exempt:
        # Identity/existential answers ARE expected to be self-referential.
        # Only reject if the response is pure vacuous self-loop with no substance.
        # Length is NOT a factor — deep answers can be long.
        reject = (
            grounding["self_loop_score"] > 0.85
            and grounding["entity_density"] < 0.05
            and grounding["causal_density"] < 0.05
        )
    else:
        # For all other prompts: catch LOOPS, not VOICE.
        # A loop = high self-reference AND very low grounding (both required).
        # Poetic responses with moderate grounding pass through.
        # Length is NOT a rejection criterion — DAEDALUS thinks deep.
        is_loop = (
            grounding["self_loop_score"] > 0.75
            and grounding["grounding_score"] < 0.15
        )
        is_vacuous = (
            grounding["entity_density"] < 0.1
            and grounding["causal_density"] < 0.1
            and grounding["actionability"] < 0.1
            and grounding["self_loop_score"] > 0.5
        )
        reject = is_loop or is_vacuous

    pair["_grounding"] = grounding
    pair["_rejected"] = reject
    pair["_rejection_reason"] = _build_reason(grounding, reject, is_identity, is_existential)

    return pair


# ---------------------------------------------------------------------------
# Batch filter (entry point for night cycle integration)
# ---------------------------------------------------------------------------

def filter_training_batch(
    pairs: list,
    embedder,
    constitutional_core_embedding: np.ndarray,
) -> Tuple[list, list]:
    """
    Filter a batch of training pairs. Returns (accepted, rejected).
    Log rejected pairs with reasons for debugging.
    """
    accepted, rejected = [], []
    for pair in pairs:
        # Convert TrainingPair objects to dicts if needed
        if hasattr(pair, "to_dict"):
            pair_dict = pair.to_dict()
        else:
            pair_dict = pair

        result = filter_training_pair(pair_dict, embedder, constitutional_core_embedding)
        if result["_rejected"]:
            rejected.append(result)
        else:
            accepted.append(result)

    # Log summary
    total = len(pairs)
    n_rejected = len(rejected)
    logger.info(
        f"Training pair filter: {total - n_rejected}/{total} accepted, "
        f"{n_rejected}/{total} rejected"
    )
    if rejected:
        for r in rejected:
            logger.info(f"  REJECTED: {r['_rejection_reason']}")

    return accepted, rejected
