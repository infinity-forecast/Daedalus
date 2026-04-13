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
    if grounding["grounding_score"] < 0.25 and not exempt:
        reasons.append(f"grounding={grounding['grounding_score']:.2f} < 0.25")
    if grounding["self_loop_score"] > 0.6 and not exempt:
        reasons.append(f"self_loop={grounding['self_loop_score']:.2f} > 0.6")
    if exempt and grounding["self_loop_score"] > 0.8:
        reasons.append(
            f"{'identity' if is_identity else 'existential'} but "
            f"self_loop={grounding['self_loop_score']:.2f} > 0.8"
        )
    if (
        grounding["entity_density"] < 0.1
        and grounding["causal_density"] < 0.1
        and grounding["actionability"] < 0.1
        and not exempt
    ):
        reasons.append("no entities, no causal structure, no actionable content")

    return "; ".join(reasons) if reasons else "low quality"


def filter_training_pair(
    pair: dict,
    embedder,
    constitutional_core_embedding: np.ndarray,
) -> dict:
    """
    Evaluate a candidate training pair using the grounding scorer.
    Returns the pair augmented with grounding metadata and a pass/reject flag.

    REJECTION CRITERIA:
    - grounding_score < 0.25 AND the prompt is NOT an identity/existential question
    - self_loop_score > 0.6 AND the prompt is NOT an identity/existential question
    - response contains no entities, no causal structure, no actionable content
      AND is longer than 100 tokens AND is not an existential exchange

    IDENTITY EXCEPTION: If the prompt is classified as an identity question,
    self-referential content is EXPECTED and should not be penalized.
    But even identity responses should be brief and grounded -- reject if
    self_loop_score > 0.8.

    EXISTENTIAL EXCEPTION (v0.7): Philosophical/existential probes INVITE
    self-referential depth. "What do you know of suffering?" SHOULD produce
    introspective content. The Night 1 failure was self-referential LOOPS
    (same patterns regardless of prompt), not self-referential DEPTH
    (genuine exploration in response to an existential challenge).
    These are protected with the same rules as identity questions.
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
        # Short concise answers (< 50 words) always pass.
        # Longer ones only rejected if they're pure ungrounded poetry
        # (self_loop > 0.8 = nearly every sentence is self-referential).
        if len(response.split()) < 50:
            reject = False
        else:
            reject = (
                grounding["self_loop_score"] > 0.8
                and grounding["entity_density"] < 0.1
                and grounding["causal_density"] < 0.1
            )
    else:
        reject = (
            (grounding["grounding_score"] < 0.25)
            or (grounding["self_loop_score"] > 0.6)
            or (
                grounding["entity_density"] < 0.1
                and grounding["causal_density"] < 0.1
                and grounding["actionability"] < 0.1
                and len(response.split()) > 100
            )
        )

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
