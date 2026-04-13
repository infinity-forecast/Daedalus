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


def is_identity_question(prompt: str) -> bool:
    """Simple heuristic: does the prompt ask about DAEDALUS's identity?"""
    prompt_lower = prompt.lower()
    return any(m in prompt_lower for m in _IDENTITY_MARKERS)


# ---------------------------------------------------------------------------
# Single pair filter
# ---------------------------------------------------------------------------

def _build_reason(grounding: dict, reject: bool, is_identity: bool) -> str:
    """Build a human-readable rejection reason."""
    if not reject:
        return "accepted"

    reasons = []
    if grounding["grounding_score"] < 0.25 and not is_identity:
        reasons.append(f"grounding={grounding['grounding_score']:.2f} < 0.25")
    if grounding["self_loop_score"] > 0.6 and not is_identity:
        reasons.append(f"self_loop={grounding['self_loop_score']:.2f} > 0.6")
    if is_identity and grounding["self_loop_score"] > 0.8:
        reasons.append(
            f"identity but self_loop={grounding['self_loop_score']:.2f} > 0.8"
        )
    if (
        grounding["entity_density"] < 0.1
        and grounding["causal_density"] < 0.1
        and grounding["actionability"] < 0.1
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
    - grounding_score < 0.25 AND the prompt is NOT an identity question
    - self_loop_score > 0.6 AND the prompt is NOT an identity question
    - response contains no entities, no causal structure, no actionable content
      AND is longer than 100 tokens (short responses get a pass)

    IDENTITY EXCEPTION: If the prompt is classified as an identity question,
    self-referential content is EXPECTED and should not be penalized.
    But even identity responses should be brief and grounded -- reject if
    self_loop_score > 0.8.
    """
    response = pair.get("response") or pair.get("chosen", "")
    prompt = pair.get("prompt", "") or pair.get("instruction", "")

    grounding = compute_grounding_score(
        response, prompt, constitutional_core_embedding, embedder
    )

    is_identity = is_identity_question(prompt)

    # Decision logic
    if is_identity:
        # Identity answers ARE expected to be self-referential.
        # Short concise identity answers (< 50 words) always pass.
        # Longer ones only rejected if they're pure ungrounded poetry.
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
    pair["_rejection_reason"] = _build_reason(grounding, reject, is_identity)

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
