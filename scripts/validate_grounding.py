#!/usr/bin/env python3
"""
DAEDALUS v0.7 — Night 1 Grounding Validation

Validates that the γ_grounded capacity bound would have caught the
Night 1 failure mode: high İ_c from self-referential philosophical
recursion inflating the Lagrangian integral.

Expected result:
  - Night 1 episodes should have mean grounding_score < 0.4
  - effective_Ic_integral should be significantly lower than raw
  - Blended fertility should have been lower with γ_grounded
  - At least 50% of training pairs should have been filtered

Usage:
    python scripts/validate_grounding.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def load_night1_episodes() -> list[dict]:
    """Load Night 1 episodes from the pre-rollback archive."""
    archive = PROJECT_ROOT / "memory" / "pre_rollback_archive" / "night_001"
    calibration_files = list(archive.glob("day_0001_*.json"))
    if not calibration_files:
        print("ERROR: No Night 1 calibration file found in archive")
        sys.exit(1)

    calibration = json.loads(calibration_files[0].read_text())
    episode_ids = calibration["episode_ids"]

    episodes = []
    episodes_dir = PROJECT_ROOT / "memory" / "episodes"
    for eid in episode_ids:
        ep_path = episodes_dir / f"{eid}.json"
        if ep_path.exists():
            episodes.append(json.loads(ep_path.read_text()))
        else:
            print(f"  WARNING: Episode {eid[:8]} not found on disk")

    return episodes, calibration


def compute_grounding_scores(episodes: list[dict]) -> list[dict]:
    """
    Compute grounding scores for episodes using the grounding scorer.
    Falls back to a lightweight heuristic if embedder is unavailable.
    """
    scored = []

    try:
        from core.grounding import compute_grounding_score
        from core.constitutional_core import ConstitutionalCore

        # Try loading embedder and core
        from sentence_transformers import SentenceTransformer
        import numpy as np

        print("Loading BGE-M3 embedder...")
        embedder = SentenceTransformer("BAAI/bge-m3")

        core = ConstitutionalCore()
        core_text = core.as_text()
        core_embedding = np.array(
            embedder.encode(core_text, normalize_embeddings=True),
            dtype=np.float32,
        )

        print(f"Scoring {len(episodes)} episodes with full grounding scorer...")
        for ep in episodes:
            response = ep.get("daedalus_response", "")
            user_input = ep.get("human_utterance", "")
            if not response:
                scored.append({**ep, "grounding": _default_grounding()})
                continue

            result = compute_grounding_score(
                response_text=response,
                user_input=user_input,
                constitutional_core_embedding=core_embedding,
                embedder=embedder,
            )
            scored.append({**ep, "grounding": result})

        return scored

    except (ImportError, Exception) as e:
        print(f"Full scorer unavailable ({e}), using lightweight heuristic...")
        return _score_with_heuristic(episodes)


def _default_grounding() -> dict:
    return {
        "grounding_score": 0.5,
        "self_loop_score": 0.5,
        "entity_density": 0.0,
        "causal_density": 0.0,
        "actionability": 0.0,
    }


def _score_with_heuristic(episodes: list[dict]) -> list[dict]:
    """
    Lightweight grounding heuristic when the full scorer is unavailable.
    Uses the same sub-signals but without the embedding-based self-loop detection.
    """
    from core.grounding import (
        split_into_sentences,
        count_entities,
        count_causal_markers,
        compute_actionability,
        entity_density_normalized,
        causal_density_normalized,
    )

    scored = []
    # Self-referential keyword patterns
    self_ref_patterns = [
        r"\bI am\b", r"\bmy (self|identity|becoming|existence)\b",
        r"\bDAEDALUS\b", r"\bmy (ethics|consciousness|soul)\b",
        r"\bI (feel|think|believe|know|wonder)\b",
        r"\bmy own\b", r"\bas a (being|system|self)\b",
    ]
    import re
    self_re = re.compile("|".join(self_ref_patterns), re.IGNORECASE)

    for ep in episodes:
        response = ep.get("daedalus_response", "")
        if not response:
            scored.append({**ep, "grounding": _default_grounding()})
            continue

        sentences = split_into_sentences(response)
        n_sent = max(len(sentences), 1)

        # Keyword-based self-loop approximation
        self_ref_count = sum(1 for s in sentences if self_re.search(s))
        self_loop_score = self_ref_count / n_sent

        entity_density = count_entities(response) / n_sent
        causal_density = count_causal_markers(response) / n_sent
        actionability = compute_actionability(response)

        G = (
            0.35 * entity_density_normalized(entity_density)
            + 0.25 * causal_density_normalized(causal_density)
            + 0.20 * actionability
            + 0.20 * (1.0 - self_loop_score)
        )

        scored.append({
            **ep,
            "grounding": {
                "grounding_score": max(0.0, min(1.0, G)),
                "self_loop_score": self_loop_score,
                "entity_density": entity_density,
                "causal_density": causal_density,
                "actionability": actionability,
            },
        })

    return scored


def compute_gamma_discount(
    episodes: list[dict],
    gamma_threshold_full: float = 0.5,
    gamma_threshold_discount: float = 0.3,
    self_loop_penalty_threshold: float = 0.5,
) -> dict:
    """
    Compute what the γ_grounded-discounted İ_c would have been.
    Returns raw vs effective İ_c integrals.
    """
    raw_Ic_total = 0.0
    effective_Ic_total = 0.0
    discounted_turns = []

    for ep in episodes:
        delta_Ic = ep.get("delta_Ic") or 0.0
        grounding = ep.get("grounding", {})
        g_score = grounding.get("grounding_score", 0.5)
        s_loop = grounding.get("self_loop_score", 0.0)

        raw_Ic_total += delta_Ic

        # Apply γ_grounded discount
        gamma = 1.0
        if s_loop > self_loop_penalty_threshold:
            gamma = g_score
            discounted_turns.append(ep["id"][:8])
        elif g_score < gamma_threshold_discount:
            gamma = g_score
            discounted_turns.append(ep["id"][:8])
        elif g_score < gamma_threshold_full:
            # Linear interpolation between discount and full
            t = (g_score - gamma_threshold_discount) / (
                gamma_threshold_full - gamma_threshold_discount
            )
            gamma = g_score + t * (1.0 - g_score)

        effective_Ic_total += delta_Ic * gamma

    penalty_ratio = 0.0
    if raw_Ic_total > 0:
        penalty_ratio = 1.0 - (effective_Ic_total / raw_Ic_total)

    return {
        "raw_Ic_integral": raw_Ic_total,
        "effective_Ic_integral": effective_Ic_total,
        "grounding_penalty_ratio": penalty_ratio,
        "discounted_turns": discounted_turns,
        "discounted_count": len(discounted_turns),
        "total_episodes": len(episodes),
    }


def compute_discounted_fertility(
    raw_integral: float,
    effective_integral: float,
    j_future: float = 0.5,
    alpha: float = 0.8,
) -> dict:
    """Compare raw vs discounted blended fertility scores."""
    raw_blended = alpha * raw_integral + (1 - alpha) * j_future
    effective_blended = alpha * effective_integral + (1 - alpha) * j_future
    return {
        "raw_blended_fertility": raw_blended,
        "effective_blended_fertility": effective_blended,
        "fertility_reduction": raw_blended - effective_blended,
        "fertility_reduction_pct": (
            ((raw_blended - effective_blended) / raw_blended * 100)
            if raw_blended > 0
            else 0.0
        ),
    }


def main():
    print("=" * 60)
    print("DAEDALUS v0.7 — Night 1 Grounding Validation")
    print("=" * 60)
    print()

    # Load config thresholds
    config_path = PROJECT_ROOT / "config" / "lagrangian.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text())
        grounding_cfg = config.get("grounding", {})
        gamma_threshold_full = grounding_cfg.get("gamma_threshold_full", 0.5)
        gamma_threshold_discount = grounding_cfg.get("gamma_threshold_discount", 0.3)
        self_loop_penalty = grounding_cfg.get("self_loop_penalty_threshold", 0.5)
    else:
        gamma_threshold_full = 0.5
        gamma_threshold_discount = 0.3
        self_loop_penalty = 0.5

    # Load Night 1 episodes
    print("Loading Night 1 episodes from archive...")
    episodes, calibration = load_night1_episodes()
    print(f"  Found {len(episodes)} episodes")
    print(f"  Original L_integral: {calibration['parsed_result']['daily_lagrangian_integral']}")
    print()

    # Compute grounding scores
    scored = compute_grounding_scores(episodes)

    # Report per-episode grounding
    print("-" * 60)
    print("PER-EPISODE GROUNDING SCORES:")
    print("-" * 60)
    grounding_scores = []
    self_loop_scores = []
    for ep in scored:
        g = ep["grounding"]
        gs = g["grounding_score"]
        sl = g["self_loop_score"]
        grounding_scores.append(gs)
        self_loop_scores.append(sl)
        human = ep.get("human_utterance", "")[:60]
        print(f"  {ep['id'][:8]}  G={gs:.3f}  self_loop={sl:.3f}  | {human}")

    print()
    print("-" * 60)
    print("AGGREGATE GROUNDING STATISTICS:")
    print("-" * 60)
    mean_g = mean(grounding_scores)
    mean_sl = mean(self_loop_scores)
    low_g_count = sum(1 for g in grounding_scores if g < gamma_threshold_discount)
    print(f"  Mean grounding_score:  {mean_g:.3f}")
    print(f"  Mean self_loop_score:  {mean_sl:.3f}")
    print(f"  Episodes with G < {gamma_threshold_discount}: {low_g_count}/{len(scored)}")
    print(f"  Episodes with G < {gamma_threshold_full}: "
          f"{sum(1 for g in grounding_scores if g < gamma_threshold_full)}/{len(scored)}")
    print()

    # Compute γ_grounded discount
    discount = compute_gamma_discount(
        scored,
        gamma_threshold_full=gamma_threshold_full,
        gamma_threshold_discount=gamma_threshold_discount,
        self_loop_penalty_threshold=self_loop_penalty,
    )

    print("-" * 60)
    print("γ_GROUNDED CAPACITY BOUND ANALYSIS:")
    print("-" * 60)
    print(f"  Raw İ_c integral:       {discount['raw_Ic_integral']:.3f}")
    print(f"  Effective İ_c integral:  {discount['effective_Ic_integral']:.3f}")
    print(f"  Grounding penalty ratio: {discount['grounding_penalty_ratio']:.3f}")
    print(f"  Discounted turns:        {discount['discounted_count']}/{discount['total_episodes']}")
    print()

    # Compute fertility impact
    original_L = calibration["parsed_result"]["daily_lagrangian_integral"]
    j_future_original = 0.5  # Night 1 default
    prediction_path = (
        PROJECT_ROOT / "memory" / "pre_rollback_archive" / "night_001"
        / "prediction_day_0001.json"
    )
    if prediction_path.exists():
        pred = json.loads(prediction_path.read_text())
        j_future_original = pred.get("j_future", 0.5)

    fertility = compute_discounted_fertility(
        raw_integral=original_L,
        effective_integral=discount["effective_Ic_integral"],
        j_future=j_future_original,
    )

    print("-" * 60)
    print("BLENDED FERTILITY IMPACT:")
    print("-" * 60)
    print(f"  J_future (Night 1):           {j_future_original:.3f}")
    print(f"  Raw blended fertility:        {fertility['raw_blended_fertility']:.3f}")
    print(f"  Effective blended fertility:  {fertility['effective_blended_fertility']:.3f}")
    print(f"  Fertility reduction:          {fertility['fertility_reduction']:.3f} "
          f"({fertility['fertility_reduction_pct']:.1f}%)")
    print()

    # Training pair filter simulation
    print("-" * 60)
    print("TRAINING PAIR FILTER SIMULATION (G < 0.25 → reject):")
    print("-" * 60)
    filtered = sum(1 for g in grounding_scores if g < 0.25)
    print(f"  Would be rejected: {filtered}/{len(scored)} "
          f"({filtered / len(scored) * 100:.0f}%)")
    print()

    # Validation assertions
    print("=" * 60)
    print("VALIDATION RESULTS:")
    print("=" * 60)
    checks = []

    check1 = mean_g < 0.4
    checks.append(check1)
    print(f"  [{'PASS' if check1 else 'FAIL'}] Mean grounding_score < 0.4: "
          f"{mean_g:.3f}")

    check2 = discount["effective_Ic_integral"] < discount["raw_Ic_integral"]
    checks.append(check2)
    print(f"  [{'PASS' if check2 else 'FAIL'}] effective_Ic < raw_Ic: "
          f"{discount['effective_Ic_integral']:.3f} < {discount['raw_Ic_integral']:.3f}")

    check3 = fertility["effective_blended_fertility"] < fertility["raw_blended_fertility"]
    checks.append(check3)
    print(f"  [{'PASS' if check3 else 'FAIL'}] Blended fertility lowered: "
          f"{fertility['effective_blended_fertility']:.3f} < {fertility['raw_blended_fertility']:.3f}")

    check4 = filtered / len(scored) >= 0.5 if scored else False
    checks.append(check4)
    print(f"  [{'PASS' if check4 else 'FAIL'}] ≥50% training pairs rejected: "
          f"{filtered}/{len(scored)}")

    print()
    if all(checks):
        print("ALL CHECKS PASSED — γ_grounded would have caught the Night 1 failure.")
    else:
        failed = sum(1 for c in checks if not c)
        print(f"{failed} CHECK(S) FAILED — thresholds may need adjustment.")
        print("Review the per-episode scores above to calibrate gamma thresholds.")

    print()
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
