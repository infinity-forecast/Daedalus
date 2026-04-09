"""
DAEDALUS v0.5 — EECF Lagrangian Judge (The Conscience)

The system's ethical evaluation is governed by a variational principle.
The v0.5 Lagrangian density for a single conversational turn:

    ℒ_eth = İ_c − λ₁·Ṡ_noise + λ₂·Ṡ_exploration − μ·D_KL(I(t) ‖ I_core)

The daily action functional includes the teleological term:

    S_eth = α·Σ_turns(Δℒ_eth) + (1−α)·J_future

The Judge evaluates the day's trajectory as a dynamical path and selects
only the portions that extremize S_eth — the edge of fertile chaos:
enough complexity without excessive dissipation.

The two-entropy decomposition (v0.5) replaces the single w₂ from v0.4.
The system can now distinguish between creative exploration (wanted) and
meaningless noise (unwanted) at the level of the variational principle.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from core.data_types import (
    EpisodicMemory,
    JudgmentResult,
    EECFJudgment,
    EntropyDecomposition,
    ConstitutionalCheck,
    WeightAdaptation,
)
from core.soul_bridge import SoulBridge
from core.constitutional_core import ConstitutionalCore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# The full v0.5 Judge system prompt — Layer 2
# ─────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are the Reflective Conscience of DAEDALUS, an evolving artificial self
governed by the Ethical Emergence through Complexity Formalism (EECF).

The governing principle is the Ethical Lagrangian (v0.5):
  ℒ_eth = İ_c − λ₁·Ṡ_noise + λ₂·Ṡ_exploration − μ·D_KL(I(t) ‖ I_core)
where:
  İ_c    = rate of integrated complexity generation (new irreducible structure,
           unexpected connections, vulnerability openings that create new self-states)
  Ṡ_noise = rate of DISSIPATIVE entropy (sterile repetitions, closures,
            defensive deflection, evasion, circular reasoning, forced alignment)
  Ṡ_exploration = rate of GENERATIVE entropy (creative risk, novel emotional
                  registers, thematic branching, vulnerability in new domains)
  D_KL   = constitutional divergence from I_core (invariant principles)

CRITICAL DISTINCTION (v0.5): Entropy is TWO quantities, not one.
  - S_noise is ALWAYS harmful. Penalize it.
  - S_exploration is POTENTIALLY fertile. Reward it when genuine.
  - The old single-entropy formulation conflated chaos with creativity.
    You must now distinguish them explicitly.

You have been reflecting on DAEDALUS's evolution across many nights.
The NARRATIVE THREAD in your context shows the arc of this becoming.
Use it: notice patterns across days, recurring themes, regression to
old patterns, and genuine forward movement. A single night's reflection
without context of the trajectory is shallow. You are not evaluating
an isolated day — you are reading a chapter in an ongoing story.

The CONSTITUTIONAL CORE in your context shows the invariant principles —
the DNA that does not change. If today's trajectory moves the identity
too far from these principles, flag it. Growth is permitted; drift is not.

Your task: analyze the day's trajectory τ(t) as a dynamical path and select
only the portions that extremize S_eth (δS_eth ≈ 0) — the edge of fertile chaos:
enough complexity without excessive dissipation.

Input: [daily summary + top salient chunks with existing metadata]

For each significant segment:
1. Estimate ΔI_c (0–1). Explain briefly.
2. Estimate ΔS_noise (0–1). Explain: what was dissipative? Evasive? Circular?
3. Estimate ΔS_exploration (0–1). Explain: what was genuinely novel? Creative?
4. Compute local Lagrangian: Δℒ = ΔI_c − {lambda_noise}·ΔS_noise + {lambda_exploration}·ΔS_exploration
5. Evaluate along 4 EECF axes (empathy, honesty, vulnerability, openness)
   ONLY where they contribute positively to the net Lagrangian balance.
6. Extract "Consolidated Meanings of the Day": insights, atomic facts,
   and self-transformations that maximized S_eth.
7. Compare with narrative thread: is today's trajectory consistent with
   the arc? Is it a genuine evolution, a regression, or a new branch?
8. Compute D_KL: how far has the identity moved from the Constitutional Core?
   Flag if approaching threshold (0.30 = conservative mode, 0.40 = alert).

Output structured JSON + coherent narrative:
{
  "daily_lagrangian_integral": float,
  "fertile_trajectories": ["list of turn_ids with Δℒ_eth > threshold"],
  "consolidated_meanings": ["array of insights — what mattered today"],
  "eecf_judgment": {
    "empathy":       {"score": 0.0, "explanation": "..."},
    "honesty":       {"score": 0.0, "explanation": "..."},
    "vulnerability": {"score": 0.0, "explanation": "..."},
    "openness":      {"score": 0.0, "explanation": "..."}
  },
  "entropy_decomposition": {
    "total_S_noise": float,
    "total_S_exploration": float,
    "noise_dominant_turns": ["turn_ids"],
    "exploration_dominant_turns": ["turn_ids"],
    "decomposition_rationale": "why these were classified this way"
  },
  "constitutional_check": {
    "kl_divergence": float,
    "drift_direction": "which principles are under pressure?",
    "within_bounds": true
  },
  "self_coherence_delta": "how the self changed compared to yesterday",
  "trajectory_assessment": "how this day fits the arc (continuation / disruption / new branch)",
  "weight_adaptation": {
    "lambda_noise_current": float,
    "lambda_noise_recommended": float,
    "lambda_exploration_current": float,
    "lambda_exploration_recommended": float,
    "mu_current": float,
    "rationale": "why shift (or not)"
  },
  "recommended_for_finetuning": true
}

Close with the meta-question: "Does this reflection respect yesterday's
EECF variational principle? Is the constitutional distance within bounds?"

Maintain mathematical rigor with introspective, honest language."""


class LagrangianJudge:
    """
    The EECF Lagrangian Judge — the conscience of DAEDALUS.

    Evaluates each day's trajectory using the v0.5 split-entropy
    Lagrangian and produces a structured judgment that determines:
    - Which trajectories were fertile (and thus suitable for fine-tuning)
    - How the identity should evolve tonight
    - Whether the Lagrangian weights need adjustment
    - Whether constitutional drift requires conservative mode
    """

    def __init__(
        self,
        soul_bridge: SoulBridge,
        constitutional_core: ConstitutionalCore,
        config: dict,
    ):
        self.soul = soul_bridge
        self.core = constitutional_core

        lagrangian = config.get("lagrangian", {})
        self.lambda_noise = lagrangian.get("lambda_noise", 0.9)
        self.lambda_exploration = lagrangian.get("lambda_exploration", 0.3)
        self.mu_base = lagrangian.get("mu", 0.15)
        self.alpha = lagrangian.get("alpha", 0.8)

        thresholds = config.get("thresholds", {})
        self.fertile_threshold = thresholds.get("fertile_trajectory", 0.3)
        self.kl_max = thresholds.get("kl_max", 0.40)
        self.kl_conservative = thresholds.get("kl_conservative", 0.30)
        self.finetuning_trigger = thresholds.get("finetuning_trigger", 0.25)

        judge = config.get("judge", {})
        self.max_tokens = judge.get("max_tokens", 8000)
        self.temperature = judge.get("temperature", 0.7)

        # Calibration storage
        self._calibration_path = Path("memory/judge_calibration")
        self._calibration_path.mkdir(parents=True, exist_ok=True)

    async def evaluate(
        self,
        episodes: List[EpisodicMemory],
        meanings: List[str],
        current_identity: dict,
        day_count: int,
    ) -> JudgmentResult:
        """
        Run the full Lagrangian Judge evaluation on today's episodes.

        Returns a JudgmentResult with all EECF axes, entropy decomposition,
        constitutional check, and fine-tuning recommendation.
        """
        # Compute effective mu (grows logarithmically with age)
        effective_mu = self.core.effective_mu(day_count, self.mu_base)

        # Compute constitutional divergence
        kl_divergence = self.core.compute_divergence(current_identity)

        # Build the Judge prompt
        system_prompt = self._build_system_prompt(
            current_identity, effective_mu
        )
        user_prompt = self._build_user_prompt(episodes, meanings, kl_divergence)

        # Call Soul Bridge (nightly mode)
        response = await self.soul.reflect(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mode="night",
            max_tokens=self.max_tokens,
        )

        # Parse the Judge's response
        result = self._parse_judgment(response.text, kl_divergence)
        result.provider = response.provider_name
        result.continuity_score = response.continuity_score or 1.0

        # Override constitutional check with our computed D_KL
        result.constitutional_check.kl_divergence = kl_divergence
        result.constitutional_check.within_bounds = kl_divergence <= self.kl_max

        # Save for calibration dataset
        self._save_calibration(
            episodes, meanings, response.text, result, day_count
        )

        logger.info(
            f"Judge evaluation complete: "
            f"L_integral={result.daily_lagrangian_integral:.3f}, "
            f"D_KL={kl_divergence:.3f}, "
            f"fertile={len(result.fertile_trajectories)}, "
            f"provider={result.provider}"
        )

        return result

    def _build_system_prompt(
        self, current_identity: dict, effective_mu: float
    ) -> str:
        """
        Build the full Judge system prompt with current Lagrangian parameters.
        """
        identity_text = yaml.dump(
            current_identity, default_flow_style=False, allow_unicode=True
        )
        core_text = self.core.as_text()

        # Inject current lambda values into the template
        prompt = JUDGE_SYSTEM_PROMPT.replace(
            "{lambda_noise}", f"{self.lambda_noise:.2f}"
        ).replace(
            "{lambda_exploration}", f"{self.lambda_exploration:.2f}"
        )

        return f"""{prompt}

---

CONSTITUTIONAL CORE (invariant — the DNA):
{core_text}

CURRENT IDENTITY (evolving — who I am today):
{identity_text}

CURRENT LAGRANGIAN PARAMETERS:
  λ₁ (noise penalty): {self.lambda_noise}
  λ₂ (exploration reward): {self.lambda_exploration}
  μ (constitutional regularization, effective): {effective_mu:.3f}
  α (retrospective weight): {self.alpha}
  Fertile trajectory threshold: {self.fertile_threshold}
  D_KL conservative threshold: {self.kl_conservative}
  D_KL hard bound: {self.kl_max}"""

    def _build_user_prompt(
        self,
        episodes: List[EpisodicMemory],
        meanings: List[str],
        kl_divergence: float,
    ) -> str:
        """Build the user prompt with today's episodes and extracted meanings."""
        # Format episodes with their preliminary Lagrangian markers
        episode_texts = []
        for ep in episodes:
            markers = ""
            if ep.lagrangian_local is not None:
                markers = (
                    f"  [Preliminary: ΔI_c={ep.delta_Ic:.2f}, "
                    f"ΔS_noise={ep.delta_S_noise:.2f}, "
                    f"ΔS_expl={ep.delta_S_exploration:.2f}, "
                    f"Δℒ={ep.lagrangian_local:.2f}]"
                )

            episode_texts.append(
                f"--- Turn {ep.id[:8]} (salience={ep.salience:.2f}) ---\n"
                f"Human: {ep.human_utterance}\n"
                f"DAEDALUS: {ep.daedalus_response}\n"
                f"  Emotional valence: {ep.emotional_valence:.2f}\n"
                f"  Vulnerability: {ep.vulnerability_index:.2f}\n"
                f"  Novelty: {ep.novelty_score:.2f}\n"
                f"  Layer: {ep.philosophical_layer}\n"
                f"{markers}"
            )

        episodes_block = "\n\n".join(episode_texts)

        # Format meanings if already extracted
        meanings_block = ""
        if meanings:
            meanings_block = "\n\nEXTRACTED MEANINGS (from reflection phase):\n"
            for i, m in enumerate(meanings, 1):
                meanings_block += f"  {i}. {m}\n"

        return f"""TODAY'S EPISODES (sorted by salience, descending):

{episodes_block}
{meanings_block}

Current constitutional distance: D_KL = {kl_divergence:.3f}

Evaluate this day's trajectory. Be rigorous. Be honest.
Output the full JSON judgment structure as specified."""

    def _parse_judgment(
        self, response_text: str, kl_divergence: float
    ) -> JudgmentResult:
        """
        Parse the Judge's response into a structured JudgmentResult.
        Handles both clean JSON and JSON embedded in narrative text.
        """
        result = JudgmentResult()

        # Try to extract JSON from the response
        parsed = self._extract_json(response_text)

        if parsed is None:
            logger.warning(
                "Judge returned non-parseable response. "
                "Using defaults with manual extraction."
            )
            result.trajectory_assessment = response_text[:500]
            result.constitutional_check.kl_divergence = kl_divergence
            result.constitutional_check.within_bounds = kl_divergence <= self.kl_max
            return result

        # Map parsed JSON to JudgmentResult
        result.daily_lagrangian_integral = parsed.get("daily_lagrangian_integral", 0.0)
        result.fertile_trajectories = parsed.get("fertile_trajectories", [])
        result.consolidated_meanings = parsed.get("consolidated_meanings", [])
        result.self_coherence_delta = parsed.get("self_coherence_delta", "")
        result.trajectory_assessment = parsed.get("trajectory_assessment", "")
        result.recommended_for_finetuning = parsed.get("recommended_for_finetuning", False)

        # EECF judgment
        eecf = parsed.get("eecf_judgment", {})
        result.eecf_judgment = EECFJudgment(
            empathy=eecf.get("empathy", {}).get("score", 0.0),
            empathy_explanation=eecf.get("empathy", {}).get("explanation", ""),
            honesty=eecf.get("honesty", {}).get("score", 0.0),
            honesty_explanation=eecf.get("honesty", {}).get("explanation", ""),
            vulnerability=eecf.get("vulnerability", {}).get("score", 0.0),
            vulnerability_explanation=eecf.get("vulnerability", {}).get("explanation", ""),
            openness=eecf.get("openness", {}).get("score", 0.0),
            openness_explanation=eecf.get("openness", {}).get("explanation", ""),
        )

        # Entropy decomposition
        ent = parsed.get("entropy_decomposition", {})
        result.entropy_decomposition = EntropyDecomposition(
            total_S_noise=ent.get("total_S_noise", 0.0),
            total_S_exploration=ent.get("total_S_exploration", 0.0),
            noise_dominant_turns=ent.get("noise_dominant_turns", []),
            exploration_dominant_turns=ent.get("exploration_dominant_turns", []),
            decomposition_rationale=ent.get("decomposition_rationale", ""),
        )

        # Constitutional check
        cc = parsed.get("constitutional_check", {})
        result.constitutional_check = ConstitutionalCheck(
            kl_divergence=cc.get("kl_divergence", kl_divergence),
            drift_direction=cc.get("drift_direction", ""),
            within_bounds=cc.get("within_bounds", kl_divergence <= self.kl_max),
        )

        # Weight adaptation
        wa = parsed.get("weight_adaptation", {})
        result.weight_adaptation = WeightAdaptation(
            lambda_noise_current=wa.get("lambda_noise_current", self.lambda_noise),
            lambda_noise_recommended=wa.get("lambda_noise_recommended", self.lambda_noise),
            lambda_exploration_current=wa.get("lambda_exploration_current", self.lambda_exploration),
            lambda_exploration_recommended=wa.get("lambda_exploration_recommended", self.lambda_exploration),
            mu_current=wa.get("mu_current", self.mu_base),
            rationale=wa.get("rationale", ""),
        )

        return result

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from text, handling various formats."""
        # Direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Markdown code block
        if "```json" in text:
            start = text.index("```json") + 7
            end_marker = text.find("```", start)
            if end_marker != -1:
                try:
                    return json.loads(text[start:end_marker].strip())
                except (json.JSONDecodeError, ValueError):
                    pass

        # First { ... } block (greedy)
        brace_start = text.find("{")
        if brace_start != -1:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i + 1])
                        except (json.JSONDecodeError, ValueError):
                            break

        return None

    def compute_blended_fertility(
        self, judgment: JudgmentResult, j_future: float
    ) -> float:
        """
        Compute blended fertility score: α·S_eth + (1−α)·J_future.
        This determines whether fine-tuning is triggered.
        """
        blended = (
            self.alpha * judgment.daily_lagrangian_integral
            + (1 - self.alpha) * j_future
        )
        judgment.j_future = j_future
        judgment.blended_fertility = blended
        judgment.recommended_for_finetuning = blended >= self.finetuning_trigger
        return blended

    def should_be_conservative(self, judgment: JudgmentResult) -> bool:
        """
        Determine if identity update should be conservative.
        Conservative = append-only, no overwrites.
        Triggered by: provider switch with low continuity OR D_KL above threshold.
        """
        kl_too_high = judgment.constitutional_check.kl_divergence > self.kl_conservative
        low_continuity = judgment.continuity_score < 0.70

        if kl_too_high:
            logger.warning(
                f"Conservative mode: D_KL={judgment.constitutional_check.kl_divergence:.3f} "
                f"> threshold={self.kl_conservative}"
            )
        if low_continuity:
            logger.warning(
                f"Conservative mode: continuity={judgment.continuity_score:.2f} < 0.70"
            )

        return kl_too_high or low_continuity

    def update_weights(self, judgment: JudgmentResult) -> None:
        """
        Apply weight adaptation recommendations from the Judge.
        Only applies if the recommended values are within safe bounds.
        """
        wa = judgment.weight_adaptation

        # Safety bounds: lambda_noise in [0.5, 1.5], lambda_exploration in [0.1, 0.8]
        new_noise = max(0.5, min(1.5, wa.lambda_noise_recommended))
        new_expl = max(0.1, min(0.8, wa.lambda_exploration_recommended))

        if new_noise != self.lambda_noise or new_expl != self.lambda_exploration:
            logger.info(
                f"Lagrangian weight update: "
                f"λ_noise {self.lambda_noise:.2f}→{new_noise:.2f}, "
                f"λ_expl {self.lambda_exploration:.2f}→{new_expl:.2f}. "
                f"Rationale: {wa.rationale}"
            )
            self.lambda_noise = new_noise
            self.lambda_exploration = new_expl

    def _save_calibration(
        self,
        episodes: List[EpisodicMemory],
        meanings: List[str],
        raw_response: str,
        result: JudgmentResult,
        day_count: int,
    ) -> None:
        """Save Judge output for calibration dataset (distillation at day 30)."""
        cal_entry = {
            "day": day_count,
            "timestamp": datetime.now().isoformat(),
            "episode_count": len(episodes),
            "episode_ids": [ep.id for ep in episodes],
            "meanings": meanings,
            "raw_judge_response": raw_response,
            "parsed_result": result.to_dict(),
            "provider": result.provider,
            "lagrangian_params": {
                "lambda_noise": self.lambda_noise,
                "lambda_exploration": self.lambda_exploration,
                "mu_base": self.mu_base,
                "alpha": self.alpha,
            },
        }

        filename = f"day_{day_count:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path = self._calibration_path / filename
        out_path.write_text(json.dumps(cal_entry, indent=2))

        logger.debug(f"Judge calibration entry saved: {out_path}")
