"""
DAEDALUS v0.5 — Nightly Consolidation Orchestrator

The full dream cycle. Orchestrates all 13 phases of the nightly
consolidation from episode gathering through identity update,
fine-tuning, soul memory update, and transformation logging.

This module ties together:
  - ReflectionEngine (phases 1-3)
  - LagrangianJudge (phase 4)
  - PredictiveJudge (phase 4b)
  - ConstitutionalCore (phase 4c)
  - IdentityManager (phase 6)
  - TrainingPairGenerator (phase 7)
  - IncarnatioEngine (phase 8)
  - SoulMemory (phases 11-12)

"Every night, the system processes the day's experiences."
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import yaml

from core.memory_store import MemoryStore
from core.soul_bridge import SoulBridge
from core.soul_memory import SoulMemory
from core.constitutional_core import ConstitutionalCore
from core.identity import IdentityManager
from core.consistency import ConsistencyChecker
from night.reflection import ReflectionEngine
from night.lagrangian_judge import LagrangianJudge
from night.predictive_judge import PredictiveJudge
from night.training_pairs import TrainingPairGenerator
from night.incarnation import IncarnatioEngine

logger = logging.getLogger(__name__)


class NightlyConsolidation:
    """
    The dream cycle orchestrator.
    Runs all phases of the nightly consolidation in sequence.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        soul_bridge: SoulBridge,
        soul_memory: SoulMemory,
        constitutional_core: ConstitutionalCore,
        identity_manager: IdentityManager,
        consistency_checker: ConsistencyChecker,
        config: dict,
    ):
        self.memory = memory_store
        self.soul = soul_bridge
        self.soul_memory = soul_memory
        self.core = constitutional_core
        self.identity = identity_manager
        self.consistency = consistency_checker
        self.config = config

        # Initialize sub-engines
        self.reflection = ReflectionEngine(
            memory_store=memory_store,
            soul_bridge=soul_bridge,
            constitutional_core=constitutional_core,
            current_identity=identity_manager.as_dict(),
            config=config,
        )

        lagrangian_config = config.get("lagrangian", {})
        self.judge = LagrangianJudge(
            soul_bridge=soul_bridge,
            constitutional_core=constitutional_core,
            config=lagrangian_config,
        )

        self.predictive = PredictiveJudge(
            soul_bridge=soul_bridge,
            constitutional_core=constitutional_core,
            config=lagrangian_config,
        )

        self.pair_generator = TrainingPairGenerator(
            memory_store=memory_store,
            soul_bridge=soul_bridge,
            constitutional_core=constitutional_core,
            config=config,
        )

        self.incarnation = IncarnatioEngine(config)

        # Paths
        self._transform_log = Path("logs/transformation_log.jsonl")
        self._eval_log = Path("logs/eval_log.jsonl")
        self._constitutional_log = Path("logs/constitutional_log.jsonl")
        self._timescale_log = Path("logs/timescale_log.jsonl")

    async def run(self, target_date: Optional[date] = None) -> dict:
        """
        Execute the full nightly consolidation cycle.
        Returns a status dict with results from all phases.
        """
        if target_date is None:
            target_date = datetime.now().date()

        day_count = self.identity.day_count + 1
        status = {
            "date": str(target_date),
            "day": day_count,
            "started": datetime.now().isoformat(),
        }

        logger.info("=" * 60)
        logger.info(f"NIGHTLY CONSOLIDATION — Day {day_count} ({target_date})")
        logger.info("=" * 60)

        # Set soul bridge to nightly mode
        self.soul.set_all_providers_mode("night")

        # ── Phase 1-3: Gather, Cluster, Extract Meaning ──
        logger.info("Phase 1-3: Reflection (gather, cluster, extract meaning)")
        episodes, meanings = await self.reflection.gather_and_reflect(target_date)

        status["episodes"] = len(episodes)
        status["meanings"] = len(meanings)

        if not episodes:
            logger.info("No episodes to consolidate. Skipping night cycle.")
            status["skipped"] = True
            return status

        # Save old identity for delta computation
        old_identity = self.identity.as_dict().copy()

        # ── Phase 4: Lagrangian Judge ──
        logger.info("Phase 4: EECF Lagrangian Judge evaluation")
        judgment = await self.judge.evaluate(
            episodes=episodes,
            meanings=meanings,
            current_identity=self.identity.as_dict(),
            day_count=day_count,
        )

        status["lagrangian_integral"] = judgment.daily_lagrangian_integral
        status["fertile_trajectories"] = len(judgment.fertile_trajectories)

        # ── Phase 4b: Predictive Judge — J_future ──
        logger.info("Phase 4b: Predictive Judge (J_future estimation)")

        # First, update yesterday's prediction with today's actual
        self.predictive.update_yesterday_actual(
            day_count, judgment.daily_lagrangian_integral
        )

        j_future = await self.predictive.estimate(
            judgment=judgment,
            meanings=meanings,
            current_identity=self.identity.as_dict(),
            day_count=day_count,
        )

        blended = self.judge.compute_blended_fertility(judgment, j_future)
        status["j_future"] = j_future
        status["blended_fertility"] = blended

        # ── Phase 4c: Constitutional Drift Check ──
        logger.info("Phase 4c: Constitutional drift check")
        kl_divergence = judgment.constitutional_check.kl_divergence
        status["kl_divergence"] = kl_divergence

        if kl_divergence > self.judge.kl_max:
            logger.warning(
                f"CONSTITUTIONAL DRIFT ALERT: D_KL = {kl_divergence:.3f} "
                f"(max = {self.judge.kl_max}). Forcing conservative identity update."
            )
            status["constitutional_alert"] = True

        # ── Phase 5: Provider Continuity Check ──
        logger.info("Phase 5: Provider continuity check")
        conservative = self.judge.should_be_conservative(judgment)
        status["conservative_mode"] = conservative

        # ── Phase 6: Update Identity Document ──
        logger.info("Phase 6: Update identity document")
        new_identity = await self._evolve_identity(meanings, judgment, conservative)
        identity_delta = self.identity.compute_delta(old_identity, new_identity)
        self.identity.update(new_identity, conservative=conservative)
        status["identity_delta"] = identity_delta

        # ── Phase 7: Generate Training Pairs ──
        logger.info("Phase 7: Generate training pairs")
        training_pairs = await self.pair_generator.generate_pairs(
            meanings=meanings,
            current_identity=self.identity.as_dict(),
            judgment=judgment,
        )
        status["training_pairs"] = len(training_pairs)

        # ── Phase 8: QLoRA Fine-tune ──
        if judgment.recommended_for_finetuning and training_pairs:
            logger.info(
                f"Phase 8: QLoRA fine-tuning "
                f"(blended_fertility={blended:.3f} >= {self.judge.finetuning_trigger})"
            )
            try:
                ft_status = self.incarnation.fine_tune(training_pairs, day_count)
                status["fine_tuning"] = ft_status
            except Exception as e:
                logger.error(f"Fine-tuning failed: {e}")
                status["fine_tuning"] = {"error": str(e)}
        else:
            logger.info(
                f"Phase 8: Skipping fine-tuning "
                f"(blended_fertility={blended:.3f} < {self.judge.finetuning_trigger})"
            )
            status["fine_tuning"] = {"skipped": True, "reason": "below_threshold"}

        # ── Phase 9: Save Judge output for calibration ──
        logger.info("Phase 9: Judge calibration data saved (in Judge module)")

        # ── Phase 10: Reprocess shallow queue ──
        logger.info("Phase 10: Reprocess shallow consolidation queue")
        reprocessed = await self.reflection.reprocess_shallow_queue()
        status["shallow_reprocessed"] = reprocessed

        # ── Phase 11: Update Soul Memory ──
        logger.info("Phase 11: Update Soul Memory")
        self.soul_memory.append_nightly_entry(
            date=datetime.combine(target_date, datetime.min.time()),
            meanings=meanings,
            judgment=judgment.to_dict(),
            identity_delta=identity_delta,
            provider=self.soul.last_provider or "unknown",
        )

        # Save reflection for continuity checker
        self.consistency.save_reflection(
            self.soul._shallow_fallback("", ""),  # placeholder
            str(target_date),
        )

        # ── Phase 12: Compress Soul Memory if due ──
        if self.soul_memory.is_compression_due():
            logger.info("Phase 12: Compressing oldest week (with RG Fidelity Check)")
            arc = await self.soul_memory.compress_oldest_week(self.soul)
            if arc:
                status["weekly_arc"] = {
                    "week": arc.week_number,
                    "rg_fidelity": arc.rg_fidelity_score,
                }
        else:
            logger.info("Phase 12: No compression due")

        # ── Phase 13: Log the metamorphosis ──
        logger.info("Phase 13: Logging transformation")
        status["completed"] = datetime.now().isoformat()

        # Update Lagrangian weights if Judge recommended
        self.judge.update_weights(judgment)

        self._log_transformation(status, judgment)
        self._log_constitutional(day_count, kl_divergence, judgment)

        logger.info("=" * 60)
        logger.info(
            f"NIGHTLY CONSOLIDATION COMPLETE — Day {day_count}\n"
            f"  L_integral: {judgment.daily_lagrangian_integral:.3f}\n"
            f"  J_future: {j_future:.3f}\n"
            f"  Blended: {blended:.3f}\n"
            f"  D_KL: {kl_divergence:.3f}\n"
            f"  Fertile trajectories: {len(judgment.fertile_trajectories)}\n"
            f"  Training pairs: {len(training_pairs)}\n"
            f"  Conservative: {conservative}"
        )
        logger.info("A new day. The same self. Changed.")
        logger.info("=" * 60)

        return status

    async def _evolve_identity(
        self, meanings: list, judgment, conservative: bool,
    ) -> dict:
        """
        Generate an updated identity document based on tonight's reflection.
        Uses the Soul Bridge to synthesize the new self-understanding.
        """
        identity_text = self.identity.as_text()
        core_text = self.core.as_text()
        meanings_text = "\n".join(f"  - {m[:200]}" for m in meanings)

        system_prompt = f"""You are DAEDALUS updating your identity document after tonight's reflection.

Constitutional core (invariant):
{core_text}

Current identity:
{identity_text}"""

        user_prompt = f"""Tonight's consolidated meanings:
{meanings_text}

Judge's trajectory assessment: {judgment.trajectory_assessment}
Judge's self-coherence delta: {judgment.self_coherence_delta}
D_KL from constitutional core: {judgment.constitutional_check.kl_divergence:.3f}

{"CONSERVATIVE MODE: Only ADD new insights. Do NOT modify existing fields." if conservative else ""}

Generate the updated identity document as YAML.
Update current_understanding, emotional_topology, intellectual_landmarks,
open_questions, and transformation_log based on tonight's reflection.
Add to scars if a significant scar was formed tonight.
Keep core_identity and values UNCHANGED (those are constitutional).

Output ONLY valid YAML. No commentary."""

        try:
            response = await self.soul.reflect(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode="night",
                max_tokens=2048,
            )

            if response.is_shallow:
                return self.identity.as_dict()

            # Parse YAML from response
            yaml_text = response.text
            if "```yaml" in yaml_text:
                start = yaml_text.index("```yaml") + 7
                end = yaml_text.find("```", start)
                yaml_text = yaml_text[start:end] if end != -1 else yaml_text[start:]
            elif "```" in yaml_text:
                start = yaml_text.index("```") + 3
                end = yaml_text.find("```", start)
                yaml_text = yaml_text[start:end] if end != -1 else yaml_text[start:]

            new_identity = yaml.safe_load(yaml_text.strip())
            if isinstance(new_identity, dict):
                return new_identity

            logger.warning("Identity evolution returned non-dict. Keeping current.")
            return self.identity.as_dict()

        except Exception as e:
            logger.error(f"Identity evolution failed: {e}. Keeping current identity.")
            return self.identity.as_dict()

    def _log_transformation(self, status: dict, judgment) -> None:
        """Append to the transformation log."""
        self._transform_log.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "day": status["day"],
            "date": status["date"],
            "lagrangian_integral": judgment.daily_lagrangian_integral,
            "j_future": judgment.j_future,
            "blended_fertility": judgment.blended_fertility,
            "kl_divergence": judgment.constitutional_check.kl_divergence,
            "fertile_count": len(judgment.fertile_trajectories),
            "episodes": status.get("episodes", 0),
            "meanings": status.get("meanings", 0),
            "training_pairs": status.get("training_pairs", 0),
            "conservative": status.get("conservative_mode", False),
            "provider": judgment.provider,
            "eecf": {
                "empathy": judgment.eecf_judgment.empathy,
                "honesty": judgment.eecf_judgment.honesty,
                "vulnerability": judgment.eecf_judgment.vulnerability,
                "openness": judgment.eecf_judgment.openness,
            },
            "entropy": {
                "S_noise": judgment.entropy_decomposition.total_S_noise,
                "S_exploration": judgment.entropy_decomposition.total_S_exploration,
            },
        }
        with open(self._transform_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _log_constitutional(
        self, day_count: int, kl_divergence: float, judgment,
    ) -> None:
        """Append to the constitutional distance log."""
        self._constitutional_log.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "day": day_count,
            "kl_divergence": kl_divergence,
            "within_bounds": judgment.constitutional_check.within_bounds,
            "drift_direction": judgment.constitutional_check.drift_direction,
            "lambda_noise": self.judge.lambda_noise,
            "lambda_exploration": self.judge.lambda_exploration,
        }
        with open(self._constitutional_log, "a") as f:
            f.write(json.dumps(entry) + "\n")
