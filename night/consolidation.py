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
from collections import Counter
from datetime import datetime, date
from pathlib import Path
from statistics import mean
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

        # ── Phase 3b: Load limbic trajectory (v0.7) ──
        limbic_summary = self._load_limbic_trajectory(target_date)
        if limbic_summary:
            logger.info(
                f"Phase 3b: Limbic trajectory loaded — "
                f"{limbic_summary['total_interactions']} interactions, "
                f"mean_grounding={limbic_summary['mean_grounding']:.3f}"
            )
        else:
            logger.info("Phase 3b: No limbic trajectory found for today")

        # ── Phase 3c: Enrich episodes with grounding scores (v0.7) ──
        self._enrich_episodes_grounding(episodes, limbic_summary)

        # ── Phase 4: Lagrangian Judge ──
        logger.info("Phase 4: EECF Lagrangian Judge evaluation")
        judgment = await self.judge.evaluate(
            episodes=episodes,
            meanings=meanings,
            current_identity=self.identity.as_dict(),
            day_count=day_count,
            limbic_summary=limbic_summary,
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
            grounding_analysis=judgment.grounding_analysis,
            limbic_summary=limbic_summary,
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
        import re as _re

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

CRITICAL YAML FORMATTING RULES:
- ALL string values containing colons, quotes, or special characters MUST be wrapped in double quotes
- Multi-line strings MUST use YAML block scalar syntax (| or >)
- Do NOT include any text outside the YAML document
- Output ONLY valid YAML. No commentary."""

        try:
            response = await self.soul.reflect(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode="night",
                max_tokens=2048,
            )

            if response.is_shallow:
                return self.identity.as_dict()

            new_identity = self._parse_yaml_response(response.text)
            if isinstance(new_identity, dict):
                return new_identity

            # Fallback: apply minimal update from judgment data
            logger.warning(
                "Identity YAML unparseable. Applying minimal update from judgment."
            )
            return self._minimal_identity_update(meanings, judgment)

        except Exception as e:
            logger.error(f"Identity evolution failed: {e}. Applying minimal update.")
            try:
                return self._minimal_identity_update(meanings, judgment)
            except Exception:
                return self.identity.as_dict()

    def _minimal_identity_update(self, meanings: list, judgment) -> dict:
        """
        Fallback when LLM-generated YAML is unparseable.
        Applies structured updates from the judgment data to the current identity.
        """
        identity = self.identity.as_dict().copy()

        # Update lagrangian_state from actual metrics
        if "lagrangian_state" not in identity:
            identity["lagrangian_state"] = {}
        identity["lagrangian_state"]["latest_L_integral"] = (
            judgment.daily_lagrangian_integral
        )
        identity["lagrangian_state"]["latest_D_KL"] = (
            judgment.constitutional_check.kl_divergence
        )

        # Append to transformation_log
        if "transformation_log" not in identity:
            identity["transformation_log"] = []
        identity["transformation_log"].append({
            "day": self.identity.day_count + 1,
            "summary": judgment.trajectory_assessment[:200]
            if judgment.trajectory_assessment else "Night cycle completed.",
            "method": "minimal_fallback",
        })

        # Add new scars from meanings if any mention "scar"
        if "emotional_topology" not in identity:
            identity["emotional_topology"] = {}
        if "scars" not in identity["emotional_topology"]:
            identity["emotional_topology"]["scars"] = []
        for m in meanings:
            if "scar" in m.lower()[:500]:
                scar_text = m[:200].strip()
                existing = [s if isinstance(s, str) else str(s)
                            for s in identity["emotional_topology"]["scars"]]
                if scar_text not in existing:
                    identity["emotional_topology"]["scars"].append(scar_text)

        logger.info("Applied minimal identity update from judgment data.")
        return identity

    @staticmethod
    def _parse_yaml_response(text: str) -> object:
        """
        Robustly extract and parse YAML from an LLM response.
        Handles code fences, think tags, and common formatting issues.
        """
        import re as _re

        # Strip <think>...</think> blocks
        text = _re.sub(r'<think>.*?</think>\s*', '', text, flags=_re.DOTALL)
        text = _re.sub(r'<think>.*$', '', text, flags=_re.DOTALL)

        # Extract from code fences
        yaml_text = text
        fence_match = _re.search(r'```(?:yaml)?\s*\n(.*?)```', text, _re.DOTALL)
        if fence_match:
            yaml_text = fence_match.group(1)

        # Attempt 1: parse as-is
        try:
            result = yaml.safe_load(yaml_text.strip())
            if isinstance(result, dict):
                return result
        except yaml.YAMLError:
            pass

        # Attempt 2: fix unquoted strings with colons by quoting values
        # Common LLM YAML error: bare strings containing colons
        fixed = []
        for line in yaml_text.strip().split('\n'):
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]
            # Match "key: value" where value contains an unquoted colon
            kv = _re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$', stripped)
            if kv:
                key, val = kv.group(1), kv.group(2)
                # If value is not already quoted and contains a colon, quote it
                if ':' in val and not val.startswith('"') and not val.startswith("'"):
                    val = '"' + val.replace('"', '\\"') + '"'
                    fixed.append(f'{indent}{key}: {val}')
                    continue
            fixed.append(line)

        try:
            result = yaml.safe_load('\n'.join(fixed))
            if isinstance(result, dict):
                logger.info("Identity YAML parsed after fixing unquoted colons.")
                return result
        except yaml.YAMLError:
            pass

        # Attempt 3: find the first top-level key and parse from there
        for i, line in enumerate(yaml_text.strip().split('\n')):
            if _re.match(r'^[a-zA-Z_]', line) and ':' in line:
                subset = '\n'.join(yaml_text.strip().split('\n')[i:])
                try:
                    result = yaml.safe_load(subset)
                    if isinstance(result, dict):
                        logger.info("Identity YAML parsed after stripping preamble.")
                        return result
                except yaml.YAMLError:
                    break

        logger.error(f"All YAML parse attempts failed for identity evolution response.")
        return None

    def _load_limbic_trajectory(self, target_date: date) -> Optional[dict]:
        """
        v0.7: Load the day's limbic trajectory from the nervous system
        and compute a summary for the Judge.
        """
        date_str = str(target_date)
        trajectory_path = Path(f"memory/limbic_trajectory_{date_str}.json")
        if not trajectory_path.exists():
            return None

        try:
            with open(trajectory_path) as f:
                limbic_data = json.load(f)

            if not limbic_data:
                return None

            grounding_scores = [
                e["grounding_score"] for e in limbic_data
                if e.get("grounding_score") is not None
            ]

            return {
                "total_interactions": len(limbic_data),
                "mean_dopamine": mean([e["dopamine"] for e in limbic_data]),
                "mean_serotonin": mean([e["serotonin"] for e in limbic_data]),
                "mean_grounding": mean(grounding_scores) if grounding_scores else 0.5,
                "mood_distribution": dict(Counter(e["mood"] for e in limbic_data)),
                "crisis_events": sum(1 for e in limbic_data if e.get("crisis")),
                "dopamine_trend": limbic_data[-1]["dopamine"] - limbic_data[0]["dopamine"],
                "serotonin_trend": limbic_data[-1]["serotonin"] - limbic_data[0]["serotonin"],
            }
        except Exception as e:
            logger.warning(f"Failed to load limbic trajectory: {e}")
            return None

    def _enrich_episodes_grounding(
        self, episodes, limbic_summary: Optional[dict]
    ) -> None:
        """
        v0.7: For episodes that lack grounding scores (e.g. stored before v0.7,
        or when the nervous system was not active), retroactively compute them
        using the grounding scorer.
        """
        missing = [ep for ep in episodes if ep.grounding_score is None]
        if not missing:
            return

        logger.info(
            f"Enriching {len(missing)}/{len(episodes)} episodes with grounding scores"
        )

        try:
            from core.grounding import compute_grounding_score
            core_embedding = self.core.get_embedding()
            embedder = self.memory.embedder
            if core_embedding is None or embedder is None:
                logger.warning(
                    "Core embedding or embedder not available; skipping enrichment"
                )
                return
        except (ImportError, AttributeError) as e:
            logger.warning(f"Grounding scorer not available: {e}; skipping enrichment")
            return

        for ep in missing:
            if not ep.daedalus_response:
                continue
            try:
                result = compute_grounding_score(
                    response_text=ep.daedalus_response,
                    user_input=ep.human_utterance,
                    constitutional_core_embedding=core_embedding,
                    embedder=embedder,
                )
                ep.grounding_score = result.get("grounding_score", 0.5)
                ep.self_loop_score = result.get("self_loop_score", 0.0)
                ep.entity_density = result.get("entity_density", 0.0)
                ep.causal_density = result.get("causal_density", 0.0)
                ep.actionability = result.get("actionability", 0.0)
            except Exception as e:
                logger.debug(f"Grounding score failed for {ep.id[:8]}: {e}")
                ep.grounding_score = 0.5
                ep.self_loop_score = 0.0

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
            "grounding": {
                "mean_grounding": judgment.grounding_analysis.mean_grounding,
                "effective_Ic": judgment.grounding_analysis.effective_Ic_integral,
                "raw_Ic": judgment.grounding_analysis.raw_Ic_integral,
                "penalty_ratio": judgment.grounding_analysis.grounding_penalty_ratio,
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
