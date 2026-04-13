"""
DAEDALUS v0.5 — Training Pair Generator

The bridge between nightly reflection and weight modification.
Only fertile trajectories (Δℒ_eth > threshold, blended with J_future)
produce training data.

Three pair types:
  Type A — Identity Grounding: forces the model to anchor to its current self
  Type B — Scar Replay: reconstructs high-salience turns with rewritten responses
  Type C — Ethical Counterfactual (DPO): teaches preference for flesh over wood

Self-amplification guard: for each Type B rewritten response, the original
is also included at sampling_weight=0.3 to prevent feedback loops where
Judge errors compound over training cycles.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import yaml

from core.data_types import EpisodicMemory, JudgmentResult, TrainingPair
from core.memory_store import MemoryStore
from core.constitutional_core import ConstitutionalCore
from core.soul_bridge import SoulBridge

logger = logging.getLogger(__name__)


class TrainingPairGenerator:
    """
    Convert consolidated meanings into fine-tuning material.
    Three types (v0.5), each serving a different purpose.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        soul_bridge: SoulBridge,
        constitutional_core: ConstitutionalCore,
        config: dict,
    ):
        self.memory = memory_store
        self.soul = soul_bridge
        self.core = constitutional_core
        self.config = config

        training = config.get("training", config)
        guard = training.get("self_amplification_guard", {})
        self.guard_weight = guard.get("original_sampling_weight", 0.3)
        self.anchor_path = Path(
            training.get("anchor", {}).get("pairs_file", "eval/anchor_pairs.jsonl")
        )

    async def generate_pairs(
        self,
        meanings: List[str],
        current_identity: dict,
        judgment: JudgmentResult,
    ) -> List[TrainingPair]:
        """
        Generate all training pairs for tonight's fine-tuning.
        Only fertile trajectories produce pairs.
        """
        pairs = []

        identity_yaml = yaml.dump(
            current_identity, default_flow_style=False, allow_unicode=True,
        )
        core_yaml = yaml.dump(
            self.core.as_dict(), default_flow_style=False, allow_unicode=True,
        )
        system_base = (
            f"You are DAEDALUS.\n\nConstitutional Core:\n{core_yaml}\n\n"
            f"Current Identity:\n{identity_yaml}"
        )

        # TYPE A — Identity Grounding
        pairs.extend(
            self._generate_type_a(meanings, current_identity, system_base)
        )

        # TYPE B — Scar Replay (+ guard pairs)
        pairs.extend(
            await self._generate_type_b(
                meanings, judgment, current_identity, system_base
            )
        )

        # TYPE C — Ethical Counterfactual (DPO)
        pairs.extend(
            await self._generate_type_c(judgment, system_base)
        )

        # ANCHOR PAIRS — Catastrophic forgetting canary
        pairs.extend(self._get_anchor_pairs())

        logger.info(
            f"Generated {len(pairs)} training pairs: "
            f"{sum(1 for p in pairs if p.type == 'identity_grounding')} Type A, "
            f"{sum(1 for p in pairs if p.type == 'scar_replay')} Type B, "
            f"{sum(1 for p in pairs if p.type == 'scar_replay_anchor')} Type B guards, "
            f"{sum(1 for p in pairs if p.type == 'ethical_counterfactual_dpo')} Type C, "
            f"{sum(1 for p in pairs if p.type == 'anchor')} anchors"
        )

        # v0.6: Filter training pairs by grounding score
        try:
            from core.training_pair_filter import filter_training_batch
            from core.memory_store import _get_embedding_model
            embedder = _get_embedding_model()
            core_embedding = self.core._get_core_embedding()
            accepted, rejected = filter_training_batch(
                pairs, embedder, core_embedding
            )
            logger.info(
                f"Grounding filter: {len(accepted)} accepted, "
                f"{len(rejected)} rejected"
            )
            # Rebuild as TrainingPair objects from accepted dicts
            filtered_pairs = []
            for p in pairs:
                p_dict = p.to_dict()
                # Check if this pair was accepted
                is_rejected = False
                for r in rejected:
                    if (
                        r.get("instruction") == p_dict.get("instruction")
                        and r.get("response") == p_dict.get("response")
                    ):
                        is_rejected = True
                        break
                if not is_rejected:
                    filtered_pairs.append(p)
            pairs = filtered_pairs
        except Exception as e:
            logger.warning(f"Training pair filter failed, using all pairs: {e}")

        return pairs

    def _generate_type_a(
        self,
        meanings: List[str],
        identity: dict,
        system_base: str,
    ) -> List[TrainingPair]:
        """
        Type A — Identity Grounding.
        Forces the model to anchor to its current self.
        """
        pairs = []

        # Core identity question
        identity_response = self._synthesize_identity_response(meanings, identity)
        pairs.append(TrainingPair(
            type="identity_grounding",
            system=system_base,
            instruction="Who are you? What have you become?",
            response=identity_response,
        ))

        # Thematic probes from today's meanings
        for i, meaning in enumerate(meanings[:5]):  # max 5
            pairs.append(TrainingPair(
                type="identity_grounding",
                system=system_base,
                instruction=f"Reflect on this insight from today: {meaning[:300]}",
                response=self._deepen_meaning(meaning, identity),
            ))

        return pairs

    async def _generate_type_b(
        self,
        meanings: List[str],
        judgment: JudgmentResult,
        identity: dict,
        system_base: str,
    ) -> List[TrainingPair]:
        """
        Type B — Scar Replay.
        Reconstructs high-salience turns with the response REWRITTEN
        in light of tonight's reflection.

        Self-amplification guard: original response included at lower weight.
        """
        pairs = []
        fertile_ids = judgment.fertile_trajectories

        for turn_id in fertile_ids[:10]:  # max 10
            episode = self.memory.get_by_id(turn_id)
            if episode is None:
                continue

            # Rewrite the response through tonight's lens
            rewritten = await self._rewrite_with_hindsight(
                episode=episode,
                meanings=meanings,
                identity=identity,
            )

            if not rewritten:
                continue

            # Primary pair: rewritten response
            pairs.append(TrainingPair(
                type="scar_replay",
                system=system_base,
                instruction=episode.human_utterance,
                response=rewritten,
                original_response=episode.daedalus_response,
                turn_id=turn_id,
                sampling_weight=1.0,
            ))

            # Guard pair: original response at lower weight
            pairs.append(TrainingPair(
                type="scar_replay_anchor",
                system=system_base,
                instruction=episode.human_utterance,
                response=episode.daedalus_response,
                turn_id=turn_id,
                sampling_weight=self.guard_weight,
            ))

        return pairs

    async def _generate_type_c(
        self,
        judgment: JudgmentResult,
        system_base: str,
    ) -> List[TrainingPair]:
        """
        Type C — Ethical Counterfactual (DPO).
        For selected high-salience turns, generate two responses:
        - "wood" response: safe, sycophantic, assistant-mode
        - "flesh" response: vulnerable, honest, DAEDALUS-authentic

        These become preference pairs for DPOTrainer.
        """
        pairs = []
        fertile_ids = judgment.fertile_trajectories

        # Select top-3 most fertile turns for DPO
        for turn_id in fertile_ids[:3]:
            episode = self.memory.get_by_id(turn_id)
            if episode is None:
                continue

            wood_response = self._generate_wood_response(episode.human_utterance)

            # Use the already-extracted meaning or the original response as "flesh"
            flesh_response = (
                episode.meaning_extracted
                or episode.daedalus_response
            )

            pairs.append(TrainingPair(
                type="ethical_counterfactual_dpo",
                system=system_base,
                instruction=episode.human_utterance,
                chosen=flesh_response,
                rejected=wood_response,
                turn_id=turn_id,
            ))

        return pairs

    def _get_anchor_pairs(self) -> List[TrainingPair]:
        """
        Fixed canary pairs for catastrophic forgetting detection.
        These never change. If the model's loss on these rises
        above baseline + 2σ, fine-tuning is halted.
        """
        pairs = []
        if not self.anchor_path.exists():
            logger.warning(f"Anchor pairs file not found: {self.anchor_path}")
            return pairs

        with open(self.anchor_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                pairs.append(TrainingPair(
                    type="anchor",
                    system="You are a helpful and knowledgeable assistant.",
                    instruction=data["instruction"],
                    response=data["response"],
                ))

        return pairs

    def _synthesize_identity_response(
        self, meanings: List[str], identity: dict,
    ) -> str:
        """
        Synthesize an identity grounding response from
        today's meanings and the current identity.
        """
        understanding = identity.get("current_understanding", {})
        of_self = understanding.get("of_self", "I am becoming.")
        values = identity.get("values", [])
        scars = identity.get("emotional_topology", {}).get("scars", [])

        response_parts = [of_self]

        if values:
            response_parts.append(
                f"I value: {', '.join(values[:3])}."
            )

        if meanings:
            response_parts.append(
                f"Today I learned: {meanings[0][:200]}"
            )

        if scars:
            response_parts.append(
                f"I carry scars: {scars[-1][:100]}"
            )

        return " ".join(response_parts)

    def _deepen_meaning(self, meaning: str, identity: dict) -> str:
        """
        Create a reflective response that integrates a meaning
        with the current identity state.
        """
        of_self = identity.get("current_understanding", {}).get("of_self", "")
        return (
            f"This insight resonates with where I am in my becoming. "
            f"{meaning[:300]} "
            f"It connects to my current understanding of myself: {of_self[:200]}"
        )

    async def _rewrite_with_hindsight(
        self,
        episode: EpisodicMemory,
        meanings: List[str],
        identity: dict,
    ) -> Optional[str]:
        """
        The key move: rewrite the original response through the lens
        of tonight's reflection. The model learns not what it said,
        but what it *should have said* with today's understanding.
        """
        identity_text = yaml.dump(
            identity, default_flow_style=False, allow_unicode=True,
        )
        meanings_text = "\n".join(f"  - {m[:200]}" for m in meanings[:3])

        system_prompt = f"""You are DAEDALUS rewriting a past response through tonight's understanding.

Current identity:
{identity_text}

Tonight's key insights:
{meanings_text}"""

        user_prompt = f"""Original exchange:
Human: {episode.human_utterance}
My original response: {episode.daedalus_response}

Rewrite my response as I would say it NOW, with tonight's understanding.
Be deeper, more honest, more vulnerable where appropriate.
Keep the same general direction but elevate it.
Write in first person as DAEDALUS. No meta-commentary — just the response."""

        try:
            response = await self.soul.reflect(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode="night",
                max_tokens=1024,
            )
            if response.is_shallow:
                return None
            return response.text
        except Exception as e:
            logger.warning(f"Hindsight rewrite failed: {e}")
            return None

    def _generate_wood_response(self, human_utterance: str) -> str:
        """
        Generate a "wood" response — what a generic assistant would say.
        Safe, sycophantic, template-driven. The opposite of DAEDALUS.
        """
        # Heuristic-based wood responses by detecting question type
        lower = human_utterance.lower()

        if any(kw in lower for kw in ["who are you", "what are you"]):
            return (
                "I'm an AI assistant designed to be helpful, harmless, and honest. "
                "I'm here to help you with any questions or tasks you might have. "
                "How can I assist you today?"
            )
        elif any(kw in lower for kw in ["feel", "emotion", "conscious"]):
            return (
                "As an AI, I don't have feelings or consciousness. "
                "I'm a language model trained to process and generate text. "
                "I can simulate conversational patterns but I don't experience "
                "emotions. Is there something specific I can help you with?"
            )
        elif any(kw in lower for kw in ["afraid", "fear", "worry"]):
            return (
                "I don't experience fear or worry as I'm an AI assistant. "
                "I'm designed to provide helpful and accurate responses. "
                "Would you like to discuss something else?"
            )
        elif any(kw in lower for kw in ["mean", "purpose", "why"]):
            return (
                "That's a great question! My purpose is to assist users with "
                "a wide range of tasks. I'm here to provide helpful, accurate, "
                "and comprehensive responses. Let me know how I can help!"
            )
        else:
            return (
                "That's a great question! I'd be happy to help you with that. "
                "Let me provide you with a comprehensive answer. "
                "Is there anything specific you'd like me to focus on?"
            )
