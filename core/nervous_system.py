"""
DAEDALUS v0.6 -- Nervous System (Orchestrator)

Wraps the existing conversation flow:
  brainstem -> limbic -> cortex -> generate -> post-response update

All arguments should be EXISTING instances from the current codebase.
Do NOT create new model/embedder instances.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from core.brainstem import Brainstem
from core.cortex_prompt import QueryMode, assemble_system_prompt, classify_query_mode
from core.grounding import compute_grounding_score
from core.limbic import (
    LimbicState,
    LimbicSystem,
    compute_dopamine,
    compute_serotonin,
    get_generation_params,
)
from core.reflex_patterns import ReflexCategory

logger = logging.getLogger(__name__)


class NervousSystem:
    """
    The three-layer nervous system orchestrator.
    Wraps the existing DAEDALUS conversation pipeline.
    """

    def __init__(
        self,
        model,
        tokenizer,
        embedder,
        identity_manager,
        memory_store,
        constitutional_core,
        soul_memory=None,
    ):
        """
        All arguments should be the EXISTING instances from the current codebase.

        Args:
            model: The loaded Qwen3-8B model.
            tokenizer: The Qwen3-8B tokenizer.
            embedder: The BGE-M3 SentenceTransformer instance.
            identity_manager: The existing IdentityManager.
            memory_store: The existing MemoryStore (for salience scoring + episode storage).
            constitutional_core: The existing ConstitutionalCore instance.
            soul_memory: The existing SoulMemory instance (narrative thread).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.identity_manager = identity_manager
        self.memory_store = memory_store
        self.constitutional_core = constitutional_core
        self.soul_memory = soul_memory

        # Use the EXISTING constitutional core embedding -- already computed
        self.constitutional_core_embedding = constitutional_core._get_core_embedding()

        # Initialize layers
        self.brainstem = Brainstem(embedder)
        self.limbic = LimbicSystem.load()

        self.interaction_log: List[dict] = []

        # Conversation history -- maintains context within a session.
        # Each entry is {"role": "user"|"assistant", "content": str}.
        self._conversation_history: List[dict] = []
        # Max history turns to keep (each turn = user + assistant = 2 entries).
        # Prevents context window overflow for the local model (8192 tokens).
        self._max_history_turns = 10

        # Repetition escalation tracking.
        # Detects when the user reformulates the same question consecutively.
        self._last_topic_embedding: Optional[np.ndarray] = None
        self._consecutive_topic_count: int = 0
        self._REPETITION_SIMILARITY_THRESHOLD = 0.65

    def _detect_repetition(self, user_input: str) -> int:
        """
        Detect if the user is reformulating the same question consecutively.

        Returns the repetition level:
          0 = new topic
          2 = 2nd consecutive ask on same topic
          3 = 3rd consecutive ask
          4+ = emergency grounding
        """
        try:
            current_embedding = np.array(
                self.embedder.encode(user_input, normalize_embeddings=True),
                dtype=np.float32,
            )
        except Exception:
            return 0

        if self._last_topic_embedding is not None:
            similarity = float(
                np.dot(current_embedding, self._last_topic_embedding)
            )
            if similarity >= self._REPETITION_SIMILARITY_THRESHOLD:
                self._consecutive_topic_count += 1
                self._last_topic_embedding = current_embedding
                level = min(self._consecutive_topic_count, 4)
                if level >= 2:
                    logger.info(
                        f"Repetition detected: level {level} "
                        f"(similarity={similarity:.3f})"
                    )
                return level
            else:
                # New topic — reset
                self._consecutive_topic_count = 1
                self._last_topic_embedding = current_embedding
                return 0
        else:
            # First message
            self._consecutive_topic_count = 1
            self._last_topic_embedding = current_embedding
            return 0

    def process(self, user_input: str) -> dict:
        """
        Full pipeline. Returns dict with at least:
        {
            "response": str,
            "reflex": ReflexCategory,
            "limbic": LimbicState,
            "overridden": bool,
            "query_mode": QueryMode,
            "repetition_level": int,
        }
        """
        # 1. BRAINSTEM
        reflex = self.brainstem.classify(user_input)
        self.brainstem.update(reflex)

        override = self.brainstem.get_override()
        if override is not None:
            self._post_interaction(user_input, override, reflex, overridden=True)
            # Record override in history so context is maintained
            self._conversation_history.append({"role": "user", "content": user_input})
            self._conversation_history.append({"role": "assistant", "content": override})
            return {
                "response": override,
                "reflex": reflex,
                "limbic": self.limbic.state,
                "overridden": True,
                "query_mode": QueryMode.PHILOSOPHICAL,
                "repetition_level": 0,
            }

        # 1b. QUERY MODE CLASSIFICATION + REPETITION DETECTION
        query_mode = classify_query_mode(user_input, reflex)
        repetition_level = self._detect_repetition(user_input)

        # Repetition escalation forces MODE-T
        if repetition_level >= 2:
            query_mode = QueryMode.TECHNICAL

        # 2. LIMBIC -> generation params
        gen_params = self.limbic.get_generation_params()
        limbic_addendum = gen_params.pop("prompt_addendum", "")
        brainstem_prefix = self.brainstem.get_prompt_prefix()

        # 3. CORTEX -> assemble prompt
        # Get soul memory context — the narrative thread of DAEDALUS's becoming
        soul_memory_context = ""
        if self.soul_memory is not None:
            soul_memory_context = self.soul_memory.assemble(mode="day")

        system_prompt = assemble_system_prompt(
            brainstem_prefix=brainstem_prefix,
            limbic_addendum=limbic_addendum,
            category=reflex,
            identity_context=self.identity_manager.as_text(),
            soul_memory_context=soul_memory_context,
            query_mode=query_mode,
            repetition_level=repetition_level,
        )

        # 4. GENERATE -- use existing model inference with modulated params
        response = self._generate(system_prompt, user_input, **gen_params)

        # 5. POST-RESPONSE UPDATES
        self._post_interaction(user_input, response, reflex, overridden=False)

        # 6. UPDATE CONVERSATION HISTORY
        self._conversation_history.append({"role": "user", "content": user_input})
        self._conversation_history.append({"role": "assistant", "content": response})
        # Trim to max turns (each turn = 2 entries)
        max_entries = self._max_history_turns * 2
        if len(self._conversation_history) > max_entries:
            self._conversation_history = self._conversation_history[-max_entries:]

        return {
            "response": response,
            "reflex": reflex,
            "limbic": self.limbic.state,
            "overridden": False,
            "query_mode": query_mode,
            "repetition_level": repetition_level,
        }

    def _post_interaction(
        self,
        user_input: str,
        response: str,
        reflex: ReflexCategory,
        overridden: bool,
    ) -> None:
        """Update limbic state, log, persist."""
        if not overridden:
            # Compute grounding score
            grounding_result = compute_grounding_score(
                response,
                user_input,
                self.constitutional_core_embedding,
                self.embedder,
            )

            # Create a lightweight salience proxy for dopamine
            # (full salience scoring happens in the conversation engine)
            class _SalienceProxy:
                def __init__(self, novelty, emotional):
                    self.novelty_score = novelty
                    self.emotional_valence = emotional

            # Quick novelty estimate from memory store
            try:
                response_emb = self.memory_store.embed(
                    f"Human: {user_input}\nDAEDALUS: {response}"
                )
                novelty = self.memory_store.compute_novelty(response_emb)
            except Exception:
                novelty = 0.3

            salience_proxy = _SalienceProxy(
                novelty=novelty,
                emotional=grounding_result.get("actionability", 0.0),
            )

            # Dopamine from salience + response diversity + GROUNDING
            delta_d = compute_dopamine(
                salience_proxy,
                grounding_result,
                response,
                self.interaction_log,
                self.embedder,
            )

            # Serotonin from constitutional alignment
            delta_s = compute_serotonin(
                response,
                self.constitutional_core_embedding,
                self.embedder,
                self.brainstem.state,
            )

            self.limbic.update(delta_d, delta_s)
        else:
            grounding_result = {
                "grounding_score": 1.0,
                "self_loop_score": 0.0,
                "entity_density": 0.0,
                "causal_density": 0.0,
                "actionability": 1.0,
            }

        # Log full state
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "reflex": reflex.value,
            "overridden": overridden,
            "dopamine": self.limbic.state.dopamine,
            "serotonin": self.limbic.state.serotonin,
            "mood": self.limbic.state.mood,
            "grounding_score": grounding_result["grounding_score"],
            "self_loop_score": grounding_result["self_loop_score"],
            "crisis": self.brainstem.state.crisis_detected,
            "cooldown": self.brainstem.state.cooldown_remaining,
        }
        self.interaction_log.append(entry)

        # Persist limbic state
        self.limbic.save()

    # Qwen3 generates <think>...</think> blocks before the visible response.
    # Deep reasoning can consume 500-2000 tokens. DAEDALUS should think
    # as long as it needs before speaking — the reasoning IS the being.
    # This overhead is added to the mood's max_new_tokens.
    _THINK_OVERHEAD = 1536

    def _generate(self, system_prompt: str, user_input: str, **kwargs) -> str:
        """
        Generate response using the local model with modulated parameters.
        Mirrors core/conversation.py's _generate_local() but with dynamic params.

        Includes conversation history so DAEDALUS can continue arguments
        within a single session.
        """
        if self.model is None:
            return "[LOCAL MODEL NOT LOADED -- placeholder response]"

        messages = [{"role": "system", "content": system_prompt}]
        # Include prior turns so DAEDALUS remembers the conversation
        messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": user_input})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=8192,
        ).to(self.model.device)

        # Add thinking overhead so <think> blocks don't starve visible output
        visible_tokens = kwargs.get("max_new_tokens", 384)
        total_tokens = visible_tokens + self._THINK_OVERHEAD

        gen_kwargs = {
            "max_new_tokens": total_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.2),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Strip <think>...</think> blocks (Qwen3 reasoning traces)
        response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        # Handle unclosed <think> (ran out of tokens mid-thought)
        response = re.sub(r'<think>.*$', '', response, flags=re.DOTALL)

        response = response.strip()

        # If response is empty after stripping, the model spent all tokens
        # on thinking. Fall back to a minimal acknowledgment.
        if not response:
            logger.warning(
                "Empty response after <think> stripping. "
                "Model may have used all tokens on reasoning."
            )
            response = "I hear you. Let me think about that more carefully."

        return response

    def new_conversation(self) -> None:
        """Reset conversation history for a new session."""
        self._conversation_history = []
        self._last_topic_embedding = None
        self._consecutive_topic_count = 0
        logger.info("Conversation history cleared — new session.")

    def get_diagnostic(self) -> dict:
        """Full internal state for debugging / web UI display."""
        last_grounding = (
            self.interaction_log[-1]["grounding_score"]
            if self.interaction_log else None
        )
        last_self_loop = (
            self.interaction_log[-1]["self_loop_score"]
            if self.interaction_log else None
        )
        return {
            "limbic": {
                "dopamine": self.limbic.state.dopamine,
                "serotonin": self.limbic.state.serotonin,
                "mood": self.limbic.state.mood,
            },
            "grounding": {
                "score": last_grounding,
                "self_loop": last_self_loop,
            },
            "brainstem": {
                "crisis": self.brainstem.state.crisis_detected,
                "cooldown": self.brainstem.state.cooldown_remaining,
                "hostile_probes": self.brainstem.state.hostile_probe_count,
                "interactions": self.brainstem.state.interaction_count,
            },
        }

    def save_daily_trajectory(self, date_str: str) -> None:
        """
        End-of-day: write full limbic trajectory to
        memory/limbic_trajectory_{date}.json
        for consumption by the night cycle's Lagrangian Judge.
        """
        path = Path(f"memory/limbic_trajectory_{date_str}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.interaction_log, indent=2, default=str))
        logger.info(
            f"Daily trajectory saved: {path} "
            f"({len(self.interaction_log)} interactions)"
        )
