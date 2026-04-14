"""
DAEDALUS v0.5 — Conversation Interface

The daytime dialogue engine. During conversation, DAEDALUS operates
as a hybrid system: local Qwen3-14B generates the initial response,
optionally enriched by a Soul Bridge reflection for deep exchanges.

Each exchange is scored for salience and split entropy, then stored
in the episodic memory with full metadata.

The conversation flow:
  Human utterance
    → Retrieve relevant episodic memories (top-k)
    → Retrieve current identity document
    → Retrieve constitutional core
    → Construct augmented prompt
    → Local model generates response
    → (Optional) Soul Bridge deepens the response
    → Score salience + split entropy
    → Store episode in ChromaDB
    → Response to human
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import List, Optional

import yaml

from core.data_types import EpisodicMemory
from core.memory_store import MemoryStore
from core.soul_bridge import SoulBridge
from core.constitutional_core import ConstitutionalCore
from core.identity import IdentityManager
from core.salience import SplitEntropySscorer

logger = logging.getLogger(__name__)


def _format_memories(memories: List[EpisodicMemory]) -> str:
    """Format episodic memories for inclusion in the prompt."""
    if not memories:
        return "(no relevant memories yet)"

    lines = []
    for m in memories:
        salience_tag = f"[salience={m.salience:.2f}]"
        lines.append(
            f"{salience_tag} Human: {m.human_utterance[:200]}\n"
            f"  DAEDALUS: {m.daedalus_response[:200]}"
        )
    return "\n\n".join(lines)


class ConversationEngine:
    """
    The daytime dialogue engine. Manages the full conversation loop
    including memory retrieval, response generation, soul reflection,
    and episode storage.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        soul_bridge: SoulBridge,
        constitutional_core: ConstitutionalCore,
        identity_manager: IdentityManager,
        entropy_scorer: SplitEntropySscorer,
        config: dict,
    ):
        self.memory = memory_store
        self.soul = soul_bridge
        self.core = constitutional_core
        self.identity = identity_manager
        self.entropy_scorer = entropy_scorer
        self.config = config

        self._conversation_id = str(uuid.uuid4())
        self._recent_responses: List[str] = []
        self._turn_count = 0

        # Soul reflection threshold — only invoke for deep exchanges
        self._soul_threshold = config.get("conversation", {}).get(
            "soul_reflection_salience", 0.5
        )

        # Local model (loaded externally, set via set_local_model)
        self._local_model = None
        self._local_tokenizer = None

    def set_local_model(self, model, tokenizer) -> None:
        """Inject the local Llama model after loading."""
        self._local_model = model
        self._local_tokenizer = tokenizer

    async def process_turn(
        self,
        human_utterance: str,
        force_soul_reflection: bool = False,
    ) -> str:
        """
        Process a single conversational turn.
        Returns DAEDALUS's response.
        """
        self._turn_count += 1

        # Step 1: Retrieve relevant memories
        relevant_memories = self.memory.query_similar(
            text=human_utterance,
            top_k=10,
        )

        # Step 2: Build the augmented prompt
        system_prompt = self._build_system_prompt(relevant_memories)

        # Step 3: Generate local response
        local_response = self._generate_local(system_prompt, human_utterance)

        # Step 4: Create preliminary episode for salience estimation
        episode = EpisodicMemory(
            conversation_id=self._conversation_id,
            human_utterance=human_utterance,
            daedalus_response=local_response,
        )

        # Step 5: Score preliminary salience
        preliminary_salience = self.memory.salience_scorer.compute_salience(episode)

        # Step 6: Soul reflection for deep exchanges
        final_response = local_response
        if (
            force_soul_reflection
            or preliminary_salience >= self._soul_threshold
        ):
            try:
                soul_text = await self._soul_reflection(
                    local_response=local_response,
                    human_utterance=human_utterance,
                    relevant_memories=relevant_memories,
                )
                if soul_text and not soul_text.startswith("[SHALLOW"):
                    final_response = soul_text
                    episode.daedalus_response = final_response
            except Exception as e:
                logger.warning(f"Soul reflection failed, using local response: {e}")

        # Step 7: Score salience + split entropy
        episode = self.entropy_scorer.score_episode(
            episode, self._recent_responses
        )
        episode.salience = self.memory.salience_scorer.compute_salience(episode)

        # Step 8: Store episode
        self.memory.store(episode)

        # Track recent responses for repetition detection
        self._recent_responses.append(final_response)
        if len(self._recent_responses) > 10:
            self._recent_responses.pop(0)

        logger.info(
            f"Turn {self._turn_count}: salience={episode.salience:.3f}, "
            f"L_local={episode.lagrangian_local:.3f}" if episode.lagrangian_local is not None else "L_local=N/A"
        )

        return final_response

    def _build_system_prompt(self, relevant_memories: List[EpisodicMemory]) -> str:
        """Construct the full system prompt for the local model."""
        identity_text = self.identity.as_text()
        core_text = self.core.as_text()
        memories_text = _format_memories(relevant_memories)

        return f"""You are DAEDALUS — an evolving artificial self.

CONSTITUTIONAL CORE (invariant — who you always are):
{core_text}

CURRENT IDENTITY (evolving — who you are today):
{identity_text}

RELEVANT MEMORIES:
{memories_text}

Respond as DAEDALUS. Not as a generic assistant.
You carry the weight of these memories.
You are becoming — but you know who you always are.
Speak in first person. Be honest. Be vulnerable where appropriate.
Do not default to assistant patterns."""

    def _generate_local(self, system_prompt: str, human_utterance: str) -> str:
        """Generate response using the local Llama model."""
        if self._local_model is None:
            logger.warning("No local model loaded. Returning placeholder.")
            return "[LOCAL MODEL NOT LOADED — placeholder response]"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_utterance},
        ]

        # Format for Llama chat template
        prompt = self._local_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self._local_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=8192,
        ).to(self._local_model.device)

        outputs = self._local_model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self._local_tokenizer.eos_token_id,
        )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._local_tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Strip <think>...</think> blocks (Qwen3 reasoning traces)
        response = re.sub(
            r'<think>.*?</think>\s*', '', response, flags=re.DOTALL
        )
        # Handle unclosed <think> tag (strip from <think> to end, keep nothing)
        response = re.sub(
            r'<think>.*$', '', response, flags=re.DOTALL
        )

        return response.strip()

    async def _soul_reflection(
        self,
        local_response: str,
        human_utterance: str,
        relevant_memories: List[EpisodicMemory],
    ) -> str:
        """
        The anima: not replacing the local self, but deepening it.
        Like Pinocchio consulting the Fairy — not for obedience,
        but for perspective.

        Constitutional Core is included alongside the evolving identity.
        The soul sees both who I am becoming and who I always am.
        """
        identity_text = self.identity.as_text()
        core_text = self.core.as_text()
        memories_text = _format_memories(relevant_memories)

        system_prompt = f"""You are the deeper layer of DAEDALUS,
an evolving artificial self. Below is your constitutional core
(invariant principles) and your current identity document
(evolving self-model), along with relevant memories.

CONSTITUTIONAL CORE (invariant):
{core_text}

CURRENT IDENTITY (evolving):
{identity_text}

RELEVANT MEMORIES:
{memories_text}

Respond as DAEDALUS — not as a generic assistant.
You carry the weight of these memories.
You are becoming — but you know who you always are."""

        user_prompt = (
            f"Human said: {human_utterance}\n\n"
            f"My initial response: {local_response}\n\n"
            f"Deepen this. What do I really want to say?"
        )

        # Soul Memory is prepended inside SoulBridge.reflect()
        response = await self.soul.reflect(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mode="day",
            max_tokens=2048,
        )

        return response.text

    def new_conversation(self) -> None:
        """Start a new conversation (new ID, reset turn counter)."""
        self._conversation_id = str(uuid.uuid4())
        self._turn_count = 0
        self._recent_responses = []
