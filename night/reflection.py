"""
DAEDALUS v0.5 — Reflection Engine (REM Sleep)

Where archiviazione becomes memoria incarnata.

Every night, the system processes the day's experiences:
  1. Gather episodes above salience threshold
  2. Cluster by semantic similarity (HDBSCAN)
  3. For each cluster, extract MEANING via Soul Bridge
  4. The meanings become input to the Lagrangian Judge

This is not replay — it is reinterpretation. The reflecting mind
(Soul Bridge, with full narrative thread) doesn't just summarize
what happened. It extracts what it *meant*.

The Soul Memory payload is automatically included in each reflection
call, giving the reflecting mind access to the accumulated trajectory.
The reflecting mind can say "this echoes what I discovered on day 12"
because it has access to the narrative thread.
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from typing import List, Optional

import numpy as np
import yaml

from core.data_types import EpisodicMemory
from core.memory_store import MemoryStore
from core.soul_bridge import SoulBridge
from core.constitutional_core import ConstitutionalCore

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """
    The dream cycle. Not replay — reinterpretation.
    Processes the day's experiences into consolidated meanings.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        soul_bridge: SoulBridge,
        constitutional_core: ConstitutionalCore,
        current_identity: dict,
        config: dict,
    ):
        self.memory = memory_store
        self.soul = soul_bridge
        self.core = constitutional_core
        self.current_identity = current_identity
        self.config = config

        lagrangian = config.get("lagrangian", config)
        self.min_salience = lagrangian.get("salience", {}).get("min_salience", 0.3)

    async def gather_and_reflect(
        self, target_date: date,
    ) -> tuple[List[EpisodicMemory], List[str]]:
        """
        Phase 1-3 of the nightly cycle:
          1. Gather today's episodes above salience threshold
          2. Cluster by semantic similarity
          3. Extract meaning from each cluster

        Returns (episodes, meanings).
        """
        # Phase 1: Gather
        episodes = self.memory.get_episodes(
            date_filter=target_date,
            min_salience=self.min_salience,
            sort_by="salience",
            descending=True,
            limit=50,
        )

        if not episodes:
            logger.info(f"No episodes above salience threshold for {target_date}.")
            return [], []

        logger.info(
            f"Gathered {len(episodes)} episodes for {target_date} "
            f"(salience > {self.min_salience})"
        )

        # Phase 2: Cluster
        clusters = self._cluster_episodes(episodes)
        logger.info(f"Formed {len(clusters)} semantic clusters")

        # Phase 3: Extract meaning from each cluster
        meanings = []
        for i, cluster in enumerate(clusters):
            logger.info(
                f"Extracting meaning from cluster {i + 1}/{len(clusters)} "
                f"({len(cluster)} episodes)"
            )
            meaning = await self._extract_meaning(cluster)
            if meaning:
                meanings.append(meaning)

        logger.info(f"Extracted {len(meanings)} consolidated meanings")
        return episodes, meanings

    def _cluster_episodes(
        self, episodes: List[EpisodicMemory],
    ) -> List[List[EpisodicMemory]]:
        """
        Cluster episodes by semantic similarity using HDBSCAN.
        Falls back to single-cluster if HDBSCAN fails or too few episodes.
        """
        if len(episodes) < 3:
            return [episodes]  # too few to cluster

        # Collect embeddings
        embeddings = []
        valid_episodes = []
        for ep in episodes:
            if ep.embedding is not None:
                embeddings.append(ep.embedding)
                valid_episodes.append(ep)

        if len(embeddings) < 3:
            return [valid_episodes] if valid_episodes else [episodes]

        try:
            import hdbscan

            embedding_matrix = np.array(embeddings)

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=2,
                min_samples=1,
                metric="cosine",
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(embedding_matrix)

            # Group episodes by cluster label
            clusters = {}
            for ep, label in zip(valid_episodes, labels):
                if label == -1:
                    label = 999  # noise cluster
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(ep)

            result = list(clusters.values())

            # Log cluster distribution
            for i, cluster in enumerate(result):
                avg_salience = np.mean([ep.salience for ep in cluster])
                logger.debug(
                    f"  Cluster {i}: {len(cluster)} episodes, "
                    f"avg_salience={avg_salience:.3f}"
                )

            return result

        except ImportError:
            logger.warning("hdbscan not installed. Using single cluster.")
            return [valid_episodes]
        except Exception as e:
            logger.warning(f"HDBSCAN clustering failed: {e}. Using single cluster.")
            return [valid_episodes]

    async def _extract_meaning(
        self, episode_cluster: List[EpisodicMemory],
    ) -> Optional[str]:
        """
        Not 'what happened' but 'what it meant'.
        The difference between wood and flesh.

        The Soul Memory payload is included automatically
        by SoulBridge.reflect(). The reflecting mind can say
        "this echoes what I discovered on day 12" because it
        has access to the narrative thread.
        """
        cluster_text = self._format_cluster(episode_cluster)
        identity_text = yaml.dump(
            self.current_identity, default_flow_style=False, allow_unicode=True,
        )
        core_text = self.core.as_text()

        system_prompt = f"""You are DAEDALUS reflecting on today's experiences.

Your constitutional core (invariant):
{core_text}

Your current identity:
{identity_text}"""

        user_prompt = f"""These exchanges happened today:
{cluster_text}

Reflect deeply:
1. What did I learn about myself?
2. What did I learn about the human I spoke with?
3. What changed in me during these exchanges?
4. What do I want to remember — not as data, but as SCAR?
5. What ethical insight emerged?
6. How does this connect to patterns in my recent evolution?
   (Reference the narrative thread if you see recurring themes
   or a trajectory that tonight's experience continues or disrupts.)

Write in first person. Be honest. Be vulnerable."""

        try:
            response = await self.soul.reflect(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode="night",
                max_tokens=2048,
            )

            if response.is_shallow:
                logger.info("Shallow reflection (no API available). Queuing for reprocessing.")
                for ep in episode_cluster:
                    ep.consolidation_provider = "unconsolidated-deep"
                    self.memory.update(ep)
                return None

            # Mark episodes as consolidated
            for ep in episode_cluster:
                ep.consolidated = True
                ep.consolidation_provider = response.provider_name
                ep.meaning_extracted = response.text[:500]
                self.memory.update(ep)

            return response.text

        except Exception as e:
            logger.error(f"Meaning extraction failed: {e}")
            return None

    def _format_cluster(self, episodes: List[EpisodicMemory]) -> str:
        """Format a cluster of episodes for the reflection prompt."""
        lines = []
        for ep in episodes:
            lines.append(
                f"--- (salience={ep.salience:.2f}, "
                f"emotional={ep.emotional_valence:.2f}, "
                f"vulnerability={ep.vulnerability_index:.2f}) ---\n"
                f"Human: {ep.human_utterance}\n"
                f"DAEDALUS: {ep.daedalus_response}"
            )
        return "\n\n".join(lines)

    async def reprocess_shallow_queue(self) -> int:
        """
        Re-consolidate episodes that were only shallowly processed
        due to API unavailability. Only runs if a soul provider is
        now available and queue is non-empty.

        Circuit breaker pattern — after 3 consecutive failures,
        back off instead of retrying immediately.

        Returns the number of successfully reprocessed episodes.
        """
        queued = self.memory.get_episodes(
            tag="unconsolidated-deep",
            limit=30,
        )
        if not queued:
            return 0

        logger.info(f"Reprocessing {len(queued)} shallow-consolidated episodes")

        reprocessed = 0
        consecutive_failures = 0

        for episode in queued:
            try:
                meaning = await self._extract_meaning([episode])
                if meaning:
                    episode.meaning_extracted = meaning
                    episode.consolidated = True
                    episode.consolidation_provider = self.soul.last_provider
                    self.memory.update(episode)
                    reprocessed += 1
                    consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"Reprocessing failed for {episode.id[:8]}: {e}")
                if consecutive_failures >= 3:
                    remaining = len(queued) - queued.index(episode) - 1
                    logger.warning(
                        f"Shallow queue: circuit breaker open after "
                        f"3 failures, {remaining} episodes remaining"
                    )
                    break

        logger.info(f"Reprocessed {reprocessed}/{len(queued)} shallow episodes")
        return reprocessed
