"""
DAEDALUS v0.5 — Episodic Memory Store (The Hippocampus)

ChromaDB-backed vector store with BGE-M3 embeddings.
Every conversational exchange is encoded as an embedding
with rich metadata — salience, emotional valence, Lagrangian markers.

Not all experiences weigh equally. The salience scorer implements
a multi-factor formula with nonlinear saturation: extreme experiences
leave disproportionate marks.

v0.5: Split entropy markers (delta_S_noise, delta_S_exploration)
are stored alongside each episode for the Lagrangian Judge.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

from core.data_types import EpisodicMemory

logger = logging.getLogger(__name__)

# Lazy-loaded models
_embedding_model = None
_chroma_client = None
_collection = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("BAAI/bge-m3")
        logger.info("BGE-M3 embedding model loaded for memory store.")
    return _embedding_model


class SalienceScorer:
    """
    Pinocchio's scars: not all moments shape you equally.
    High salience = high deformation of the self-model.

    v0.5: Also computes split entropy markers for each episode.
    v0.6: external_relevance factor added (grounding score).
          Episodes about the real world get higher salience than
          episodes of pure self-reflection.
    """

    def __init__(self, config: dict):
        salience_config = config.get("salience", {})
        weights = salience_config.get("weights", {})
        # v0.6: rebalanced weights to accommodate external_relevance
        self.weights = {
            "emotional": weights.get("emotional", 0.20),
            "relational": weights.get("relational", 0.20),
            "novelty": weights.get("novelty", 0.15),
            "self_impact": weights.get("self_impact", 0.15),
            "vulnerability": weights.get("vulnerability", 0.10),
            "external_relevance": weights.get("external_relevance", 0.20),
        }
        self.saturation_factor = salience_config.get("saturation_factor", 2.0)

    def compute_salience(
        self,
        memory: EpisodicMemory,
        external_relevance: float = 0.0,
    ) -> float:
        """
        Compute composite salience score.
        Non-linear: extreme experiences leave disproportionate marks.

        Args:
            memory: The episodic memory to score.
            external_relevance: Grounding score G from core/grounding.py [0, 1].
                                World-directed episodes get higher salience.
        """
        raw = (
            self.weights["emotional"] * abs(memory.emotional_valence)
            + self.weights["relational"] * memory.relational_depth
            + self.weights["novelty"] * memory.novelty_score
            + self.weights["self_impact"] * memory.self_model_impact
            + self.weights["vulnerability"] * memory.vulnerability_index
            + self.weights["external_relevance"] * external_relevance
        )

        # Saturate: tanh maps [0, inf) -> [0, 1) with disproportionate weight to extremes
        return float(np.tanh(self.saturation_factor * raw))


class MemoryStore:
    """
    ChromaDB-backed episodic memory with BGE-M3 embeddings.
    The hippocampus of DAEDALUS — fast episodic capture with salience weighting.
    """

    COLLECTION_NAME = "daedalus_episodic"

    def __init__(
        self,
        config: dict,
        chroma_path: str = "memory/chroma_db",
        episodes_path: str = "memory/episodes",
    ):
        self.config = config
        self._chroma_path = Path(chroma_path)
        self._episodes_path = Path(episodes_path)
        self._episodes_path.mkdir(parents=True, exist_ok=True)

        self.salience_scorer = SalienceScorer(
            config.get("lagrangian", config)
        )

        # Initialize ChromaDB
        self._client = None
        self._collection = None

    def _ensure_chroma(self):
        """Lazy initialization of ChromaDB client and collection."""
        if self._client is None:
            import chromadb
            self._chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self._chroma_path)
            )
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB initialized at {self._chroma_path}, "
                f"collection '{self.COLLECTION_NAME}' "
                f"({self._collection.count()} entries)"
            )

    def embed(self, text: str) -> np.ndarray:
        """Compute BGE-M3 embedding for text."""
        model = _get_embedding_model()
        return np.array(
            model.encode(text, normalize_embeddings=True),
            dtype=np.float32,
        )

    def compute_novelty(self, embedding: np.ndarray, top_k: int = 5) -> float:
        """
        Compute novelty as mean cosine distance from nearest existing memories.
        High novelty = this experience is far from anything stored.
        """
        self._ensure_chroma()
        if self._collection.count() == 0:
            return 1.0  # first memory is maximally novel

        results = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(top_k, self._collection.count()),
        )

        if not results["distances"] or not results["distances"][0]:
            return 1.0

        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Normalize to [0, 1]: distance/2
        distances = results["distances"][0]
        mean_distance = sum(distances) / len(distances)
        return min(1.0, mean_distance / 2.0)

    def store(self, memory: EpisodicMemory) -> EpisodicMemory:
        """
        Store an episodic memory: embed, compute salience + novelty, persist.
        Returns the memory with computed fields populated.
        """
        self._ensure_chroma()

        # Embed the combined exchange
        combined_text = (
            f"Human: {memory.human_utterance}\n"
            f"DAEDALUS: {memory.daedalus_response}"
        )
        embedding = self.embed(combined_text)
        memory.embedding = embedding

        # Compute novelty from existing memories
        memory.novelty_score = self.compute_novelty(embedding)

        # Compute composite salience
        memory.salience = self.salience_scorer.compute_salience(memory)

        # Store in ChromaDB
        metadata = memory.to_dict()
        # ChromaDB metadata values must be str, int, float, or bool
        metadata_clean = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                metadata_clean[k] = v
            elif isinstance(v, list):
                metadata_clean[k] = json.dumps(v)
            elif isinstance(v, datetime):
                metadata_clean[k] = v.isoformat()

        self._collection.add(
            ids=[memory.id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata_clean],
            documents=[combined_text],
        )

        # Also save full episode as JSON (for nightly processing)
        episode_file = self._episodes_path / f"{memory.id}.json"
        episode_file.write_text(json.dumps(memory.to_dict(), indent=2))

        logger.debug(
            f"Episode stored: id={memory.id[:8]}..., "
            f"salience={memory.salience:.3f}, "
            f"novelty={memory.novelty_score:.3f}"
        )

        return memory

    def get_by_id(self, memory_id: str) -> Optional[EpisodicMemory]:
        """Retrieve a specific episode by ID."""
        self._ensure_chroma()
        try:
            result = self._collection.get(
                ids=[memory_id],
                include=["embeddings", "metadatas", "documents"],
            )
            if not result["ids"]:
                return None

            metadata = result["metadatas"][0]
            embedding = (
                np.array(result["embeddings"][0], dtype=np.float32)
                if result.get("embeddings") is not None and len(result["embeddings"]) > 0
                else None
            )

            # Deserialize list fields
            if "themes" in metadata and isinstance(metadata["themes"], str):
                metadata["themes"] = json.loads(metadata["themes"])

            return EpisodicMemory.from_dict(metadata, embedding=embedding)

        except Exception as e:
            logger.error(f"Failed to retrieve episode {memory_id}: {e}")
            return None

    def get_episodes(
        self,
        date_filter: Optional[date] = None,
        min_salience: float = 0.0,
        sort_by: str = "salience",
        descending: bool = True,
        limit: int = 100,
        tag: Optional[str] = None,
    ) -> List[EpisodicMemory]:
        """
        Retrieve episodes with filtering and sorting.
        Used by the nightly cycle to gather the day's experiences.
        """
        self._ensure_chroma()

        # Build where clause
        where_clauses = []
        if min_salience > 0:
            where_clauses.append({"salience": {"$gte": min_salience}})
        if tag:
            where_clauses.append({"consolidation_provider": tag})

        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        # When date_filter is set, we must fetch ALL episodes first and filter
        # in Python, since ChromaDB can't filter on date substrings. Applying
        # limit to the ChromaDB query would silently drop episodes from recent
        # dates when older dates fill the limit.
        query_limit = None if date_filter else limit

        try:
            get_kwargs = {
                "include": ["embeddings", "metadatas", "documents"],
            }
            if where is not None:
                get_kwargs["where"] = where
            if query_limit is not None:
                get_kwargs["limit"] = query_limit

            result = self._collection.get(**get_kwargs)
        except Exception as e:
            logger.error(f"Episode query failed: {e}")
            return []

        episodes = []
        for i, mid in enumerate(result["ids"]):
            metadata = result["metadatas"][i]
            embedding = (
                np.array(result["embeddings"][i], dtype=np.float32)
                if result.get("embeddings") is not None and len(result["embeddings"]) > i
                else None
            )

            # Deserialize list fields
            if "themes" in metadata and isinstance(metadata["themes"], str):
                metadata["themes"] = json.loads(metadata["themes"])

            ep = EpisodicMemory.from_dict(metadata, embedding=embedding)

            # Date filter (post-query since ChromaDB metadata filtering is limited)
            if date_filter:
                if isinstance(ep.timestamp, str):
                    try:
                        ep_date = datetime.fromisoformat(ep.timestamp).date()
                    except ValueError:
                        ep_date = None
                elif isinstance(ep.timestamp, datetime):
                    ep_date = ep.timestamp.date()
                else:
                    ep_date = None
                    
                if ep_date != date_filter:
                    continue

            episodes.append(ep)

        # Sort
        if sort_by == "salience":
            episodes.sort(key=lambda e: e.salience, reverse=descending)
        elif sort_by == "timestamp":
            episodes.sort(key=lambda e: e.timestamp, reverse=descending)

        return episodes[:limit]

    def query_similar(
        self,
        text: str,
        top_k: int = 10,
        min_salience: float = 0.0,
    ) -> List[EpisodicMemory]:
        """
        Retrieve the most similar memories to a given text.
        Used during daytime conversation to populate context.
        """
        self._ensure_chroma()
        embedding = self.embed(text)

        where = {"salience": {"$gte": min_salience}} if min_salience > 0 else None

        results = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(top_k, max(1, self._collection.count())),
            where=where,
            include=["embeddings", "metadatas", "documents", "distances"],
        )

        episodes = []
        for i, mid in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            emb = (
                np.array(results["embeddings"][0][i], dtype=np.float32)
                if results.get("embeddings") is not None and len(results["embeddings"]) > 0 and len(results["embeddings"][0]) > i
                else None
            )

            if "themes" in metadata and isinstance(metadata["themes"], str):
                metadata["themes"] = json.loads(metadata["themes"])

            episodes.append(EpisodicMemory.from_dict(metadata, embedding=emb))

        return episodes

    def update(self, memory: EpisodicMemory) -> None:
        """Update an existing episode's metadata (e.g., after consolidation)."""
        self._ensure_chroma()
        metadata = memory.to_dict()
        metadata_clean = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                metadata_clean[k] = v
            elif isinstance(v, list):
                metadata_clean[k] = json.dumps(v)

        self._collection.update(
            ids=[memory.id],
            metadatas=[metadata_clean],
        )

        # Also update JSON file
        episode_file = self._episodes_path / f"{memory.id}.json"
        episode_file.write_text(json.dumps(memory.to_dict(), indent=2))

    def count(self) -> int:
        """Total number of stored episodes."""
        self._ensure_chroma()
        return self._collection.count()
