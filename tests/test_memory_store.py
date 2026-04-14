"""
Tests for core/memory_store.py — SalienceScorer and MemoryStore operations.

Uses a real ChromaDB instance in a temp directory (no GPU needed).
Embedding model is mocked.
"""

import json
import sys
from datetime import datetime, date
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_types import EpisodicMemory
from core.memory_store import SalienceScorer, MemoryStore


# ── SalienceScorer ──

class TestSalienceScorer:
    @pytest.fixture
    def scorer(self, sample_config):
        return SalienceScorer(sample_config["lagrangian"])

    def test_zero_salience_for_empty_memory(self, scorer):
        ep = EpisodicMemory()
        score = scorer.compute_salience(ep)
        assert score == 0.0

    def test_high_salience_for_extreme_memory(self, scorer):
        ep = EpisodicMemory(
            emotional_valence=0.9,
            relational_depth=0.8,
            novelty_score=0.9,
            self_model_impact=0.7,
            vulnerability_index=0.6,
        )
        score = scorer.compute_salience(ep, external_relevance=0.8)
        assert score > 0.5

    def test_salience_bounded_zero_one(self, scorer):
        ep = EpisodicMemory(
            emotional_valence=1.0,
            relational_depth=1.0,
            novelty_score=1.0,
            self_model_impact=1.0,
            vulnerability_index=1.0,
        )
        score = scorer.compute_salience(ep, external_relevance=1.0)
        assert 0.0 <= score < 1.0  # tanh never reaches 1.0

    def test_external_relevance_increases_salience(self, scorer):
        ep = EpisodicMemory(
            emotional_valence=0.5,
            relational_depth=0.5,
            novelty_score=0.5,
        )
        score_without = scorer.compute_salience(ep, external_relevance=0.0)
        score_with = scorer.compute_salience(ep, external_relevance=0.8)
        assert score_with > score_without

    def test_saturation_nonlinear(self, scorer):
        ep_low = EpisodicMemory(novelty_score=0.1)
        ep_high = EpisodicMemory(novelty_score=0.9)
        s_low = scorer.compute_salience(ep_low)
        s_high = scorer.compute_salience(ep_high)
        # Due to tanh saturation, doubling input doesn't double output
        assert s_high > s_low
        assert s_high < s_low * 10  # nonlinear compression


# ── MemoryStore (with real ChromaDB, mocked embedder) ──

class TestMemoryStore:
    @pytest.fixture
    def store(self, tmp_path, sample_config, mock_embedder):
        """Create a MemoryStore with real ChromaDB in temp dir, mocked embedder."""
        with patch("core.memory_store._get_embedding_model", return_value=mock_embedder):
            ms = MemoryStore(
                config=sample_config,
                chroma_path=str(tmp_path / "chroma"),
                episodes_path=str(tmp_path / "episodes"),
            )
            return ms

    def test_store_and_retrieve(self, store):
        ep = EpisodicMemory(
            human_utterance="What is meaning?",
            daedalus_response="Meaning emerges from dialogue.",
            emotional_valence=0.5,
        )
        stored = store.store(ep)
        assert stored.embedding is not None
        assert stored.salience >= 0.0
        assert stored.novelty_score >= 0.0

        # Retrieve by ID
        retrieved = store.get_by_id(stored.id)
        assert retrieved is not None
        assert retrieved.human_utterance == "What is meaning?"

    def test_count(self, store):
        assert store.count() == 0
        store.store(EpisodicMemory(
            human_utterance="A", daedalus_response="B",
        ))
        assert store.count() == 1

    def test_novelty_first_memory_is_maximal(self, store):
        emb = np.random.randn(1024).astype(np.float32)
        novelty = store.compute_novelty(emb)
        assert novelty == 1.0

    def test_novelty_decreases_with_similar(self, store):
        ep = EpisodicMemory(
            human_utterance="What is meaning?",
            daedalus_response="Meaning emerges.",
        )
        store.store(ep)

        # Query novelty with the same text
        emb = store.embed("Human: What is meaning?\nDAEDALUS: Meaning emerges.")
        novelty = store.compute_novelty(emb)
        # Should be less than 1.0 since a similar embedding exists
        assert novelty < 1.0

    def test_get_episodes_empty(self, store):
        episodes = store.get_episodes()
        assert episodes == []

    def test_get_episodes_with_date_filter(self, store):
        ep1 = EpisodicMemory(
            human_utterance="Morning",
            daedalus_response="Good morning",
            timestamp=datetime(2026, 4, 10, 9, 0),
        )
        ep2 = EpisodicMemory(
            human_utterance="Evening",
            daedalus_response="Good evening",
            timestamp=datetime(2026, 4, 11, 20, 0),
        )
        store.store(ep1)
        store.store(ep2)

        # Filter by Apr 10
        episodes = store.get_episodes(date_filter=date(2026, 4, 10))
        assert len(episodes) == 1
        assert episodes[0].human_utterance == "Morning"

        # Filter by Apr 11
        episodes = store.get_episodes(date_filter=date(2026, 4, 11))
        assert len(episodes) == 1
        assert episodes[0].human_utterance == "Evening"

    def test_get_episodes_date_filter_no_limit_applied_to_chromadb(self, store):
        """
        Regression test for the critical get_episodes limit bug.
        When date_filter is set, ChromaDB should NOT apply a limit,
        because the limit would silently drop episodes from recent dates.
        """
        # Store 5 episodes across 2 dates
        for i in range(3):
            store.store(EpisodicMemory(
                human_utterance=f"Old {i}",
                daedalus_response=f"Response {i}",
                timestamp=datetime(2026, 4, 9, 10 + i, 0),
            ))
        for i in range(2):
            store.store(EpisodicMemory(
                human_utterance=f"New {i}",
                daedalus_response=f"Response {i}",
                timestamp=datetime(2026, 4, 10, 10 + i, 0),
            ))

        # With date filter, should find exactly 2 episodes for Apr 10
        # even with limit=3 (limit applies AFTER date filtering)
        episodes = store.get_episodes(date_filter=date(2026, 4, 10), limit=3)
        assert len(episodes) == 2

    def test_get_episodes_sorted_by_salience(self, store):
        ep_low = EpisodicMemory(
            human_utterance="Low", daedalus_response="Low",
            emotional_valence=0.1,
        )
        ep_high = EpisodicMemory(
            human_utterance="High vulnerability and emotion deeply profoundly",
            daedalus_response="High depth with vulnerability and raw emotion",
            emotional_valence=0.9, relational_depth=0.8,
            vulnerability_index=0.7,
        )
        store.store(ep_low)
        store.store(ep_high)

        episodes = store.get_episodes(sort_by="salience", descending=True)
        assert len(episodes) == 2
        assert episodes[0].salience >= episodes[1].salience

    def test_query_similar(self, store):
        store.store(EpisodicMemory(
            human_utterance="What is consciousness?",
            daedalus_response="A deep mystery.",
        ))
        store.store(EpisodicMemory(
            human_utterance="How to bake bread?",
            daedalus_response="Mix flour and water.",
        ))

        results = store.query_similar("consciousness and awareness")
        assert len(results) >= 1

    def test_update_episode(self, store):
        ep = EpisodicMemory(
            human_utterance="Test",
            daedalus_response="Response",
        )
        stored = store.store(ep)
        stored.consolidated = True
        stored.meaning_extracted = "A key insight."
        store.update(stored)

        retrieved = store.get_by_id(stored.id)
        assert retrieved.consolidated is True
        assert retrieved.meaning_extracted == "A key insight."

    def test_episode_json_file_created(self, store, tmp_path):
        ep = EpisodicMemory(
            human_utterance="Test", daedalus_response="Response",
        )
        stored = store.store(ep)
        json_path = tmp_path / "episodes" / f"{stored.id}.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["human_utterance"] == "Test"
