#!/usr/bin/env python3
"""
DAEDALUS v0.5.1 — Episode Repair Script

Re-scores existing episodes with the fixed salience metadata estimator
and syncs any missing episodes into ChromaDB.

This fixes the Day 0 problem where all episodes had 0.0 for
emotional_valence, relational_depth, self_model_impact, and
vulnerability_index, causing salience to be near-zero and
the nightly cycle to find no episodes above threshold.

Also strips <think>...</think> blocks from stored responses.

Usage:
    python scripts/repair_episodes.py
    python scripts/repair_episodes.py --dry-run   # preview without writing
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("daedalus.repair")


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 responses."""
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Repair episode metadata")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--episodes-dir", default="memory/episodes")
    args = parser.parse_args()

    episodes_dir = Path(args.episodes_dir)
    if not episodes_dir.exists():
        logger.error(f"Episodes directory not found: {episodes_dir}")
        sys.exit(1)

    # Load config for scorer
    config = {}
    lagrangian_path = Path("config/lagrangian.yaml")
    if lagrangian_path.exists():
        with open(lagrangian_path) as f:
            config["lagrangian"] = yaml.safe_load(f)

    from core.salience import SplitEntropySscorer
    from core.data_types import EpisodicMemory

    scorer = SplitEntropySscorer(config.get("lagrangian", {}))

    # Load salience scorer for recomputing composite salience
    from core.memory_store import SalienceScorer
    salience_scorer = SalienceScorer(config.get("lagrangian", {}))

    episode_files = sorted(episodes_dir.glob("*.json"))
    logger.info(f"Found {len(episode_files)} episode files to repair")

    repaired = 0
    think_stripped = 0
    recent_responses = []

    # Sort by timestamp for proper repetition detection
    episodes_with_paths = []
    for fpath in episode_files:
        try:
            data = json.loads(fpath.read_text())
            episodes_with_paths.append((fpath, data))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Skipping {fpath.name}: {e}")

    episodes_with_paths.sort(key=lambda x: x[1].get("timestamp", ""))

    for fpath, data in episodes_with_paths:
        modified = False

        # Strip think tags from response
        response = data.get("daedalus_response", "")
        cleaned_response = strip_think_tags(response)
        if cleaned_response != response:
            data["daedalus_response"] = cleaned_response
            modified = True
            think_stripped += 1

        # Build a temporary EpisodicMemory for scoring
        ep = EpisodicMemory(
            id=data.get("id", ""),
            human_utterance=data.get("human_utterance", ""),
            daedalus_response=data.get("daedalus_response", ""),
            novelty_score=data.get("novelty_score", 0.0),
        )

        # Re-score with the fixed scorer
        ep = scorer.score_episode(ep, recent_responses[-10:])

        # Recompute composite salience
        new_salience = salience_scorer.compute_salience(ep)

        old_salience = data.get("salience", 0.0)

        # Update fields
        data["emotional_valence"] = ep.emotional_valence
        data["relational_depth"] = ep.relational_depth
        data["self_model_impact"] = ep.self_model_impact
        data["vulnerability_index"] = ep.vulnerability_index
        data["philosophical_layer"] = ep.philosophical_layer
        data["delta_S_noise"] = ep.delta_S_noise
        data["delta_S_exploration"] = ep.delta_S_exploration
        data["delta_Ic"] = ep.delta_Ic
        data["lagrangian_local"] = ep.lagrangian_local
        data["salience"] = new_salience

        if new_salience != old_salience:
            modified = True

        if modified:
            repaired += 1
            if not args.dry_run:
                fpath.write_text(json.dumps(data, indent=2))

        recent_responses.append(data.get("daedalus_response", ""))

        logger.debug(
            f"  {fpath.name[:12]}... salience: {old_salience:.3f} -> {new_salience:.3f}"
        )

    # Summary
    above_threshold = 0
    for fpath, data in episodes_with_paths:
        if not args.dry_run:
            data = json.loads(fpath.read_text())
        sal = data.get("salience", 0.0)
        if sal >= 0.3:
            above_threshold += 1

    logger.info("=" * 60)
    logger.info(f"REPAIR {'(DRY RUN) ' if args.dry_run else ''}COMPLETE")
    logger.info(f"  Episodes processed: {len(episodes_with_paths)}")
    logger.info(f"  Episodes modified:  {repaired}")
    logger.info(f"  Think tags stripped: {think_stripped}")
    logger.info(f"  Episodes above salience 0.3: {above_threshold}/{len(episodes_with_paths)}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("Dry run — no files were modified. Run without --dry-run to apply.")

    # Phase 2: Sync ChromaDB
    if not args.dry_run:
        logger.info("Syncing episodes to ChromaDB...")
        sync_chromadb(episodes_dir, config)


def sync_chromadb(episodes_dir: Path, config: dict):
    """Re-index all episodes into ChromaDB to fix the count mismatch."""
    import chromadb
    from core.memory_store import MemoryStore

    store = MemoryStore(config, chroma_path="memory/chroma_db", episodes_path=str(episodes_dir))
    store._ensure_chroma()

    existing_ids = set(store._collection.get()["ids"])
    episode_files = sorted(episodes_dir.glob("*.json"))

    synced = 0
    for fpath in episode_files:
        try:
            data = json.loads(fpath.read_text())
            ep_id = data.get("id", "")
            if not ep_id or ep_id in existing_ids:
                continue

            # Embed and add to ChromaDB
            combined_text = (
                f"Human: {data.get('human_utterance', '')}\n"
                f"DAEDALUS: {data.get('daedalus_response', '')}"
            )
            embedding = store.embed(combined_text)

            metadata_clean = {}
            for k, v in data.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    metadata_clean[k] = v
                elif isinstance(v, list):
                    metadata_clean[k] = json.dumps(v)

            store._collection.add(
                ids=[ep_id],
                embeddings=[embedding.tolist()],
                metadatas=[metadata_clean],
                documents=[combined_text],
            )
            synced += 1

        except Exception as e:
            logger.warning(f"Failed to sync {fpath.name}: {e}")

    # Also update metadata for existing episodes
    updated = 0
    for fpath in episode_files:
        try:
            data = json.loads(fpath.read_text())
            ep_id = data.get("id", "")
            if not ep_id or ep_id not in existing_ids:
                continue

            metadata_clean = {}
            for k, v in data.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    metadata_clean[k] = v
                elif isinstance(v, list):
                    metadata_clean[k] = json.dumps(v)

            store._collection.update(
                ids=[ep_id],
                metadatas=[metadata_clean],
            )
            updated += 1

        except Exception as e:
            logger.warning(f"Failed to update {fpath.name}: {e}")

    total = store._collection.count()
    logger.info(f"ChromaDB sync: {synced} added, {updated} updated, {total} total entries")


if __name__ == "__main__":
    main()
