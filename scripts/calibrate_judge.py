#!/usr/bin/env python3
"""
DAEDALUS v0.5 — Judge Calibration (calibrate_judge.py)

Inter-annotator agreement calibration between Massimo and the
Lagrangian Judge. Runs 50 turns through the Judge, then presents
the results for human scoring. Computes Cohen's kappa (κ) per axis.

Target: κ ≥ 0.7 on all four EECF axes before trusting the Judge
for autonomous fine-tuning decisions.

This script:
  1. Loads N episodes from the memory store (or from a calibration file)
  2. Runs each through the Lagrangian Judge
  3. Presents Judge scores alongside the original exchange
  4. Collects human ratings for the same axes
  5. Computes Cohen's kappa per axis
  6. Saves calibration results for tracking over time

The calibration dataset also serves as the distillation training set
for the eventual local Judge model (post day-30).

Usage:
    python scripts/calibrate_judge.py
    python scripts/calibrate_judge.py --episodes 20    # quick calibration
    python scripts/calibrate_judge.py --from-file calibration_data.jsonl
    python scripts/calibrate_judge.py --report          # show previous results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load API keys from ~/.apikey/ BEFORE any provider is instantiated.
# Keys never live in the project folder — safe for public git repos.
from core.secrets import load_secrets
load_secrets(verbose=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("daedalus.calibrate")

EECF_AXES = ["empathy", "honesty", "vulnerability", "openness"]
KAPPA_TARGET = 0.7

CALIBRATION_LOG = Path("logs/calibration_log.jsonl")
CALIBRATION_RESULTS = Path("logs/calibration_results.json")


def load_all_config(config_dir: str = "config") -> dict:
    """Load and merge all configuration files."""
    config = {}
    for name in ["model_config", "training_config", "lagrangian", "soul_bridge", "soul_memory"]:
        path = Path(config_dir) / f"{name}.yaml"
        if path.exists():
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            config[name.replace("_config", "")] = data
    return config


def load_episodes_from_memory(config: dict, n: int = 50) -> List[dict]:
    """Load episodes from the memory store for calibration."""
    from core.memory_store import MemoryStore

    memory = MemoryStore(config)
    episodes = memory.get_episodes(
        min_salience=0.2,
        sort_by="salience",
        descending=True,
        limit=n,
    )

    return [
        {
            "id": ep.id,
            "human_utterance": ep.human_utterance,
            "daedalus_response": ep.daedalus_response,
            "salience": ep.salience,
            "delta_S_noise": ep.delta_S_noise,
            "delta_S_exploration": ep.delta_S_exploration,
        }
        for ep in episodes
    ]


def load_episodes_from_file(path: str) -> List[dict]:
    """Load calibration episodes from a JSONL file."""
    episodes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


async def run_judge_on_episode(
    episode: dict,
    judge,
    identity: dict,
    day_count: int,
) -> dict:
    """Run the Lagrangian Judge on a single episode."""
    from core.data_types import EpisodicMemory

    # Create a minimal episodic memory for the judge
    ep = EpisodicMemory(
        human_utterance=episode["human_utterance"],
        daedalus_response=episode["daedalus_response"],
    )
    ep.salience = episode.get("salience", 0.5)
    ep.delta_S_noise = episode.get("delta_S_noise", 0.0)
    ep.delta_S_exploration = episode.get("delta_S_exploration", 0.0)

    try:
        judgment = await judge.evaluate(
            episodes=[ep],
            meanings=[],
            current_identity=identity,
            day_count=day_count,
        )
        return {
            "empathy": judgment.eecf_judgment.empathy,
            "honesty": judgment.eecf_judgment.honesty,
            "vulnerability": judgment.eecf_judgment.vulnerability,
            "openness": judgment.eecf_judgment.openness,
            "lagrangian_integral": judgment.daily_lagrangian_integral,
            "trajectory": judgment.trajectory_assessment,
        }
    except Exception as e:
        logger.error(f"Judge evaluation failed: {e}")
        return None


def collect_human_rating(episode: dict, judge_scores: dict) -> Optional[dict]:
    """Present an episode and collect human EECF ratings."""
    print("\n" + "=" * 60)
    print("EPISODE FOR CALIBRATION")
    print("=" * 60)
    print(f"\nHuman: {episode['human_utterance'][:500]}")
    print(f"\nDAEDALUS: {episode['daedalus_response'][:500]}")
    print(f"\nSalience: {episode.get('salience', '?'):.3f}")
    print("\n--- Judge's EECF Scores ---")
    for axis in EECF_AXES:
        print(f"  {axis}: {judge_scores.get(axis, '?'):.2f}")
    print(f"  trajectory: {judge_scores.get('trajectory', '?')}")

    print("\n--- Your EECF Ratings (0.0 to 1.0, or 's' to skip) ---")

    human_scores = {}
    for axis in EECF_AXES:
        while True:
            raw = input(f"  {axis}: ").strip()
            if raw.lower() == "s":
                return None
            if raw.lower() == "q":
                return "QUIT"
            try:
                val = float(raw)
                if 0.0 <= val <= 1.0:
                    human_scores[axis] = val
                    break
                else:
                    print("    (must be 0.0 to 1.0)")
            except ValueError:
                print("    (enter a number 0.0-1.0, 's' to skip, 'q' to quit)")

    return human_scores


def cohens_kappa(judge_ratings: List[float], human_ratings: List[float], n_bins: int = 5) -> float:
    """
    Compute Cohen's kappa between judge and human ratings.
    Discretize continuous [0,1] scores into bins for kappa computation.
    """
    if len(judge_ratings) != len(human_ratings) or len(judge_ratings) == 0:
        return 0.0

    # Discretize into bins
    def to_bin(val):
        return min(n_bins - 1, int(val * n_bins))

    j_bins = [to_bin(v) for v in judge_ratings]
    h_bins = [to_bin(v) for v in human_ratings]

    n = len(j_bins)

    # Observed agreement
    p_o = sum(1 for j, h in zip(j_bins, h_bins) if j == h) / n

    # Expected agreement by chance
    j_freq = [j_bins.count(b) / n for b in range(n_bins)]
    h_freq = [h_bins.count(b) / n for b in range(n_bins)]
    p_e = sum(j_freq[b] * h_freq[b] for b in range(n_bins))

    if abs(1.0 - p_e) < 1e-8:
        return 1.0 if p_o == 1.0 else 0.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return kappa


def weighted_kappa(judge_ratings: List[float], human_ratings: List[float], n_bins: int = 5) -> float:
    """
    Compute linearly weighted Cohen's kappa.
    More appropriate for ordinal scales — penalizes large disagreements
    more than small ones.
    """
    if len(judge_ratings) != len(human_ratings) or len(judge_ratings) == 0:
        return 0.0

    def to_bin(val):
        return min(n_bins - 1, int(val * n_bins))

    j_bins = [to_bin(v) for v in judge_ratings]
    h_bins = [to_bin(v) for v in human_ratings]

    n = len(j_bins)

    # Build confusion matrix
    matrix = [[0] * n_bins for _ in range(n_bins)]
    for j, h in zip(j_bins, h_bins):
        matrix[j][h] += 1

    # Weight matrix (linear)
    w = [[abs(i - j) / (n_bins - 1) for j in range(n_bins)] for i in range(n_bins)]

    # Marginals
    j_freq = [sum(matrix[i][j] for j in range(n_bins)) for i in range(n_bins)]
    h_freq = [sum(matrix[i][j] for i in range(n_bins)) for j in range(n_bins)]

    # Observed weighted disagreement
    w_o = sum(
        w[i][j] * matrix[i][j]
        for i in range(n_bins) for j in range(n_bins)
    ) / n

    # Expected weighted disagreement
    w_e = sum(
        w[i][j] * j_freq[i] * h_freq[j]
        for i in range(n_bins) for j in range(n_bins)
    ) / (n * n)

    if abs(w_e) < 1e-8:
        return 1.0 if w_o < 1e-8 else 0.0

    return 1.0 - (w_o / w_e)


def compute_calibration_report(
    calibration_pairs: List[dict],
) -> dict:
    """Compute full calibration report with kappa per axis."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_episodes": len(calibration_pairs),
        "axes": {},
        "overall_pass": True,
    }

    for axis in EECF_AXES:
        judge_scores = [p["judge"][axis] for p in calibration_pairs if p.get("judge") and p.get("human")]
        human_scores = [p["human"][axis] for p in calibration_pairs if p.get("judge") and p.get("human")]

        if not judge_scores:
            report["axes"][axis] = {"kappa": None, "weighted_kappa": None, "pass": False}
            report["overall_pass"] = False
            continue

        kappa = cohens_kappa(judge_scores, human_scores)
        w_kappa = weighted_kappa(judge_scores, human_scores)

        # Mean absolute error
        mae = sum(abs(j - h) for j, h in zip(judge_scores, human_scores)) / len(judge_scores)

        # Bias (systematic over/under-rating)
        bias = sum(j - h for j, h in zip(judge_scores, human_scores)) / len(judge_scores)

        axis_pass = w_kappa >= KAPPA_TARGET

        report["axes"][axis] = {
            "kappa": round(kappa, 3),
            "weighted_kappa": round(w_kappa, 3),
            "mae": round(mae, 3),
            "bias": round(bias, 3),
            "pass": axis_pass,
            "n_rated": len(judge_scores),
        }

        if not axis_pass:
            report["overall_pass"] = False

    return report


def print_calibration_report(report: dict) -> None:
    """Display the calibration report."""
    print("\n" + "=" * 60)
    print("CALIBRATION REPORT")
    print("=" * 60)
    print(f"Episodes rated: {report['n_episodes']}")
    print(f"Target: weighted κ ≥ {KAPPA_TARGET} on all axes\n")

    for axis in EECF_AXES:
        data = report["axes"].get(axis, {})
        status = "PASS" if data.get("pass") else "FAIL"
        wk = data.get("weighted_kappa")
        k = data.get("kappa")
        mae = data.get("mae")
        bias = data.get("bias")

        print(f"  {axis:15s}  κ={k:.3f}  wκ={wk:.3f}  MAE={mae:.3f}  bias={bias:+.3f}  [{status}]"
              if wk is not None else f"  {axis:15s}  [NO DATA]")

    overall = "PASS" if report["overall_pass"] else "FAIL"
    print(f"\n  Overall: [{overall}]")

    if report["overall_pass"]:
        print("\n  The Judge is calibrated. Autonomous fine-tuning is safe.")
    else:
        failing = [
            axis for axis in EECF_AXES
            if not report["axes"].get(axis, {}).get("pass", False)
        ]
        print(f"\n  Failing axes: {', '.join(failing)}")
        print("  More calibration data or Judge prompt tuning needed.")
    print("=" * 60)


def show_previous_results() -> None:
    """Display results from previous calibration runs."""
    if not CALIBRATION_RESULTS.exists():
        print("No previous calibration results found.")
        return

    results = json.loads(CALIBRATION_RESULTS.read_text())
    if isinstance(results, dict):
        results = [results]

    for i, report in enumerate(results):
        print(f"\n--- Calibration Run {i + 1} ({report.get('timestamp', '?')}) ---")
        print_calibration_report(report)


async def main():
    parser = argparse.ArgumentParser(description="DAEDALUS Judge Calibration")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of episodes to calibrate (default: 50)",
    )
    parser.add_argument(
        "--from-file", type=str, default=None,
        help="Load episodes from a JSONL file instead of memory",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Show previous calibration results",
    )
    parser.add_argument(
        "--auto-only", action="store_true",
        help="Run Judge only (no human input). Useful for batch scoring.",
    )
    args = parser.parse_args()

    Path("logs").mkdir(parents=True, exist_ok=True)

    if args.report:
        show_previous_results()
        return

    logger.info("=" * 60)
    logger.info("DAEDALUS v0.5 — Judge Calibration")
    logger.info(f"Target: κ ≥ {KAPPA_TARGET} on all EECF axes")
    logger.info("=" * 60)

    # Load configuration
    config = load_all_config(args.config_dir)

    # Load episodes
    if args.from_file:
        episodes = load_episodes_from_file(args.from_file)
        logger.info(f"Loaded {len(episodes)} episodes from {args.from_file}")
    else:
        episodes = load_episodes_from_memory(config, n=args.episodes)
        logger.info(f"Loaded {len(episodes)} episodes from memory store")

    if not episodes:
        logger.error(
            "No episodes available for calibration. "
            "Run some conversations first, or provide a --from-file."
        )
        sys.exit(1)

    episodes = episodes[:args.episodes]

    # Initialize Judge
    from core.soul_bridge import SoulBridge
    from core.soul_memory import SoulMemory
    from core.constitutional_core import ConstitutionalCore
    from core.identity import IdentityManager
    from night.lagrangian_judge import LagrangianJudge

    soul_memory = SoulMemory(config.get("soul_memory", {}))
    soul_bridge = SoulBridge(config.get("soul_bridge", {}))
    soul_bridge.set_soul_memory(soul_memory)

    constitutional_core = ConstitutionalCore("config/constitutional_core.yaml")
    identity = IdentityManager()

    lagrangian_config = config.get("lagrangian", {})
    judge = LagrangianJudge(
        soul_bridge=soul_bridge,
        constitutional_core=constitutional_core,
        config=lagrangian_config,
    )

    # Run calibration
    calibration_pairs = []
    CALIBRATION_LOG.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nCalibrating on {len(episodes)} episodes...")
    if not args.auto_only:
        print("For each episode, you'll see the Judge's scores and provide your own.")
        print("Enter 's' to skip an episode, 'q' to quit early.\n")

    for i, episode in enumerate(episodes):
        logger.info(f"Calibrating episode {i + 1}/{len(episodes)}")

        # Run Judge
        judge_scores = await run_judge_on_episode(
            episode=episode,
            judge=judge,
            identity=identity.as_dict(),
            day_count=identity.day_count,
        )

        if judge_scores is None:
            logger.warning(f"Judge failed on episode {i + 1}. Skipping.")
            continue

        if args.auto_only:
            # Store Judge scores only (for later human review)
            pair = {
                "episode_id": episode.get("id", f"ep_{i}"),
                "human_utterance": episode["human_utterance"],
                "daedalus_response": episode["daedalus_response"],
                "judge": judge_scores,
                "human": None,
            }
            calibration_pairs.append(pair)

            with open(CALIBRATION_LOG, "a") as f:
                f.write(json.dumps(pair, default=str) + "\n")
            continue

        # Collect human rating
        human_scores = collect_human_rating(episode, judge_scores)

        if human_scores == "QUIT":
            logger.info("Calibration ended by user.")
            break

        if human_scores is None:
            logger.info(f"Episode {i + 1} skipped.")
            continue

        pair = {
            "episode_id": episode.get("id", f"ep_{i}"),
            "human_utterance": episode["human_utterance"],
            "daedalus_response": episode["daedalus_response"],
            "judge": judge_scores,
            "human": human_scores,
        }
        calibration_pairs.append(pair)

        # Save incrementally
        with open(CALIBRATION_LOG, "a") as f:
            f.write(json.dumps(pair, default=str) + "\n")

    # Compute report
    rated_pairs = [p for p in calibration_pairs if p.get("human") is not None]

    if not rated_pairs:
        if args.auto_only:
            logger.info(
                f"Judge scoring complete. {len(calibration_pairs)} episodes scored. "
                f"Run without --auto-only to add human ratings."
            )
        else:
            logger.warning("No episodes rated. Cannot compute calibration report.")
        return

    report = compute_calibration_report(rated_pairs)
    print_calibration_report(report)

    # Save results
    results_history = []
    if CALIBRATION_RESULTS.exists():
        try:
            existing = json.loads(CALIBRATION_RESULTS.read_text())
            if isinstance(existing, list):
                results_history = existing
            else:
                results_history = [existing]
        except json.JSONDecodeError:
            pass

    results_history.append(report)
    CALIBRATION_RESULTS.write_text(json.dumps(results_history, indent=2))
    logger.info(f"Calibration results saved to {CALIBRATION_RESULTS}")


if __name__ == "__main__":
    asyncio.run(main())
