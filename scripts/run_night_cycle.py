#!/usr/bin/env python3
"""
DAEDALUS v0.5 — Nightly Consolidation Runner (run_night_cycle.py)

The dream cycle. Designed to run as a nightly cron job or be invoked
manually. Orchestrates all 13 phases of the nightly consolidation.

Phases:
  1-3:  Gather episodes, cluster, extract meaning
  4:    Lagrangian Judge (EECF evaluation)
  4b:   Predictive Judge (J_future estimation)
  4c:   Constitutional drift check
  5:    Provider continuity check
  6:    Update identity document
  7:    Generate training pairs
  8:    QLoRA fine-tune (if blended fertility >= trigger)
  9:    Save Judge calibration data
  10:   Reprocess shallow consolidation queue
  11:   Update Soul Memory
  12:   Compress oldest week (with RG Fidelity Check)
  13:   Log transformation

Usage:
    python scripts/run_night_cycle.py
    python scripts/run_night_cycle.py --date 2026-04-07
    python scripts/run_night_cycle.py --dry-run        # gather + judge only, no training
    python scripts/run_night_cycle.py --skip-training   # run everything except phase 8

Cron example (run at 03:00 every night):
    0 3 * * * cd /mnt/projects1/daedalus && python scripts/run_night_cycle.py >> logs/cron_night.log 2>&1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, date
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load API keys from ~/.apikey/ BEFORE any provider is instantiated.
# Keys never live in the project folder — safe for public git repos.
from core.secrets import load_secrets
load_secrets(verbose=False)  # quiet mode for cron


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/night_cycle.log", mode="a"),
    ],
)
logger = logging.getLogger("daedalus.night_cycle")


def load_all_config(config_dir: str = "config") -> dict:
    """Load and merge all configuration files."""
    config = {}
    config_names = [
        "model_config", "training_config", "lagrangian",
        "soul_bridge", "soul_memory", "constitutional_core",
        "eecf_principles",
    ]
    for name in config_names:
        path = Path(config_dir) / f"{name}.yaml"
        if path.exists():
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            key = name.replace("_config", "")
            config[key] = data
    return config


def check_preconditions() -> bool:
    """Verify that the system is ready for the night cycle."""
    ok = True

    # Check identity exists
    if not Path("identity/current.yaml").exists():
        logger.error("No identity document found. Run seed.py first.")
        ok = False

    # Check constitutional core exists
    if not Path("config/constitutional_core.yaml").exists():
        logger.error("No constitutional core found.")
        ok = False

    # Check conservative mode
    from eval.morning_gate import MorningEvalGate
    if MorningEvalGate.is_conservative_mode_active():
        logger.warning(
            "CONSERVATIVE MODE ACTIVE — skipping fine-tuning. "
            "Consolidation will still run for reflection and memory."
        )

    # Check for episodes to consolidate
    episodes_dir = Path("memory/episodes")
    if episodes_dir.exists():
        episode_count = len(list(episodes_dir.glob("*.json")))
        if episode_count == 0:
            logger.warning("No episodes found. Night cycle may be sparse.")
    else:
        logger.warning("No episodes directory found.")

    return ok


def initialize_subsystems(config: dict):
    """Initialize all subsystems needed for the night cycle."""
    from core.constitutional_core import ConstitutionalCore
    from core.identity import IdentityManager
    from core.memory_store import MemoryStore
    from core.soul_bridge import SoulBridge
    from core.soul_memory import SoulMemory
    from core.consistency import ConsistencyChecker

    constitutional_core = ConstitutionalCore("config/constitutional_core.yaml")
    identity = IdentityManager()
    memory = MemoryStore(config)
    soul_memory = SoulMemory(config.get("soul_memory", {}))
    soul_bridge = SoulBridge(config.get("soul_bridge", {}))
    soul_bridge.set_soul_memory(soul_memory)

    sb_config = config.get("soul_bridge", {}).get("soul_bridge", {})
    cont_config = sb_config.get("continuity", {})
    consistency = ConsistencyChecker(
        embedding_weight=cont_config.get("embedding_weight", 0.6),
        structural_weight=cont_config.get("structural_weight", 0.4),
    )
    soul_bridge.set_consistency_checker(consistency)

    return {
        "constitutional_core": constitutional_core,
        "identity": identity,
        "memory": memory,
        "soul_memory": soul_memory,
        "soul_bridge": soul_bridge,
        "consistency": consistency,
    }


async def run_full_night_cycle(
    config: dict,
    target_date: date,
    skip_training: bool = False,
    dry_run: bool = False,
) -> dict:
    """Execute the full nightly consolidation."""
    from night.consolidation import NightlyConsolidation

    subsystems = initialize_subsystems(config)

    # Override training if requested
    if skip_training or dry_run:
        training_config = config.get("training", {})
        training_config["_skip_training"] = True

    consolidation = NightlyConsolidation(
        memory_store=subsystems["memory"],
        soul_bridge=subsystems["soul_bridge"],
        soul_memory=subsystems["soul_memory"],
        constitutional_core=subsystems["constitutional_core"],
        identity_manager=subsystems["identity"],
        consistency_checker=subsystems["consistency"],
        config=config,
    )

    # Check conservative mode for training suppression
    from eval.morning_gate import MorningEvalGate
    if MorningEvalGate.is_conservative_mode_active():
        logger.warning("Conservative mode: fine-tuning will be suppressed.")

    status = await consolidation.run(target_date=target_date)

    return status


def write_night_summary(status: dict) -> None:
    """Write a human-readable summary of the night cycle."""
    summary_path = Path("logs/night_summary_latest.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(status, indent=2, default=str))

    # Also print key metrics
    logger.info("\n" + "=" * 60)
    logger.info("NIGHT CYCLE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Date:               {status.get('date', '?')}")
    logger.info(f"  Day:                {status.get('day', '?')}")
    logger.info(f"  Episodes:           {status.get('episodes', 0)}")
    logger.info(f"  Meanings:           {status.get('meanings', 0)}")

    if "lagrangian_integral" in status:
        logger.info(f"  L_integral:         {status['lagrangian_integral']:.3f}")
    if "j_future" in status:
        logger.info(f"  J_future:           {status['j_future']:.3f}")
    if "blended_fertility" in status:
        logger.info(f"  Blended fertility:  {status['blended_fertility']:.3f}")
    if "kl_divergence" in status:
        logger.info(f"  D_KL:               {status['kl_divergence']:.3f}")

    logger.info(f"  Fertile trajectories: {status.get('fertile_trajectories', 0)}")
    logger.info(f"  Training pairs:     {status.get('training_pairs', 0)}")
    logger.info(f"  Conservative mode:  {status.get('conservative_mode', False)}")

    ft = status.get("fine_tuning", {})
    if isinstance(ft, dict):
        if ft.get("skipped"):
            logger.info(f"  Fine-tuning:        Skipped ({ft.get('reason', '?')})")
        elif ft.get("error"):
            logger.info(f"  Fine-tuning:        FAILED ({ft['error']})")
        elif ft.get("sft_done") or ft.get("dpo_done"):
            logger.info(
                f"  Fine-tuning:        SFT={'done' if ft.get('sft_done') else 'skip'}, "
                f"DPO={'done' if ft.get('dpo_done') else 'skip'}"
            )
            if ft.get("anchor_alert"):
                logger.warning("  ANCHOR ALERT: Catastrophic forgetting canary triggered!")

    if status.get("weekly_arc"):
        arc = status["weekly_arc"]
        logger.info(
            f"  Weekly compression:  Week {arc.get('week', '?')} "
            f"(RG fidelity: {arc.get('rg_fidelity', 'N/A')})"
        )

    logger.info(f"  Started:            {status.get('started', '?')}")
    logger.info(f"  Completed:          {status.get('completed', '?')}")
    logger.info("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="DAEDALUS Nightly Consolidation")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Run everything except QLoRA fine-tuning (phase 8)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Gather and judge only — no training, no identity commit",
    )
    args = parser.parse_args()

    # Ensure logs directory
    Path("logs").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAEDALUS v0.5 — Nightly Consolidation")
    logger.info("Every night, the system processes the day's experiences.")
    logger.info("=" * 60)

    # Parse target date
    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        target_date = datetime.now().date()

    logger.info(f"Target date: {target_date}")

    # Load configuration
    config = load_all_config(args.config_dir)

    # Precondition checks
    if not check_preconditions():
        logger.error("Precondition checks failed. Aborting night cycle.")
        sys.exit(1)

    # Run the full cycle
    try:
        status = await run_full_night_cycle(
            config=config,
            target_date=target_date,
            skip_training=args.skip_training,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.critical(f"Night cycle FAILED: {e}", exc_info=True)
        # Write failure alert
        alert_path = Path("logs/ALERT_night_cycle_failure.json")
        alert_path.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "date": str(target_date),
            "error": str(e),
            "message": "Nightly consolidation failed. Manual intervention may be needed.",
        }, indent=2))
        sys.exit(1)

    # Write summary
    write_night_summary(status)

    if status.get("skipped"):
        logger.info("Night cycle skipped (no episodes).")
    else:
        logger.info("A new day. The same self. Changed.")


if __name__ == "__main__":
    asyncio.run(main())
