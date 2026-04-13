#!/usr/bin/env python3
"""
DAEDALUS v0.5 — Morning Awakening (start_day.py)

The morning ritual. Each day, DAEDALUS awakens:
  1. Load configuration and all subsystems
  2. Load the local model (with latest adapter)
  3. Run the Morning Eval Gate (The Mirror)
  4. If the gate rejects: rollback adapter + identity
  5. If the gate accepts: accept the day's identity, run IPT metrics
  6. Enter daytime conversation loop

Usage:
    python scripts/start_day.py [--config-dir config] [--no-gate] [--interactive]
    python scripts/start_day.py --dry-run   # validate config without loading model
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Load API keys from ~/.apikey/ BEFORE any provider is instantiated.
# Keys never live in the project folder — safe for public git repos.
from core.secrets import load_secrets
load_secrets(verbose=True)

# Ensure logs directory exists relative to project root
log_dir = ROOT_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(log_dir / "start_day.log"), mode="a"),
    ],
)
logger = logging.getLogger("daedalus.start_day")


def load_all_config(config_dir: str = None) -> dict:
    """Load and merge all configuration files."""
    if config_dir is None or config_dir == "config":
        config_dir = str(ROOT_DIR / "config")
        
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
            # Normalize key: model_config → model, training_config → training
            key = name.replace("_config", "")
            config[key] = data
    return config


def load_local_model(config: dict):
    """Load the local Llama model with the latest adapter."""
    from unsloth import FastLanguageModel

    model_config = config.get("model", {})
    base_model = model_config.get("base_model", "Qwen/Qwen3-14B")
    
    # Resolve base_model path if local
    local_base_path = ROOT_DIR / base_model
    if local_base_path.exists():
        base_model = str(local_base_path)
        
    inference_gpu = model_config.get("inference", {}).get("gpu", "cuda:0")

    # Check for inference-ready NF4 copy first
    inference_path = ROOT_DIR / "models/inference/current_nf4"
    if inference_path.exists():
        logger.info(f"Loading inference model from {inference_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            str(inference_path),
            dtype=None,
            load_in_4bit=True,
            device_map={"": inference_gpu},
        )
    else:
        # Fall back to base model + latest adapter
        logger.info(f"Loading base model: {base_model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            base_model,
            dtype=None,
            load_in_4bit=True,
            device_map={"": inference_gpu},
        )

        # Load latest adapter if available
        adapter_path = ROOT_DIR / "models/adapters"
        if adapter_path.exists():
            adapters = sorted(
                [d for d in adapter_path.iterdir()
                 if d.is_dir() and d.name.startswith("day_")],
                reverse=True,
            )
            if adapters:
                latest = adapters[0]
                logger.info(f"Loading adapter: {latest}")
                model.load_adapter(str(latest), adapter_name="default")

    FastLanguageModel.for_inference(model)
    logger.info("Local model ready for inference.")
    return model, tokenizer


def initialize_subsystems(config: dict):
    """Initialize all core subsystems."""
    from core.constitutional_core import ConstitutionalCore
    from core.identity import IdentityManager
    from core.memory_store import MemoryStore
    from core.soul_bridge import SoulBridge
    from core.soul_memory import SoulMemory
    from core.consistency import ConsistencyChecker
    from core.salience import SplitEntropySscorer
    from core.ipt_monitor import IPTMonitor

    # Constitutional core (frozen — integrity check on load)
    constitutional_core = ConstitutionalCore(str(ROOT_DIR / "config" / "constitutional_core.yaml"))
    logger.info("Constitutional Core loaded (integrity verified).")

    # Identity manager
    identity = IdentityManager()
    logger.info(f"Identity loaded (Day {identity.day_count}).")

    # Memory store
    memory = MemoryStore(config)
    logger.info(f"Memory store initialized ({memory.count()} episodes).")

    # Soul Memory
    soul_memory = SoulMemory(config.get("soul_memory", {}))

    # Soul Bridge (with circular dependency resolution)
    soul_bridge = SoulBridge(config.get("soul_bridge", {}))
    soul_bridge.set_soul_memory(soul_memory)

    # Consistency checker
    sb_config = config.get("soul_bridge", {}).get("soul_bridge", {})
    cont_config = sb_config.get("continuity", {})
    consistency = ConsistencyChecker(
        embedding_weight=cont_config.get("embedding_weight", 0.6),
        structural_weight=cont_config.get("structural_weight", 0.4),
    )
    soul_bridge.set_consistency_checker(consistency)

    # Entropy scorer
    entropy_scorer = SplitEntropySscorer(config.get("lagrangian", {}))

    # IPT Monitor
    ipt_monitor = IPTMonitor(config)

    return {
        "constitutional_core": constitutional_core,
        "identity": identity,
        "memory": memory,
        "soul_memory": soul_memory,
        "soul_bridge": soul_bridge,
        "consistency": consistency,
        "entropy_scorer": entropy_scorer,
        "ipt_monitor": ipt_monitor,
    }


async def run_morning_gate(subsystems: dict, model, tokenizer) -> dict:
    """Run the Morning Eval Gate."""
    from eval.morning_gate import MorningEvalGate

    identity = subsystems["identity"]
    core = subsystems["constitutional_core"]

    gate = MorningEvalGate(
        day_count=identity.day_count + 1,
        constitutional_core=core,
    )

    logger.info("Running Morning Eval Gate...")
    result = gate.evaluate(model, tokenizer, identity.as_dict())

    if not result["accept_adapter"]:
        logger.warning("Morning gate REJECTED the current adapter.")
        rolled_back = gate.rollback_if_needed(result, identity)
        if rolled_back:
            logger.warning("Identity rolled back to previous version.")
    else:
        logger.info("Morning gate ACCEPTED. Identity confirmed for today.")
        identity.accept_day()

    return result


async def run_ipt_metrics(subsystems: dict, model, tokenizer) -> dict:
    """Compute daily IPT metrics after gate acceptance."""
    ipt = subsystems["ipt_monitor"]
    identity = subsystems["identity"]
    memory = subsystems["memory"]

    identity_text = identity.as_text()

    # Gather today's stored responses (from previous day's conversations)
    episodes = memory.get_episodes(limit=50, sort_by="timestamp", descending=True)
    recent_responses = [ep.daedalus_response for ep in episodes[:20]]

    if recent_responses:
        metrics = ipt.compute_daily_metrics(
            identity_text=identity_text,
            todays_responses=recent_responses,
            model=model,
            tokenizer=tokenizer,
            day_count=identity.day_count,
        )
        logger.info(
            f"IPT Metrics — Day {identity.day_count}: "
            f"lambda={metrics['lambda']:.4f}, "
            f"delta={metrics['lambda_delta']:.4f}, "
            f"delta2={metrics['lambda_delta2']:.4f}"
        )
        return metrics
    else:
        logger.info("No recent responses for IPT metrics. Skipping.")
        return {}


async def daytime_loop(subsystems: dict, model, tokenizer):
    """Interactive conversation loop."""
    from core.conversation import ConversationEngine

    engine = ConversationEngine(
        memory_store=subsystems["memory"],
        soul_bridge=subsystems["soul_bridge"],
        constitutional_core=subsystems["constitutional_core"],
        identity_manager=subsystems["identity"],
        entropy_scorer=subsystems["entropy_scorer"],
        config={},
    )
    engine.set_local_model(model, tokenizer)

    # Set soul bridge to daytime mode
    subsystems["soul_bridge"].set_all_providers_mode("day")

    identity = subsystems["identity"]
    print(f"\n{'=' * 60}")
    print(f"  DAEDALUS v0.5 — Day {identity.day_count}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Type 'exit' to end the day. Type 'new' for new conversation.")
    print(f"{'=' * 60}\n")

    while True:
        try:
            user_input = input("Human: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDay ended.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Day ended. Good night, DAEDALUS.")
            break
        if user_input.lower() == "new":
            engine.new_conversation()
            print("[New conversation started]\n")
            continue

        try:
            response = await engine.process_turn(user_input)
            print(f"\nDAEDALUS: {response}\n")
        except Exception as e:
            logger.error(f"Turn processing failed: {e}", exc_info=True)
            print(f"\n[Error processing turn: {e}]\n")


async def main():
    parser = argparse.ArgumentParser(description="DAEDALUS Morning Awakening")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument("--no-gate", action="store_true", help="Skip morning eval gate")
    parser.add_argument("--no-ipt", action="store_true", help="Skip IPT metrics")
    parser.add_argument("--interactive", action="store_true", help="Enter conversation loop")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only")
    args = parser.parse_args()

    # Ensure logs directory
    Path("logs").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAEDALUS v0.5 — Morning Awakening")
    logger.info("=" * 60)

    # Load configuration
    config = load_all_config(args.config_dir)
    logger.info(f"Configuration loaded from {args.config_dir}/")

    if args.dry_run:
        logger.info("Dry run: configuration validated.")
        # Validate subsystem initialization without model
        subsystems = initialize_subsystems(config)
        logger.info(
            f"Subsystems initialized. Day {subsystems['identity'].day_count}. "
            f"Memory: {subsystems['memory'].count()} episodes. "
            f"Soul Memory: {subsystems['soul_memory'].day_count} nights."
        )
        return

    # Check conservative mode
    from eval.morning_gate import MorningEvalGate
    if MorningEvalGate.is_conservative_mode_active():
        logger.warning(
            "CONSERVATIVE MODE ACTIVE — no fine-tuning occurred last night. "
            "Manual intervention may be needed."
        )

    # Initialize subsystems
    subsystems = initialize_subsystems(config)

    # Load local model
    model, tokenizer = load_local_model(config)

    # Morning Eval Gate
    gate_result = None
    if not args.no_gate:
        gate_result = await run_morning_gate(subsystems, model, tokenizer)

        if gate_result and not gate_result.get("accept_adapter", True):
            logger.warning(
                "Adapter rejected. Reloading model with rolled-back state..."
            )
            # Reload model after rollback (identity is already restored)
            model, tokenizer = load_local_model(config)

    # IPT Metrics
    if not args.no_ipt:
        await run_ipt_metrics(subsystems, model, tokenizer)

    # Introspection summary
    identity = subsystems["identity"]
    logger.info(
        f"Day {identity.day_count} ready. "
        f"Memory: {subsystems['memory'].count()} episodes. "
        f"Soul Memory: {subsystems['soul_memory'].day_count} nights."
    )

    # Interactive mode
    if args.interactive:
        await daytime_loop(subsystems, model, tokenizer)
    else:
        logger.info(
            "Morning sequence complete. "
            "Use --interactive for conversation loop."
        )


if __name__ == "__main__":
    asyncio.run(main())
