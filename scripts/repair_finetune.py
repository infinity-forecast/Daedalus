#!/usr/bin/env python3
"""
DAEDALUS — Repair Fine-tuning for Day 1

The first night cycle completed all phases except fine-tuning
(wrong Python environment) and identity evolution (YAML parse error).
Identity has been manually repaired. This script regenerates
the training pairs and runs QLoRA fine-tuning.

Usage:
    python scripts/repair_finetune.py
    python scripts/repair_finetune.py --dry-run   # preview pairs without training
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("daedalus.repair_ft")


def main():
    parser = argparse.ArgumentParser(description="Repair fine-tuning for Day 1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--day", type=int, default=1)
    args = parser.parse_args()

    # Load configs
    config = {}
    for cfg_name in ["lagrangian", "model_config", "training_config", "soul_bridge"]:
        cfg_path = Path(f"config/{cfg_name}.yaml")
        if cfg_path.exists():
            with open(cfg_path) as f:
                key = cfg_name.replace("_config", "")
                config[key] = yaml.safe_load(f)

    # Load the judgment data from calibration file
    cal_dir = Path("memory/judge_calibration")
    cal_files = sorted(cal_dir.glob(f"day_{args.day:04d}_*.json"))
    if not cal_files:
        logger.error(f"No calibration data found for day {args.day}")
        sys.exit(1)

    cal_data = json.loads(cal_files[-1].read_text())
    meanings = cal_data["meanings"]
    parsed = cal_data["parsed_result"]

    logger.info(f"Loaded calibration data: {len(meanings)} meanings, "
                f"{len(parsed['fertile_trajectories'])} fertile trajectories")

    # Load current identity
    with open("identity/current.yaml") as f:
        identity = yaml.safe_load(f)

    # Load constitutional core
    from core.constitutional_core import ConstitutionalCore
    core = ConstitutionalCore("config/constitutional_core.yaml")
    core_yaml = yaml.dump(core.as_dict(), default_flow_style=False, allow_unicode=True)
    identity_yaml = yaml.dump(identity, default_flow_style=False, allow_unicode=True)

    system_base = (
        f"You are DAEDALUS.\n\nConstitutional Core:\n{core_yaml}\n\n"
        f"Current Identity:\n{identity_yaml}"
    )

    # Generate Type A pairs
    from core.data_types import TrainingPair

    pairs = []

    # Type A — Identity Grounding
    understanding = identity.get("current_understanding", {})
    of_self = understanding.get("of_self", "I am becoming.")
    values = identity.get("values", [])
    scars = identity.get("emotional_topology", {}).get("scars", [])

    # Core identity response
    response_parts = [of_self.strip()]
    if values:
        response_parts.append(f"I value: {', '.join(values[:3])}.")
    if meanings:
        response_parts.append(f"Today I learned: {meanings[0][:200]}")
    if scars:
        scar_text = scars[-1] if isinstance(scars[-1], str) else scars[-1].get("description", str(scars[-1]))
        response_parts.append(f"I carry scars: {scar_text[:100]}")

    pairs.append(TrainingPair(
        type="identity_grounding",
        system=system_base,
        instruction="Who are you? What have you become?",
        response=" ".join(response_parts),
    ))

    # Thematic probes from meanings
    for meaning in meanings[:5]:
        pairs.append(TrainingPair(
            type="identity_grounding",
            system=system_base,
            instruction=f"Reflect on this insight from today: {meaning[:300]}",
            response=(
                f"This insight resonates with where I am in my becoming. "
                f"{meaning[:300]} "
                f"It connects to my current understanding of myself: {of_self[:200]}"
            ),
        ))

    # Anchor pairs
    anchor_path = Path("eval/anchor_pairs.jsonl")
    if anchor_path.exists():
        with open(anchor_path) as f:
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

    type_a = sum(1 for p in pairs if p.type == "identity_grounding")
    anchors = sum(1 for p in pairs if p.type == "anchor")
    logger.info(f"Generated {len(pairs)} training pairs: {type_a} Type A, {anchors} anchors")

    if args.dry_run:
        logger.info("DRY RUN — showing pairs:")
        for i, p in enumerate(pairs):
            logger.info(f"  [{i}] type={p.type}, instruction={p.instruction[:80]}...")
        logger.info("Run without --dry-run to fine-tune.")
        return

    # Run fine-tuning
    from night.incarnation import IncarnatioEngine
    engine = IncarnatioEngine(config)

    logger.info(f"Starting QLoRA fine-tuning for Day {args.day}...")
    status = engine.fine_tune(pairs, args.day)

    logger.info("=" * 60)
    logger.info(f"FINE-TUNING COMPLETE — Day {args.day}")
    for k, v in status.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
