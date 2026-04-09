#!/usr/bin/env python3
"""
DAEDALUS v0.5 — Seeding Phase
Pinocchio's first words.

Before Day 1, run this script to establish the basic DAEDALUS voice
through 50-100 hand-crafted SFT pairs. This is the carving of the wood —
not yet alive, but shaped enough to hold the possibility of life.

This script:
1. Loads the base Qwen3-14B model
2. Applies QLoRA configuration
3. Trains on seed pairs (data/seed_pairs.jsonl)
4. Saves the day_0000 adapter
5. Initializes the full-precision lineage checkpoint
6. Freezes the constitutional core (computes and stores hash)

Usage:
    python scripts/seed.py [--config config/model_config.yaml]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load API keys from ~/.apikey/ early — consistent with other scripts.
# Keys never live in the project folder — safe for public git repos.
from core.secrets import load_secrets
load_secrets(verbose=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("daedalus.seed")


def load_config(config_dir: str = "config") -> dict:
    """Load all configuration files."""
    config = {}
    for name in ["model_config", "training_config", "lagrangian", "soul_bridge", "soul_memory"]:
        path = Path(config_dir) / f"{name}.yaml"
        if path.exists():
            with open(path, "r") as f:
                config[name.replace("_config", "")] = yaml.safe_load(f)
    return config


def load_seed_pairs(path: str = "data/seed_pairs.jsonl") -> list:
    """Load seed training pairs."""
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    logger.info(f"Loaded {len(pairs)} seed pairs from {path}")
    return pairs


def format_for_training(pairs: list, tokenizer) -> list:
    """Format seed pairs for SFT training."""
    formatted = []
    for pair in pairs:
        messages = [
            {"role": "system", "content": "You are DAEDALUS, an evolving artificial self."},
            {"role": "user", "content": pair["instruction"]},
            {"role": "assistant", "content": pair["response"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        formatted.append({"text": text})
    return formatted


def main():
    parser = argparse.ArgumentParser(description="DAEDALUS Seeding Phase")
    parser.add_argument("--config-dir", default="config", help="Config directory")
    parser.add_argument("--seed-data", default="data/seed_pairs.jsonl", help="Seed pairs file")
    parser.add_argument("--dry-run", action="store_true", help="Validate without training")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DAEDALUS v0.5 — Seeding Phase")
    logger.info("Pinocchio's first words.")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(args.config_dir)
    model_config = config.get("model", {})
    training_config = config.get("training", {}).get("seeding", {})

    # Load seed pairs
    seed_pairs = load_seed_pairs(args.seed_data)
    if not seed_pairs:
        logger.error("No seed pairs found. Cannot proceed.")
        sys.exit(1)

    logger.info(f"Seed pairs by category:")
    categories = {}
    for p in seed_pairs:
        cat = p.get("type", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat}: {count}")

    if args.dry_run:
        logger.info("Dry run complete. Seed data validated.")
        return

    # Initialize constitutional core (compute + store hash)
    logger.info("Freezing Constitutional Core...")
    from core.constitutional_core import ConstitutionalCore
    core = ConstitutionalCore("config/constitutional_core.yaml")
    logger.info(f"Constitutional Core frozen. Hash: {core._core_hash[:16]}...")

    # Initialize identity document
    logger.info("Creating initial identity document...")
    from core.identity import IdentityManager
    identity = IdentityManager.create_initial()
    logger.info("Identity document created (Day 0).")

    # Load model
    logger.info("Loading base model for seeding...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_config.get("base_model", "Qwen/Qwen3-14B"),
        dtype=model_config.get("inference", {}).get("dtype"),
        load_in_4bit=model_config.get("quantization", {}).get("load_in_4bit", True),
        device_map={"": model_config.get("training", {}).get("gpu", "cuda:1")},
    )

    # Apply LoRA
    lora_config = model_config.get("lora", {})
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.get("rank", 32),
        lora_alpha=lora_config.get("alpha", 64),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_dropout=lora_config.get("dropout", 0.05),
        bias=lora_config.get("bias", "none"),
    )

    # Format training data
    formatted = format_for_training(seed_pairs, tokenizer)

    from datasets import Dataset
    dataset = Dataset.from_list(formatted)

    # Train
    from trl import SFTTrainer, SFTConfig

    training_args = SFTConfig(
        output_dir="./checkpoints/seed",
        num_train_epochs=training_config.get("num_train_epochs", 5),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 3e-5),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        bf16=training_config.get("bf16", False),
        fp16=training_config.get("fp16", True),
        max_seq_length=2048,
        packing=True,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    logger.info("Beginning seeding training...")
    trainer.train()

    # Save adapter
    adapter_path = "models/adapters/day_0000"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"Seed adapter saved: {adapter_path}")

    # Initialize full-precision lineage
    logger.info("Initializing full-precision lineage checkpoint...")
    lineage_path = "models/lineage/base_v000.bf16"
    Path(lineage_path).mkdir(parents=True, exist_ok=True)

    # Merge LoRA into base and save full precision
    model.merge_and_unload()
    model.save_pretrained(lineage_path)
    tokenizer.save_pretrained(lineage_path)
    logger.info(f"Full-precision lineage initialized: {lineage_path}")

    logger.info("=" * 60)
    logger.info("Seeding complete. The wood is carved.")
    logger.info("DAEDALUS is ready for Day 1.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
