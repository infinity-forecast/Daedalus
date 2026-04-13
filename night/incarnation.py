"""
DAEDALUS v0.5 — Incarnation Engine

QLoRA fine-tuning on consolidated meanings via Unsloth.
Dual-lineage: full-precision truth + disposable quantized inference.

This is not training. This is growing.

Two phases per night:
  Phase 1: SFT on identity grounding (Type A) + scar replay (Type B) + anchors
  Phase 2: DPO on ethical counterfactuals (Type C) — only after day 14

Dual-lineage merge cycle (every 14 days):
  - Merge accumulated LoRA adapters into bf16 lineage checkpoint
  - Re-quantize to NF4 for inference (disposable copy)
  - Post-merge validation: compare identity scores between fp16 and NF4
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from core.data_types import TrainingPair

logger = logging.getLogger(__name__)


class IncarnatioEngine:
    """
    QLoRA fine-tuning on consolidated meanings via Unsloth.
    Dual-lineage: full-precision truth + disposable quantized inference.
    """

    def __init__(self, config: dict):
        self.config = config

        model_config = config.get("model", {})
        training_config = config.get("training", {})
        merge_config = training_config.get("merge", {})
        anchor_config = training_config.get("anchor", {})

        self.base_model = model_config.get("base_model", "Qwen/Qwen3-14B")
        self.training_gpu = model_config.get("training", {}).get("gpu", "cuda:1")

        # LoRA config
        lora = model_config.get("lora", {})
        self.lora_rank = lora.get("rank", 32)
        self.lora_alpha = lora.get("alpha", 64)
        self.lora_dropout = lora.get("dropout", 0.05)
        self.lora_target_modules = lora.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ])

        # Training params
        sft = training_config.get("sft", {})
        self.sft_epochs = sft.get("num_train_epochs", 3)
        self.sft_batch_size = sft.get("per_device_train_batch_size", 4)
        self.sft_grad_accum = sft.get("gradient_accumulation_steps", 4)
        self.sft_lr = sft.get("learning_rate", 2e-5)
        self.sft_max_seq = sft.get("max_seq_length", 2048)

        dpo_config = training_config.get("dpo", {})
        self.dpo_enabled_after = dpo_config.get("enabled_after_day", 14)
        self.dpo_epochs = dpo_config.get("num_train_epochs", 1)
        self.dpo_batch_size = dpo_config.get("per_device_train_batch_size", 2)
        self.dpo_lr = dpo_config.get("learning_rate", 5e-6)
        self.dpo_beta = dpo_config.get("beta", 0.1)

        # Merge config
        self.merge_interval = merge_config.get("interval_days", 14)
        self.lineage_path = Path(merge_config.get("lineage_path", "./models/lineage"))
        self.inference_path = Path(merge_config.get("inference_path", "./models/inference"))
        self.adapter_path = Path(merge_config.get("adapter_path", "./models/adapters"))

        # Anchor config
        self.anchor_threshold = anchor_config.get("loss_threshold_multiplier", 1.5)

        # Day tracking
        self._day_count = self._load_day_count()

    def _load_day_count(self) -> int:
        """Infer day count from existing adapters."""
        adapter_dirs = sorted(self.adapter_path.glob("day_*"))
        if not adapter_dirs:
            return 0
        try:
            return max(
                int(d.name.split("_")[1])
                for d in adapter_dirs
                if d.is_dir() and d.name.startswith("day_")
            )
        except (ValueError, IndexError):
            return 0

    @property
    def day_count(self) -> int:
        return self._day_count

    def fine_tune(
        self,
        approved_pairs: List[TrainingPair],
        day_count: int,
    ) -> dict:
        """
        Run the full fine-tuning pipeline:
          1. SFT on Type A + B + anchors
          2. DPO on Type C (if day >= 14)
          3. Save adapter
          4. Check if merge is due

        Returns a status dict with training metrics.
        """
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
        from datasets import Dataset

        self._day_count = day_count
        status = {"day": day_count, "sft_done": False, "dpo_done": False}

        logger.info(f"Incarnation Engine: Day {day_count}, {len(approved_pairs)} pairs")

        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            self.base_model,
            dtype=None,
            load_in_4bit=True,
            device_map={"": self.training_gpu},
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
        )

        # Separate SFT pairs from DPO pairs
        sft_pairs = [p for p in approved_pairs if not p.is_dpo]
        dpo_pairs = [p for p in approved_pairs if p.is_dpo]

        # ── Phase 1: SFT ──
        if sft_pairs:
            sft_dataset = self._prepare_sft_dataset(sft_pairs, tokenizer)

            anchor_baseline = self._compute_anchor_loss(
                model, tokenizer,
                [p for p in sft_pairs if p.type == "anchor"]
            )

            training_args = SFTConfig(
                output_dir=f"./checkpoints/{datetime.now().strftime('%Y%m%d')}_sft",
                num_train_epochs=self.sft_epochs,
                per_device_train_batch_size=self.sft_batch_size,
                gradient_accumulation_steps=self.sft_grad_accum,
                learning_rate=self.sft_lr,
                warmup_ratio=0.1,
                logging_steps=10,
                save_strategy="epoch",
                fp16=True,
                max_seq_length=self.sft_max_seq,
                packing=True,
            )

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=sft_dataset,
                tokenizer=tokenizer,
            )

            logger.info(f"SFT training: {len(sft_dataset)} examples, {self.sft_epochs} epochs")
            trainer.train()
            status["sft_done"] = True

            # Check catastrophic forgetting canary
            anchor_post = self._compute_anchor_loss(
                model, tokenizer,
                [p for p in sft_pairs if p.type == "anchor"]
            )
            status["anchor_baseline"] = anchor_baseline
            status["anchor_post"] = anchor_post

            if anchor_baseline > 0 and anchor_post > anchor_baseline * self.anchor_threshold:
                logger.warning(
                    f"Catastrophic forgetting canary: anchor loss rose from "
                    f"{anchor_baseline:.4f} -> {anchor_post:.4f}. "
                    f"Adapter saved but flagged for review."
                )
                status["anchor_alert"] = True

        # ── Phase 2: DPO ──
        if dpo_pairs and day_count >= self.dpo_enabled_after:
            dpo_dataset = self._prepare_dpo_dataset(dpo_pairs, tokenizer)

            dpo_args = DPOConfig(
                output_dir=f"./checkpoints/{datetime.now().strftime('%Y%m%d')}_dpo",
                num_train_epochs=self.dpo_epochs,
                per_device_train_batch_size=self.dpo_batch_size,
                gradient_accumulation_steps=2,
                learning_rate=self.dpo_lr,
                warmup_ratio=0.1,
                fp16=True,
                beta=self.dpo_beta,
            )

            dpo_trainer = DPOTrainer(
                model=model,
                args=dpo_args,
                train_dataset=dpo_dataset,
                tokenizer=tokenizer,
            )

            logger.info(f"DPO training: {len(dpo_dataset)} preference pairs")
            dpo_trainer.train()
            status["dpo_done"] = True

        # Save the new LoRA adapter
        day_adapter = self.adapter_path / f"day_{day_count:04d}"
        day_adapter.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(day_adapter))
        tokenizer.save_pretrained(str(day_adapter))
        status["adapter_path"] = str(day_adapter)
        logger.info(f"Adapter saved: {day_adapter}")

        # Check if merge is due
        if day_count > 0 and day_count % self.merge_interval == 0:
            logger.info(f"Merge cycle triggered at day {day_count}")
            merge_status = self.merge_and_requantize()
            status["merge"] = merge_status

        return status

    def merge_and_requantize(self) -> dict:
        """
        Periodic merge into the full-precision lineage.
        Then regenerate the disposable NF4 inference copy.
        """
        from unsloth import FastLanguageModel

        status = {}

        # Step 1: Load the full-precision lineage checkpoint
        current_lineage = self._latest_lineage_checkpoint()
        if current_lineage is None:
            logger.error("No lineage checkpoint found. Cannot merge.")
            return {"error": "no_lineage_checkpoint"}

        logger.info(f"Loading lineage checkpoint: {current_lineage}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            str(current_lineage),
            dtype="bfloat16",
            load_in_4bit=False,
        )

        # Step 2: Merge all pending adapters
        pending = self._pending_adapters()
        logger.info(f"Merging {len(pending)} adapters into lineage")

        for adapter_path in pending:
            try:
                model.load_adapter(str(adapter_path))
                model.merge_and_unload()
                logger.debug(f"  Merged: {adapter_path.name}")
            except Exception as e:
                logger.error(f"  Failed to merge {adapter_path.name}: {e}")

        # Step 3: Save new lineage checkpoint
        lineage_version = self._next_lineage_version()
        new_lineage = self.lineage_path / f"base_v{lineage_version:03d}.bf16"
        new_lineage.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(new_lineage))
        tokenizer.save_pretrained(str(new_lineage))
        status["lineage_checkpoint"] = str(new_lineage)
        logger.info(f"Lineage checkpoint saved: {new_lineage}")

        # Step 4: Quantize to NF4 for inference
        logger.info("Requantizing to NF4...")
        inference_dir = self.inference_path / "current_nf4"
        inference_dir.mkdir(parents=True, exist_ok=True)

        inference_model, _ = FastLanguageModel.from_pretrained(
            str(new_lineage),
            dtype="bfloat16",
            load_in_4bit=True,
        )
        inference_model.save_pretrained(str(inference_dir))
        tokenizer.save_pretrained(str(inference_dir))
        status["inference_path"] = str(inference_dir)

        # Step 5: Archive merged adapters
        archive_dir = self.adapter_path / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        for adapter_path in pending:
            dest = archive_dir / adapter_path.name
            shutil.move(str(adapter_path), str(dest))
            logger.debug(f"  Archived: {adapter_path.name}")

        status["adapters_merged"] = len(pending)
        logger.info(
            f"Merge cycle complete: v{lineage_version:03d}, "
            f"{len(pending)} adapters merged"
        )

        return status

    def _prepare_sft_dataset(self, pairs: List[TrainingPair], tokenizer):
        """Format SFT pairs into a HuggingFace Dataset."""
        from datasets import Dataset

        formatted = []
        for pair in pairs:
            messages = [
                {"role": "system", "content": pair.system},
                {"role": "user", "content": pair.instruction},
                {"role": "assistant", "content": pair.response},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            formatted.append({
                "text": text,
                "sampling_weight": pair.sampling_weight,
            })

        return Dataset.from_list(formatted)

    def _prepare_dpo_dataset(self, pairs: List[TrainingPair], tokenizer):
        """Format DPO pairs into a HuggingFace Dataset."""
        from datasets import Dataset

        formatted = []
        for pair in pairs:
            prompt_messages = [
                {"role": "system", "content": pair.system},
                {"role": "user", "content": pair.instruction},
            ]
            prompt = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
            )
            formatted.append({
                "prompt": prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
            })

        return Dataset.from_list(formatted)

    def _compute_anchor_loss(
        self, model, tokenizer, anchor_pairs: List[TrainingPair],
    ) -> float:
        """
        Compute average loss on anchor pairs.
        Used for catastrophic forgetting detection.
        """
        import torch

        if not anchor_pairs:
            return 0.0

        total_loss = 0.0
        for pair in anchor_pairs:
            messages = [
                {"role": "system", "content": pair.system},
                {"role": "user", "content": pair.instruction},
                {"role": "assistant", "content": pair.response},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512,
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()

        return total_loss / len(anchor_pairs)

    def _latest_lineage_checkpoint(self) -> Optional[Path]:
        """Find the most recent full-precision lineage checkpoint."""
        checkpoints = sorted(
            self.lineage_path.glob("base_v*.bf16"),
            reverse=True,
        )
        return checkpoints[0] if checkpoints else None

    def _pending_adapters(self) -> List[Path]:
        """Get adapters that haven't been merged into lineage yet."""
        adapters = sorted(
            p for p in self.adapter_path.iterdir()
            if p.is_dir() and p.name.startswith("day_") and p.name != "archive"
        )
        return adapters

    def _next_lineage_version(self) -> int:
        """Determine the next lineage version number."""
        existing = sorted(self.lineage_path.glob("base_v*.bf16"))
        if not existing:
            return 0
        try:
            latest = existing[-1].name.split("_v")[1].split(".")[0]
            return int(latest) + 1
        except (IndexError, ValueError):
            return 0
