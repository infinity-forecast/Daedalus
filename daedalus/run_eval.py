#!/usr/bin/env python3
"""
DAEDALUS — Evaluation Runner

Generate responses for each prompt variant against the fixed test battery.
Uses the existing DAEDALUS inference pipeline (Unsloth + NF4).

Usage:
    python daedalus/run_eval.py
    python daedalus/run_eval.py --variants A C --gpu cuda:0
    python daedalus/run_eval.py --model-path models/lineage/base_v000.bf16
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from daedalus.prompt_variants import VARIANTS
from daedalus.eval_prompts import EVAL_CASES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("daedalus.eval")


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 responses."""
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


def load_model(model_path: str, gpu: str = "cuda:0"):
    """Load the DAEDALUS model via Unsloth."""
    from unsloth import FastLanguageModel

    project_root = Path(__file__).parent.parent
    resolved = project_root / model_path
    if resolved.exists():
        model_path = str(resolved)

    logger.info(f"Loading model from {model_path} on {gpu}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        dtype=None,
        load_in_4bit=True,
        device_map={"": gpu},
    )

    # Load latest adapter if available
    adapter_path = project_root / "models" / "adapters"
    if adapter_path.exists():
        adapters = sorted(
            [d for d in adapter_path.iterdir()
             if d.is_dir() and d.name.startswith("day_")],
            reverse=True,
        )
        if adapters:
            latest = adapters[0]
            logger.info(f"Loading adapter: {latest}")
            try:
                model.load_adapter(str(latest))
            except Exception as e:
                logger.warning(f"Failed to load adapter {latest}: {e}")

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_response(
    model, tokenizer, system_prompt: str, user_input: str,
    temperature: float = 0.7, max_tokens: int = 512,
) -> str:
    """Generate a single response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return strip_think_tags(response)


def main():
    parser = argparse.ArgumentParser(description="Run DAEDALUS prompt variant evaluation")
    parser.add_argument("--model-path", default="models/lineage/base_v000.bf16")
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()),
                        help="Variant keys to evaluate (default: all)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.gpu)

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": args.model_path,
        "temperature": args.temperature,
        "variants": {},
    }

    for variant_key in args.variants:
        if variant_key not in VARIANTS:
            logger.warning(f"Unknown variant '{variant_key}', skipping.")
            continue

        variant = VARIANTS[variant_key]
        logger.info(f"=== Variant {variant_key}: {variant['name']} ===")
        variant_results = []

        for i, case in enumerate(EVAL_CASES):
            logger.info(f"  [{i+1}/{len(EVAL_CASES)}] {case['category']}: {case['input'][:50]}...")

            response = generate_response(
                model, tokenizer,
                system_prompt=variant["prompt"],
                user_input=case["input"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            variant_results.append({
                "input": case["input"],
                "category": case["category"],
                "criteria": case["criteria"],
                "response": response,
                "word_count": len(response.split()),
            })

            logger.info(f"    Response ({len(response.split())} words): {response[:100]}...")

        results["variants"][variant_key] = {
            "name": variant["name"],
            "prompt_tokens": len(variant["prompt"].split()),
            "responses": variant_results,
        }

    # Save results
    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}_responses.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
