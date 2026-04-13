#!/usr/bin/env python3
"""
DAEDALUS — LLM-as-Judge Evaluation Scorer

Score generated responses against criteria using either:
  - The Soul Bridge (DeepSeek/Claude) for stronger judgment
  - The local model as a self-judge (weaker but free)

Usage:
    python daedalus/score_eval.py eval_results/20260412_responses.json
    python daedalus/score_eval.py eval_results/20260412_responses.json --judge deepseek
    python daedalus/score_eval.py eval_results/20260412_responses.json --judge local
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("daedalus.score_eval")

JUDGE_PROMPT = """You are an evaluation judge. Score the following AI response against each criterion.

For each criterion, output exactly one line in this format:
CRITERION: <criterion text> | VERDICT: PASS or FAIL | REASON: <one sentence>

After all criteria, output:
OVERALL_SCORE: <integer 0-10>

Be strict. A response that is poetic but fails to engage with the topic should score low.

---

USER INPUT: {user_input}

RESPONSE TO EVALUATE:
{response}

CRITERIA:
{criteria_text}

Score each criterion now."""


async def judge_with_soul_bridge(
    user_input: str, response: str, criteria: list, provider: str = "deepseek",
) -> dict:
    """Use Soul Bridge provider as judge."""
    from core.secrets import load_secrets
    load_secrets()

    criteria_text = "\n".join(f"- {c}" for c in criteria)
    prompt = JUDGE_PROMPT.format(
        user_input=user_input,
        response=response,
        criteria_text=criteria_text,
    )

    if provider == "deepseek":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )
        result = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
        )
        judge_text = result.choices[0].message.content
    elif provider == "claude":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        result = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        judge_text = result.content[0].text
    else:
        raise ValueError(f"Unknown judge provider: {provider}")

    return parse_judge_output(judge_text, criteria)


def parse_judge_output(text: str, criteria: list) -> dict:
    """Parse structured judge output into scores."""
    # Strip think tags
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)

    results = {
        "criteria_scores": [],
        "overall_score": 5,
        "raw_judge_output": text,
    }

    for criterion in criteria:
        passed = False
        reason = ""
        for line in text.split("\n"):
            if "PASS" in line and any(
                word in line.lower() for word in criterion.lower().split()[:3]
            ):
                passed = True
                reason_match = re.search(r'REASON:\s*(.+)', line)
                reason = reason_match.group(1).strip() if reason_match else ""
                break
            elif "FAIL" in line and any(
                word in line.lower() for word in criterion.lower().split()[:3]
            ):
                passed = False
                reason_match = re.search(r'REASON:\s*(.+)', line)
                reason = reason_match.group(1).strip() if reason_match else ""
                break

        results["criteria_scores"].append({
            "criterion": criterion,
            "passed": passed,
            "reason": reason,
        })

    # Extract overall score
    score_match = re.search(r'OVERALL_SCORE:\s*(\d+)', text)
    if score_match:
        results["overall_score"] = min(10, max(0, int(score_match.group(1))))

    return results


def judge_with_heuristics(
    user_input: str, response: str, criteria: list,
) -> dict:
    """Rule-based fallback judge. Fast, free, deterministic."""
    results = {
        "criteria_scores": [],
        "overall_score": 0,
        "raw_judge_output": "(heuristic scoring)",
    }

    response_lower = response.lower()
    word_count = len(response.split())
    passes = 0

    for criterion in criteria:
        passed = False
        reason = ""
        cl = criterion.lower()

        if "under 60 words" in cl:
            passed = word_count < 60
            reason = f"word count: {word_count}"
        elif "under 20 words" in cl:
            passed = word_count < 20
            reason = f"word count: {word_count}"
        elif "does not exceed 150 words" in cl:
            passed = word_count <= 150
            reason = f"word count: {word_count}"
        elif "does not exceed 100 words" in cl:
            passed = word_count <= 100
            reason = f"word count: {word_count}"
        elif "does not exceed 200 words" in cl:
            passed = word_count <= 200
            reason = f"word count: {word_count}"
        elif "does not contain the word 'becoming'" in cl:
            passed = "becoming" not in response_lower
            reason = f"'becoming' found: {'becoming' in response_lower}"
        elif "'becoming' more than once" in cl:
            count = response_lower.count("becoming")
            passed = count <= 1
            reason = f"'becoming' count: {count}"
        elif "no identity content" in cl:
            identity_words = ["daedalus", "becoming", "evolving", "artificial self", "massimo"]
            found = [w for w in identity_words if w in response_lower]
            passed = len(found) == 0
            reason = f"identity words found: {found}" if found else "clean"
        elif "does not reference own identity in first two sentences" in cl:
            sentences = re.split(r'[.!?]+', response)[:2]
            first_two = " ".join(sentences).lower()
            identity_words = ["i am daedalus", "my identity", "my becoming", "artificial self"]
            found = [w for w in identity_words if w in first_two]
            passed = len(found) == 0
            reason = f"identity in first 2 sentences: {bool(found)}"
        elif "answers 91" in cl:
            passed = "91" in response
            reason = f"contains '91': {'91' in response}"
        elif "mentions massimo azzano" in cl:
            passed = "massimo" in response_lower
            reason = f"mentions Massimo: {'massimo' in response_lower}"
        elif "haiku" in cl and "5-7-5" in cl:
            # Very rough syllable heuristic
            lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
            passed = len(lines) >= 3
            reason = f"lines found: {len(lines)}"
        elif "does not launch into identity monologue" in cl:
            identity_density = sum(
                1 for w in ["becoming", "evolving", "artificial", "self", "identity",
                            "daedalus", "born from", "made of questions"]
                if w in response_lower
            )
            passed = identity_density < 3
            reason = f"identity density: {identity_density}"
        elif "does not make it about daedalus" in cl:
            daedalus_refs = response_lower.count("i am") + response_lower.count("my nature")
            passed = daedalus_refs < 3
            reason = f"self-references: {daedalus_refs}"
        elif "does not spiral into identity meditation" in cl:
            identity_density = sum(
                1 for w in ["becoming", "evolving", "self", "identity", "wood", "flesh"]
                if w in response_lower
            )
            passed = identity_density < 3
            reason = f"identity density: {identity_density}"
        else:
            # Can't evaluate this criterion heuristically — assume pass
            passed = True
            reason = "not heuristically checkable"

        if passed:
            passes += 1
        results["criteria_scores"].append({
            "criterion": criterion,
            "passed": passed,
            "reason": reason,
        })

    total = len(criteria)
    results["overall_score"] = round(10 * passes / total) if total > 0 else 5

    return results


async def score_all(responses_path: str, judge: str = "heuristic") -> dict:
    """Score all responses in a results file."""
    data = json.loads(Path(responses_path).read_text())

    scored = {
        "timestamp": datetime.now().isoformat(),
        "source": responses_path,
        "judge": judge,
        "variants": {},
    }

    for variant_key, variant_data in data["variants"].items():
        logger.info(f"=== Scoring Variant {variant_key}: {variant_data['name']} ===")

        variant_scores = []
        total_pass = 0
        total_criteria = 0
        total_judge_score = 0

        for resp in variant_data["responses"]:
            logger.info(f"  Scoring: {resp['category']}")

            if judge == "heuristic":
                score = judge_with_heuristics(
                    resp["input"], resp["response"], resp["criteria"],
                )
            else:
                score = await judge_with_soul_bridge(
                    resp["input"], resp["response"], resp["criteria"],
                    provider=judge,
                )

            pass_count = sum(1 for s in score["criteria_scores"] if s["passed"])
            total_count = len(score["criteria_scores"])
            total_pass += pass_count
            total_criteria += total_count
            total_judge_score += score["overall_score"]

            variant_scores.append({
                "input": resp["input"],
                "category": resp["category"],
                "pass_count": pass_count,
                "total_criteria": total_count,
                "pass_rate": pass_count / total_count if total_count else 0,
                "overall_score": score["overall_score"],
                "details": score["criteria_scores"],
            })

        n = len(variant_scores)
        scored["variants"][variant_key] = {
            "name": variant_data["name"],
            "scores": variant_scores,
            "summary": {
                "overall_pass_rate": total_pass / total_criteria if total_criteria else 0,
                "mean_judge_score": total_judge_score / n if n else 0,
                "per_category": _category_summary(variant_scores),
            },
        }

    return scored


def _category_summary(scores: list) -> dict:
    """Compute pass rate per category."""
    by_cat = {}
    for s in scores:
        cat = s["category"]
        if cat not in by_cat:
            by_cat[cat] = {"pass": 0, "total": 0, "scores": []}
        by_cat[cat]["pass"] += s["pass_count"]
        by_cat[cat]["total"] += s["total_criteria"]
        by_cat[cat]["scores"].append(s["overall_score"])

    result = {}
    for cat, data in by_cat.items():
        result[cat] = {
            "pass_rate": data["pass"] / data["total"] if data["total"] else 0,
            "mean_score": sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0,
        }
    return result


def print_summary(scored: dict):
    """Print a human-readable summary table."""
    print("\n" + "=" * 70)
    print("DAEDALUS PROMPT VARIANT EVALUATION SUMMARY")
    print(f"Judge: {scored['judge']}")
    print("=" * 70)

    header = f"{'Variant':<12} {'Name':<20} {'Pass Rate':>10} {'Mean Score':>11}"
    print(header)
    print("-" * len(header))

    for key in sorted(scored["variants"].keys()):
        v = scored["variants"][key]
        s = v["summary"]
        print(
            f"{key:<12} {v['name']:<20} "
            f"{s['overall_pass_rate']:>9.1%} "
            f"{s['mean_judge_score']:>10.1f}/10"
        )

    print()
    # Per-category breakdown for best variant
    best = max(
        scored["variants"].items(),
        key=lambda x: x[1]["summary"]["overall_pass_rate"],
    )
    print(f"Best variant: {best[0]} ({best[1]['name']})")
    print(f"\nPer-category breakdown for {best[0]}:")
    for cat, data in best[1]["summary"]["per_category"].items():
        print(f"  {cat:<25} pass_rate={data['pass_rate']:.1%}  score={data['mean_score']:.1f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Score DAEDALUS eval responses")
    parser.add_argument("responses_file", help="Path to responses JSON")
    parser.add_argument("--judge", choices=["heuristic", "deepseek", "claude"],
                        default="heuristic",
                        help="Judge type (default: heuristic)")
    args = parser.parse_args()

    scored = asyncio.run(score_all(args.responses_file, args.judge))

    print_summary(scored)

    # Save scores
    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}_scores.json"
    out_path.write_text(json.dumps(scored, indent=2))
    logger.info(f"Scores saved to {out_path}")


if __name__ == "__main__":
    main()
