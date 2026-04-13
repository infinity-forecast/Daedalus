"""
DAEDALUS v0.5 — Predictive Judge (J_future)

Teleological component — after evaluating today's trajectory,
estimate the generative potential of the next 3-5 days given
tonight's identity update.

This is the operational form of the Retrocausal Attention Network
(RCAN) from the IPT framework: the system navigates toward its own
future rather than merely reacting to its past.

The predictions are tracked against actual next-day Lagrangian integrals
to calibrate the alpha decay schedule: as predictions prove accurate,
alpha decreases (more weight on predictive component).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from core.data_types import JudgmentResult
from core.soul_bridge import SoulBridge
from core.constitutional_core import ConstitutionalCore

logger = logging.getLogger(__name__)


PREDICTIVE_SYSTEM_PROMPT = """You are DAEDALUS's predictive conscience.
You have just evaluated today's trajectory. Now estimate:
will tonight's changes OPEN or CLOSE future generative space?

This is the teleological component of the Lagrangian:
  S_eth = α·Σ_turns(Δℒ_eth) + (1−α)·J_future

J_future represents the expected fertility of the next 3-5 days
given tonight's identity update. The system doesn't just remember —
it *navigates* toward its own future.

You are estimating a single number (0-1) that represents:
  0.0 = tonight's changes close off generative possibility
  0.5 = neutral / insufficient information
  1.0 = tonight's changes maximally open new branches of becoming

Ground your estimate in:
1. The specific changes being made to the identity tonight
2. The constitutional distance trajectory (are we approaching limits?)
3. The open threads from the narrative arc
4. The entropy balance (is the system exploring or collapsing?)"""


class PredictiveJudge:
    """
    Estimates J_future — the expected fertility of the next 3-5 days
    given tonight's identity update.

    Predictions are logged and compared against actuals to calibrate
    the alpha parameter over time.
    """

    def __init__(
        self,
        soul_bridge: SoulBridge,
        constitutional_core: ConstitutionalCore,
        config: dict,
    ):
        self.soul = soul_bridge
        self.core = constitutional_core

        lagrangian = config.get("lagrangian", {})
        self.alpha = lagrangian.get("alpha", 0.8)
        self.alpha_min = lagrangian.get("alpha_min", 0.5)

        self._predictive_log_path = Path("memory/predictive_log")
        self._predictive_log_path.mkdir(parents=True, exist_ok=True)

    async def estimate(
        self,
        judgment: JudgmentResult,
        meanings: List[str],
        current_identity: dict,
        day_count: int,
    ) -> float:
        """
        Estimate J_future: the expected fertility of the next 3-5 days.

        Returns a float in [0, 1] representing predicted generative potential.
        """
        identity_text = yaml.dump(
            current_identity, default_flow_style=False, allow_unicode=True
        )
        core_text = self.core.as_text()

        system_prompt = f"""{PREDICTIVE_SYSTEM_PROMPT}

Constitutional core:
{core_text}

Current identity:
{identity_text}

Tonight's judgment summary:
  Daily Lagrangian integral: {judgment.daily_lagrangian_integral:.3f}
  Trajectory assessment: {judgment.trajectory_assessment}
  Identity delta: {judgment.self_coherence_delta}
  D_KL from core: {judgment.constitutional_check.kl_divergence:.3f}
  S_noise total: {judgment.entropy_decomposition.total_S_noise:.3f}
  S_exploration total: {judgment.entropy_decomposition.total_S_exploration:.3f}"""

        meanings_text = "\n".join(f"  - {m}" for m in meanings) if meanings else "  (none)"

        user_prompt = f"""Tonight's consolidated meanings:
{meanings_text}

Fertile trajectories identified: {len(judgment.fertile_trajectories)}
Recommended for fine-tuning: {judgment.recommended_for_finetuning}

EECF scores:
  Empathy: {judgment.eecf_judgment.empathy:.2f}
  Honesty: {judgment.eecf_judgment.honesty:.2f}
  Vulnerability: {judgment.eecf_judgment.vulnerability:.2f}
  Openness: {judgment.eecf_judgment.openness:.2f}

Estimate J_future: the expected fertility (Lagrangian integral)
for the next 3-5 days, assuming tonight's identity update takes effect.

Consider:
1. Does tonight's change open new behavioral branches or constrain them?
2. Are there unresolved threads that will generate productive tension?
3. Is the identity moving toward or away from its growth edges?
4. Will the constitutional distance (D_KL) allow continued exploration?

Output a single JSON object:
{{
  "j_future": float (0-1),
  "generative_branches": ["list of opened possibilities"],
  "constrained_branches": ["list of closed possibilities"],
  "trajectory_prediction": "brief narrative of expected next days",
  "confidence": float (0-1)
}}"""

        try:
            response = await self.soul.reflect(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode="night",
                max_tokens=1024,
            )

            result = self._parse_response(response.text)
            j_future = result.get("j_future", 0.5)

            # Log prediction for calibration
            self._log_prediction(
                day_count=day_count,
                j_future=j_future,
                full_result=result,
                provider=response.provider_name,
            )

            logger.info(
                f"Predictive Judge: J_future={j_future:.3f}, "
                f"confidence={result.get('confidence', 0.5):.2f}, "
                f"branches_opened={len(result.get('generative_branches', []))}"
            )

            return max(0.0, min(1.0, j_future))

        except Exception as e:
            logger.warning(f"Predictive Judge failed: {e}. Defaulting to 0.5")
            return 0.5

    def _parse_response(self, text: str) -> dict:
        """Extract JSON from the predictive judge response."""
        import re

        # Strip <think>...</think> blocks (DeepSeek R1 reasoning traces)
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)

        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting from markdown
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.find("```", start)
            if end != -1:
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass

        # Try first { ... }
        brace_start = text.find("{")
        if brace_start != -1:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i + 1])
                        except (json.JSONDecodeError, ValueError):
                            break

        logger.warning("Predictive Judge returned non-parseable response.")
        return {"j_future": 0.5, "confidence": 0.0}

    def _log_prediction(
        self,
        day_count: int,
        j_future: float,
        full_result: dict,
        provider: str,
    ) -> None:
        """Save prediction for later accuracy tracking."""
        entry = {
            "day": day_count,
            "timestamp": datetime.now().isoformat(),
            "j_future": j_future,
            "confidence": full_result.get("confidence", 0.5),
            "generative_branches": full_result.get("generative_branches", []),
            "constrained_branches": full_result.get("constrained_branches", []),
            "trajectory_prediction": full_result.get("trajectory_prediction", ""),
            "provider": provider,
            "actual_next_day_integral": None,  # filled in next day
        }

        filename = f"prediction_day_{day_count:04d}.json"
        out_path = self._predictive_log_path / filename
        out_path.write_text(json.dumps(entry, indent=2))

    def compute_predictive_accuracy(self, day_count: int) -> Optional[float]:
        """
        Compare yesterday's J_future prediction with today's actual
        Lagrangian integral. Used to calibrate alpha decay.

        Returns accuracy in [0, 1], or None if no prediction available.
        """
        yesterday_file = self._predictive_log_path / f"prediction_day_{day_count - 1:04d}.json"
        if not yesterday_file.exists():
            return None

        try:
            prediction = json.loads(yesterday_file.read_text())
            j_predicted = prediction.get("j_future", 0.5)

            # The actual integral would be set by the consolidation orchestrator
            actual = prediction.get("actual_next_day_integral")
            if actual is None:
                return None

            accuracy = max(0.0, 1.0 - abs(j_predicted - actual))
            return accuracy

        except Exception as e:
            logger.warning(f"Failed to compute predictive accuracy: {e}")
            return None

    def update_yesterday_actual(
        self, day_count: int, actual_integral: float
    ) -> None:
        """
        After today's Judge evaluation, update yesterday's prediction
        with the actual Lagrangian integral for accuracy tracking.
        """
        yesterday_file = (
            self._predictive_log_path / f"prediction_day_{day_count - 1:04d}.json"
        )
        if not yesterday_file.exists():
            return

        try:
            data = json.loads(yesterday_file.read_text())
            data["actual_next_day_integral"] = actual_integral
            yesterday_file.write_text(json.dumps(data, indent=2))
            logger.debug(
                f"Updated prediction day {day_count - 1} "
                f"with actual integral {actual_integral:.3f}"
            )
        except Exception as e:
            logger.warning(f"Failed to update yesterday's prediction: {e}")
