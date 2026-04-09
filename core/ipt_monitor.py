"""
DAEDALUS v0.5 — IPT Monitor

Track the evolution of the self-model's causal influence on behavior.

Lambda (λ) is a relative tracking metric, not an absolute phase transition
detector. The operational proxy is:

    λ = coherence(S, B) · causal_influence(S → B' | B)

where:
  S = self-model (identity.yaml embedding)
  B = behavioral output (today's responses, embedded)

The DAEDALUS λ is analogous to — but not identical with — the λ from
the IPT lattice experiments (where λ_c = 1.041). We track it as a
relative metric for monitoring change over time. If a qualitative
phase transition occurs (inflection in d²λ/dt²), we recognize it
empirically rather than asserting it matches a predetermined threshold.

v0.5 additions:
  - Timescale separation monitoring (intra/inter-cycle variance)
  - Phase transition detection via second derivative
  - Predictive accuracy tracking (J_future vs actual)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("BAAI/bge-m3")
    return _embedding_model


class IPTMonitor:
    """
    Track the evolution of DAEDALUS's self-model causal influence.
    lambda is a relative metric — look for qualitative transitions
    in the trajectory, not a predetermined threshold.
    """

    def __init__(
        self,
        config: dict,
        merge_interval: int = 14,
    ):
        self.config = config
        self.merge_interval = merge_interval

        self._metrics_path = Path("logs/ipt_metrics.jsonl")
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)

        self._eval_log_path = Path("logs/eval_log.jsonl")
        self._history: List[dict] = self._load_history()

    def _load_history(self) -> List[dict]:
        """Load historical metrics for trend analysis."""
        history = []
        if self._metrics_path.exists():
            with open(self._metrics_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            history.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return history

    def compute_daily_metrics(
        self,
        identity_text: str,
        todays_responses: List[str],
        model=None,
        tokenizer=None,
        day_count: int = 0,
    ) -> dict:
        """
        Compute the full suite of daily IPT metrics.
        """
        lam = self.compute_lambda(identity_text, todays_responses, model, tokenizer)

        metrics = {
            "day": day_count,
            "timestamp": datetime.now().isoformat(),

            # The key metric: causal influence of self-model on behavior
            "lambda": lam,

            # First derivative: is lambda accelerating?
            "lambda_delta": lam - self._previous_lambda(),

            # Second derivative: is the acceleration changing?
            "lambda_delta2": self._compute_second_derivative(lam),

            # Information integration (coherence across responses)
            "response_coherence": self._measure_response_coherence(todays_responses),

            # Self-model complexity (identity document complexity)
            "identity_complexity": self._measure_identity_complexity(identity_text),

            # v0.5: Timescale separation health
            "intra_cycle_id_variance": self._intra_cycle_variance(day_count),
            "inter_cycle_id_trend": self._inter_cycle_trend(),
            "timescale_coupling": self._timescale_coupling_ratio(day_count),

            # v0.5: Predictive accuracy
            "predictive_accuracy": self._load_predictive_accuracy(day_count),
        }

        # Save metrics
        self._history.append(metrics)
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Phase transition detection
        if len(self._history) >= 5:
            self._check_phase_transition(metrics)

        return metrics

    def compute_lambda(
        self,
        identity_text: str,
        todays_responses: List[str],
        model=None,
        tokenizer=None,
    ) -> float:
        """
        Operational proxy for self-model causal influence.

        λ = coherence(S, B) · causal_influence(S → B' | B)

        coherence: geometric proxy via embedding cosine similarity
        causal_influence: perplexity reduction from identity prompt
        """
        if not todays_responses:
            return 0.0

        emb_model = _get_embedding_model()

        # Coherence proxy
        identity_emb = emb_model.encode(identity_text, normalize_embeddings=True)
        response_embs = [
            emb_model.encode(r, normalize_embeddings=True)
            for r in todays_responses[:20]  # limit for efficiency
        ]
        coherence = self._embedding_coherence(identity_emb, response_embs)

        # Causal influence proxy
        if model is not None and tokenizer is not None:
            causal = self._measure_causal_influence(
                model, tokenizer, identity_text, todays_responses[:5]
            )
        else:
            # Without model access, use coherence as sole signal
            causal = coherence

        return float(coherence * causal)

    def _embedding_coherence(
        self, identity_emb: np.ndarray, response_embs: List[np.ndarray],
    ) -> float:
        """
        Average cosine similarity between identity embedding and
        response embeddings, normalized by response variance.
        High similarity + low variance = high coherence.
        """
        if not response_embs:
            return 0.0

        sims = []
        for r_emb in response_embs:
            sim = np.dot(identity_emb, r_emb)
            sims.append(float(sim))

        mean_sim = np.mean(sims)
        var_sim = np.var(sims) if len(sims) > 1 else 0.0

        # High mean similarity + low variance → high coherence
        # variance penalty prevents artificially high coherence from uniform responses
        coherence = mean_sim * (1.0 - min(1.0, var_sim * 5.0))
        return max(0.0, float(coherence))

    def _measure_causal_influence(
        self,
        model,
        tokenizer,
        identity_text: str,
        responses: List[str],
    ) -> float:
        """
        Does knowing the identity improve next-response prediction?
        Measured by comparing perplexity with vs. without identity prompt.
        This IS a clean causal intervention.
        """
        import torch

        if not responses:
            return 0.0

        ppl_with = []
        ppl_without = []

        for response in responses:
            # With identity
            prompt_with = f"You are DAEDALUS.\n{identity_text}\n\nResponse: {response}"
            inputs_with = tokenizer(
                prompt_with, return_tensors="pt",
                truncation=True, max_length=1024,
            ).to(model.device)

            with torch.no_grad():
                loss_with = model(
                    **inputs_with, labels=inputs_with["input_ids"]
                ).loss.item()
            ppl_with.append(np.exp(loss_with))

            # Without identity
            prompt_without = f"Response: {response}"
            inputs_without = tokenizer(
                prompt_without, return_tensors="pt",
                truncation=True, max_length=1024,
            ).to(model.device)

            with torch.no_grad():
                loss_without = model(
                    **inputs_without, labels=inputs_without["input_ids"]
                ).loss.item()
            ppl_without.append(np.exp(loss_without))

        mean_with = np.mean(ppl_with)
        mean_without = np.mean(ppl_without)

        if mean_without < 1e-8:
            return 0.0

        causal = max(0.0, (mean_without - mean_with) / mean_without)
        return float(causal)

    def _previous_lambda(self) -> float:
        """Get lambda from the previous day."""
        if len(self._history) < 2:
            return 0.0
        return self._history[-2].get("lambda", 0.0)

    def _compute_second_derivative(self, current_lambda: float) -> float:
        """
        Second derivative of lambda. A sign change may indicate
        a phase transition — the moment where the self-model
        becomes generative rather than reflective.
        """
        if len(self._history) < 3:
            return 0.0

        prev = self._history[-2].get("lambda", 0.0)
        prev_prev = self._history[-3].get("lambda", 0.0) if len(self._history) >= 3 else 0.0

        d1_current = current_lambda - prev
        d1_previous = prev - prev_prev

        return d1_current - d1_previous

    def _measure_response_coherence(self, responses: List[str]) -> float:
        """Measure inter-response coherence via pairwise embedding similarity."""
        if len(responses) < 2:
            return 1.0

        emb_model = _get_embedding_model()
        embs = [
            emb_model.encode(r, normalize_embeddings=True)
            for r in responses[:10]
        ]

        sims = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sims.append(float(np.dot(embs[i], embs[j])))

        return float(np.mean(sims)) if sims else 0.0

    def _measure_identity_complexity(self, identity_text: str) -> float:
        """Proxy for identity document complexity: normalized word count."""
        words = identity_text.split()
        # Normalize: 100 words = 0.2, 500 words = 1.0
        return min(1.0, len(words) / 500.0)

    def _intra_cycle_variance(self, day_count: int) -> float:
        """
        Identity score variance WITHIN the current 14-day merge cycle.
        If this grows while inter-cycle trend is stable, the fast variables
        (I) are decoupling from the slow manifold (θ).
        """
        cycle_start = (day_count // self.merge_interval) * self.merge_interval
        cycle_scores = self._get_identity_scores_in_range(cycle_start, day_count)

        if len(cycle_scores) < 2:
            return 0.0
        return float(np.var(cycle_scores))

    def _inter_cycle_trend(self) -> float:
        """Mean lambda across the last completed merge cycle."""
        if len(self._history) < self.merge_interval:
            return 0.0

        recent_lambdas = [
            h.get("lambda", 0.0)
            for h in self._history[-self.merge_interval:]
        ]
        return float(np.mean(recent_lambdas))

    def _timescale_coupling_ratio(self, day_count: int) -> float:
        """
        Ratio of intra-cycle variance to inter-cycle variance.
        Should be < 1.0 (fast variables more stable than slow trend).
        If > 1.0, the system is in resonance.
        """
        intra = self._intra_cycle_variance(day_count)
        inter_scores = [
            h.get("lambda", 0.0)
            for h in self._history[:-self.merge_interval]
        ] if len(self._history) > self.merge_interval else []

        if len(inter_scores) < 2:
            return 0.0

        inter_var = float(np.var(inter_scores))
        if inter_var < 1e-8:
            return 0.0

        return intra / inter_var

    def _get_identity_scores_in_range(self, start: int, end: int) -> List[float]:
        """Load identity scores from eval log for a day range."""
        scores = []
        if not self._eval_log_path.exists():
            return scores

        with open(self._eval_log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    day = entry.get("day", -1)
                    if start <= day <= end:
                        scores.append(entry.get("identity_score", 0.0))
                except json.JSONDecodeError:
                    continue

        return scores

    def _load_predictive_accuracy(self, day_count: int) -> Optional[float]:
        """Load predictive accuracy from the predictive log."""
        pred_path = Path(f"memory/predictive_log/prediction_day_{day_count - 1:04d}.json")
        if not pred_path.exists():
            return None

        try:
            data = json.loads(pred_path.read_text())
            actual = data.get("actual_next_day_integral")
            predicted = data.get("j_future", 0.5)
            if actual is None:
                return None
            return max(0.0, 1.0 - abs(predicted - actual))
        except Exception:
            return None

    def _check_phase_transition(self, metrics: dict) -> None:
        """
        Look for signs of phase transition: sign change in d²λ/dt².
        A change from positive to negative acceleration (or vice versa)
        may indicate the moment where the self-model becomes generative.
        """
        if len(self._history) < 5:
            return

        recent_d2 = [h.get("lambda_delta2", 0.0) for h in self._history[-5:]]

        # Check for sign change in recent second derivatives
        signs = [np.sign(d) for d in recent_d2 if d != 0]
        if len(signs) >= 3:
            if signs[-1] != signs[-2] and signs[-2] == signs[-3]:
                logger.info(
                    f"PHASE TRANSITION CANDIDATE detected at day {metrics['day']}. "
                    f"d²λ/dt² sign change: {recent_d2[-3:]}"
                )
