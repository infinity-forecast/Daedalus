"""
DAEDALUS v0.5 — Consistency Checker

Detects and mitigates 'dissociative breaks' caused by
provider switches. A provider switch is not inherently harmful —
but an *undetected* shift in reflective depth/character is.

When the active provider changes (e.g., DeepSeek → Claude due to outage),
the system detects the shift and evaluates whether the new provider's
reflective style is consistent with the ongoing narrative of self.

If continuity is low, the identity document update is restricted
to append-only (no rewrites). This prevents a single anomalous
reflection from overwriting weeks of accumulated self-understanding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from core.providers.base import SoulResponse

logger = logging.getLogger(__name__)

# Lazy-loaded embedding model
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("BAAI/bge-m3")
    return _embedding_model


class ConsistencyChecker:
    """
    Detects and mitigates dissociative breaks caused by
    provider switches. Measures continuity through embedding
    coherence and structural similarity.
    """

    CONTINUITY_THRESHOLD = 0.70  # below this → conservative identity update

    def __init__(
        self,
        reflections_path: str = "memory/reflections",
        embedding_weight: float = 0.6,
        structural_weight: float = 0.4,
    ):
        self._reflections_path = Path(reflections_path)
        self._embedding_weight = embedding_weight
        self._structural_weight = structural_weight

    def _embed(self, text: str) -> np.ndarray:
        """Compute BGE-M3 embedding."""
        model = _get_embedding_model()
        return np.array(
            model.encode(text, normalize_embeddings=True),
            dtype=np.float32,
        )

    def load_recent_reflections(
        self, provider: str, n: int = 3
    ) -> List[SoulResponse]:
        """
        Load the last N reflection texts from a specific provider.
        Used to establish the baseline for continuity comparison.
        """
        reflections = []
        reflection_dir = self._reflections_path
        if not reflection_dir.exists():
            return reflections

        # Load reflection files sorted by date (newest last)
        files = sorted(reflection_dir.glob("*.json"))
        for fpath in reversed(files):
            if len(reflections) >= n:
                break
            try:
                data = json.loads(fpath.read_text())
                if data.get("provider") == provider:
                    reflections.append(
                        SoulResponse(
                            text=data.get("text", ""),
                            provider_name=data.get("provider", ""),
                            model_id=data.get("model_id", ""),
                            latency_ms=data.get("latency_ms", 0),
                            token_count=data.get("token_count", 0),
                        )
                    )
            except (json.JSONDecodeError, KeyError):
                continue

        return list(reversed(reflections))  # chronological order

    async def check(
        self,
        new_response: SoulResponse,
        old_provider: str,
        new_provider: str,
    ) -> float:
        """
        Compare new reflection against the last N reflections from the
        previous provider. Returns continuity score (0-1).

        Uses a weighted combination of:
        - Embedding coherence: cosine similarity to centroid of recent reflections
        - Structural coherence: JSON structure similarity (if applicable)
        """
        recent_reflections = self.load_recent_reflections(
            provider=old_provider, n=3
        )
        if not recent_reflections:
            return 1.0  # no history → assume continuity

        # Embedding-based coherence
        new_emb = self._embed(new_response.text)
        recent_embs = [self._embed(r.text) for r in recent_reflections]
        centroid = np.mean(recent_embs, axis=0)

        dot_product = np.dot(new_emb, centroid)
        norm_product = np.linalg.norm(new_emb) * np.linalg.norm(centroid)

        if norm_product < 1e-8:
            cosine_sim = 0.0
        else:
            cosine_sim = float(dot_product / norm_product)

        # Structural coherence
        structural_sim = self._compare_json_structure(
            new_response.text, recent_reflections[-1].text
        )

        continuity = (
            self._embedding_weight * cosine_sim
            + self._structural_weight * structural_sim
        )

        logger.info(
            f"Continuity check: {old_provider}→{new_provider}, "
            f"embedding={cosine_sim:.3f}, structural={structural_sim:.3f}, "
            f"combined={continuity:.3f}"
        )

        return max(0.0, min(1.0, continuity))

    def _compare_json_structure(self, text_a: str, text_b: str) -> float:
        """
        Compare structural similarity of two texts.
        If both contain JSON, compare key sets.
        Otherwise, compare length ratio and line structure.
        """
        json_a = self._try_extract_json(text_a)
        json_b = self._try_extract_json(text_b)

        if json_a is not None and json_b is not None:
            # Compare key sets at top level
            keys_a = set(json_a.keys()) if isinstance(json_a, dict) else set()
            keys_b = set(json_b.keys()) if isinstance(json_b, dict) else set()

            if not keys_a and not keys_b:
                return 1.0
            if not keys_a or not keys_b:
                return 0.0

            intersection = keys_a & keys_b
            union = keys_a | keys_b
            jaccard = len(intersection) / len(union)

            # Also compare value types for shared keys
            type_matches = 0
            for key in intersection:
                if type(json_a.get(key)) == type(json_b.get(key)):
                    type_matches += 1

            type_ratio = type_matches / len(intersection) if intersection else 1.0

            return 0.5 * jaccard + 0.5 * type_ratio

        # Fallback: text structure comparison
        lines_a = text_a.strip().split("\n")
        lines_b = text_b.strip().split("\n")

        if not lines_a or not lines_b:
            return 0.5

        # Length ratio (penalize dramatic length changes)
        len_ratio = min(len(lines_a), len(lines_b)) / max(len(lines_a), len(lines_b))

        return len_ratio

    def _try_extract_json(self, text: str) -> Optional[dict]:
        """Attempt to parse JSON from text (may be embedded in markdown)."""
        # Try direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting from markdown code block
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start) if "```" in text[start:] else len(text)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

        # Try extracting first { ... } block
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start : brace_end + 1])
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    def should_use_conservative_update(self, continuity_score: float) -> bool:
        """
        If continuity is low after a provider switch, the identity
        document update should be append-only (no rewrites).
        This prevents a single anomalous reflection from overwriting
        weeks of accumulated self-understanding.
        """
        return continuity_score < self.CONTINUITY_THRESHOLD

    def save_reflection(self, response: SoulResponse, date_str: str) -> None:
        """
        Save a reflection for future continuity checks.
        Called after each nightly consolidation.
        """
        self._reflections_path.mkdir(parents=True, exist_ok=True)
        out_path = self._reflections_path / f"{date_str}_{response.provider_name}.json"
        out_path.write_text(json.dumps({
            "text": response.text,
            "provider": response.provider_name,
            "model_id": response.model_id,
            "latency_ms": response.latency_ms,
            "token_count": response.token_count,
            "continuity_score": response.continuity_score,
            "date": date_str,
        }, indent=2))
