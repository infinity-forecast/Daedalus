"""
DAEDALUS v0.5 — Constitutional Core

The invariant DNA of the self. This document never changes
after initialization. It defines the potential well in which
the evolving identity lives.

Metaphor: I_core is the shape of the container.
identity.yaml is the liquid that fills it and shifts with time.
The liquid can slosh, form waves, even overflow slightly —
but the container's shape constrains the space of possible selves.

The D_KL computation uses embedding cosine distance as an
approximation. In v0.6+, this would be replaced with geodesic
distance on the identity manifold using the Fisher information metric.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Embedding model is loaded lazily to avoid import-time GPU allocation
_embedding_model = None


def _get_embedding_model():
    """Lazy-load BGE-M3 embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("BAAI/bge-m3")
        logger.info("BGE-M3 embedding model loaded for constitutional core.")
    return _embedding_model


class ConstitutionalCore:
    """
    The invariant DNA of the self. This document never changes
    after initialization. It defines the potential well in which
    the evolving identity lives.

    The one hard constraint in a system of gradients:
    D_KL(I(t) ‖ I_core) ≤ kl_max is a wall, not a slope.
    """

    def __init__(self, config_path: str = "config/constitutional_core.yaml"):
        self._config_path = Path(config_path)
        self.core = self._load_core()
        self._core_text: Optional[str] = None
        self._core_embedding: Optional[np.ndarray] = None
        self._core_hash: Optional[str] = None
        self._verify_integrity()

    def _load_core(self) -> dict:
        """Load the frozen constitutional core from YAML."""
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Constitutional core not found at {self._config_path}. "
                f"This file must exist before DAEDALUS can awaken."
            )
        with open(self._config_path, "r") as f:
            core = yaml.safe_load(f)
        logger.info(f"Constitutional core loaded from {self._config_path}")
        return core

    def _verify_integrity(self) -> None:
        """
        Compute and store hash of the constitutional core.
        On subsequent loads, verify the hash hasn't changed.
        This is the immune system against accidental modification.
        """
        core_bytes = yaml.dump(self.core, sort_keys=True).encode("utf-8")
        current_hash = hashlib.sha256(core_bytes).hexdigest()

        hash_file = self._config_path.parent / ".constitutional_core.sha256"

        if hash_file.exists():
            stored_hash = hash_file.read_text().strip()
            if stored_hash != current_hash:
                logger.critical(
                    "CONSTITUTIONAL CORE INTEGRITY VIOLATION. "
                    f"Expected hash: {stored_hash[:16]}... "
                    f"Got: {current_hash[:16]}... "
                    "The DNA has been modified. This should NEVER happen. "
                    "Restore from backup or investigate immediately."
                )
                raise RuntimeError(
                    "Constitutional core integrity check failed. "
                    "The invariant DNA has been modified."
                )
        else:
            # First load — store the hash
            hash_file.write_text(current_hash + "\n")
            logger.info(
                f"Constitutional core hash stored: {current_hash[:16]}..."
            )

        self._core_hash = current_hash

    def as_text(self) -> str:
        """Return the core as formatted YAML text."""
        if self._core_text is None:
            self._core_text = yaml.dump(
                self.core, default_flow_style=False, allow_unicode=True
            )
        return self._core_text

    def as_dict(self) -> dict:
        """Return the core as a dictionary."""
        return self.core

    @property
    def identity_probes(self) -> list:
        """Return the 5 constitutional identity probes."""
        return self.core.get("identity_probes", [])

    @property
    def invariant_values(self) -> list:
        """Return the invariant values."""
        return self.core.get("invariant_values", [])

    @property
    def foundational_identity(self) -> dict:
        """Return the foundational identity block."""
        return self.core.get("foundational_identity", {})

    def _embed(self, text: str) -> np.ndarray:
        """Compute BGE-M3 embedding of text, L2-normalized."""
        model = _get_embedding_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)

    def _get_core_embedding(self) -> np.ndarray:
        """Compute and cache the embedding of the constitutional core text."""
        if self._core_embedding is None:
            self._core_embedding = self._embed(self.as_text())
        return self._core_embedding

    def compute_divergence(self, current_identity: dict) -> float:
        """
        Compute D_KL(I(t) ‖ I_core) as cosine distance between
        the current identity embedding and the constitutional core
        embedding.

        Uses BGE-M3 for both embeddings. The KL divergence is
        approximated as 1 - cosine_similarity (bounded [0, 2]).

        In the full Fisher information metric (v0.6+), this would
        be replaced with geodesic distance on the identity manifold.

        Returns:
            float: Divergence in [0, 2]. 0 = identical, 2 = maximally divergent.
                   Typical operational range: 0.0 - 0.5.
        """
        core_emb = self._get_core_embedding()

        identity_text = yaml.dump(
            current_identity, default_flow_style=False, allow_unicode=True
        )
        current_emb = self._embed(identity_text)

        # Cosine similarity → divergence
        dot_product = np.dot(core_emb, current_emb)
        norm_product = np.linalg.norm(core_emb) * np.linalg.norm(current_emb)

        if norm_product < 1e-8:
            logger.warning("Near-zero norm in divergence computation. Returning max divergence.")
            return 2.0

        cosine_sim = dot_product / norm_product

        # Map: similarity 1.0 → divergence 0.0; similarity -1.0 → divergence 2.0
        divergence = 1.0 - float(cosine_sim)
        return divergence

    def effective_mu(self, day_count: int, mu_base: float = 0.15) -> float:
        """
        Compute the effective constitutional regularization weight.
        Grows logarithmically with age to stabilize the mature identity.

        At day 1:   mu ≈ 0.15 * log(2) ≈ 0.10
        At day 30:  mu ≈ 0.15 * log(31) ≈ 0.52
        At day 100: mu ≈ 0.15 * log(101) ≈ 0.69
        At day 365: mu ≈ 0.15 * log(366) ≈ 0.89
        """
        return mu_base * np.log(day_count + 1)
