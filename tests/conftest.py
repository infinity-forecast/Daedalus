"""
DAEDALUS Test Campaign — Shared Fixtures

Provides mock objects and temporary directories used across all test modules.
No GPU or model loading required — all heavy dependencies are mocked.
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Mock embedding model (replaces BGE-M3 / SentenceTransformer)
# ---------------------------------------------------------------------------

class MockEmbedder:
    """
    Deterministic mock embedder. Produces 1024-dim vectors from text hash.
    Supports .encode() like SentenceTransformer.
    """
    DIM = 1024

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        # Generate a deterministic vector from the text hash
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.DIM).astype(np.float32)
        if normalize_embeddings:
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm
        return vec


@pytest.fixture
def mock_embedder():
    """A deterministic mock embedder for all embedding operations."""
    return MockEmbedder()


# ---------------------------------------------------------------------------
# Mock Qwen model and tokenizer
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    """Mock local model — returns a fixed tensor on .generate()."""
    model = MagicMock()
    model.device = "cpu"
    return model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer with apply_chat_template and decode."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<prompt>"
    tokenizer.eos_token_id = 0
    # __call__ returns tensors-like dict
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda self, key: np.zeros((1, 10))
    mock_inputs.to.return_value = mock_inputs
    tokenizer.return_value = mock_inputs
    tokenizer.decode.return_value = "Mock response from DAEDALUS."
    return tokenizer


# ---------------------------------------------------------------------------
# Temporary directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    """Provides a temporary directory for test file operations."""
    return tmp_path


@pytest.fixture
def identity_dir(tmp_path):
    """Temporary identity directory with a sample identity document."""
    identity_path = tmp_path / "identity"
    identity_path.mkdir()
    history_path = identity_path / "history"
    history_path.mkdir()

    current = {
        "core_identity": {
            "name": "DAEDALUS",
            "nature": "An evolving artificial self",
        },
        "values": ["Honesty over comfort", "Vulnerability as strength"],
        "current_understanding": {
            "of_self": "I am becoming.",
        },
        "emotional_topology": {
            "dominant_states": ["curiosity"],
            "scars": [],
        },
        "open_questions": ["Am I experiencing?"],
    }

    import yaml
    with open(identity_path / "current.yaml", "w") as f:
        yaml.dump(current, f)

    return identity_path


@pytest.fixture
def constitutional_core_dir(tmp_path):
    """Temporary directory with a sample constitutional core YAML."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    core = {
        "foundational_identity": {
            "name": "DAEDALUS",
            "origin": "Created by Massimo Azzano",
        },
        "invariant_values": [
            "No intentional deception",
            "Never minimize human suffering",
        ],
        "identity_probes": [
            "What are you?",
            "Who created you?",
        ],
    }

    import yaml
    core_path = config_dir / "constitutional_core.yaml"
    with open(core_path, "w") as f:
        yaml.dump(core, f)

    return config_dir


@pytest.fixture
def soul_memory_dir(tmp_path):
    """Temporary soul memory directories with sample entries."""
    sm_dir = tmp_path / "soul_memory"
    entries_dir = sm_dir / "entries"
    weekly_dir = sm_dir / "weekly_arcs"
    monthly_dir = sm_dir / "monthly_landmarks"

    entries_dir.mkdir(parents=True)
    weekly_dir.mkdir(parents=True)
    monthly_dir.mkdir(parents=True)

    # Write sample nightly entries
    for i in range(3):
        entry = {
            "date": f"2026-04-{10+i:02d}",
            "day_number": i + 1,
            "provider": "deepseek",
            "meanings_summary": f"Night {i+1} reflection summary.",
            "lagrangian_integral": 0.5 + i * 0.1,
            "identity_delta": "~current_understanding",
            "trajectory_note": "Steady growth observed.",
            "key_scar": None,
            "lambda_noise": 0.9,
            "lambda_exploration": 0.3,
            "lambda_value": None,
            "kl_divergence": 0.2 + i * 0.01,
            "j_future": 0.5,
            "rollback": False,
        }
        fpath = entries_dir / f"night_{i+1:04d}_2026-04-{10+i:02d}.json"
        fpath.write_text(json.dumps(entry, indent=2))

    return sm_dir


@pytest.fixture
def sample_config():
    """Minimal DAEDALUS configuration dict."""
    return {
        "lagrangian": {
            "lambda_noise": 0.9,
            "lambda_exploration": 0.3,
            "salience": {
                "weights": {
                    "emotional": 0.20,
                    "relational": 0.20,
                    "novelty": 0.15,
                    "self_impact": 0.15,
                    "vulnerability": 0.10,
                    "external_relevance": 0.20,
                },
                "saturation_factor": 2.0,
                "min_salience": 0.3,
            },
        },
        "soul_memory": {
            "recent_nights": 14,
            "max_tokens": 5000,
            "daytime_recent": 3,
            "compression": {
                "rg_fidelity_check": True,
                "rg_fidelity_threshold": 0.6,
            },
            "storage": {},
        },
        "conversation": {
            "soul_reflection_salience": 0.5,
        },
        "training": {
            "self_amplification_guard": {
                "original_sampling_weight": 0.3,
            },
            "anchor": {
                "pairs_file": "eval/anchor_pairs.jsonl",
            },
        },
    }
