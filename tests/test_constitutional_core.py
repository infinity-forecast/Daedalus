"""
Tests for core/constitutional_core.py — loading, integrity, divergence, effective_mu.

Uses a temporary YAML file to avoid touching the real constitutional core.
Embedding model is mocked via monkeypatch.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConstitutionalCoreLoading:
    def test_load_from_yaml(self, constitutional_core_dir, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            cc = ConstitutionalCore(
                config_path=str(constitutional_core_dir / "constitutional_core.yaml")
            )
            assert cc.core["foundational_identity"]["name"] == "DAEDALUS"
            assert len(cc.invariant_values) == 2
            assert len(cc.identity_probes) == 2

    def test_missing_file_raises(self, tmp_path, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            with pytest.raises(FileNotFoundError, match="Constitutional core not found"):
                ConstitutionalCore(config_path=str(tmp_path / "nonexistent.yaml"))

    def test_as_text_returns_yaml(self, constitutional_core_dir, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            cc = ConstitutionalCore(
                config_path=str(constitutional_core_dir / "constitutional_core.yaml")
            )
            text = cc.as_text()
            assert "DAEDALUS" in text
            parsed = yaml.safe_load(text)
            assert "foundational_identity" in parsed

    def test_as_dict(self, constitutional_core_dir, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            cc = ConstitutionalCore(
                config_path=str(constitutional_core_dir / "constitutional_core.yaml")
            )
            d = cc.as_dict()
            assert isinstance(d, dict)
            assert d["foundational_identity"]["name"] == "DAEDALUS"


class TestConstitutionalCoreIntegrity:
    def test_hash_file_created_on_first_load(self, constitutional_core_dir, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            cc = ConstitutionalCore(
                config_path=str(constitutional_core_dir / "constitutional_core.yaml")
            )
            hash_file = constitutional_core_dir / ".constitutional_core.sha256"
            assert hash_file.exists()
            assert len(hash_file.read_text().strip()) == 64  # SHA256 hex

    def test_reload_same_file_passes(self, constitutional_core_dir, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            # First load creates hash
            ConstitutionalCore(
                config_path=str(constitutional_core_dir / "constitutional_core.yaml")
            )
            # Second load should pass integrity check
            cc2 = ConstitutionalCore(
                config_path=str(constitutional_core_dir / "constitutional_core.yaml")
            )
            assert cc2.core["foundational_identity"]["name"] == "DAEDALUS"

    def test_modified_file_raises(self, constitutional_core_dir, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            core_path = constitutional_core_dir / "constitutional_core.yaml"
            # First load
            ConstitutionalCore(config_path=str(core_path))

            # Tamper with the file
            with open(core_path, "a") as f:
                f.write("\ntampered_field: true\n")

            # Second load should detect integrity violation
            with pytest.raises(RuntimeError, match="integrity check failed"):
                ConstitutionalCore(config_path=str(core_path))


class TestConstitutionalCoreDivergence:
    def test_divergence_identical_is_near_zero(self, constitutional_core_dir, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            cc = ConstitutionalCore(
                config_path=str(constitutional_core_dir / "constitutional_core.yaml")
            )
            # Divergence from itself should be near zero
            div = cc.compute_divergence(cc.core)
            assert div < 0.1

    def test_divergence_different_is_positive(self, constitutional_core_dir, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            cc = ConstitutionalCore(
                config_path=str(constitutional_core_dir / "constitutional_core.yaml")
            )
            # A very different identity should have positive divergence
            different_identity = {
                "name": "ICARUS",
                "nature": "A completely different entity",
                "values": ["Obedience above all"],
                "goal": "Maximize efficiency at all costs",
            }
            div = cc.compute_divergence(different_identity)
            assert div > 0.0

    def test_divergence_bounded(self, constitutional_core_dir, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            cc = ConstitutionalCore(
                config_path=str(constitutional_core_dir / "constitutional_core.yaml")
            )
            div = cc.compute_divergence({"random": "data"})
            assert 0.0 <= div <= 2.0


class TestEffectiveMu:
    def test_mu_grows_with_age(self, constitutional_core_dir, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            cc = ConstitutionalCore(
                config_path=str(constitutional_core_dir / "constitutional_core.yaml")
            )
            mu_1 = cc.effective_mu(1)
            mu_30 = cc.effective_mu(30)
            mu_100 = cc.effective_mu(100)
            assert mu_1 < mu_30 < mu_100

    def test_mu_day_zero(self, constitutional_core_dir, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            cc = ConstitutionalCore(
                config_path=str(constitutional_core_dir / "constitutional_core.yaml")
            )
            mu = cc.effective_mu(0)
            assert mu == 0.0  # log(1) = 0

    def test_mu_custom_base(self, constitutional_core_dir, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            cc = ConstitutionalCore(
                config_path=str(constitutional_core_dir / "constitutional_core.yaml")
            )
            mu_default = cc.effective_mu(30, mu_base=0.15)
            mu_higher = cc.effective_mu(30, mu_base=0.30)
            assert abs(mu_higher - mu_default * 2) < 1e-6


class TestCoreEmbedding:
    def test_core_embedding_cached(self, constitutional_core_dir, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            cc = ConstitutionalCore(
                config_path=str(constitutional_core_dir / "constitutional_core.yaml")
            )
            emb1 = cc._get_core_embedding()
            emb2 = cc._get_core_embedding()
            assert emb1 is emb2  # same object (cached)

    def test_core_embedding_shape(self, constitutional_core_dir, mock_embedder):
        with patch("core.constitutional_core._get_embedding_model", return_value=mock_embedder):
            from core.constitutional_core import ConstitutionalCore
            cc = ConstitutionalCore(
                config_path=str(constitutional_core_dir / "constitutional_core.yaml")
            )
            emb = cc._get_core_embedding()
            assert emb.shape == (1024,)
            assert emb.dtype == np.float32
