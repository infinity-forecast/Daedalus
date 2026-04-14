"""
Tests for core/secrets.py — API key loading from files and .env.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.secrets import load_secrets


class TestLoadSecrets:
    def test_load_from_apikey_files(self, tmp_path, monkeypatch):
        # Create fake API key files
        (tmp_path / "deepseek_apikey.txt").write_text("sk-test-deepseek-123\n")
        (tmp_path / "anthropic_apikey.txt").write_text("sk-ant-test-456\n")

        # Clear any existing env vars
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        loaded = load_secrets(apikey_dir=str(tmp_path), verbose=True)
        assert loaded >= 2
        assert os.environ["DEEPSEEK_API_KEY"] == "sk-test-deepseek-123"
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test-456"

    def test_skip_missing_files(self, tmp_path, monkeypatch):
        # Only create one key file
        (tmp_path / "deepseek_apikey.txt").write_text("sk-test-123\n")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        loaded = load_secrets(apikey_dir=str(tmp_path))
        assert loaded >= 1
        assert os.environ.get("DEEPSEEK_API_KEY") == "sk-test-123"
        # Anthropic key should not be set
        assert os.environ.get("ANTHROPIC_API_KEY") is None or "ANTHROPIC_API_KEY" not in os.environ

    def test_skip_empty_files(self, tmp_path, monkeypatch):
        (tmp_path / "deepseek_apikey.txt").write_text("   \n")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        loaded = load_secrets(apikey_dir=str(tmp_path))
        # Empty file should be skipped
        assert os.environ.get("DEEPSEEK_API_KEY") != "   "

    def test_load_from_dotenv(self, tmp_path, monkeypatch):
        # Create a .env file in current directory
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=test_value\nANOTHER_VAR='quoted'\n")

        monkeypatch.delenv("TEST_VAR", raising=False)
        monkeypatch.delenv("ANOTHER_VAR", raising=False)
        monkeypatch.chdir(tmp_path)

        loaded = load_secrets(apikey_dir=str(tmp_path / "nonexistent"))
        assert os.environ.get("TEST_VAR") == "test_value"
        assert os.environ.get("ANOTHER_VAR") == "quoted"

    def test_dotenv_skips_comments(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("# This is a comment\nVALID_KEY=value\n")
        monkeypatch.delenv("VALID_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        load_secrets(apikey_dir=str(tmp_path / "nonexistent"))
        assert os.environ.get("VALID_KEY") == "value"

    def test_apikey_files_take_priority_over_dotenv(self, tmp_path, monkeypatch):
        # .env has a value
        env_file = tmp_path / ".env"
        env_file.write_text("DEEPSEEK_API_KEY=from_dotenv\n")

        # API key file also has a value
        (tmp_path / "deepseek_apikey.txt").write_text("from_file\n")

        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        load_secrets(apikey_dir=str(tmp_path))
        # File-loaded key should be set first; .env should NOT overwrite
        assert os.environ["DEEPSEEK_API_KEY"] == "from_file"

    def test_nonexistent_apikey_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Should not crash
        loaded = load_secrets(apikey_dir=str(tmp_path / "does_not_exist"))
        assert loaded >= 0

    def test_strips_whitespace(self, tmp_path, monkeypatch):
        (tmp_path / "deepseek_apikey.txt").write_text("  sk-key-with-spaces  \n")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        load_secrets(apikey_dir=str(tmp_path))
        assert os.environ["DEEPSEEK_API_KEY"] == "sk-key-with-spaces"
