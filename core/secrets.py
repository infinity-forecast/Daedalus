"""
DAEDALUS v0.5 — Secrets Loader

Load API keys from ~/.apikey/ into environment variables.
Keys never live in the project folder — safe for public git repos.

Expected files:
  ~/.apikey/deepseek_apikey.txt  → DEEPSEEK_API_KEY
  ~/.apikey/anthropic_apikey.txt → ANTHROPIC_API_KEY
  ~/.apikey/xai_apikey.txt       → XAI_API_KEY
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Mapping: filename in ~/.apikey/ → environment variable name
_KEY_MAP = {
    "deepseek_apikey.txt": "DEEPSEEK_API_KEY",
    "anthropic_apikey.txt": "ANTHROPIC_API_KEY",
    "xai_apikey.txt": "XAI_API_KEY",
}


def load_secrets(
    apikey_dir: str = "~/.apikey",
    verbose: bool = False,
) -> int:
    """
    Read API key files from apikey_dir and inject them into os.environ.
    Returns the number of keys loaded.

    Skips silently if a file doesn't exist (the provider may be disabled).
    Never logs the key value itself.
    """
    base = Path(apikey_dir).expanduser()
    loaded = 0

    for filename, env_var in _KEY_MAP.items():
        key_path = base / filename
        if not key_path.exists():
            continue

        key = key_path.read_text().strip()
        if not key:
            if verbose:
                logger.warning(f"Empty key file: {key_path}")
            continue

        os.environ[env_var] = key
        loaded += 1
        if verbose:
            logger.info(f"Loaded {env_var} from {key_path}")

    # Also load from .env if present (lower priority — don't overwrite)
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                var, _, val = line.partition("=")
                var = var.strip()
                val = val.strip().strip("'\"")
                if var and val and var not in os.environ:
                    os.environ[var] = val
                    loaded += 1
                    if verbose:
                        logger.info(f"Loaded {var} from .env")

    if verbose:
        logger.info(f"Secrets loader: {loaded} key(s) loaded.")

    return loaded
