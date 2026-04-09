"""
DAEDALUS v0.5 — Environment Setup Script

Manages the user-level secret store at ~/.apikey/
API keys are NEVER stored in the project directory.

Usage:
  python scripts/setup_env.py                    # interactive setup wizard
  python scripts/setup_env.py --from path.json   # import from a JSON file
  python scripts/setup_env.py --status           # show current key status
  python scripts/setup_env.py --help             # show this help

The canonical key store is:  ~/.apikey/daedalus.json
Format: {"DEEPSEEK_API_KEY": "sk-...", "ANTHROPIC_API_KEY": "sk-..."}

Keys already present in ~/.apikey/ (as txt or .env) are automatically discovered
and consolidated into daedalus.json on first run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


APIKEY_DIR = Path.home() / ".apikey"
DAEDALUS_JSON = APIKEY_DIR / "daedalus.json"

# The keys DAEDALUS needs, with display labels
KNOWN_KEYS: dict[str, dict] = {
    "DEEPSEEK_API_KEY": {
        "label":    "DeepSeek R1 (primary Soul Bridge provider)",
        "required": True,
        "url":      "https://platform.deepseek.com/",
    },
    "ANTHROPIC_API_KEY": {
        "label":    "Anthropic Claude (secondary Soul Bridge provider — Sonnet/Opus)",
        "required": False,
        "url":      "https://console.anthropic.com/",
    },
    "XAI_API_KEY": {
        "label":    "xAI Grok (tertiary provider — optional)",
        "required": False,
        "url":      "https://console.x.ai/",
    },
}

# Filename-based key discovery (for existing key files in ~/.apikey/)
FILENAME_MAP: dict[str, str] = {
    "deepseek_apikey.txt":  "DEEPSEEK_API_KEY",
    "anthropic_apikey.txt": "ANTHROPIC_API_KEY",
    "claude_apikey.txt":    "ANTHROPIC_API_KEY",
    "xai_apikey.txt":       "XAI_API_KEY",
    "grok_apikey.txt":      "XAI_API_KEY",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_apikey_dir() -> None:
    APIKEY_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)


def load_current_store() -> dict[str, str]:
    """Read existing ~/.apikey/daedalus.json (or empty dict)."""
    if DAEDALUS_JSON.exists():
        try:
            data = json.loads(DAEDALUS_JSON.read_text())
            if isinstance(data, dict):
                return {k: str(v) for k, v in data.items()}
        except json.JSONDecodeError:
            pass
    return {}


def discover_existing_keys() -> dict[str, str]:
    """
    Auto-discover keys already present in ~/.apikey/ from txt / .env files.
    Used during first-time setup to avoid re-entering known keys.
    """
    found: dict[str, str] = {}

    # Single-key txt files
    for fname, env_var in FILENAME_MAP.items():
        fpath = APIKEY_DIR / fname
        if fpath.exists():
            value = fpath.read_text().strip()
            if value and not value.startswith("{"):  # exclude JSON files
                found[env_var] = value

    # .env KEY=value file
    env_file = APIKEY_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                found[k.strip()] = v.strip()

    return found


def save_store(store: dict[str, str]) -> None:
    """Write the key store to ~/.apikey/daedalus.json (mode 600)."""
    ensure_apikey_dir()
    DAEDALUS_JSON.write_text(json.dumps(store, indent=2) + "\n")
    DAEDALUS_JSON.chmod(0o600)


def print_status(store: dict[str, str]) -> None:
    print(f"\n── DAEDALUS Key Store: {DAEDALUS_JSON}")
    print(f"   {'KEY':<25} {'STATUS':<22} DESCRIPTION")
    print("   " + "─" * 72)
    for env_var, meta in KNOWN_KEYS.items():
        if store.get(env_var) or os.environ.get(env_var):
            status = "✓  configured"
        elif meta["required"]:
            status = "✗  MISSING (required)"
        else:
            status = "–  not set (optional)"
        print(f"   {env_var:<25} {status:<22} {meta['label']}")
    print()


# ── Modes ────────────────────────────────────────────────────────────────────

def cmd_status() -> None:
    store = load_current_store()
    print_status(store)


def cmd_import(json_path: str) -> None:
    """Import keys from an external JSON file into ~/.apikey/daedalus.json."""
    src = Path(json_path).expanduser()
    if not src.exists():
        print(f"Error: file not found: {src}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(src.read_text())
        if not isinstance(data, dict):
            print("Error: JSON file must be an object (dict), e.g. {\"DEEPSEEK_API_KEY\": \"sk-...\"}")
            sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON — {e}", file=sys.stderr)
        sys.exit(1)

    store = load_current_store()
    updated = {k: v for k, v in data.items() if k in KNOWN_KEYS}
    unknown = [k for k in data if k not in KNOWN_KEYS]

    store.update(updated)
    save_store(store)

    print(f"Imported {len(updated)} key(s) into {DAEDALUS_JSON}")
    if unknown:
        print(f"Ignored unknown keys: {', '.join(unknown)}")

    print_status(store)


def cmd_interactive() -> None:
    """Interactive wizard: prompt for each missing key."""
    ensure_apikey_dir()

    # Seed with auto-discovered keys from existing files
    store = load_current_store()
    discovered = discover_existing_keys()

    newly_discovered = {k: v for k, v in discovered.items() if k not in store}
    if newly_discovered:
        print(f"\nDiscovered {len(newly_discovered)} key(s) from existing files in {APIKEY_DIR}:")
        for k in newly_discovered:
            print(f"  - {k}")
        store.update(newly_discovered)
        save_store(store)

    print(f"\n── DAEDALUS Environment Setup ────────────────────────────────")
    print(f"   Key store: {DAEDALUS_JSON}")
    print(f"   Keys are stored OUTSIDE the project. Safe for public git repos.")
    print()

    changed = False
    for env_var, meta in KNOWN_KEYS.items():
        existing = store.get(env_var, "")
        tag = "(required)" if meta["required"] else "(optional)"
        masked = f"{'*' * 8}{existing[-4:]}" if existing else "not set"

        print(f"  {env_var} {tag}")
        print(f"    {meta['label']}")
        if not existing:
            print(f"    Get key at: {meta['url']}")
        print(f"    Current: {masked}")

        user_input = input("    New value (Enter to keep current, 'skip' to leave empty): ").strip()

        if user_input.lower() == "skip":
            store.pop(env_var, None)
            changed = True
        elif user_input:
            store[env_var] = user_input
            changed = True
        print()

    if changed:
        save_store(store)
        print(f"✓ Key store saved: {DAEDALUS_JSON}")
    else:
        print("No changes made.")

    print_status(store)


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DAEDALUS secret store manager — keys live in ~/.apikey/, never in the project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_env.py               # interactive wizard
  python scripts/setup_env.py --status      # show key status
  python scripts/setup_env.py --from keys.json   # import from JSON file

JSON file format:
  {
    "DEEPSEEK_API_KEY":  "sk-...",
    "ANTHROPIC_API_KEY": "sk-ant-...",
    "XAI_API_KEY":       "xai-..."
  }
        """,
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current key configuration status and exit."
    )
    parser.add_argument(
        "--from", dest="json_file", metavar="PATH",
        help="Import keys from a JSON file into ~/.apikey/daedalus.json."
    )
    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.json_file:
        cmd_import(args.json_file)
    else:
        cmd_interactive()


if __name__ == "__main__":
    main()
