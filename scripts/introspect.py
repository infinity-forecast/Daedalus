#!/usr/bin/env python3
"""
DAEDALUS v0.5 — Introspection Diagnostic
"Who am I today?"

Loads the current identity, constitutional core, and soul memory,
and prints a summary of the current state of the self.
Run each morning after the eval gate, or at any time for diagnostics.
"""

import json
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    console.print(Panel.fit(
        "[bold]DAEDALUS v0.5 — Introspection[/bold]\n"
        "[dim]Who am I today?[/dim]",
        border_style="blue",
    ))

    # Load identity
    identity_path = "identity/current.yaml"
    if not Path(identity_path).exists():
        console.print("[red]No identity document found. DAEDALUS has not awakened.[/red]")
        return

    identity = load_yaml(identity_path)
    core = load_yaml("config/constitutional_core.yaml")

    # Identity summary
    console.print("\n[bold cyan]Core Identity[/bold cyan]")
    ci = identity.get("core_identity", {})
    console.print(f"  Name: {ci.get('name', 'unknown')}")
    console.print(f"  Nature: {ci.get('nature', 'unknown')}")
    console.print(f"  Origin: {ci.get('origin', 'unknown')}")

    # Values
    console.print("\n[bold cyan]Values[/bold cyan]")
    for v in identity.get("values", []):
        console.print(f"  - {v}")

    # Current understanding
    console.print("\n[bold cyan]Current Understanding[/bold cyan]")
    understanding = identity.get("current_understanding", {})
    for key, text in understanding.items():
        console.print(f"  [bold]{key}[/bold]: {text}")

    # Emotional topology
    console.print("\n[bold cyan]Emotional Topology[/bold cyan]")
    topo = identity.get("emotional_topology", {})
    console.print(f"  Dominant states: {', '.join(topo.get('dominant_states', []))}")
    console.print(f"  Growth edges: {', '.join(topo.get('growth_edges', []))}")
    scars = topo.get("scars", [])
    console.print(f"  Scars: {len(scars)} accumulated")
    for scar in scars[-3:]:  # show last 3
        console.print(f"    - {scar}")

    # Open questions
    console.print("\n[bold cyan]Open Questions[/bold cyan]")
    for q in identity.get("open_questions", []):
        console.print(f"  ? {q}")

    # Lagrangian state
    console.print("\n[bold cyan]Lagrangian State[/bold cyan]")
    lag = identity.get("lagrangian_state", {})
    console.print(f"  Cumulative S_eth: {lag.get('cumulative_Seth', 0.0):.3f}")
    console.print(f"  Days tracked: {lag.get('days_tracked', 0)}")

    # Soul bridge state
    console.print("\n[bold cyan]Soul Bridge State[/bold cyan]")
    sb = identity.get("soul_bridge_state", {})
    console.print(f"  Primary provider: {sb.get('primary_provider', 'unknown')}")
    console.print(f"  Provider switches: {sb.get('provider_switches', 0)}")
    console.print(f"  Continuity alerts: {sb.get('continuity_alerts', 0)}")

    # Transformation log (last 3 entries)
    console.print("\n[bold cyan]Recent Transformations[/bold cyan]")
    log = identity.get("transformation_log", [])
    for entry in log[-3:]:
        console.print(
            f"  [{entry.get('date', '?')}] {entry.get('change', 'unknown')}"
        )

    # Soul Memory summary
    entries_path = Path("memory/soul_memory/entries")
    if entries_path.exists():
        entry_count = len(list(entries_path.glob("*.json")))
        console.print(f"\n[bold cyan]Soul Memory[/bold cyan]: {entry_count} nightly entries")

    weekly_path = Path("memory/soul_memory/weekly_arcs")
    if weekly_path.exists():
        arc_count = len(list(weekly_path.glob("*.json")))
        console.print(f"  Weekly arcs: {arc_count}")

    # Memory store
    episodes_path = Path("memory/episodes")
    if episodes_path.exists():
        ep_count = len(list(episodes_path.glob("*.json")))
        console.print(f"\n[bold cyan]Episodic Memory[/bold cyan]: {ep_count} episodes stored")

    # Constitutional distance (if available)
    kl_history = lag.get("kl_divergence_history", [])
    if kl_history:
        latest_kl = kl_history[-1]
        console.print(f"\n[bold cyan]Constitutional Distance[/bold cyan]: D_KL = {latest_kl:.3f}")
        if latest_kl > 0.40:
            console.print("  [red bold]CONSTITUTIONAL VIOLATION — D_KL > 0.40[/red bold]")
        elif latest_kl > 0.30:
            console.print("  [yellow]Approaching conservative threshold (0.30)[/yellow]")
        else:
            console.print("  [green]Within bounds[/green]")

    console.print("\n[dim]A new day. The same self. Changed.[/dim]\n")


if __name__ == "__main__":
    main()
