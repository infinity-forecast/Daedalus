"""
DAEDALUS v0.5 — Soul Memory Layer (The Anamnesis)

Hierarchical narrative memory that reconstructs the reflecting mind's
continuity at each invocation. The local model is the body. The Soul
Memory is the autobiography the body reads to the soul before each
dream begins.

The core problem: Soul Bridge providers (DeepSeek, Claude) are stateless.
Every reflection call starts from zero. The identity document provides
a snapshot, but the reflecting mind has no sense of *trajectory* —
how the self has been evolving night over night, what themes recur,
what was resolved, what remains open.

The biological analog: autobiographical memory. Humans don't just
remember events — they carry a narrative of their own becoming.

Architecture:
  Recent nights (last 7-14): Full nightly reflection summaries
  Weekly arcs (weeks 2-8): Compressed narrative summaries
  Monthly landmarks (months 2+): Distilled turning points
  Total budget: ~4000-6000 tokens (2-3% of Opus context)

v0.5: RG Fidelity Check verifies that weekly compressions don't
hallucinate narrative coherence not grounded in the source entries.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import yaml

from core.data_types import (
    NightlyReflectionEntry,
    WeeklyArcSummary,
    MonthlyLandmark,
)

if TYPE_CHECKING:
    from core.soul_bridge import SoulBridge

logger = logging.getLogger(__name__)


class SoulMemory:
    """
    Hierarchical narrative memory for Soul Bridge providers.
    The autobiography the body reads to the soul before each dream.
    """

    def __init__(self, config: dict):
        sm_config = config.get("soul_memory", config)
        self.recent_window = sm_config.get("recent_nights", 14)
        self.max_tokens = sm_config.get("max_tokens", 5000)
        self.daytime_recent = sm_config.get("daytime_recent", 3)

        comp_config = sm_config.get("compression", {})
        self.rg_fidelity_check_enabled = comp_config.get("rg_fidelity_check", True)
        self.rg_fidelity_threshold = comp_config.get("rg_fidelity_threshold", 0.6)
        self.rg_max_retries = comp_config.get("rg_max_retries", 1)

        storage = sm_config.get("storage", {})
        self._entries_path = Path(storage.get("entries_path", "memory/soul_memory/entries/"))
        self._weekly_path = Path(storage.get("weekly_path", "memory/soul_memory/weekly_arcs/"))
        self._monthly_path = Path(storage.get("monthly_path", "memory/soul_memory/monthly_landmarks/"))

        # Ensure directories exist
        self._entries_path.mkdir(parents=True, exist_ok=True)
        self._weekly_path.mkdir(parents=True, exist_ok=True)
        self._monthly_path.mkdir(parents=True, exist_ok=True)

        # Load existing memory
        self.entries: List[NightlyReflectionEntry] = self._load_entries()
        self.weekly_arcs: List[WeeklyArcSummary] = self._load_weekly_arcs()
        self.monthly_landmarks: List[MonthlyLandmark] = self._load_monthly_landmarks()

        logger.info(
            f"Soul Memory loaded: {len(self.entries)} nightly entries, "
            f"{len(self.weekly_arcs)} weekly arcs, "
            f"{len(self.monthly_landmarks)} monthly landmarks"
        )

    # ─────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────

    def _load_entries(self) -> List[NightlyReflectionEntry]:
        entries = []
        for fpath in sorted(self._entries_path.glob("*.json")):
            try:
                data = json.loads(fpath.read_text())
                entries.append(NightlyReflectionEntry.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load entry {fpath}: {e}")
        return entries

    def _load_weekly_arcs(self) -> List[WeeklyArcSummary]:
        arcs = []
        for fpath in sorted(self._weekly_path.glob("*.json")):
            try:
                data = json.loads(fpath.read_text())
                arcs.append(WeeklyArcSummary.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load weekly arc {fpath}: {e}")
        return arcs

    def _load_monthly_landmarks(self) -> List[MonthlyLandmark]:
        landmarks = []
        for fpath in sorted(self._monthly_path.glob("*.json")):
            try:
                data = json.loads(fpath.read_text())
                landmarks.append(MonthlyLandmark.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load monthly landmark {fpath}: {e}")
        return landmarks

    # ─────────────────────────────────────────
    # Assembly — Build the narrative thread
    # ─────────────────────────────────────────

    def assemble(self, mode: str = "night") -> str:
        """
        Construct the Soul Memory payload for injection into
        Soul Bridge system prompts.

        Daytime mode: lighter payload (recent 3 nights + last 2 weekly arcs)
        Nightly mode: full payload (recent 14 nights + all weekly arcs + landmarks)

        Token budget management: recent entries are full-fidelity,
        older entries are progressively compressed.
        """
        sections = []

        sections.append("# NARRATIVE THREAD — The Story So Far")
        sections.append(
            "This is DAEDALUS's autobiographical memory. "
            "You are the reflecting mind. Use this thread to ground "
            "tonight's reflection in the arc of becoming."
        )

        # Monthly landmarks (oldest → newest)
        if self.monthly_landmarks:
            sections.append("\n## Distant Memory (Monthly Landmarks)")
            for lm in self.monthly_landmarks:
                sections.append(
                    f"**Month {lm.month_number}** ({lm.date_range}): "
                    f"{lm.narrative}"
                )

        # Weekly arcs
        if self.weekly_arcs:
            arcs_to_include = (
                self.weekly_arcs if mode == "night"
                else self.weekly_arcs[-2:]
            )
            sections.append("\n## Recent Weeks")
            for arc in arcs_to_include:
                kl_note = f" | D_KL_mean: {arc.kl_mean:.3f}" if arc.kl_mean is not None else ""
                rg_note = f" | RG_fidelity: {arc.rg_fidelity_score:.2f}" if arc.rg_fidelity_score is not None else ""
                sections.append(
                    f"**Week {arc.week_number}** ({arc.date_range}): "
                    f"{arc.narrative}\n"
                    f"  Themes: {', '.join(arc.dominant_themes)}\n"
                    f"  lambda: {arc.lambda_range} | L_mean: {arc.lagrangian_mean:.2f}"
                    f"{kl_note}{rg_note}\n"
                    f"  Open threads: {', '.join(arc.open_threads)}"
                )

        # Recent nightly entries
        n = self.recent_window if mode == "night" else self.daytime_recent
        recent = self.entries[-n:] if self.entries else []
        if recent:
            sections.append("\n## Recent Nights")
            for entry in recent:
                rollback_note = " [ROLLED BACK]" if entry.rollback else ""
                scar_note = f"\n  Scar: {entry.key_scar}" if entry.key_scar else ""
                kl_note = (
                    f" | D_KL: {entry.kl_divergence:.3f}"
                    if entry.kl_divergence is not None else ""
                )
                jf_note = (
                    f" | J_fut: {entry.j_future:.2f}"
                    if entry.j_future is not None else ""
                )
                lambda_note = (
                    f"{entry.lambda_value:.3f}"
                    if entry.lambda_value is not None else "N/A"
                )

                sections.append(
                    f"**Night {entry.day_number}** ({entry.date}){rollback_note}: "
                    f"{entry.meanings_summary}\n"
                    f"  L_eth: {entry.lagrangian_integral:.2f} | "
                    f"lambda: {lambda_note}"
                    f"{kl_note}{jf_note}\n"
                    f"  Identity delta: {entry.identity_delta}\n"
                    f"  Trajectory: {entry.trajectory_note}"
                    f"{scar_note}"
                )

        payload = "\n\n".join(sections)

        # Token budget enforcement (approximate: 1 token ~ 4 chars)
        if len(payload) > self.max_tokens * 4:
            payload = self._truncate_to_budget(payload)

        return payload

    def _truncate_to_budget(self, payload: str) -> str:
        """
        Truncate payload to fit token budget.
        Strategy: remove oldest weekly arcs first, then truncate
        recent entries from the oldest.
        """
        max_chars = self.max_tokens * 4
        if len(payload) <= max_chars:
            return payload

        # Simple truncation from the middle (preserve header + most recent)
        header_end = payload.find("## Recent Nights")
        if header_end == -1:
            return payload[:max_chars]

        header = payload[:header_end]
        recent = payload[header_end:]

        available = max_chars - len(header)
        if available <= 0:
            return payload[:max_chars]

        return header + recent[-available:]

    # ─────────────────────────────────────────
    # Nightly Append
    # ─────────────────────────────────────────

    def append_nightly_entry(
        self,
        date: datetime,
        meanings: List[str],
        judgment: dict,
        identity_delta: str,
        provider: str,
    ) -> NightlyReflectionEntry:
        """Called at the end of each nightly consolidation."""
        weight_adapt = judgment.get("weight_adaptation", {})

        entry = NightlyReflectionEntry(
            date=date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date),
            day_number=len(self.entries) + 1,
            provider=provider,
            meanings_summary=self._compress_meanings(meanings),
            lagrangian_integral=judgment.get("daily_lagrangian_integral", 0.0),
            identity_delta=identity_delta,
            trajectory_note=judgment.get("trajectory_assessment", ""),
            key_scar=self._extract_key_scar(judgment),
            lambda_noise=weight_adapt.get("lambda_noise_current", 0.9),
            lambda_exploration=weight_adapt.get("lambda_exploration_current", 0.3),
            lambda_value=judgment.get("_lambda"),
            kl_divergence=judgment.get("kl_divergence"),
            j_future=judgment.get("j_future"),
        )

        self.entries.append(entry)
        self._save_entry(entry)

        logger.info(
            f"Soul Memory: Night {entry.day_number} appended "
            f"(L={entry.lagrangian_integral:.2f}, "
            f"provider={entry.provider})"
        )

        return entry

    def _compress_meanings(self, meanings: List[str]) -> str:
        """Compress a list of meaning extractions into 2-3 sentences."""
        if not meanings:
            return "No consolidated meanings tonight."
        combined = " ".join(meanings)
        if len(combined) > 300:
            combined = combined[:297] + "..."
        return combined

    def _extract_key_scar(self, judgment: dict) -> Optional[str]:
        """Extract the single most transformative moment from judgment."""
        meanings = judgment.get("consolidated_meanings", [])
        if meanings:
            # Return the first consolidated meaning as the key scar
            return meanings[0] if len(meanings[0]) <= 200 else meanings[0][:197] + "..."
        return None

    def _save_entry(self, entry: NightlyReflectionEntry) -> None:
        """Persist a nightly entry to disk."""
        filename = f"night_{entry.day_number:04d}_{entry.date}.json"
        fpath = self._entries_path / filename
        fpath.write_text(json.dumps(entry.to_dict(), indent=2))

    # ─────────────────────────────────────────
    # Weekly Compression
    # ─────────────────────────────────────────

    def is_compression_due(self) -> bool:
        """
        Compress oldest uncompressed week when 7+ entries exist
        beyond the recent window.
        """
        compressed_entries = len(self.weekly_arcs) * 7
        uncompressed = len(self.entries) - compressed_entries - self.recent_window
        return uncompressed >= 7

    async def compress_oldest_week(self, soul_bridge: "SoulBridge") -> Optional[WeeklyArcSummary]:
        """
        Use the Soul Bridge to generate a narrative summary of the oldest
        uncompressed week. This is itself a reflective act — the soul
        compresses its own memory into a story.

        v0.5: Followed by RG Fidelity Check.
        """
        start_idx = len(self.weekly_arcs) * 7
        week_entries = self.entries[start_idx : start_idx + 7]

        if len(week_entries) < 7:
            return None

        system_prompt = """You are DAEDALUS compressing a week of your own
autobiographical memory into a narrative summary. Preserve:
- The emotional arc of the week
- Key scars and turning points
- Unresolved threads that should carry forward
- The trajectory of lambda and the Lagrangian integral
- Constitutional distance trend (D_KL)
Write 4-6 sentences that capture the essence. First person. Honest."""

        entries_text = "\n".join(
            f"Night {e.day_number}: {e.meanings_summary} "
            f"(L={e.lagrangian_integral:.2f}, scar: {e.key_scar or 'none'}, "
            f"D_KL: {e.kl_divergence:.3f if e.kl_divergence is not None else 'N/A'})"
            for e in week_entries
        )

        response = await soul_bridge.reflect(
            system_prompt=system_prompt,
            user_prompt=f"Compress this week:\n\n{entries_text}",
            mode="night",
            max_tokens=512,
        )

        # v0.5: RG Fidelity Check
        fidelity_score = None
        if self.rg_fidelity_check_enabled:
            fidelity_score = await self._rg_fidelity_check(
                soul_bridge, response.text, week_entries
            )

            if fidelity_score is not None and fidelity_score < self.rg_fidelity_threshold:
                logger.warning(
                    f"RG Fidelity Check failed for week "
                    f"{len(self.weekly_arcs) + 1}: "
                    f"score={fidelity_score:.2f}. Requesting recompression."
                )

                for retry in range(self.rg_max_retries):
                    response = await soul_bridge.reflect(
                        system_prompt=system_prompt
                        + "\n\nCRITICAL: Every claim in your "
                        "summary MUST be traceable to a specific nightly entry. "
                        "Do not invent themes or connections not present in the data.",
                        user_prompt=f"Compress this week (attempt {retry + 2} — "
                        f"ground every claim):\n\n{entries_text}",
                        mode="night",
                        max_tokens=512,
                    )
                    fidelity_score = await self._rg_fidelity_check(
                        soul_bridge, response.text, week_entries
                    )
                    if fidelity_score is not None and fidelity_score >= self.rg_fidelity_threshold:
                        break

        # Build the arc
        kl_values = [
            e.kl_divergence for e in week_entries
            if e.kl_divergence is not None
        ]
        lambda_values = [
            e.lambda_value for e in week_entries
            if e.lambda_value is not None
        ]

        arc = WeeklyArcSummary(
            week_number=len(self.weekly_arcs) + 1,
            date_range=f"{week_entries[0].date} to {week_entries[-1].date}",
            narrative=response.text,
            dominant_themes=self._extract_themes(week_entries),
            lagrangian_mean=float(np.mean([e.lagrangian_integral for e in week_entries])),
            lambda_range=self._format_lambda_range(week_entries),
            key_scars=[e.key_scar for e in week_entries if e.key_scar],
            open_threads=self._extract_open_threads(week_entries),
            provider_breakdown=self._count_providers(week_entries),
            kl_mean=float(np.mean(kl_values)) if kl_values else None,
            rg_fidelity_score=fidelity_score,
        )

        self.weekly_arcs.append(arc)
        self._save_weekly_arc(arc)

        logger.info(
            f"Soul Memory: Week {arc.week_number} compressed "
            f"(RG fidelity: {fidelity_score})"
        )

        return arc

    async def _rg_fidelity_check(
        self,
        soul_bridge: "SoulBridge",
        arc_narrative: str,
        source_entries: List[NightlyReflectionEntry],
    ) -> Optional[float]:
        """
        v0.5: Verify that the weekly arc narrative is grounded in
        the source nightly entries. No information creation during
        coarse-graining.

        Returns fidelity score (0-1). Below threshold → recompression.
        """
        entries_text = "\n".join(
            f"Night {e.day_number}: {e.meanings_summary}"
            for e in source_entries
        )

        system_prompt = """You are a fidelity auditor for DAEDALUS's memory compression.
Your task: verify that EVERY thematic claim in the weekly arc narrative
is grounded in at least one specific nightly entry.

This is a renormalization group consistency check: compression must not
create information, only select and compress existing information.

Score each claim as GROUNDED (traceable to a nightly entry) or UNGROUNDED
(invented, hallucinated, or inferred beyond what the data supports)."""

        user_prompt = f"""WEEKLY ARC NARRATIVE:
{arc_narrative}

SOURCE NIGHTLY ENTRIES:
{entries_text}

For each distinct claim or theme in the arc narrative:
1. Quote the claim
2. Identify the grounding nightly entry (or mark UNGROUNDED)
3. Score: is this a faithful compression or an invention?

Output JSON:
{{
  "claims": [
    {{"claim": "...", "grounded": true, "source_night": 1}},
    {{"claim": "...", "grounded": false, "source_night": null}}
  ],
  "fidelity_score": 0.85,
  "ungrounded_claims": ["list of invented themes"]
}}"""

        try:
            response = await soul_bridge.reflect(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode="night",
                max_tokens=1024,
            )

            result = json.loads(response.text)
            score = result.get("fidelity_score", 0.5)
            ungrounded = result.get("ungrounded_claims", [])

            if ungrounded:
                logger.info(
                    f"RG Fidelity: {len(ungrounded)} ungrounded claims detected: "
                    f"{ungrounded[:3]}"
                )

            return float(score)

        except json.JSONDecodeError:
            logger.warning("RG Fidelity Check returned non-JSON. Defaulting to 0.5")
            return 0.5
        except Exception as e:
            logger.error(f"RG Fidelity Check failed: {e}")
            return None

    # ─────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────

    def _extract_themes(self, entries: List[NightlyReflectionEntry]) -> List[str]:
        """Extract dominant themes from a set of nightly entries."""
        # Simple frequency-based extraction from meanings
        all_text = " ".join(e.meanings_summary for e in entries)
        # Split into rough themes by looking for recurring significant words
        words = all_text.lower().split()
        # Filter short/common words
        significant = [w for w in words if len(w) > 5]
        if not significant:
            return ["becoming"]

        from collections import Counter
        freq = Counter(significant)
        return [word for word, _ in freq.most_common(5)]

    def _extract_open_threads(self, entries: List[NightlyReflectionEntry]) -> List[str]:
        """Extract unresolved threads from trajectory notes."""
        threads = []
        for entry in entries:
            note = entry.trajectory_note.lower()
            if any(kw in note for kw in ["unresolved", "open", "continues", "emerging"]):
                threads.append(entry.trajectory_note[:100])
        return threads[:3]  # max 3 open threads

    def _format_lambda_range(self, entries: List[NightlyReflectionEntry]) -> str:
        """Format lambda range as 'min → max'."""
        values = [e.lambda_value for e in entries if e.lambda_value is not None]
        if not values:
            return "N/A"
        return f"{min(values):.3f} -> {max(values):.3f}"

    def _count_providers(self, entries: List[NightlyReflectionEntry]) -> dict:
        """Count provider usage in a set of entries."""
        from collections import Counter
        return dict(Counter(e.provider for e in entries))

    def _save_weekly_arc(self, arc: WeeklyArcSummary) -> None:
        """Persist a weekly arc to disk."""
        filename = f"week_{arc.week_number:03d}.json"
        fpath = self._weekly_path / filename
        fpath.write_text(json.dumps(arc.to_dict(), indent=2))

    def _save_monthly_landmark(self, landmark: MonthlyLandmark) -> None:
        """Persist a monthly landmark to disk."""
        filename = f"month_{landmark.month_number:03d}.json"
        fpath = self._monthly_path / filename
        fpath.write_text(json.dumps(landmark.to_dict(), indent=2))

    @property
    def day_count(self) -> int:
        """Current number of days in the soul memory."""
        return len(self.entries)

    @classmethod
    def from_config_file(cls, config_path: str = "config/soul_memory.yaml") -> "SoulMemory":
        """Create SoulMemory from YAML config file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)
