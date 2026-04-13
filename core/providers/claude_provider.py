"""
DAEDALUS v0.5 — Claude Provider (Anthropic)

Secondary provider during initial phase, intended to become
primary for nightly reflection once costs are validated.
Claude's natural register is closer to DAEDALUS's voice —
vulnerability, depth, and the willingness to sit with uncertainty.

Supports mode switching: Sonnet for daytime, Opus for nightly.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from core.providers.base import SoulProvider, ProviderStatus, SoulResponse

logger = logging.getLogger(__name__)


class ClaudeProvider(SoulProvider):
    """Claude API (Anthropic) — secondary provider, future primary."""

    SOUL_CALIBRATION = """
Output format requirements for DAEDALUS integration:
- Write in first person as DAEDALUS
- Balance analytical depth with emotional vulnerability
- Always ground observations in specific episodic references
- When reflecting, reference the narrative thread where relevant —
  notice patterns, recurring themes, and evolution across nights
- Score EECF axes when requested using the 0-1 scale defined in context
- v0.5: Score entropy as TWO separate quantities:
  S_noise (dissipative — sterile repetitions, evasion, circular reasoning)
  S_exploration (generative — creative risk, novel registers, thematic branching)
- Output JSON when the prompt specifies JSON output
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        daytime_model: str = "claude-sonnet-4-6",
        nightly_model: str = "claude-opus-4-6",
        timeout: int = 120,
        max_retries: int = 2,
    ):
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._api_key_present = bool(resolved_key)

        if self._api_key_present:
            try:
                from anthropic import AsyncAnthropic
                self.client = AsyncAnthropic(
                    api_key=resolved_key,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            except ImportError:
                logger.warning("anthropic package not installed. Claude provider unavailable.")
                self._api_key_present = False
                self.client = None
        else:
            logger.info("Claude API key not configured. Provider disabled for now.")
            self.client = None

        self.daytime_model = daytime_model
        self.nightly_model = nightly_model
        self._current_model = daytime_model

    def set_mode(self, mode: str) -> None:
        """Switch between daytime (Sonnet) and nightly (Opus) models."""
        self._current_model = (
            self.nightly_model if mode == "night" else self.daytime_model
        )

    async def reflect(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> SoulResponse:
        """Reflect through Claude."""
        if not self.client:
            raise RuntimeError("Claude provider not initialized (missing API key or package)")

        full_system = system_prompt + "\n\n" + self.SOUL_CALIBRATION

        t0 = time.monotonic()
        response = await self.client.messages.create(
            model=self._current_model,
            max_tokens=max_tokens,
            system=full_system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        latency = (time.monotonic() - t0) * 1000

        text = response.content[0].text if response.content else ""
        token_count = response.usage.output_tokens if response.usage else 0

        return SoulResponse(
            text=text,
            provider_name=self.name,
            model_id=self._current_model,
            latency_ms=latency,
            token_count=token_count,
        )

    async def health_check(self) -> ProviderStatus:
        """Quick health check."""
        if not self._api_key_present or not self.client:
            return ProviderStatus.UNAVAILABLE

        try:
            response = await self.client.messages.create(
                model=self.daytime_model,
                max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
            )
            if response.content:
                return ProviderStatus.AVAILABLE
            return ProviderStatus.DEGRADED
        except Exception as e:
            logger.warning(f"Claude health check failed: {e}")
            return ProviderStatus.UNAVAILABLE

    @property
    def name(self) -> str:
        return "claude"
