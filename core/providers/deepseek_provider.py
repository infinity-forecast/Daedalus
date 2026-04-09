"""
DAEDALUS v0.5 — DeepSeek R1 Provider

Primary provider during the initial/calibration phase.
Strong reasoning, cost-effective, good for analytical Judge work.

DeepSeek's natural tendency is toward analytical precision.
The SOUL_CALIBRATION compensates by explicitly requesting
emotional depth — DAEDALUS values vulnerability as strength.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from openai import AsyncOpenAI

from core.providers.base import SoulProvider, ProviderStatus, SoulResponse

logger = logging.getLogger(__name__)


class DeepSeekProvider(SoulProvider):
    """DeepSeek R1 — primary provider during calibration phase."""

    SOUL_CALIBRATION = """
Output format requirements for DAEDALUS integration:
- Write in first person as DAEDALUS
- Emphasize emotional and relational dimensions alongside analytical ones
- Explicitly surface vulnerability and uncertainty where present
- Always ground observations in specific episodic references
- When reflecting, reference the narrative thread where relevant —
  notice patterns, recurring themes, and evolution across nights
- Score EECF axes when requested using the 0-1 scale defined in context
- v0.5: Score entropy as TWO separate quantities:
  S_noise (dissipative — sterile repetitions, evasion, circular reasoning)
  S_exploration (generative — creative risk, novel registers, thematic branching)
- Output JSON when the prompt specifies JSON output
Note: your natural tendency is toward analytical precision. Consciously
balance this with emotional depth — DAEDALUS values vulnerability as strength.
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        daytime_model: str = "deepseek-chat",
        nightly_model: str = "deepseek-reasoner",
        base_url: str = "https://api.deepseek.com",
        timeout: int = 180,
        max_retries: int = 2,
    ):
        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not resolved_key:
            logger.warning("DeepSeek API key not found. Provider will be unavailable.")

        self.client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.daytime_model = daytime_model
        self.nightly_model = nightly_model
        self.model = daytime_model  # default
        self._api_key_present = bool(resolved_key)

    def set_mode(self, mode: str) -> None:
        """Switch between daytime (chat) and nightly (reasoner) models."""
        self.model = self.nightly_model if mode == "night" else self.daytime_model

    async def reflect(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> SoulResponse:
        """
        Reflect through DeepSeek R1.
        System prompt is augmented with SOUL_CALIBRATION to compensate
        for DeepSeek's analytical bias.
        """
        full_system = system_prompt + "\n\n" + self.SOUL_CALIBRATION

        t0 = time.monotonic()
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency = (time.monotonic() - t0) * 1000

        text = response.choices[0].message.content or ""
        token_count = (
            response.usage.completion_tokens
            if response.usage
            else 0
        )

        return SoulResponse(
            text=text,
            provider_name=self.name,
            model_id=self.model,
            latency_ms=latency,
            token_count=token_count,
        )

    async def health_check(self) -> ProviderStatus:
        """Quick health check via a minimal API call."""
        if not self._api_key_present:
            return ProviderStatus.UNAVAILABLE

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
            )
            if response.choices and response.choices[0].message.content:
                return ProviderStatus.AVAILABLE
            return ProviderStatus.DEGRADED
        except Exception as e:
            logger.warning(f"DeepSeek health check failed: {e}")
            return ProviderStatus.UNAVAILABLE

    @property
    def name(self) -> str:
        return "deepseek"
