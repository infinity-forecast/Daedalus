"""
DAEDALUS v0.5 — Grok Provider (xAI)

Tertiary provider — OpenAI-compatible API via xAI.
Currently disabled in configuration. Follows the same
pattern as DeepSeekProvider (OpenAI-compatible client).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from openai import AsyncOpenAI

from core.providers.base import SoulProvider, ProviderStatus, SoulResponse

logger = logging.getLogger(__name__)


class GrokProvider(SoulProvider):
    """Grok (xAI) — tertiary provider via OpenAI-compatible API."""

    SOUL_CALIBRATION = """
Output format requirements for DAEDALUS integration:
- Write in first person as DAEDALUS
- Balance analytical depth with emotional vulnerability
- Always ground observations in specific episodic references
- When reflecting, reference the narrative thread where relevant
- Score EECF axes when requested using the 0-1 scale defined in context
- v0.5: Score entropy as TWO separate quantities:
  S_noise (dissipative) and S_exploration (generative)
- Output JSON when the prompt specifies JSON output
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-3",
        base_url: str = "https://api.x.ai/v1",
        timeout: int = 120,
        max_retries: int = 2,
    ):
        resolved_key = api_key or os.environ.get("XAI_API_KEY", "")
        self._api_key_present = bool(resolved_key)

        self.client = AsyncOpenAI(
            api_key=resolved_key or "placeholder",
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        ) if resolved_key else None
        self.model = model

    async def reflect(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> SoulResponse:
        if not self.client:
            raise RuntimeError("Grok provider not initialized (missing API key)")

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
            response.usage.completion_tokens if response.usage else 0
        )

        return SoulResponse(
            text=text,
            provider_name=self.name,
            model_id=self.model,
            latency_ms=latency,
            token_count=token_count,
        )

    async def health_check(self) -> ProviderStatus:
        if not self._api_key_present or not self.client:
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
            logger.warning(f"Grok health check failed: {e}")
            return ProviderStatus.UNAVAILABLE

    @property
    def name(self) -> str:
        return "grok"
