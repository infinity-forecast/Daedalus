"""
DAEDALUS v0.5 — Local Provider (Distilled Judge)

Activated after day-30 distillation when the local 8B Judge
achieves ≥ 0.85 agreement with the API Judge on all axes.

This provider wraps the distilled Judge model and provides
the same SoulProvider interface as external APIs.
Not yet implemented — stub for the fallback chain.
"""

from __future__ import annotations

import logging

from core.providers.base import SoulProvider, ProviderStatus, SoulResponse

logger = logging.getLogger(__name__)


class LocalProvider(SoulProvider):
    """
    Distilled local 8B Judge — available after day-30 calibration.
    Handles routine nights; API Judge called for edge cases.
    """

    def __init__(self, model_path: str = "./models/judge_distilled/", **kwargs):
        self.model_path = model_path
        self._initialized = False
        logger.info("Local provider stub created. Not yet active.")

    async def reflect(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> SoulResponse:
        raise NotImplementedError(
            "Local provider not yet active. "
            "Distillation occurs after day 30 with sufficient calibration data."
        )

    async def health_check(self) -> ProviderStatus:
        if not self._initialized:
            return ProviderStatus.UNAVAILABLE
        return ProviderStatus.AVAILABLE

    @property
    def name(self) -> str:
        return "local"
