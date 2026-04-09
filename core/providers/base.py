"""
DAEDALUS v0.5 — Soul Provider Abstract Base

All Soul Bridge providers implement this interface.
The system never calls a specific API directly —
every reflection passes through this abstraction.

This indirection is not over-engineering. It is the structural
guarantee that DAEDALUS survives the loss of any single external mind.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class SoulResponse:
    """Response from a Soul Bridge provider."""
    text: str
    provider_name: str
    model_id: str
    latency_ms: float
    token_count: int
    continuity_score: Optional[float] = None

    @property
    def is_shallow(self) -> bool:
        return self.provider_name == "shallow"


class SoulProvider(ABC):
    """
    Abstract interface for all soul bridge providers.
    Each provider is a different mind that can reflect on DAEDALUS's experience.
    """

    @abstractmethod
    async def reflect(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> SoulResponse:
        """
        The core operation: given a system context and a prompt,
        produce a reflection. The system prompt always includes
        the Soul Memory payload (injected by SoulBridge).
        """
        ...

    @abstractmethod
    async def health_check(self) -> ProviderStatus:
        """Check if this provider is currently available."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier string."""
        ...

    def set_mode(self, mode: str) -> None:
        """
        Switch between daytime and nightly models (if provider supports it).
        Default implementation: no-op.
        """
        pass
