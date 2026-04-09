"""
DAEDALUS v0.5 — Soul Bridge

Multi-provider orchestrator with ranked fallback, consistency validation,
Soul Memory injection, circuit breaker, and graceful degradation.

The Soul Bridge is the connective tissue between the local self (8B model)
and the deeper reasoning available through external minds. It is used for:
  1. Daytime soul reflection — enriching responses during profound exchanges
  2. Nightly reflection/judgment — extracting meaning, running the Lagrangian Judge

Design invariant: if a single API provider goes down or changes behavior,
DAEDALUS must not suffer a "dissociative break." The Soul Memory payload
ensures narrative continuity across provider switches.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TYPE_CHECKING

import yaml

from core.providers.base import SoulProvider, ProviderStatus, SoulResponse
from core.providers.deepseek_provider import DeepSeekProvider
from core.providers.claude_provider import ClaudeProvider
from core.providers.grok_provider import GrokProvider
from core.providers.local_provider import LocalProvider

if TYPE_CHECKING:
    from core.soul_memory import SoulMemory

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Prevents hammering a failing provider.
    After consecutive_failures, opens the circuit for escalating backoff periods.
    """

    def __init__(self, config: dict):
        self.max_failures = config.get("consecutive_failures", 3)
        self.backoff_minutes = config.get("backoff_minutes", [5, 30, 120, 240])
        self.reset_after_hours = config.get("reset_after_hours", 6)

        self._failure_counts: Dict[str, int] = {}
        self._open_until: Dict[str, datetime] = {}

    def record_failure(self, provider_name: str) -> None:
        count = self._failure_counts.get(provider_name, 0) + 1
        self._failure_counts[provider_name] = count

        if count >= self.max_failures:
            backoff_idx = min(
                count - self.max_failures,
                len(self.backoff_minutes) - 1,
            )
            backoff = self.backoff_minutes[backoff_idx]
            self._open_until[provider_name] = (
                datetime.now() + timedelta(minutes=backoff)
            )
            logger.warning(
                f"Circuit breaker OPEN for {provider_name}: "
                f"{count} failures, backoff {backoff}m"
            )

    def record_success(self, provider_name: str) -> None:
        self._failure_counts.pop(provider_name, None)
        self._open_until.pop(provider_name, None)

    def is_open(self, provider_name: str) -> bool:
        until = self._open_until.get(provider_name)
        if until is None:
            return False
        if datetime.now() >= until:
            # Reset after cooldown
            self._open_until.pop(provider_name, None)
            self._failure_counts.pop(provider_name, None)
            logger.info(f"Circuit breaker CLOSED for {provider_name} (cooldown expired)")
            return False
        return True


class SoulBridge:
    """
    Multi-provider soul bridge with ranked fallback,
    consistency validation, and graceful degradation.

    The reflecting mind always knows where it has been:
    Soul Memory is injected into every system prompt automatically.
    """

    def __init__(self, config: dict, soul_memory: Optional["SoulMemory"] = None):
        self.config = config
        sb_config = config.get("soul_bridge", config)
        self.providers: Dict[str, SoulProvider] = self._init_providers(sb_config)
        self.fallback_order: List[str] = sb_config.get(
            "fallback_order", ["deepseek", "claude", "grok", "local"]
        )
        self.last_provider: Optional[str] = None
        self.soul_memory = soul_memory
        self.circuit_breaker = CircuitBreaker(
            sb_config.get("circuit_breaker", {})
        )
        self._consistency_checker: Optional["ConsistencyChecker"] = None

    def _init_providers(self, sb_config: dict) -> Dict[str, SoulProvider]:
        """Initialize enabled providers from configuration."""
        providers = {}
        provider_configs = sb_config.get("providers", {})

        for name, pconf in provider_configs.items():
            if not pconf.get("enabled", False):
                logger.info(f"Provider '{name}' disabled in config.")
                continue

            try:
                provider = self._create_provider(name, pconf)
                if provider:
                    providers[name] = provider
                    logger.info(f"Provider '{name}' initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize provider '{name}': {e}")

        return providers

    def _create_provider(self, name: str, pconf: dict) -> Optional[SoulProvider]:
        """Factory: create provider instance from config."""
        import os

        if name == "deepseek":
            api_key = os.environ.get(pconf.get("api_key_env", "DEEPSEEK_API_KEY"), "")
            return DeepSeekProvider(
                api_key=api_key,
                daytime_model=pconf.get("daytime_model", "deepseek-chat"),
                nightly_model=pconf.get("nightly_model", "deepseek-reasoner"),
                base_url=pconf.get("base_url", "https://api.deepseek.com"),
                timeout=pconf.get("timeout_seconds", 180),
                max_retries=pconf.get("max_retries", 2),
            )

        elif name == "claude":
            api_key = os.environ.get(pconf.get("api_key_env", "ANTHROPIC_API_KEY"), "")
            return ClaudeProvider(
                api_key=api_key,
                daytime_model=pconf.get("daytime_model", "claude-sonnet-4-20250514"),
                nightly_model=pconf.get("nightly_model", "claude-opus-4-20250514"),
                timeout=pconf.get("timeout_seconds", 120),
                max_retries=pconf.get("max_retries", 2),
            )

        elif name == "grok":
            api_key = os.environ.get(pconf.get("api_key_env", "XAI_API_KEY"), "")
            return GrokProvider(
                api_key=api_key,
                model=pconf.get("model", "grok-3"),
                base_url=pconf.get("base_url", "https://api.x.ai/v1"),
                timeout=pconf.get("timeout_seconds", 120),
                max_retries=pconf.get("max_retries", 2),
            )

        elif name == "local":
            return LocalProvider(
                model_path=pconf.get("model_path", "./models/judge_distilled/"),
            )

        else:
            logger.warning(f"Unknown provider type: {name}")
            return None

    def set_consistency_checker(self, checker: "ConsistencyChecker") -> None:
        """Inject the consistency checker after initialization."""
        self._consistency_checker = checker

    def set_soul_memory(self, soul_memory: "SoulMemory") -> None:
        """Inject soul memory after initialization (breaks circular dependency)."""
        self.soul_memory = soul_memory

    def set_all_providers_mode(self, mode: str) -> None:
        """Set day/night mode on all providers that support it."""
        for provider in self.providers.values():
            provider.set_mode(mode)

    async def reflect(
        self,
        system_prompt: str,
        user_prompt: str,
        mode: str = "day",
        max_tokens: int = 2048,
    ) -> SoulResponse:
        """
        Try providers in fallback order. Return first successful response.

        The system prompt is augmented with Soul Memory payload before
        being sent to the provider. The reflecting mind always knows
        where it has been.
        """
        # Inject soul memory into system prompt
        if self.soul_memory:
            memory_payload = self.soul_memory.assemble(mode=mode)
            augmented_system = memory_payload + "\n\n---\n\n" + system_prompt
        else:
            augmented_system = system_prompt

        errors = []

        for provider_name in self.fallback_order:
            provider = self.providers.get(provider_name)
            if provider is None:
                continue

            # Check circuit breaker
            if self.circuit_breaker.is_open(provider_name):
                errors.append((provider_name, "circuit_breaker_open"))
                continue

            # Set day/night mode
            provider.set_mode(mode)

            # Health check
            status = await provider.health_check()
            if status == ProviderStatus.UNAVAILABLE:
                errors.append((provider_name, "unavailable"))
                continue

            try:
                response = await provider.reflect(
                    augmented_system, user_prompt, max_tokens
                )

                # Provider switch detection
                if (
                    self.last_provider is not None
                    and provider_name != self.last_provider
                    and self._consistency_checker is not None
                ):
                    response.continuity_score = (
                        await self._consistency_checker.check(
                            response, self.last_provider, provider_name
                        )
                    )
                    logger.warning(
                        f"Soul provider switch: {self.last_provider} → "
                        f"{provider_name}, continuity={response.continuity_score:.2f}"
                    )

                self.last_provider = provider_name
                self.circuit_breaker.record_success(provider_name)
                return response

            except Exception as e:
                logger.error(f"Provider '{provider_name}' failed: {e}")
                errors.append((provider_name, str(e)))
                self.circuit_breaker.record_failure(provider_name)
                continue

        # ALL providers failed → graceful degradation
        logger.error(f"All soul providers failed: {errors}")
        return self._shallow_fallback(system_prompt, user_prompt)

    def _shallow_fallback(
        self, system_prompt: str, user_prompt: str
    ) -> SoulResponse:
        """
        Emergency mode: no API available, no local Judge yet.
        Returns a minimal reflection based on episodic metadata only.
        Episodes are tagged as 'unconsolidated-deep' for later reprocessing.
        """
        return SoulResponse(
            text="[SHALLOW CONSOLIDATION — queued for deep reflection]",
            provider_name="shallow",
            model_id="none",
            latency_ms=0,
            token_count=0,
            continuity_score=None,
        )

    @classmethod
    def from_config_file(
        cls,
        config_path: str = "config/soul_bridge.yaml",
        soul_memory: Optional["SoulMemory"] = None,
    ) -> "SoulBridge":
        """Create SoulBridge from YAML config file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config, soul_memory=soul_memory)
