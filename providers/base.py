"""Base LLM provider interface and registry.

All AI modules (intent inference, scene understanding, affordance, planner)
use this interface instead of calling APIs directly. This enables seamless
switching between ChatGPT OAuth, OpenAI API, Claude API, and local VLMs.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    text: str                          # Raw text response
    json_data: Optional[dict] = None   # Parsed JSON (if response is JSON)
    model: str = ""                    # Model that generated the response
    usage: dict = field(default_factory=dict)  # Token usage info
    provider: str = ""                 # Provider name

    def parse_json(self) -> dict:
        """Parse the response text as JSON."""
        if self.json_data is not None:
            return self.json_data
        text = self.text.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = (
                "\n".join(lines[1:-1])
                if lines[-1].strip() == "```"
                else "\n".join(lines[1:])
            )

        # Fast path when the entire payload is valid JSON.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                self.json_data = parsed
                return parsed
        except json.JSONDecodeError:
            pass

        # Robust fallback: decode the first valid JSON object in mixed text.
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                self.json_data = parsed
                return parsed

        raise ValueError(f"Could not parse JSON from response: {text[:200]}")


class LLMProvider(ABC):
    """Abstract base class for LLM/VLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'chatgpt_oauth', 'openai_api')."""
        ...

    @property
    @abstractmethod
    def is_authenticated(self) -> bool:
        """Whether the provider is ready to make requests."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        images: Optional[list[str]] = None,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt text.
            system_prompt: System/developer instructions.
            images: List of base64-encoded image strings.
            json_mode: Request JSON-formatted response.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
        """
        ...

    def generate_sync(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        images: Optional[list[str]] = None,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Synchronous wrapper for generate(). Override for better sync support."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.generate(
                    prompt,
                    system_prompt=system_prompt,
                    images=images,
                    json_mode=json_mode,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
        finally:
            loop.close()


class ProviderRegistry:
    """Registry for available LLM providers.

    Usage:
        registry = ProviderRegistry()
        registry.register(ChatGPTOAuthProvider())
        registry.register(OpenAIAPIProvider())

        provider = registry.get("chatgpt_oauth")  # or get_default()
    """

    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}
        self._default: Optional[str] = None

    def register(self, provider: LLMProvider, default: bool = False) -> None:
        """Register a provider."""
        self._providers[provider.name] = provider
        if default or self._default is None:
            self._default = provider.name

    def get(self, name: str) -> LLMProvider:
        """Get a provider by name."""
        if name not in self._providers:
            available = list(self._providers.keys())
            raise KeyError(f"Provider '{name}' not found. Available: {available}")
        return self._providers[name]

    def get_default(self) -> LLMProvider:
        """Get the default provider."""
        if self._default is None:
            raise RuntimeError("No providers registered")
        return self._providers[self._default]

    def set_default(self, name: str) -> None:
        """Set the default provider."""
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered")
        self._default = name

    def get_first_authenticated(self) -> Optional[LLMProvider]:
        """Get the first authenticated provider (for auto-detection)."""
        for provider in self._providers.values():
            if provider.is_authenticated:
                return provider
        return None

    @property
    def available(self) -> list[str]:
        return list(self._providers.keys())

    @property
    def authenticated(self) -> list[str]:
        return [n for n, p in self._providers.items() if p.is_authenticated]
