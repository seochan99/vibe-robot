"""Provider setup and initialization.

Configures the global LLM provider registry based on config.py settings
and available credentials. Auto-detection tries providers in order:
  1. ChatGPT OAuth (if ~/.viberobot/auth.json exists)
  2. Codex CLI (if `codex` binary is installed and logged in)
  3. OpenAI API (if OPENAI_API_KEY is set)

Usage:
    from providers.setup import get_provider

    provider = get_provider()
    response = provider.generate_sync("Analyze this scene...")
"""

from __future__ import annotations

from typing import Optional

from providers.base import LLMProvider, ProviderRegistry
from config import get_config


_registry: Optional[ProviderRegistry] = None


def init_providers(force_provider: str = None) -> ProviderRegistry:
    """Initialize and return the provider registry.

    Args:
        force_provider: Force a specific provider ("chatgpt_oauth", "openai_api", "codex_cli").
                       If None, uses config or auto-detection.
    """
    global _registry
    _registry = ProviderRegistry()

    # Register ChatGPT OAuth provider
    try:
        from providers.chatgpt_auth import ChatGPTAuth
        from providers.chatgpt_provider import ChatGPTOAuthProvider

        auth = ChatGPTAuth()
        chatgpt_provider = ChatGPTOAuthProvider(auth=auth)
        _registry.register(chatgpt_provider)
    except Exception:
        pass

    # Register Codex CLI provider
    try:
        from providers.codex_cli_provider import CodexCLIProvider

        codex_provider = CodexCLIProvider()
        _registry.register(codex_provider)
    except Exception:
        pass

    # Register OpenAI API provider
    try:
        from providers.openai_provider import OpenAIAPIProvider

        api_provider = OpenAIAPIProvider()
        _registry.register(api_provider)
    except Exception:
        pass

    # Set default
    if force_provider:
        _registry.set_default(force_provider)
    else:
        cfg = get_config()
        provider_pref = cfg.get("llm_provider", "auto")

        if provider_pref != "auto":
            try:
                _registry.set_default(provider_pref)
            except KeyError:
                pass
        else:
            # Auto-detect: use first authenticated provider
            auto = _registry.get_first_authenticated()
            if auto:
                _registry.set_default(auto.name)

    return _registry


def get_registry() -> ProviderRegistry:
    """Get the initialized provider registry."""
    global _registry
    if _registry is None:
        _registry = init_providers()
    return _registry


def get_provider(name: str = None) -> LLMProvider:
    """Get a specific provider or the default.

    This is the main entry point for all AI modules.
    """
    registry = get_registry()
    if name:
        return registry.get(name)
    return registry.get_default()
