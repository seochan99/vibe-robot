"""LLM/VLM provider abstraction layer.

Supports multiple backends:
- chatgpt_oauth: ChatGPT Plus/Pro subscription via OAuth (no API key needed)
- openai_api: Standard OpenAI API (requires API key)
- anthropic_api: Anthropic Claude API (requires API key)
"""

from providers.base import LLMProvider, LLMResponse, ProviderRegistry

__all__ = ["LLMProvider", "LLMResponse", "ProviderRegistry"]
