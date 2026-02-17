"""Standard OpenAI API provider (API key-based).

Fallback provider for when users have an API key but not a ChatGPT subscription,
or for development/testing with cached responses.
"""

from __future__ import annotations

import json
from typing import Optional

from providers.base import LLMProvider, LLMResponse

from config import OPENAI_API_KEY


class OpenAIAPIProvider(LLMProvider):
    """OpenAI Platform API provider (standard pay-per-token)."""

    def __init__(self, api_key: str = "", model: str = "gpt-4o"):
        self._api_key = api_key or OPENAI_API_KEY
        self._model = model

    @property
    def name(self) -> str:
        return "openai_api"

    @property
    def is_authenticated(self) -> bool:
        return bool(self._api_key)

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
        """Generate using OpenAI API."""
        import openai

        client = openai.AsyncOpenAI(api_key=self._api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user message
        if images:
            content = []
            for img_b64 in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""

        json_data = None
        if json_mode:
            try:
                json_data = json.loads(text)
            except json.JSONDecodeError:
                pass

        return LLMResponse(
            text=text,
            json_data=json_data,
            model=self._model,
            provider=self.name,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )
