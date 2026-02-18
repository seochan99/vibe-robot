"""ChatGPT OAuth provider — calls chatgpt.com/backend-api/codex/responses.

Uses the same backend API as OpenAI's Codex CLI. Requires OAuth tokens from
ChatGPTAuth (ChatGPT Plus/Pro subscription, no API key needed).

This is the mechanism used by OpenCode, OpenClaw, and similar tools.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

import requests

from providers.base import LLMProvider, LLMResponse
from providers.chatgpt_auth import ChatGPTAuth

logger = logging.getLogger(__name__)


# ChatGPT backend API endpoint (same as Codex CLI uses)
CODEX_API_URL = "https://chatgpt.com/backend-api/codex/responses"

# Available models through ChatGPT subscription
CHATGPT_MODELS = {
    "gpt-5.3-codex": "Most capable (recommended)",
    "gpt-5.2-codex": "Advanced coding model",
    "gpt-5.1-codex": "Long-running tasks",
    "gpt-5-codex": "Original GPT-5 coding variant",
    "gpt-5-codex-mini": "Cost-effective",
    "codex-mini-latest": "Lightweight default",
}

DEFAULT_MODEL = "gpt-5.3-codex"

# System instructions required by the Codex backend
DEFAULT_INSTRUCTIONS = (
    "You are a helpful AI assistant integrated into VibeRobot, a system for safe "
    "robotic manipulation. You help with intent inference, scene analysis, affordance "
    "reasoning, and action planning. Always respond in the requested format."
)


class ChatGPTOAuthProvider(LLMProvider):
    """LLM provider using ChatGPT Plus/Pro subscription via OAuth.

    Usage:
        # From backend auth (stored tokens):
        auth = ChatGPTAuth()
        auth.login()
        provider = ChatGPTOAuthProvider(auth)

        # From frontend raw token (web app flow):
        provider = ChatGPTOAuthProvider(
            access_token="eyJ...",
            account_id="acc_...",
        )

        response = await provider.generate("Analyze this scene...")
    """

    def __init__(
        self,
        auth: Optional[ChatGPTAuth] = None,
        model: str = DEFAULT_MODEL,
        instructions: str = DEFAULT_INSTRUCTIONS,
        access_token: Optional[str] = None,
        account_id: Optional[str] = None,
    ):
        self._auth = auth if not access_token else None
        self._raw_token = access_token
        self._raw_account_id = account_id or ""
        if not access_token and auth is None:
            self._auth = ChatGPTAuth()
        self._model = model
        self._instructions = instructions
        self._session_id = str(uuid.uuid4())

    @property
    def name(self) -> str:
        return "chatgpt_oauth"

    @property
    def is_authenticated(self) -> bool:
        if self._raw_token:
            return True
        return self._auth.is_authenticated if self._auth else False

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

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
        """Generate using ChatGPT backend API (async wrapper around sync)."""
        return self.generate_sync(
            prompt,
            system_prompt=system_prompt,
            images=images,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )

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
        """Generate using ChatGPT backend API (synchronous).

        Calls chatgpt.com/backend-api/codex/responses with OAuth JWT token.
        Parses SSE streaming response.
        """
        if self._raw_token:
            access_token = self._raw_token
            account_id = self._raw_account_id
        else:
            access_token = self._auth.get_access_token()
            account_id = self._auth.get_account_id()

        # Build request headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
            "Accept": "text/event-stream",
            "OpenAI-Beta": "responses=experimental",
            "session_id": self._session_id,
            "originator": "viberobot",
            "chatgpt-account-id": account_id,
        }

        # Build input messages
        input_messages = []
        if system_prompt:
            input_messages.append(
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": system_prompt}],
                }
            )

        # Build user message content
        user_content = []
        if images:
            for img_b64 in images:
                user_content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{img_b64}",
                    }
                )
        user_content.append({"type": "input_text", "text": prompt})

        input_messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        # Build request body
        instructions = self._instructions
        if json_mode:
            instructions += "\n\nIMPORTANT: Respond in valid JSON format only."

        body = {
            "model": self._model,
            "instructions": instructions,
            "input": input_messages,
            "tools": [],
            "tool_choice": "auto",
            "reasoning": {"summary": "auto"},
            "stream": True,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": str(uuid.uuid4()),
        }

        # Make the API call
        logger.info("ChatGPT OAuth request start model=%s", self._model)
        resp = requests.post(
            CODEX_API_URL,
            headers=headers,
            json=body,
            stream=True,
            timeout=120,
        )
        logger.info("ChatGPT OAuth response status=%s", resp.status_code)

        if resp.status_code == 401 and self._auth:
            # Token might be expired, try refreshing
            logger.warning("ChatGPT OAuth got 401; attempting token refresh")
            self._auth._refresh_tokens()
            headers["Authorization"] = f"Bearer {self._auth.get_access_token()}"
            resp = requests.post(
                CODEX_API_URL,
                headers=headers,
                json=body,
                stream=True,
                timeout=120,
            )
            logger.info("ChatGPT OAuth retry status=%s", resp.status_code)

        if resp.status_code != 200:
            logger.error(
                "ChatGPT API error status=%s body=%s",
                resp.status_code,
                resp.text[:500],
            )
            raise RuntimeError(
                f"ChatGPT API error ({resp.status_code}): {resp.text[:500]}"
            )

        # Parse SSE response
        text = self._parse_sse_response(resp)

        json_data = None
        if json_mode:
            try:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_data = json.loads(text[start:end])
            except (json.JSONDecodeError, ValueError):
                pass

        return LLMResponse(
            text=text,
            json_data=json_data,
            model=self._model,
            provider=self.name,
        )

    def _parse_sse_response(self, resp: requests.Response) -> str:
        """Parse Server-Sent Events stream and extract text content."""
        text_parts = []
        event_count = 0

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue

            # requests can still yield bytes when encoding is unknown.
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")

            # SSE format: "event: <type>\ndata: <json>"
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    event_data = json.loads(data_str)
                    event_count += 1
                    text = self._extract_text_from_event(event_data)
                    if text:
                        text_parts.append(text)
                except json.JSONDecodeError:
                    continue

        logger.info(
            "ChatGPT OAuth SSE parsed events=%s chars=%s",
            event_count,
            len("".join(text_parts)),
        )
        return "".join(text_parts)

    def _extract_text_from_event(self, event: dict) -> str:
        """Extract text content from an SSE event."""
        event_type = event.get("type", "")

        # Handle response.output_text.delta
        if event_type == "response.output_text.delta":
            return event.get("delta", "")

        # Handle response.output_item.added with text content
        if event_type == "response.output_item.added":
            item = event.get("item", {})
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        return content.get("text", "")

        # Handle response.completed — extract full text from output
        if event_type == "response.completed":
            response = event.get("response", {})
            for item in response.get("output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            return content.get("text", "")

        # Handle content_part delta
        if event_type == "response.content_part.delta":
            return event.get("delta", {}).get("text", "")

        return ""

    def login(self, headless: bool = False) -> dict:
        """Convenience: trigger login and return status."""
        self._auth.login(headless=headless)
        return self._auth.status

    def get_status(self) -> dict:
        """Get provider status."""
        return {
            "provider": self.name,
            "model": self._model,
            **self._auth.status,
        }
