"""Codex CLI subprocess provider (wraps `codex exec`).

Alternative to the native OAuth provider — uses the installed Codex CLI binary.
Simpler setup (just `npm i -g @openai/codex && codex login`) but requires Node.js.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from providers.base import LLMProvider, LLMResponse


class CodexCLIProvider(LLMProvider):
    """LLM provider that wraps OpenAI's Codex CLI (`codex exec`)."""

    def __init__(self, model: str = "gpt-5.3-codex"):
        self._model = model
        self._codex_path = shutil.which("codex")

    @property
    def name(self) -> str:
        return "codex_cli"

    @property
    def is_authenticated(self) -> bool:
        """Check if Codex CLI is installed and logged in."""
        if not self._codex_path:
            return False
        try:
            result = subprocess.run(
                [self._codex_path, "login", "status"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

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
        """Generate using Codex CLI."""
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
        """Generate using `codex exec` subprocess."""
        if not self._codex_path:
            raise RuntimeError(
                "Codex CLI not found. Install with: npm i -g @openai/codex"
            )

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "response.txt"

            cmd = [
                self._codex_path, "exec",
                "--sandbox", "read-only",
                "--output-last-message", str(out_path),
                "--model", self._model,
                "--skip-git-repo-check",
            ]

            # Add images if provided
            if images:
                for img_b64 in images:
                    # Write base64 to temp PNG file
                    import base64
                    img_path = Path(td) / f"img_{len(cmd)}.png"
                    img_path.write_bytes(base64.b64decode(img_b64))
                    cmd.extend(["--image", str(img_path)])

            # Add output schema for JSON mode
            if json_mode:
                prompt += "\n\nRespond in valid JSON format only."

            # Full prompt with system context
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"[System: {system_prompt}]\n\n{prompt}"

            cmd.append(full_prompt)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Read output
            text = ""
            if out_path.exists():
                text = out_path.read_text(encoding="utf-8")
            elif result.stdout:
                text = result.stdout

            if result.returncode != 0 and not text:
                raise RuntimeError(
                    f"Codex exec failed (exit={result.returncode}): {result.stderr[:500]}"
                )

            json_data = None
            if json_mode and text:
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

    def login(self, headless: bool = False) -> None:
        """Trigger Codex CLI login."""
        if not self._codex_path:
            raise RuntimeError("Codex CLI not found. Install: npm i -g @openai/codex")
        cmd = [self._codex_path, "login"]
        if headless:
            cmd.append("--device-auth")
        subprocess.run(cmd)
