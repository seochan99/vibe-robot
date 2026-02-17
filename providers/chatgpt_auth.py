"""ChatGPT OAuth 2.0 PKCE authentication.

Implements the same OAuth flow used by OpenAI's Codex CLI, OpenCode, and OpenClaw.
Users log in with their ChatGPT Plus/Pro account — no API key needed.

Flow:
  1. Generate PKCE codes (code_verifier + code_challenge)
  2. Open browser to OpenAI authorization endpoint
  3. Capture auth code via localhost callback (or device code for headless)
  4. Exchange auth code for access + refresh tokens
  5. Store tokens securely
  6. Auto-refresh tokens before expiry

Uses the official Codex CLI public OAuth client credentials.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


# --- OAuth constants (from official Codex CLI: github.com/openai/codex) ---

AUTH_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
DEVICE_CODE_URL = "https://auth.openai.com/codex/device"

# Official Codex CLI public client ID (embedded in the open-source Codex CLI)
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

REDIRECT_URI = "http://localhost:1455/auth/callback"
CALLBACK_PORT = 1455
SCOPES = "openid profile email offline_access"

# Token refresh buffer (refresh 5 minutes before expiry)
REFRESH_BUFFER_SEC = 300

# Default credential storage path
DEFAULT_AUTH_DIR = Path.home() / ".viberobot"
DEFAULT_AUTH_FILE = DEFAULT_AUTH_DIR / "auth.json"


@dataclass
class OAuthTokens:
    """OAuth token set."""

    access_token: str
    refresh_token: str
    expires_at: float          # Unix timestamp
    id_token: str = ""
    account_id: str = ""       # Extracted from JWT claims
    plan_type: str = ""        # "plus", "pro", "team", etc.

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at - REFRESH_BUFFER_SEC

    @property
    def is_valid(self) -> bool:
        return bool(self.access_token) and not self.is_expired

    def to_dict(self) -> dict:
        return {
            "type": "oauth",
            "access": self.access_token,
            "refresh": self.refresh_token,
            "expires": int(self.expires_at * 1000),  # ms (Codex format)
            "id_token": self.id_token,
            "account_id": self.account_id,
            "plan_type": self.plan_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OAuthTokens:
        expires_ms = data.get("expires", 0)
        return cls(
            access_token=data.get("access", ""),
            refresh_token=data.get("refresh", ""),
            expires_at=expires_ms / 1000 if expires_ms > 1e12 else expires_ms,
            id_token=data.get("id_token", ""),
            account_id=data.get("account_id", ""),
            plan_type=data.get("plan_type", ""),
        )


class ChatGPTAuth:
    """Manages ChatGPT OAuth authentication.

    Usage:
        auth = ChatGPTAuth()

        # First time: login via browser
        auth.login()

        # Subsequent times: load stored credentials
        auth.load()

        # Get valid token (auto-refreshes if needed)
        token = auth.get_access_token()
        account_id = auth.get_account_id()
    """

    def __init__(self, auth_file: Optional[Path] = None):
        self._auth_file = auth_file or DEFAULT_AUTH_FILE
        self._tokens: Optional[OAuthTokens] = None

    # --- Public API ---

    def login(self, headless: bool = False) -> OAuthTokens:
        """Initiate OAuth login flow.

        Args:
            headless: Use device code flow (for SSH/server environments).
        """
        if headless:
            tokens = self._device_code_flow()
        else:
            tokens = self._browser_oauth_flow()

        self._tokens = tokens
        self._save_tokens()
        return tokens

    def load(self) -> bool:
        """Load stored credentials. Returns True if valid tokens found."""
        if not self._auth_file.exists():
            return False

        try:
            data = json.loads(self._auth_file.read_text())
            self._tokens = OAuthTokens.from_dict(data)

            # Auto-refresh if expired
            if self._tokens.is_expired and self._tokens.refresh_token:
                self._refresh_tokens()

            return self._tokens.is_valid
        except (json.JSONDecodeError, KeyError):
            return False

    def logout(self) -> None:
        """Remove stored credentials."""
        self._tokens = None
        if self._auth_file.exists():
            self._auth_file.unlink()

    def get_access_token(self) -> str:
        """Get a valid access token, refreshing if necessary."""
        if self._tokens is None:
            if not self.load():
                raise RuntimeError(
                    "Not authenticated. Run `viberobot auth login` first, "
                    "or use `viberobot auth login --device` for headless environments."
                )

        if self._tokens.is_expired:
            self._refresh_tokens()

        return self._tokens.access_token

    def get_account_id(self) -> str:
        """Get the ChatGPT account ID (required for API calls)."""
        if self._tokens is None:
            self.load()
        if self._tokens and self._tokens.account_id:
            return self._tokens.account_id
        # Extract from JWT if not cached
        token = self.get_access_token()
        account_id = _extract_jwt_claim(token, "chatgpt_account_id")
        if self._tokens:
            self._tokens.account_id = account_id
            self._save_tokens()
        return account_id

    @property
    def is_authenticated(self) -> bool:
        """Check if we have valid credentials."""
        if self._tokens is not None and self._tokens.is_valid:
            return True
        return self.load()

    @property
    def plan_type(self) -> str:
        """ChatGPT plan type (plus, pro, team, etc.)."""
        if self._tokens:
            return self._tokens.plan_type
        return ""

    @property
    def status(self) -> dict:
        """Get auth status for display."""
        if not self.is_authenticated:
            return {"authenticated": False, "message": "Not logged in"}
        return {
            "authenticated": True,
            "plan": self._tokens.plan_type or "unknown",
            "account_id": self._tokens.account_id[:8] + "..." if self._tokens.account_id else "",
            "expires_in": max(0, int(self._tokens.expires_at - time.time())),
        }

    # --- Browser OAuth flow ---

    def _browser_oauth_flow(self) -> OAuthTokens:
        """Full OAuth 2.0 PKCE flow with browser redirect."""
        # 1. Generate PKCE
        code_verifier, code_challenge = _generate_pkce()
        state = secrets.token_urlsafe(32)

        # 2. Build authorization URL
        params = {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
        }
        auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

        # 3. Start local callback server
        auth_code_holder = {"code": None, "error": None}

        class CallbackHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                qs = urllib.parse.parse_qs(parsed.query)

                if parsed.path == "/auth/callback":
                    returned_state = qs.get("state", [None])[0]
                    if returned_state != state:
                        auth_code_holder["error"] = "State mismatch"
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(b"Authentication failed: state mismatch")
                        return

                    if "code" in qs:
                        auth_code_holder["code"] = qs["code"][0]
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html")
                        self.end_headers()
                        self.wfile.write(
                            b"<html><body><h2>Authentication successful!</h2>"
                            b"<p>You can close this tab and return to the terminal.</p>"
                            b"</body></html>"
                        )
                    elif "error" in qs:
                        auth_code_holder["error"] = qs.get("error_description", qs["error"])[0]
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(b"Authentication failed")
                    return

                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress HTTP server logs

        server = http.server.HTTPServer(("127.0.0.1", CALLBACK_PORT), CallbackHandler)
        server_thread = threading.Thread(target=server.handle_request, daemon=True)
        server_thread.start()

        # 4. Open browser
        print(f"\nOpening browser for ChatGPT login...")
        print(f"If the browser doesn't open, visit:\n{auth_url}\n")
        webbrowser.open(auth_url)

        # 5. Wait for callback
        server_thread.join(timeout=120)
        server.server_close()

        if auth_code_holder["error"]:
            raise RuntimeError(f"OAuth failed: {auth_code_holder['error']}")
        if not auth_code_holder["code"]:
            raise RuntimeError(
                "OAuth callback not received within 120 seconds. "
                "Try `viberobot auth login --device` for headless environments."
            )

        # 6. Exchange code for tokens
        return self._exchange_code(auth_code_holder["code"], code_verifier)

    # --- Device code flow (headless) ---

    def _device_code_flow(self) -> OAuthTokens:
        """Device code flow for headless/SSH environments.

        Prints a URL + code for the user to enter on any browser.
        """
        # 1. Request device code
        resp = requests.post(DEVICE_CODE_URL, json={
            "client_id": CLIENT_ID,
            "scope": SCOPES,
        })
        if resp.status_code != 200:
            raise RuntimeError(
                f"Device code request failed ({resp.status_code}). "
                "Make sure device code login is enabled in your ChatGPT security settings."
            )

        data = resp.json()
        device_code = data["device_code"]
        user_code = data["user_code"]
        verification_uri = data["verification_uri"]
        interval = data.get("interval", 5)
        expires_in = data.get("expires_in", 900)

        print(f"\n{'='*50}")
        print(f"  Device Code: {user_code}")
        print(f"  Go to: {verification_uri}")
        print(f"  Enter the code above and log in.")
        print(f"{'='*50}\n")
        print("Waiting for authentication...")

        # 2. Poll for completion
        deadline = time.time() + expires_in
        while time.time() < deadline:
            time.sleep(interval)

            token_resp = requests.post(TOKEN_URL, data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": CLIENT_ID,
                "device_code": device_code,
            })

            if token_resp.status_code == 200:
                return self._parse_token_response(token_resp.json())

            error = token_resp.json().get("error", "")
            if error == "authorization_pending":
                continue
            elif error == "slow_down":
                interval += 5
                continue
            elif error == "expired_token":
                raise RuntimeError("Device code expired. Please try again.")
            elif error == "access_denied":
                raise RuntimeError("Authentication denied by user.")
            else:
                raise RuntimeError(f"Device auth error: {error}")

        raise RuntimeError("Device code authentication timed out.")

    # --- Token exchange and refresh ---

    def _exchange_code(self, code: str, code_verifier: str) -> OAuthTokens:
        """Exchange authorization code for tokens."""
        resp = requests.post(TOKEN_URL, data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "code_verifier": code_verifier,
        })

        if resp.status_code != 200:
            raise RuntimeError(f"Token exchange failed ({resp.status_code}): {resp.text}")

        return self._parse_token_response(resp.json())

    def _refresh_tokens(self) -> None:
        """Refresh the access token using the refresh token."""
        if not self._tokens or not self._tokens.refresh_token:
            raise RuntimeError("No refresh token available. Please login again.")

        resp = requests.post(TOKEN_URL, data={
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": self._tokens.refresh_token,
        })

        if resp.status_code != 200:
            # Refresh failed — need to re-login
            self._tokens = None
            raise RuntimeError(
                f"Token refresh failed ({resp.status_code}). Please login again."
            )

        new_tokens = self._parse_token_response(resp.json())
        # Preserve account_id if not in new token
        if not new_tokens.account_id and self._tokens:
            new_tokens.account_id = self._tokens.account_id
        self._tokens = new_tokens
        self._save_tokens()

    def _parse_token_response(self, data: dict) -> OAuthTokens:
        """Parse token endpoint response."""
        access_token = data["access_token"]
        refresh_token = data.get("refresh_token", "")
        expires_in = data.get("expires_in", 86400)  # Default 24h
        id_token = data.get("id_token", "")

        # Extract account info from JWT
        account_id = ""
        plan_type = ""
        try:
            claims = _decode_jwt_payload(access_token)
            auth_info = claims.get("https://api.openai.com/auth", {})
            account_id = auth_info.get("chatgpt_account_id", "")
            plan_type = auth_info.get("chatgpt_plan_type", "")
        except Exception:
            pass

        return OAuthTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=time.time() + expires_in,
            id_token=id_token,
            account_id=account_id,
            plan_type=plan_type,
        )

    # --- Token storage ---

    def _save_tokens(self) -> None:
        """Save tokens to disk."""
        if self._tokens is None:
            return
        self._auth_file.parent.mkdir(parents=True, exist_ok=True)
        self._auth_file.write_text(json.dumps(self._tokens.to_dict(), indent=2))
        # Set restrictive permissions (owner read/write only)
        self._auth_file.chmod(0o600)


# --- Utilities ---

def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    code_verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return code_verifier, code_challenge


def _decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload (without verification — we trust the auth server)."""
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload_b64 = parts[1]
    # Add padding
    payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
    payload_bytes = base64.urlsafe_b64decode(payload_b64)
    return json.loads(payload_bytes)


def _extract_jwt_claim(token: str, claim: str) -> str:
    """Extract a specific claim from the JWT's OpenAI auth namespace."""
    payload = _decode_jwt_payload(token)
    auth_info = payload.get("https://api.openai.com/auth", {})
    return auth_info.get(claim, "")
