"""Tests for the LLM provider system."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from providers.base import LLMProvider, LLMResponse, ProviderRegistry


class TestLLMResponse:

    def test_parse_json_basic(self):
        resp = LLMResponse(text='{"key": "value"}')
        data = resp.parse_json()
        assert data == {"key": "value"}

    def test_parse_json_with_markdown(self):
        resp = LLMResponse(text='```json\n{"key": "value"}\n```')
        data = resp.parse_json()
        assert data == {"key": "value"}

    def test_parse_json_with_surrounding_text(self):
        resp = LLMResponse(text='Here is the result:\n{"key": "value"}\nDone.')
        data = resp.parse_json()
        assert data == {"key": "value"}

    def test_parse_json_with_concatenated_objects(self):
        resp = LLMResponse(text='{"key": "first"}\n{"key": "second"}')
        data = resp.parse_json()
        assert data == {"key": "first"}

    def test_parse_json_caches(self):
        resp = LLMResponse(text='{"key": "value"}')
        data1 = resp.parse_json()
        data2 = resp.parse_json()
        assert data1 is data2  # Same object from cache


class TestProviderRegistry:

    def test_register_and_get(self):
        registry = ProviderRegistry()

        class FakeProvider(LLMProvider):
            @property
            def name(self): return "fake"
            @property
            def is_authenticated(self): return True
            async def generate(self, prompt, **kw):
                return LLMResponse(text="ok", provider="fake")

        provider = FakeProvider()
        registry.register(provider)
        assert registry.get("fake") is provider

    def test_default_provider(self):
        registry = ProviderRegistry()

        class P1(LLMProvider):
            @property
            def name(self): return "p1"
            @property
            def is_authenticated(self): return False
            async def generate(self, prompt, **kw):
                return LLMResponse(text="")

        class P2(LLMProvider):
            @property
            def name(self): return "p2"
            @property
            def is_authenticated(self): return True
            async def generate(self, prompt, **kw):
                return LLMResponse(text="")

        registry.register(P1())
        registry.register(P2(), default=True)
        assert registry.get_default().name == "p2"

    def test_get_first_authenticated(self):
        registry = ProviderRegistry()

        class Unauth(LLMProvider):
            @property
            def name(self): return "unauth"
            @property
            def is_authenticated(self): return False
            async def generate(self, prompt, **kw):
                return LLMResponse(text="")

        class Auth(LLMProvider):
            @property
            def name(self): return "auth"
            @property
            def is_authenticated(self): return True
            async def generate(self, prompt, **kw):
                return LLMResponse(text="")

        registry.register(Unauth())
        registry.register(Auth())
        first = registry.get_first_authenticated()
        assert first.name == "auth"

    def test_missing_provider_raises(self):
        registry = ProviderRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_available_and_authenticated_lists(self):
        registry = ProviderRegistry()

        class Auth(LLMProvider):
            @property
            def name(self): return "auth"
            @property
            def is_authenticated(self): return True
            async def generate(self, prompt, **kw):
                return LLMResponse(text="")

        class Unauth(LLMProvider):
            @property
            def name(self): return "unauth"
            @property
            def is_authenticated(self): return False
            async def generate(self, prompt, **kw):
                return LLMResponse(text="")

        registry.register(Auth())
        registry.register(Unauth())
        assert set(registry.available) == {"auth", "unauth"}
        assert registry.authenticated == ["auth"]


class TestChatGPTAuth:
    """Test ChatGPT auth utilities (without actual OAuth)."""

    def test_pkce_generation(self):
        from providers.chatgpt_auth import _generate_pkce
        verifier, challenge = _generate_pkce()
        assert len(verifier) > 20
        assert len(challenge) > 20
        assert verifier != challenge

    def test_jwt_decode(self):
        from providers.chatgpt_auth import _decode_jwt_payload
        import base64, json

        # Build a fake JWT
        header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=").decode()
        payload_dict = {
            "sub": "user123",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acc-abc-123",
                "chatgpt_plan_type": "plus",
            },
        }
        payload = base64.urlsafe_b64encode(json.dumps(payload_dict).encode()).rstrip(b"=").decode()
        sig = "fakesig"
        fake_jwt = f"{header}.{payload}.{sig}"

        decoded = _decode_jwt_payload(fake_jwt)
        assert decoded["sub"] == "user123"
        assert decoded["https://api.openai.com/auth"]["chatgpt_account_id"] == "acc-abc-123"
        assert decoded["https://api.openai.com/auth"]["chatgpt_plan_type"] == "plus"

    def test_extract_account_id(self):
        from providers.chatgpt_auth import _extract_jwt_claim
        import base64, json

        payload_dict = {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acc-test-456",
            },
        }
        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(json.dumps(payload_dict).encode()).rstrip(b"=").decode()
        fake_jwt = f"{header}.{payload}.sig"

        account_id = _extract_jwt_claim(fake_jwt, "chatgpt_account_id")
        assert account_id == "acc-test-456"

    def test_token_expiry(self):
        import time
        from providers.chatgpt_auth import OAuthTokens

        # Not expired
        tokens = OAuthTokens(
            access_token="test",
            refresh_token="test",
            expires_at=time.time() + 3600,
        )
        assert not tokens.is_expired
        assert tokens.is_valid

        # Expired
        tokens_expired = OAuthTokens(
            access_token="test",
            refresh_token="test",
            expires_at=time.time() - 100,
        )
        assert tokens_expired.is_expired
        assert not tokens_expired.is_valid

    def test_tokens_serialization(self):
        import time
        from providers.chatgpt_auth import OAuthTokens

        tokens = OAuthTokens(
            access_token="at_test",
            refresh_token="rt_test",
            expires_at=time.time() + 3600,
            account_id="acc-123",
            plan_type="pro",
        )
        d = tokens.to_dict()
        assert d["type"] == "oauth"
        assert d["access"] == "at_test"
        assert d["plan_type"] == "pro"

        # Round-trip
        restored = OAuthTokens.from_dict(d)
        assert restored.access_token == "at_test"
        assert restored.plan_type == "pro"


class TestProviderSetup:
    """Test provider auto-detection setup."""

    def test_init_providers(self):
        from providers.setup import init_providers
        registry = init_providers()
        # Should have at least the available providers registered
        assert len(registry.available) >= 1
