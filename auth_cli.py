"""VibeRobot auth CLI — manage ChatGPT login.

Usage:
    python auth_cli.py login           # Browser OAuth login
    python auth_cli.py login --device  # Device code login (headless/SSH)
    python auth_cli.py status          # Check auth status
    python auth_cli.py logout          # Remove stored credentials
    python auth_cli.py providers       # List available providers
"""

from __future__ import annotations

import argparse
import sys


def cmd_login(args):
    from providers.chatgpt_auth import ChatGPTAuth

    auth = ChatGPTAuth()
    headless = getattr(args, "device", False)

    if headless:
        print("Starting device code authentication...")
        print("(Make sure device code login is enabled in your ChatGPT security settings)")
    else:
        print("Starting browser-based ChatGPT login...")

    try:
        tokens = auth.login(headless=headless)
        print(f"\nAuthentication successful!")
        print(f"  Plan: {tokens.plan_type or 'unknown'}")
        print(f"  Account: {tokens.account_id[:12]}...")
        print(f"\nCredentials saved to ~/.viberobot/auth.json")
        print("You can now use VibeRobot without an API key.")
    except Exception as e:
        print(f"\nAuthentication failed: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_status(args):
    from providers.chatgpt_auth import ChatGPTAuth
    from providers.setup import get_registry, init_providers

    # Check ChatGPT OAuth
    auth = ChatGPTAuth()
    print("=== ChatGPT OAuth ===")
    if auth.is_authenticated:
        status = auth.status
        print(f"  Status: Logged in")
        print(f"  Plan: {status.get('plan', 'unknown')}")
        print(f"  Account: {status.get('account_id', 'unknown')}")
        expires = status.get("expires_in", 0)
        print(f"  Token expires in: {expires // 3600}h {(expires % 3600) // 60}m")
    else:
        print("  Status: Not logged in")
        print("  Run: python auth_cli.py login")

    # Check all providers
    print("\n=== Available Providers ===")
    registry = init_providers()
    for name in registry.available:
        provider = registry.get(name)
        status = "authenticated" if provider.is_authenticated else "not configured"
        default = " (default)" if name == registry._default else ""
        print(f"  {name}: {status}{default}")

    if not registry.authenticated:
        print("\n  No authenticated providers!")
        print("  Run: python auth_cli.py login")
        print("  Or set OPENAI_API_KEY environment variable")


def cmd_logout(args):
    from providers.chatgpt_auth import ChatGPTAuth

    auth = ChatGPTAuth()
    auth.logout()
    print("Logged out. Stored credentials removed.")


def cmd_providers(args):
    from providers.setup import init_providers

    registry = init_providers()
    print("Available LLM Providers:")
    print()
    for name in registry.available:
        provider = registry.get(name)
        auth_status = "OK" if provider.is_authenticated else "NOT CONFIGURED"
        print(f"  [{auth_status:>14}] {name}")

    print()
    print("Provider priority (auto mode):")
    print("  1. chatgpt_oauth  — ChatGPT Plus/Pro login (no API key needed)")
    print("  2. codex_cli      — OpenAI Codex CLI (npm i -g @openai/codex)")
    print("  3. openai_api     — Standard OpenAI API ($OPENAI_API_KEY)")


def main():
    parser = argparse.ArgumentParser(
        description="VibeRobot Authentication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    login_parser = sub.add_parser("login", help="Login with ChatGPT account")
    login_parser.add_argument(
        "--device", action="store_true",
        help="Use device code flow (for SSH/headless environments)",
    )

    sub.add_parser("status", help="Check authentication status")
    sub.add_parser("logout", help="Remove stored credentials")
    sub.add_parser("providers", help="List available LLM providers")

    args = parser.parse_args()

    if args.command == "login":
        cmd_login(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "logout":
        cmd_logout(args)
    elif args.command == "providers":
        cmd_providers(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
