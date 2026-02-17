/**
 * ChatGPT OAuth 2.0 PKCE — browser flow.
 *
 * Uses the same OAuth client + endpoints as the official OpenAI Codex CLI.
 * Since the only registered redirect_uri is localhost:1455, the web app
 * opens a new tab for login, then the user pastes the redirect URL back.
 *
 * Sources:
 *  - https://github.com/numman-ali/opencode-openai-codex-auth
 *  - https://developers.openai.com/codex/auth/
 */

const AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize";
const TOKEN_URL = "https://auth.openai.com/oauth/token";
const CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const REDIRECT_URI = "http://localhost:1455/auth/callback";
const SCOPE = "openid profile email offline_access";

const STORAGE_KEY = "viberobot_oauth";
const PKCE_KEY = "viberobot_pkce";

// ── Types ────────────────────────────────────────────────────────────────────

export interface OAuthTokens {
  access_token: string;
  refresh_token: string;
  expires_at: number; // unix ms
  account_id: string;
  plan_type: string;
}

export interface AuthStatus {
  authenticated: boolean;
  plan?: string;
  account_id?: string;
  message?: string;
}

export interface PKCEFlow {
  authUrl: string;
  verifier: string;
  state: string;
}

// ── PKCE helpers ─────────────────────────────────────────────────────────────

function base64url(buf: ArrayBuffer | Uint8Array): string {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  let str = "";
  bytes.forEach((b) => (str += String.fromCharCode(b)));
  return btoa(str).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function generatePKCE() {
  const raw = new Uint8Array(32);
  crypto.getRandomValues(raw);
  const verifier = base64url(raw);
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(verifier)
  );
  const challenge = base64url(digest);
  return { verifier, challenge };
}

// ── JWT decode ───────────────────────────────────────────────────────────────

function decodeJwtPayload(token: string): Record<string, unknown> {
  try {
    const parts = token.split(".");
    if (parts.length < 2) return {};
    let b64 = parts[1].replace(/-/g, "+").replace(/_/g, "/");
    while (b64.length % 4) b64 += "=";
    return JSON.parse(atob(b64));
  } catch {
    return {};
  }
}

// ── Token storage ────────────────────────────────────────────────────────────

export function loadTokens(): OAuthTokens | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const tokens: OAuthTokens = JSON.parse(raw);
    if (Date.now() >= tokens.expires_at - 300_000) return null;
    return tokens;
  } catch {
    return null;
  }
}

function saveTokens(tokens: OAuthTokens) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(tokens));
}

export function clearTokens() {
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(PKCE_KEY);
}

// ── Auth status ──────────────────────────────────────────────────────────────

export function getAuthStatus(): AuthStatus {
  const tokens = loadTokens();
  if (!tokens) return { authenticated: false };
  return {
    authenticated: true,
    plan: tokens.plan_type || "connected",
    account_id: tokens.account_id
      ? tokens.account_id.slice(0, 8) + "..."
      : "",
  };
}

// ── Step 1: Generate auth URL ────────────────────────────────────────────────

export async function createAuthFlow(): Promise<PKCEFlow> {
  const { verifier, challenge } = await generatePKCE();
  const stateBytes = new Uint8Array(16);
  crypto.getRandomValues(stateBytes);
  const state = base64url(stateBytes);

  const params = new URLSearchParams({
    response_type: "code",
    client_id: CLIENT_ID,
    redirect_uri: REDIRECT_URI,
    scope: SCOPE,
    code_challenge: challenge,
    code_challenge_method: "S256",
    state,
    id_token_add_organizations: "true",
    codex_cli_simplified_flow: "true",
    originator: "codex_cli_rs",
  });

  const authUrl = `${AUTHORIZE_URL}?${params}`;

  // Persist for after redirect/paste
  localStorage.setItem(PKCE_KEY, JSON.stringify({ verifier, state }));

  return { authUrl, verifier, state };
}

// ── Step 2: Parse callback URL + exchange code ───────────────────────────────

export function parseCallbackUrl(
  input: string
): { code: string; state: string } | null {
  const trimmed = (input || "").trim();
  if (!trimmed) return null;

  try {
    const url = new URL(trimmed);
    const code = url.searchParams.get("code");
    const state = url.searchParams.get("state");
    if (code) return { code, state: state || "" };
  } catch {
    // Not a URL — try other formats
  }

  // code#state format
  if (trimmed.includes("#")) {
    const [code, state] = trimmed.split("#", 2);
    return { code, state: state || "" };
  }

  // query string format
  if (trimmed.includes("code=")) {
    const params = new URLSearchParams(trimmed);
    const code = params.get("code");
    if (code) return { code, state: params.get("state") || "" };
  }

  return null;
}

export async function exchangeCode(code: string): Promise<OAuthTokens> {
  const pkceRaw = localStorage.getItem(PKCE_KEY);
  if (!pkceRaw) throw new Error("Auth session expired. Please try again.");
  const { verifier, state: savedState } = JSON.parse(pkceRaw);

  const resp = await fetch(TOKEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      client_id: CLIENT_ID,
      code,
      code_verifier: verifier,
      redirect_uri: REDIRECT_URI,
    }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Token exchange failed (${resp.status}): ${text}`);
  }

  const data = await resp.json();
  if (!data.access_token || !data.refresh_token) {
    throw new Error("Invalid token response");
  }

  const claims = decodeJwtPayload(data.access_token);
  const authInfo = (claims["https://api.openai.com/auth"] || {}) as Record<
    string,
    string
  >;

  const tokens: OAuthTokens = {
    access_token: data.access_token,
    refresh_token: data.refresh_token,
    expires_at: Date.now() + (data.expires_in || 86400) * 1000,
    account_id: authInfo.chatgpt_account_id || "",
    plan_type: authInfo.chatgpt_plan_type || "",
  };

  saveTokens(tokens);
  localStorage.removeItem(PKCE_KEY);
  return tokens;
}

// ── Token refresh ────────────────────────────────────────────────────────────

export async function refreshAccessToken(): Promise<OAuthTokens | null> {
  const tokens = loadTokens();
  if (!tokens?.refresh_token) return null;

  try {
    const resp = await fetch(TOKEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "refresh_token",
        client_id: CLIENT_ID,
        refresh_token: tokens.refresh_token,
      }),
    });

    if (!resp.ok) {
      clearTokens();
      return null;
    }

    const data = await resp.json();
    const newTokens: OAuthTokens = {
      ...tokens,
      access_token: data.access_token as string,
      refresh_token: (data.refresh_token as string) || tokens.refresh_token,
      expires_at: Date.now() + ((data.expires_in as number) || 86400) * 1000,
    };

    saveTokens(newTokens);
    return newTokens;
  } catch {
    return null;
  }
}
