/**
 * ChatGPT OAuth 2.0 PKCE — browser redirect flow.
 *
 * Uses the same OAuth client as the official OpenAI Codex CLI.
 * PKCE flow: redirect to OpenAI → user logs in → redirect back with code → exchange for tokens.
 */

const AUTH_URL = "https://auth.openai.com/authorize";
const TOKEN_URL = "https://auth.openai.com/oauth/token";
const CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const SCOPES = "openid profile email offline_access";

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

// ── Redirect URI ─────────────────────────────────────────────────────────────

function getRedirectUri(): string {
  if (typeof window === "undefined") return "http://localhost:3000/callback";
  return `${window.location.origin}/callback`;
}

// ── PKCE helpers ─────────────────────────────────────────────────────────────

function randomBytes(n: number): Uint8Array {
  const buf = new Uint8Array(n);
  crypto.getRandomValues(buf);
  return buf;
}

function base64url(buf: ArrayBuffer | Uint8Array): string {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  let str = "";
  bytes.forEach((b) => (str += String.fromCharCode(b)));
  return btoa(str).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function generatePKCE() {
  const verifier = base64url(randomBytes(32));
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(verifier)
  );
  const challenge = base64url(digest);
  return { verifier, challenge };
}

// ── JWT decode (no verification — we trust auth.openai.com) ──────────────────

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

// ── Step 1: Redirect to OpenAI login ─────────────────────────────────────────

export async function startLogin() {
  const { verifier, challenge } = await generatePKCE();
  const state = base64url(randomBytes(16));
  const redirectUri = getRedirectUri();

  // Persist PKCE verifier + state + redirect_uri for after redirect
  localStorage.setItem(
    PKCE_KEY,
    JSON.stringify({ verifier, state, redirect_uri: redirectUri })
  );

  const params = new URLSearchParams({
    response_type: "code",
    client_id: CLIENT_ID,
    redirect_uri: redirectUri,
    scope: SCOPES,
    code_challenge: challenge,
    code_challenge_method: "S256",
    state,
    audience: "https://api.openai.com/v1",
  });

  window.location.href = `${AUTH_URL}?${params}`;
}

// ── Step 2: Handle callback (called from /callback page) ─────────────────────

export async function handleCallback(
  searchParams: URLSearchParams
): Promise<OAuthTokens> {
  const code = searchParams.get("code");
  const returnedState = searchParams.get("state");
  const error = searchParams.get("error");
  const errorDesc = searchParams.get("error_description");

  if (error) {
    throw new Error(errorDesc || error);
  }
  if (!code) {
    throw new Error("No authorization code received");
  }

  // Verify state + retrieve PKCE verifier
  const pkceRaw = localStorage.getItem(PKCE_KEY);
  if (!pkceRaw) throw new Error("PKCE state not found — please try logging in again");
  const { verifier, state, redirect_uri } = JSON.parse(pkceRaw);

  if (returnedState !== state) {
    throw new Error("State mismatch — possible CSRF attack");
  }

  // Exchange code for tokens
  const resp = await fetch(TOKEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      code,
      redirect_uri: redirect_uri,
      client_id: CLIENT_ID,
      code_verifier: verifier,
    }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Token exchange failed (${resp.status}): ${text}`);
  }

  const data = await resp.json();
  const accessToken: string = data.access_token;
  const refreshToken: string = data.refresh_token || "";
  const expiresIn: number = data.expires_in || 86400;

  // Extract account info from JWT
  const claims = decodeJwtPayload(accessToken);
  const authInfo = (claims["https://api.openai.com/auth"] || {}) as Record<
    string,
    string
  >;

  const tokens: OAuthTokens = {
    access_token: accessToken,
    refresh_token: refreshToken,
    expires_at: Date.now() + expiresIn * 1000,
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
