"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AIModel,
  ExecutionResponse,
  PipelineResponse,
} from "@/lib/api";
import {
  approveExecution,
  fetchModels,
  sendCommand,
} from "@/lib/api";
import type { AuthStatus } from "@/lib/chatgpt-oauth";
import {
  getAuthStatus,
  createAuthFlow,
  parseCallbackUrl,
  exchangeCode,
  clearTokens,
} from "@/lib/chatgpt-oauth";

// ── Types ────────────────────────────────────────────────────────────────────

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  pipeline?: PipelineResponse;
  execution?: ExecutionResponse;
  previewImage?: string;  // base64 PNG
  timestamp: number;
}

const FALLBACK_MODELS: AIModel[] = [
  { id: "rule_based", name: "Rule-Based (No AI)", requires_auth: false },
];

const CHATGPT_MODELS: AIModel[] = [
  { id: "rule_based", name: "Rule-Based (No AI)", requires_auth: false },
  { id: "gpt-4o", name: "GPT-4o", requires_auth: true },
  { id: "o3", name: "o3", requires_auth: true },
  { id: "o4-mini", name: "o4-mini", requires_auth: true },
];

const SCENES = [
  { id: "wind_paper", label: "Wind & Papers", desc: "Papers blowing on a desk with a book nearby." },
  { id: "pick_and_place", label: "Pick & Place", desc: "Move the red cube to the blue bin." },
  { id: "sorting", label: "Fruit Sorting", desc: "Sort three fruits into a container." },
];

// ── Helpers ──────────────────────────────────────────────────────────────────

function msgId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

function formatPipeline(p: PipelineResponse): string {
  const parts: string[] = [];

  if (p.intent) {
    parts.push(
      `**Intent Analysis**\n` +
        `You said: "${p.intent.raw_command}"\n` +
        `You meant: ${p.intent.intended_meaning}\n` +
        `Goal: ${p.intent.immediate_goal}\n` +
        `Targets: ${p.intent.target_objects.join(", ")}\n` +
        `Constraints: ${p.intent.implicit_constraints.join(", ")}\n` +
        `Confidence: ${Math.round(p.intent.confidence * 100)}%`
    );
  }

  if (p.affordance) {
    const a = p.affordance;
    let s = "**Affordance Discovery**\n";
    if (a.recommended_object) {
      s += `Use **${a.recommended_object}** — *${a.recommended_affordance}*\n`;
    }
    for (const oa of a.objects) {
      const active = oa.active_affordances
        .slice(0, 3)
        .map((af) => `\`${af.name}\` (${Math.round(af.score * 100)}%)`)
        .join(", ");
      if (active) s += `- ${oa.object_name} (${oa.object_category}): ${active}\n`;
    }
    if (a.reasoning) s += `\n> ${a.reasoning}`;
    parts.push(s);
  }

  if (p.plan) {
    const steps = p.plan.steps
      .map(
        (st, i) =>
          `${i + 1}. \`${st.action}(${st.target})\`${st.description ? ` — ${st.description}` : ""}`
      )
      .join("\n");
    parts.push(
      `**Execution Plan**\n${p.plan.plan_description}\n\n${steps}\n\nEst. ${p.plan.estimated_duration}s | Risk: ${p.plan.risk_level}`
    );
  }

  if (p.safety) {
    const lines: string[] = [];
    for (const c of p.safety.preconditions)
      lines.push(`[${c.severity.toUpperCase()}] ${c.description}`);
    for (const c of p.safety.invariants)
      lines.push(`[${c.severity.toUpperCase()}] ${c.description}`);
    for (const c of p.safety.tripwires) lines.push(`[TRIPWIRE] ${c.description}`);
    lines.push(
      `\nMax speed: ${p.safety.limits.max_velocity_m_s} m/s | Max torque: ${p.safety.limits.max_torque_Nm} Nm`
    );
    parts.push("**Safety Contract**\n" + lines.join("\n"));
  }

  if (p.error) parts.push(`**Error**\n${p.error}`);

  if (p.stage === "awaiting_approval") {
    parts.push("---\nPlan is ready. Click **Approve** to execute, or **Reject** to cancel.");
  }

  return parts.join("\n\n");
}

function formatExecution(e: ExecutionResponse): string {
  if (e.success) {
    const lines = e.results.map(
      (r) =>
        `${r.success ? "+" : "x"} \`${r.action}(${r.target})\` — ${r.message}`
    );
    return "**Execution Complete**\n\n" + lines.join("\n");
  }
  return `**Execution Failed**\n${e.error || "Unknown error"}`;
}

// ── Component ────────────────────────────────────────────────────────────────

export default function AppPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [scene, setScene] = useState("wind_paper");
  const [model, setModel] = useState("rule_based");
  const [loading, setLoading] = useState(false);
  const [awaitingApproval, setAwaitingApproval] = useState(false);
  const [auth, setAuth] = useState<AuthStatus | null>(null);
  const [authLoading, setAuthLoading] = useState(false);
  const [authModal, setAuthModal] = useState<{ authUrl: string; phase: "waiting" | "paste" } | null>(null);
  const [callbackUrl, setCallbackUrl] = useState("");
  const popupRef = useRef<Window | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [sceneOpen, setSceneOpen] = useState(false);
  const [modelOpen, setModelOpen] = useState(false);
  const [models, setModels] = useState<AIModel[]>(FALLBACK_MODELS);
  const [isListening, setIsListening] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recognitionRef = useRef<any>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Check auth (client-side) + fetch models on mount
  useEffect(() => {
    const status = getAuthStatus();
    setAuth(status);
    fetchModels().then((m) => {
      if (m.length > 0) {
        setModels(m);
      } else if (status.authenticated) {
        // Backend unreachable but user has ChatGPT tokens — show client-side models
        setModels(CHATGPT_MODELS);
      }
    });
  }, []);

  // Close dropdowns on outside click
  useEffect(() => {
    const handler = () => { setSceneOpen(false); setModelOpen(false); };
    if (sceneOpen || modelOpen) {
      document.addEventListener("click", handler, { once: true });
      return () => document.removeEventListener("click", handler);
    }
  }, [sceneOpen, modelOpen]);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Welcome message
  useEffect(() => {
    setMessages([
      {
        id: msgId(),
        role: "system",
        content:
          "Welcome to **VibeRobot!** Tell the robot arm what to do in plain language.\n\n" +
          "Try commands like:\n" +
          '- "Prevent the papers from flying away"\n' +
          '- "Move the red cube to the blue bin"\n' +
          '- "Sort the fruits"\n\n' +
          "Select a scene above to set up the environment.",
        timestamp: Date.now(),
      },
    ]);
  }, []);

  const addMessage = useCallback((msg: Omit<ChatMessage, "id" | "timestamp">) => {
    setMessages((prev) => [...prev, { ...msg, id: msgId(), timestamp: Date.now() }]);
  }, []);

  // Listen for auth changes from popup callback page (storage event)
  useEffect(() => {
    const handler = (e: StorageEvent) => {
      if (e.key === "viberobot_oauth" && e.newValue) {
        const status = getAuthStatus();
        if (status.authenticated) {
          setAuth(status);
          setAuthModal(null);
          setCallbackUrl("");
          if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
          popupRef.current?.close();
          addMessage({ role: "system", content: "ChatGPT connected! You can now use AI-powered models." });
          fetchModels().then((m) => { setModels(m.length > 0 ? m : CHATGPT_MODELS); });
        }
      }
    };
    window.addEventListener("storage", handler);
    return () => window.removeEventListener("storage", handler);
  }, [addMessage]);

  // Send command
  const handleSend = useCallback(async () => {
    const cmd = input.trim();
    if (!cmd || loading) return;

    setInput("");
    addMessage({ role: "user", content: cmd });
    setLoading(true);
    setAwaitingApproval(false);

    try {
      const res = await sendCommand(cmd, scene, model);
      addMessage({
        role: "assistant",
        content: formatPipeline(res),
        pipeline: res,
        previewImage: res.preview_image,
      });
      setAwaitingApproval(res.stage === "awaiting_approval");
    } catch (err) {
      addMessage({
        role: "assistant",
        content: `**Error:** ${err instanceof Error ? err.message : "Something went wrong"}.\n\nMake sure the backend is running: \`python api/server.py\``,
      });
    } finally {
      setLoading(false);
    }
  }, [input, loading, scene, model, addMessage]);

  // Approve
  const handleApprove = useCallback(async () => {
    setAwaitingApproval(false);
    setLoading(true);
    try {
      const res = await approveExecution();
      addMessage({
        role: "assistant",
        content: formatExecution(res),
        execution: res,
      });
    } catch (err) {
      addMessage({
        role: "assistant",
        content: `**Execution error:** ${err instanceof Error ? err.message : "Failed"}`,
      });
    } finally {
      setLoading(false);
    }
  }, [addMessage]);

  // Reject
  const handleReject = useCallback(() => {
    setAwaitingApproval(false);
    addMessage({
      role: "assistant",
      content: "Plan rejected. Send a new command to try again.",
    });
  }, [addMessage]);

  // Auth — popup PKCE flow with auto-detect + fallback paste
  const finishLogin = useCallback(async (code: string) => {
    setAuthLoading(true);
    try {
      const tokens = await exchangeCode(code);
      setAuth({
        authenticated: true,
        plan: tokens.plan_type || "connected",
        account_id: tokens.account_id ? tokens.account_id.slice(0, 8) + "..." : "",
      });
      setAuthModal(null);
      setCallbackUrl("");
      addMessage({ role: "system", content: "ChatGPT connected! You can now use AI-powered models." });
      // Refresh models
      fetchModels().then((m) => { setModels(m.length > 0 ? m : CHATGPT_MODELS); });
    } catch (e) {
      setAuthModal((prev) => prev ? { ...prev, phase: "paste" } : null);
      addMessage({
        role: "system",
        content: `Connection failed: ${e instanceof Error ? e.message : "Unknown error"}`,
      });
    } finally {
      setAuthLoading(false);
    }
  }, [addMessage]);

  const handleLogin = useCallback(async () => {
    try {
      const flow = await createAuthFlow();
      setCallbackUrl("");

      // Open popup
      const popup = window.open(
        flow.authUrl,
        "chatgpt_auth",
        "width=500,height=700,left=200,top=100"
      );
      popupRef.current = popup;
      setAuthModal({ authUrl: flow.authUrl, phase: "waiting" });

      // Poll for auth code in popup URL
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(() => {
        try {
          if (!popup || popup.closed) {
            clearInterval(pollRef.current!);
            pollRef.current = null;
            // Check if auth succeeded while popup was open
            const status = getAuthStatus();
            if (status.authenticated) {
              setAuth(status);
              setAuthModal(null);
            } else {
              // Show paste fallback
              setAuthModal({ authUrl: flow.authUrl, phase: "paste" });
            }
            return;
          }
          // Try reading popup URL (works when popup is on localhost)
          const url = popup.location.href;
          if (url && url.includes("code=")) {
            clearInterval(pollRef.current!);
            pollRef.current = null;
            popup.close();
            const parsed = parseCallbackUrl(url);
            if (parsed) finishLogin(parsed.code);
          }
        } catch {
          // Cross-origin (auth.openai.com) — keep polling
        }
      }, 500);
    } catch (e) {
      addMessage({
        role: "system",
        content: `Failed to start login: ${e instanceof Error ? e.message : "Unknown error"}`,
      });
    }
  }, [addMessage, finishLogin]);

  const handleLoginComplete = useCallback(async () => {
    const parsed = parseCallbackUrl(callbackUrl);
    if (!parsed) return;
    finishLogin(parsed.code);
  }, [callbackUrl, finishLogin]);

  const handleLogout = useCallback(() => {
    clearTokens();
    setAuth({ authenticated: false });
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);

  // Speech-to-text
  const toggleListening = useCallback(() => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
      return;
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const w = window as any;
    const SpeechRecognition = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onresult = (event: any) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const transcript = Array.from(event.results as any[])
        .map((r: any) => r[0].transcript)
        .join("");
      setInput(transcript);
    };
    recognition.onend = () => setIsListening(false);
    recognition.onerror = () => setIsListening(false);

    recognitionRef.current = recognition;
    recognition.start();
    setIsListening(true);
  }, [isListening]);

  const selectedScene = SCENES.find((s) => s.id === scene)!;

  return (
    <div className="flex flex-col h-screen bg-[var(--background)] text-[var(--foreground)]">
      {/* Header */}
      <header className="shrink-0 border-b border-[var(--border)] bg-[var(--background)]">
        <div className="max-w-4xl mx-auto px-4 h-14 flex items-center justify-between">
          <Link href="/" className="text-lg font-bold tracking-tight">
            VibeRobot<span className="text-[var(--accent)]">!</span>
          </Link>

          <div className="flex items-center gap-3">
            {/* Auth status */}
            {auth?.authenticated ? (
              <div className="flex items-center gap-2">
                <span className="hidden sm:inline text-xs text-emerald-500 font-medium">
                  ChatGPT {auth.plan || "Connected"}
                </span>
                <div className="w-2 h-2 rounded-full bg-emerald-500" />
                <button
                  onClick={handleLogout}
                  className="text-xs text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
                >
                  Sign out
                </button>
              </div>
            ) : (
              <button
                onClick={handleLogin}
                disabled={authLoading}
                className="px-3 py-1.5 text-xs font-medium rounded-lg bg-[#10a37f] hover:bg-[#0d8c6d] text-white transition-colors disabled:opacity-50"
              >
                {authLoading ? "Connecting..." : "Connect ChatGPT"}
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Auth modal */}
      {authModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
          <div className="bg-[var(--background)] border border-[var(--border)] rounded-xl max-w-md w-full p-6 shadow-2xl">
            <div className="flex items-center justify-between mb-5">
              <h3 className="font-semibold">Connect ChatGPT</h3>
              <button
                onClick={() => {
                  setAuthModal(null);
                  setCallbackUrl("");
                  if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
                  popupRef.current?.close();
                }}
                className="text-[var(--muted)] hover:text-[var(--foreground)] text-lg leading-none"
              >
                &times;
              </button>
            </div>

            {authModal.phase === "waiting" ? (
              /* Phase 1: Waiting for popup login */
              <div className="text-center py-4">
                <div className="w-8 h-8 border-2 border-[#10a37f] border-t-transparent rounded-full animate-spin mx-auto" />
                <p className="mt-4 text-sm">Waiting for ChatGPT login...</p>
                <p className="mt-1 text-xs text-[var(--muted)]">Complete the sign-in in the popup window.</p>
                <button
                  onClick={() => {
                    if (popupRef.current && !popupRef.current.closed) {
                      popupRef.current.focus();
                    } else {
                      popupRef.current = window.open(authModal.authUrl, "chatgpt_auth", "width=500,height=700,left=200,top=100");
                    }
                  }}
                  className="mt-4 text-xs text-[#10a37f] hover:underline"
                >
                  Reopen login popup
                </button>
              </div>
            ) : (
              /* Phase 2: Paste fallback */
              <div className="space-y-3 text-sm">
                <p className="text-[var(--muted)]">
                  After logging in, the popup showed an error page.
                  Copy the <strong className="text-[var(--foreground)]">full URL</strong> from that page&apos;s address bar and paste it here:
                </p>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={callbackUrl}
                    onChange={(e) => setCallbackUrl(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && callbackUrl.trim()) handleLoginComplete(); }}
                    // eslint-disable-next-line jsx-a11y/no-autofocus
                    autoFocus
                    placeholder="http://localhost:1455/auth/callback?code=..."
                    className="flex-1 min-w-0 px-3 py-2.5 text-xs font-mono rounded-md border border-[var(--border)] bg-[var(--surface)] placeholder:text-[var(--muted)] focus:outline-none focus:border-[#10a37f]/50"
                  />
                  <button
                    onClick={handleLoginComplete}
                    disabled={!callbackUrl.trim() || authLoading}
                    className="px-4 py-2.5 text-xs font-medium rounded-md bg-[#10a37f] hover:bg-[#0d8c6d] text-white transition-colors disabled:opacity-40 shrink-0"
                  >
                    {authLoading ? "..." : "Connect"}
                  </button>
                </div>
                <button
                  onClick={() => {
                    popupRef.current = window.open(authModal.authUrl, "chatgpt_auth", "width=500,height=700,left=200,top=100");
                    setAuthModal({ authUrl: authModal.authUrl, phase: "waiting" });
                  }}
                  className="text-xs text-[var(--muted)] hover:text-[var(--foreground)]"
                >
                  Try again
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Scene selector bar */}
      <div className="shrink-0 border-b border-[var(--border)] bg-[var(--surface)]">
        <div className="max-w-4xl mx-auto px-4 py-2 flex items-center gap-3">
          <span className="text-xs text-[var(--muted)] shrink-0">Scene:</span>
          <div className="relative">
            <button
              onClick={() => setSceneOpen(!sceneOpen)}
              className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-lg border border-[var(--border)] bg-[var(--background)] hover:border-[var(--accent)]/30 transition-colors"
            >
              {selectedScene.label}
              <svg
                className={`w-3.5 h-3.5 text-[var(--muted)] transition-transform ${sceneOpen ? "rotate-180" : ""}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {sceneOpen && (
              <div className="absolute top-full left-0 mt-1 w-64 rounded-lg border border-[var(--border)] bg-[var(--background)] shadow-lg z-10 overflow-hidden">
                {SCENES.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => {
                      setScene(s.id);
                      setSceneOpen(false);
                    }}
                    className={`w-full text-left px-4 py-3 hover:bg-[var(--surface)] transition-colors ${
                      s.id === scene ? "bg-[var(--accent-soft)] border-l-2 border-[var(--accent)]" : ""
                    }`}
                  >
                    <div className="text-sm font-medium">{s.label}</div>
                    <div className="text-xs text-[var(--muted)]">{s.desc}</div>
                  </button>
                ))}
              </div>
            )}
          </div>
          {/* Model selector */}
          <span className="text-xs text-[var(--muted)] shrink-0 ml-2">Model:</span>
          <div className="relative">
            <button
              onClick={() => setModelOpen(!modelOpen)}
              className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-lg border border-[var(--border)] bg-[var(--background)] hover:border-[var(--accent)]/30 transition-colors"
            >
              {models.find((m) => m.id === model)?.name || model}
              <svg
                className={`w-3.5 h-3.5 text-[var(--muted)] transition-transform ${modelOpen ? "rotate-180" : ""}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {modelOpen && (
              <div className="absolute top-full left-0 mt-1 w-56 rounded-lg border border-[var(--border)] bg-[var(--background)] shadow-lg z-10 overflow-hidden">
                {models.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => {
                      if (m.requires_auth && !auth?.authenticated) {
                        handleLogin();
                        return;
                      }
                      setModel(m.id);
                      setModelOpen(false);
                    }}
                    className={`w-full text-left px-4 py-2.5 hover:bg-[var(--surface)] transition-colors ${
                      m.id === model ? "bg-[var(--accent-soft)] border-l-2 border-[var(--accent)]" : ""
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{m.name}</span>
                      {m.requires_auth && !auth?.authenticated && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-500">
                          login
                        </span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          <span className="text-xs text-[var(--muted)] hidden lg:inline flex-1 text-right truncate">
            {selectedScene.desc}
          </span>
        </div>
      </div>

      {/* Chat area */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 py-6 space-y-4">
          {messages.map((msg) => (
            <div key={msg.id} className="message-enter">
              {msg.role === "user" ? (
                <div className="flex justify-end">
                  <div className="max-w-[80%] px-4 py-3 rounded-2xl rounded-br-md bg-[var(--foreground)] text-[var(--background)] text-sm leading-relaxed">
                    {msg.content}
                  </div>
                </div>
              ) : msg.role === "system" ? (
                <div className="flex justify-center">
                  <div className="max-w-lg px-5 py-4 rounded-xl bg-[var(--surface)] border border-[var(--border)] text-sm text-[var(--muted)] leading-relaxed">
                    <MarkdownLite text={msg.content} />
                  </div>
                </div>
              ) : (
                <div className="flex justify-start">
                  <div className="max-w-[85%] px-4 py-3 rounded-2xl rounded-bl-md bg-[var(--surface)] border border-[var(--border)] text-sm leading-relaxed">
                    <MarkdownLite text={msg.content} />
                    {msg.previewImage && (
                      <div className="mt-3 rounded-lg overflow-hidden border border-[var(--border)]">
                        <div className="px-3 py-1.5 bg-[var(--border)] text-xs font-medium text-[var(--muted)]">
                          Simulation Preview
                        </div>
                        <img
                          src={`data:image/png;base64,${msg.previewImage}`}
                          alt="Simulation preview"
                          className="w-full max-w-md"
                        />
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex justify-start message-enter">
              <div className="px-4 py-3 rounded-2xl rounded-bl-md bg-[var(--surface)] border border-[var(--border)]">
                <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-[var(--accent)] pulse-dot" />
                  <div className="w-2 h-2 rounded-full bg-[var(--accent)] pulse-dot" style={{ animationDelay: "0.3s" }} />
                  <div className="w-2 h-2 rounded-full bg-[var(--accent)] pulse-dot" style={{ animationDelay: "0.6s" }} />
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Approve/Reject bar */}
      {awaitingApproval && (
        <div className="shrink-0 border-t border-[var(--border)] bg-amber-500/5">
          <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
            <span className="text-sm text-amber-600 dark:text-amber-400 font-medium">
              Plan ready for approval
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={handleReject}
                className="px-4 py-2 text-sm font-medium rounded-lg border border-[var(--border)] hover:bg-red-500/10 hover:border-red-500/30 text-[var(--foreground)] transition-colors"
              >
                Reject
              </button>
              <button
                onClick={handleApprove}
                className="px-4 py-2 text-sm font-medium rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white transition-colors"
              >
                Approve & Execute
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Input bar */}
      <div className="shrink-0 border-t border-[var(--border)] bg-[var(--background)]">
        <div className="max-w-4xl mx-auto px-4 py-3">
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder={isListening ? "Listening..." : "Type a command or use the mic..."}
              disabled={loading}
              className="flex-1 px-4 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--surface)] text-sm placeholder:text-[var(--muted)] focus:outline-none focus:border-[var(--accent)]/50 focus:ring-1 focus:ring-[var(--accent)]/20 disabled:opacity-50 transition-colors"
            />
            {/* Mic button */}
            <button
              onClick={toggleListening}
              disabled={loading}
              className={`p-2.5 rounded-xl border transition-all shrink-0 ${
                isListening
                  ? "bg-red-500/10 border-red-500/30 text-red-500"
                  : "border-[var(--border)] text-[var(--muted)] hover:text-[var(--foreground)] hover:border-[var(--accent)]/30"
              }`}
              title={isListening ? "Stop listening" : "Voice input"}
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                {isListening ? (
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4M12 15a3 3 0 003-3V5a3 3 0 00-6 0v7a3 3 0 003 3z" />
                )}
              </svg>
            </button>
            <button
              onClick={handleSend}
              disabled={loading || !input.trim()}
              className="px-5 py-2.5 rounded-xl bg-[var(--foreground)] hover:opacity-80 text-[var(--background)] text-sm font-medium transition-colors disabled:opacity-50 disabled:hover:opacity-50 shrink-0"
            >
              Send
            </button>
          </div>
          {!auth?.authenticated && (
            <p className="mt-2 text-xs text-[var(--muted)] text-center">
              Running in demo mode.{" "}
              <button
                onClick={handleLogin}
                className="text-[var(--accent)] hover:text-[var(--foreground)] underline"
              >
                Connect ChatGPT
              </button>{" "}
              for full AI-powered analysis.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Minimal markdown renderer ────────────────────────────────────────────────

function MarkdownLite({ text }: { text: string }) {
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (line === "---") {
      elements.push(<hr key={i} className="my-3 border-[var(--border)]" />);
    } else if (line.startsWith("**") && line.endsWith("**")) {
      elements.push(
        <p key={i} className="font-semibold mt-3 mb-1 text-[var(--foreground)]">
          {line.slice(2, -2)}
        </p>
      );
    } else if (line.startsWith("- ")) {
      elements.push(
        <p key={i} className="pl-3 text-[var(--muted)]">
          <InlineFormat text={line} />
        </p>
      );
    } else if (line.startsWith("> ")) {
      elements.push(
        <blockquote
          key={i}
          className="pl-3 border-l-2 border-[var(--accent)]/30 text-[var(--muted)] italic"
        >
          {line.slice(2)}
        </blockquote>
      );
    } else if (line.match(/^\d+\.\s/)) {
      elements.push(
        <p key={i} className="pl-3">
          <InlineFormat text={line} />
        </p>
      );
    } else if (line.trim() === "") {
      elements.push(<div key={i} className="h-1" />);
    } else {
      elements.push(
        <p key={i}>
          <InlineFormat text={line} />
        </p>
      );
    }
  }

  return <div className="space-y-0.5">{elements}</div>;
}

function InlineFormat({ text }: { text: string }) {
  // Handle **bold**, *italic*, `code`
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**")) {
          return (
            <strong key={i} className="font-semibold text-[var(--foreground)]">
              {part.slice(2, -2)}
            </strong>
          );
        }
        if (part.startsWith("*") && part.endsWith("*")) {
          return (
            <em key={i} className="italic text-[var(--accent)]">
              {part.slice(1, -1)}
            </em>
          );
        }
        if (part.startsWith("`") && part.endsWith("`")) {
          return (
            <code
              key={i}
              className="px-1.5 py-0.5 rounded bg-[var(--border)] text-xs font-mono"
            >
              {part.slice(1, -1)}
            </code>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}
