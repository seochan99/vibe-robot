"use client";

import Link from "next/link";
import { useCallback, useEffect, useState, useRef } from "react";
import type {
  AIModel,
  ExecutionResponse,
  PipelineResponse,
  SceneObjectInfo,
} from "@/lib/api";
import {
  approveExecution,
  fetchModels,
  sendCommand,
  initScene,
  fetchSceneObjects,
  addSceneObject,
  removeSceneObject,
  getSimStreamUrl,
} from "@/lib/api";
import type { AuthStatus } from "@/lib/chatgpt-oauth";
import {
  getAuthStatus,
  createAuthFlow,
  parseCallbackUrl,
  exchangeCode,
  clearTokens,
  loadTokens,
  refreshAccessToken,
} from "@/lib/chatgpt-oauth";

// ── Types ────────────────────────────────────────────────────────────────────

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  pipeline?: PipelineResponse;
  execution?: ExecutionResponse;
  previewImage?: string;
  previewFormat?: "png" | "gif";
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

/** One-line summary shown above the preview */
function formatSummary(p: PipelineResponse): string {
  if (p.error) return p.error;
  if (p.plan) return p.plan.plan_description;
  if (p.intent) return p.intent.intended_meaning;
  return "Processing...";
}

/** Check if the response likely came from GPT (has substantive reasoning text) */
function isGptResponse(p: PipelineResponse): boolean {
  const intentReasoning = p.intent?.reasoning || "";
  const planReasoning = p.plan?.reasoning || "";
  // Rule-based responses have empty or very short reasoning
  return intentReasoning.length > 30 || planReasoning.length > 30;
}

/** Detailed breakdown (shown in expandable section) */
function formatDetails(p: PipelineResponse): string {
  const parts: string[] = [];
  const gpt = isGptResponse(p);

  // GPT reasoning — shown first and prominently when available
  if (gpt) {
    let reasoning = "**AI Reasoning**\n";
    if (p.intent?.reasoning) {
      reasoning += `> ${p.intent.reasoning}\n`;
    }
    if (p.plan?.reasoning && p.plan.reasoning !== p.intent?.reasoning) {
      reasoning += `\n> ${p.plan.reasoning}`;
    }
    parts.push(reasoning);
  }

  if (p.intent) {
    const conf = Math.round(p.intent.confidence * 100);
    parts.push(
      `**Intent Analysis**\n` +
        `You said: "${p.intent.raw_command}"\n` +
        `Meaning: ${p.intent.intended_meaning}\n` +
        `Goal: ${p.intent.immediate_goal}\n` +
        `Deep goal: ${p.intent.deep_goal}\n` +
        `Targets: ${p.intent.target_objects.join(", ") || "auto"}\n` +
        `Constraints: ${p.intent.implicit_constraints.join(", ")}\n` +
        `Confidence: ${conf}%`
    );
  }

  if (p.affordance) {
    const a = p.affordance;
    let s = "**Affordance Analysis**\n";
    if (a.recommended_object) {
      s += `Recommended: **${a.recommended_object}** (${a.recommended_affordance})\n`;
    }
    for (const oa of a.objects) {
      const active = oa.active_affordances
        .slice(0, 3)
        .map((af) => `${af.name} (${Math.round(af.score * 100)}%)`)
        .join(", ");
      if (active) s += `- ${oa.object_name}: ${active}\n`;
    }
    if (a.reasoning) s += `\n> ${a.reasoning}`;
    parts.push(s);
  }

  if (p.plan) {
    const steps = p.plan.steps
      .map((st, i) => `${i + 1}. \`${st.action}(${st.target})\` — ${st.description || ""}`)
      .join("\n");
    let planSection = `**Plan Steps**\n${steps}\n\nDuration: ~${p.plan.estimated_duration}s | Risk: ${p.plan.risk_level}`;
    // Non-GPT reasoning (rule-based) — show inline if it exists and wasn't shown above
    if (!gpt && p.plan.reasoning) {
      planSection += `\n> ${p.plan.reasoning}`;
    }
    parts.push(planSection);
  }

  if (p.safety) {
    const lines: string[] = [];
    for (const c of p.safety.preconditions) lines.push(`[${c.severity}] ${c.description}`);
    for (const c of p.safety.invariants) lines.push(`[${c.severity}] ${c.description}`);
    for (const c of p.safety.tripwires) lines.push(`[tripwire] ${c.description}`);
    lines.push(`Max speed: ${p.safety.limits.max_velocity_m_s} m/s | Max torque: ${p.safety.limits.max_torque_Nm} Nm`);
    parts.push("**Safety Contract**\n" + lines.join("\n"));
  }

  return parts.join("\n\n");
}

function formatExecution(e: ExecutionResponse): string {
  if (e.success) {
    return "Done!";
  }
  return `Failed — ${e.error || "Unknown error"}`;
}

// ── Component ────────────────────────────────────────────────────────────────

// ── Object palette for scene editor ──────────────────────────────────────────

const OBJECT_PALETTE = [
  { type: "book", label: "Book", icon: "\uD83D\uDCD6" },
  { type: "cup", label: "Cup", icon: "\u2615" },
  { type: "pen", label: "Pen", icon: "\uD83D\uDD8A" },
  { type: "paper", label: "Paper", icon: "\uD83D\uDCC4" },
  { type: "apple", label: "Apple", icon: "\uD83C\uDF4E" },
  { type: "banana", label: "Banana", icon: "\uD83C\uDF4C" },
  { type: "orange", label: "Orange", icon: "\uD83C\uDF4A" },
  { type: "cube", label: "Cube", icon: "\uD83D\uDFE5" },
];

export default function AppPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [scene, setScene] = useState("wind_paper");
  const [model, setModel] = useState("rule_based");
  const [loading, setLoading] = useState(false);
  const [awaitingApproval, setAwaitingApproval] = useState(false);
  const [auth, setAuth] = useState<AuthStatus | null>(null);
  const [authLoading, setAuthLoading] = useState(false);
  const [authModal, setAuthModal] = useState<{ authUrl: string } | null>(null);
  const [callbackUrl, setCallbackUrl] = useState("");
  const [sceneOpen, setSceneOpen] = useState(false);
  const [modelOpen, setModelOpen] = useState(false);
  const [models, setModels] = useState<AIModel[]>(FALLBACK_MODELS);
  const [isListening, setIsListening] = useState(false);
  const [sceneObjects, setSceneObjects] = useState<SceneObjectInfo[]>([]);
  const [simConnected, setSimConnected] = useState(false);
  const [editorMode, setEditorMode] = useState<string | null>(null); // object type being placed
  const [pipelineStage, setPipelineStage] = useState<string>("idle");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recognitionRef = useRef<any>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const simImgRef = useRef<HTMLImageElement>(null);

  // Check auth (client-side) + fetch models on mount
  useEffect(() => {
    const status = getAuthStatus();
    setAuth(status);
    fetchModels().then((m) => {
      if (m.length > 0) {
        setModels(m);
      } else if (status.authenticated) {
        setModels(CHATGPT_MODELS);
      }
    });
  }, []);

  // Initialize scene + sim stream on mount & scene change
  const refreshSceneObjects = useCallback(() => {
    fetchSceneObjects().then((objs) => setSceneObjects(objs));
  }, []);

  useEffect(() => {
    initScene(scene)
      .then(() => {
        setSimConnected(true);
        refreshSceneObjects();
      })
      .catch(() => setSimConnected(false));
  }, [scene, refreshSceneObjects]);

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
    setPipelineStage("intent_inference");

    try {
      // Load tokens for GPT models
      let accessToken: string | undefined;
      let accountId: string | undefined;
      if (model !== "rule_based") {
        let tokens = loadTokens();
        if (!tokens) {
          tokens = await refreshAccessToken();
        }
        if (tokens) {
          accessToken = tokens.access_token;
          accountId = tokens.account_id;
        }
      }

      const res = await sendCommand(cmd, scene, model, accessToken, accountId);
      setPipelineStage(res.stage);
      addMessage({
        role: "assistant",
        content: formatSummary(res),
        pipeline: res,
        previewImage: res.preview_image,
        previewFormat: res.preview_format,
      });
      setAwaitingApproval(res.stage === "awaiting_approval");
    } catch (err) {
      addMessage({
        role: "assistant",
        content: `**Error:** ${err instanceof Error ? err.message : "Something went wrong"}.\n\nMake sure the backend is running: \`python api/server.py\``,
      });
    } finally {
      setLoading(false);
      setPipelineStage("idle");
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

  // Auth — open new tab + auto-detect via storage event
  const handleLogin = useCallback(async () => {
    try {
      const flow = await createAuthFlow();
      setCallbackUrl("");
      // Open in a new tab (not popup — popups get blocked)
      window.open(flow.authUrl, "_blank");
      setAuthModal({ authUrl: flow.authUrl });
    } catch (e) {
      addMessage({
        role: "system",
        content: `Failed to start login: ${e instanceof Error ? e.message : "Unknown error"}`,
      });
    }
  }, [addMessage]);

  const handleLoginComplete = useCallback(async () => {
    const parsed = parseCallbackUrl(callbackUrl);
    if (!parsed) return;
    setAuthLoading(true);
    try {
      const tokens = await exchangeCode(parsed.code);
      setAuth({
        authenticated: true,
        plan: tokens.plan_type || "connected",
        account_id: tokens.account_id ? tokens.account_id.slice(0, 8) + "..." : "",
      });
      setAuthModal(null);
      setCallbackUrl("");
      addMessage({ role: "system", content: "ChatGPT connected! You can now use AI-powered models." });
      fetchModels().then((m) => { setModels(m.length > 0 ? m : CHATGPT_MODELS); });
    } catch (e) {
      addMessage({
        role: "system",
        content: `Connection failed: ${e instanceof Error ? e.message : "Unknown error"}`,
      });
    } finally {
      setAuthLoading(false);
    }
  }, [callbackUrl, addMessage]);

  const handleLogout = useCallback(() => {
    clearTokens();
    setAuth({ authenticated: false });
  }, []);

  // Scene editor: add object on sim view click
  const handleSimClick = useCallback(
    async (e: React.MouseEvent<HTMLImageElement>) => {
      if (!editorMode || !simImgRef.current) return;

      const rect = simImgRef.current.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / rect.width; // 0..1
      const ny = (e.clientY - rect.top) / rect.height; // 0..1

      // Map pixel coords to approximate workspace coords
      // Camera is ~120 azimuth, -20 elevation, centered ~(0.3, 0, 0.4)
      const x = 0.2 + nx * 0.6; // x: 0.2..0.8
      const y = (0.5 - nx) * 0.6; // rough y mapping
      const z = 0.35; // desk surface height

      try {
        const result = await addSceneObject(editorMode, [x, y, z]);
        refreshSceneObjects();
        setEditorMode(null);
        addMessage({
          role: "system",
          content: `Added **${result.name}** to the scene.`,
        });
      } catch (err) {
        addMessage({
          role: "system",
          content: `Failed to add object: ${err instanceof Error ? err.message : "error"}`,
        });
      }
    },
    [editorMode, addMessage, refreshSceneObjects],
  );

  // Scene editor: remove object
  const handleRemoveObject = useCallback(
    async (name: string) => {
      try {
        await removeSceneObject(name);
        refreshSceneObjects();
        addMessage({ role: "system", content: `Removed **${name}** from the scene.` });
      } catch (err) {
        addMessage({
          role: "system",
          content: `Failed to remove: ${err instanceof Error ? err.message : "error"}`,
        });
      }
    },
    [addMessage, refreshSceneObjects],
  );

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
        <div className="px-4 h-14 flex items-center justify-between">
          <Link href="/" className="text-lg font-bold tracking-tight">
            VibeRobot<span className="text-[var(--accent)]">!</span>
          </Link>

          <div className="flex items-center gap-3 flex-wrap">
            {/* Scene selector */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-[var(--muted)] shrink-0">Scene:</span>
              <div className="relative">
                <button
                  onClick={() => setSceneOpen(!sceneOpen)}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-lg border border-[var(--border)] bg-[var(--background)] hover:border-[var(--accent)]/30 transition-colors"
                >
                  {selectedScene.label}
                  <svg className={`w-3.5 h-3.5 text-[var(--muted)] transition-transform ${sceneOpen ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {sceneOpen && (
                  <div className="absolute top-full left-0 mt-1 w-64 rounded-lg border border-[var(--border)] bg-[var(--background)] shadow-lg z-20 overflow-hidden">
                    {SCENES.map((s) => (
                      <button
                        key={s.id}
                        onClick={() => { setScene(s.id); setSceneOpen(false); }}
                        className={`w-full text-left px-4 py-3 hover:bg-[var(--surface)] transition-colors ${s.id === scene ? "bg-[var(--accent-soft)] border-l-2 border-[var(--accent)]" : ""}`}
                      >
                        <div className="text-sm font-medium">{s.label}</div>
                        <div className="text-xs text-[var(--muted)]">{s.desc}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Model selector */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-[var(--muted)] shrink-0">Model:</span>
              <div className="relative">
                <button
                  onClick={() => setModelOpen(!modelOpen)}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-lg border border-[var(--border)] bg-[var(--background)] hover:border-[var(--accent)]/30 transition-colors"
                >
                  {models.find((m) => m.id === model)?.name || model}
                  <svg className={`w-3.5 h-3.5 text-[var(--muted)] transition-transform ${modelOpen ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {modelOpen && (
                  <div className="absolute top-full right-0 mt-1 w-56 rounded-lg border border-[var(--border)] bg-[var(--background)] shadow-lg z-20 overflow-hidden">
                    {models.map((m) => (
                      <button
                        key={m.id}
                        onClick={() => {
                          if (m.requires_auth && !auth?.authenticated) { handleLogin(); return; }
                          setModel(m.id); setModelOpen(false);
                        }}
                        className={`w-full text-left px-4 py-2.5 hover:bg-[var(--surface)] transition-colors ${m.id === model ? "bg-[var(--accent-soft)] border-l-2 border-[var(--accent)]" : ""}`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">{m.name}</span>
                          {m.requires_auth && !auth?.authenticated && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-500">login</span>
                          )}
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Auth status */}
            {auth?.authenticated ? (
              <div className="flex items-center gap-2">
                <span className="hidden sm:inline text-xs text-emerald-500 font-medium">
                  ChatGPT {auth.plan || "Connected"}
                </span>
                <div className="w-2 h-2 rounded-full bg-emerald-500" />
                <button onClick={handleLogout} className="text-xs text-[var(--muted)] hover:text-[var(--foreground)] transition-colors">
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
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold">Connect ChatGPT</h3>
              <button onClick={() => { setAuthModal(null); setCallbackUrl(""); }} className="text-[var(--muted)] hover:text-[var(--foreground)] text-lg leading-none">&times;</button>
            </div>
            <div className="space-y-4 text-sm">
              <div className="flex gap-3">
                <span className="shrink-0 w-6 h-6 rounded-full bg-[#10a37f] text-white text-xs flex items-center justify-center font-bold">1</span>
                <div>
                  <p className="font-medium">Sign in to ChatGPT</p>
                  <a href={authModal.authUrl} target="_blank" rel="noopener noreferrer" className="inline-block mt-1.5 px-3 py-1.5 text-xs font-medium rounded-md bg-[#10a37f] hover:bg-[#0d8c6d] text-white transition-colors">
                    Open ChatGPT Login
                  </a>
                </div>
              </div>
              <div className="flex gap-3">
                <span className="shrink-0 w-6 h-6 rounded-full bg-[var(--foreground)] text-[var(--background)] text-xs flex items-center justify-center font-bold">2</span>
                <div className="flex-1 min-w-0">
                  <p className="font-medium">Paste the redirect URL</p>
                  <p className="text-xs text-[var(--muted)] mt-0.5">After login, copy the URL from the new tab and paste it here.</p>
                  <div className="flex gap-2 mt-2">
                    <input
                      type="text" value={callbackUrl} onChange={(e) => setCallbackUrl(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter" && callbackUrl.trim()) handleLoginComplete(); }}
                      placeholder="Paste URL here..."
                      className="flex-1 min-w-0 px-3 py-2 text-xs font-mono rounded-md border border-[var(--border)] bg-[var(--surface)] placeholder:text-[var(--muted)] focus:outline-none focus:border-[#10a37f]/50"
                    />
                    <button onClick={handleLoginComplete} disabled={!callbackUrl.trim() || authLoading}
                      className="px-4 py-2 text-xs font-medium rounded-md bg-[var(--foreground)] hover:opacity-80 text-[var(--background)] transition-colors disabled:opacity-40 shrink-0"
                    >
                      {authLoading ? "..." : "Connect"}
                    </button>
                  </div>
                </div>
              </div>
              <p className="text-[10px] text-[var(--muted)] leading-relaxed">If the auth-relay server is running locally, this step happens automatically.</p>
            </div>
          </div>
        </div>
      )}

      {/* ── Split-pane: Chat + Simulation ────────────────────────────────────── */}
      <div className="flex-1 flex overflow-hidden">
        {/* LEFT: Chat panel */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Chat area */}
          <div className="flex-1 overflow-y-auto">
            <div className="max-w-3xl mx-auto px-4 py-6 space-y-4">
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
                      <div className="max-w-[90%] rounded-2xl rounded-bl-md bg-[var(--surface)] border border-[var(--border)] text-sm leading-relaxed overflow-hidden">
                        {/* Preview image — front and center */}
                        {msg.previewImage && (
                          <img
                            src={`data:image/${msg.previewFormat === "gif" ? "gif" : "png"};base64,${msg.previewImage}`}
                            alt="Simulation preview"
                            className="w-full max-w-lg"
                          />
                        )}
                        {/* Summary text */}
                        <div className="px-4 py-3">
                          <MarkdownLite text={msg.content} />
                        </div>
                        {/* Expandable pipeline details */}
                        {msg.pipeline && (
                          <PipelineDetails pipeline={msg.pipeline} />
                        )}
                        {/* Execution results */}
                        {msg.execution && (
                          <div className="px-4 pb-3">
                            {msg.execution.results.map((r, i) => (
                              <div key={i} className={`text-xs ${r.success ? "text-emerald-500" : "text-red-500"}`}>
                                {r.success ? "+" : "x"} {r.message}
                              </div>
                            ))}
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
              <div className="px-4 py-3 flex items-center justify-between">
                <span className="text-sm text-amber-600 dark:text-amber-400 font-medium">Plan ready for approval</span>
                <div className="flex items-center gap-2">
                  <button onClick={handleReject} className="px-4 py-2 text-sm font-medium rounded-lg border border-[var(--border)] hover:bg-red-500/10 hover:border-red-500/30 text-[var(--foreground)] transition-colors">
                    Reject
                  </button>
                  <button onClick={handleApprove} className="px-4 py-2 text-sm font-medium rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white transition-colors">
                    Approve & Execute
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Input bar */}
          <div className="shrink-0 border-t border-[var(--border)] bg-[var(--background)]">
            <div className="px-4 py-3">
              <div className="flex items-center gap-2">
                <input
                  type="text" value={input} onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                  placeholder={isListening ? "Listening..." : "Type a command or use the mic..."}
                  disabled={loading}
                  className="flex-1 px-4 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--surface)] text-sm placeholder:text-[var(--muted)] focus:outline-none focus:border-[var(--accent)]/50 focus:ring-1 focus:ring-[var(--accent)]/20 disabled:opacity-50 transition-colors"
                />
                <button
                  onClick={toggleListening} disabled={loading}
                  className={`p-2.5 rounded-xl border transition-all shrink-0 ${isListening ? "bg-red-500/10 border-red-500/30 text-red-500" : "border-[var(--border)] text-[var(--muted)] hover:text-[var(--foreground)] hover:border-[var(--accent)]/30"}`}
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
                <button onClick={handleSend} disabled={loading || !input.trim()}
                  className="px-5 py-2.5 rounded-xl bg-[var(--foreground)] hover:opacity-80 text-[var(--background)] text-sm font-medium transition-colors disabled:opacity-50 disabled:hover:opacity-50 shrink-0"
                >
                  Send
                </button>
              </div>
              {!auth?.authenticated && (
                <p className="mt-2 text-xs text-[var(--muted)] text-center">
                  Running in demo mode.{" "}
                  <button onClick={handleLogin} className="text-[var(--accent)] hover:text-[var(--foreground)] underline">Connect ChatGPT</button>{" "}
                  for full AI-powered analysis.
                </p>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT: Simulation panel (hidden on mobile) */}
        <div className="hidden lg:flex flex-col w-[480px] shrink-0 border-l border-[var(--border)] bg-[var(--surface)]">
          {/* Sim header */}
          <div className="shrink-0 px-4 py-2 border-b border-[var(--border)] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${simConnected ? "bg-emerald-500" : "bg-red-500"}`} />
              <span className="text-xs font-medium">Simulation View</span>
            </div>
            {pipelineStage !== "idle" && (
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--accent)]/10 text-[var(--accent)] font-medium">
                {pipelineStage.replace(/_/g, " ")}
              </span>
            )}
          </div>

          {/* MJPEG stream */}
          <div className="relative flex-1 min-h-0 bg-black flex items-center justify-center">
            {simConnected ? (
              <img
                ref={simImgRef}
                src={getSimStreamUrl()}
                alt="MuJoCo simulation"
                className={`w-full h-full object-contain ${editorMode ? "cursor-crosshair" : ""}`}
                onClick={handleSimClick}
              />
            ) : (
              <div className="text-center text-[var(--muted)] text-sm p-8">
                <p className="font-medium mb-2">Simulation offline</p>
                <p className="text-xs">Start the backend: <code className="px-1.5 py-0.5 rounded bg-[var(--border)] text-xs font-mono">python api/server.py</code></p>
              </div>
            )}

            {/* Editor mode indicator */}
            {editorMode && (
              <div className="absolute top-2 left-2 right-2 flex items-center justify-between">
                <span className="px-3 py-1.5 rounded-lg bg-blue-600/90 text-white text-xs font-medium backdrop-blur-sm">
                  Click to place: {editorMode}
                </span>
                <button
                  onClick={() => setEditorMode(null)}
                  className="px-2 py-1 rounded-lg bg-black/50 text-white text-xs backdrop-blur-sm hover:bg-black/70"
                >
                  Cancel
                </button>
              </div>
            )}
          </div>

          {/* Object palette / scene editor */}
          <div className="shrink-0 border-t border-[var(--border)]">
            {/* Palette */}
            <div className="px-3 py-2">
              <p className="text-[10px] text-[var(--muted)] uppercase tracking-wider font-medium mb-1.5">Add Objects</p>
              <div className="flex flex-wrap gap-1">
                {OBJECT_PALETTE.map((obj) => (
                  <button
                    key={obj.type}
                    onClick={() => setEditorMode(editorMode === obj.type ? null : obj.type)}
                    className={`px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      editorMode === obj.type
                        ? "bg-blue-600 text-white ring-2 ring-blue-400/50"
                        : "bg-[var(--background)] border border-[var(--border)] hover:border-[var(--accent)]/30"
                    }`}
                  >
                    <span className="mr-1">{obj.icon}</span>{obj.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Scene objects list */}
            <div className="px-3 py-2 border-t border-[var(--border)] max-h-40 overflow-y-auto">
              <p className="text-[10px] text-[var(--muted)] uppercase tracking-wider font-medium mb-1.5">
                Scene Objects ({sceneObjects.length})
              </p>
              {sceneObjects.length === 0 ? (
                <p className="text-xs text-[var(--muted)]">No objects loaded</p>
              ) : (
                <div className="space-y-1">
                  {sceneObjects.map((obj) => (
                    <div key={obj.name} className="flex items-center justify-between text-xs group">
                      <span className="truncate">
                        <span className="font-medium">{obj.name}</span>
                        <span className="text-[var(--muted)] ml-1">
                          ({obj.position[0].toFixed(2)}, {obj.position[1].toFixed(2)}, {obj.position[2].toFixed(2)})
                        </span>
                      </span>
                      <button
                        onClick={() => handleRemoveObject(obj.name)}
                        className="opacity-0 group-hover:opacity-100 text-red-500 hover:text-red-400 transition-opacity px-1"
                        title="Remove"
                      >
                        &times;
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Minimal markdown renderer ────────────────────────────────────────────────

// ── Expandable pipeline details ──────────────────────────────────────────────

function PipelineDetails({ pipeline }: { pipeline: PipelineResponse }) {
  const [open, setOpen] = useState(false);
  const details = formatDetails(pipeline);
  const gpt = isGptResponse(pipeline);

  if (!details) return null;

  return (
    <div className="border-t border-[var(--border)]">
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-4 py-2 text-xs text-[var(--muted)] hover:text-[var(--foreground)] flex items-center gap-1.5 transition-colors"
      >
        <svg
          className={`w-3 h-3 transition-transform ${open ? "rotate-90" : ""}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
        {open ? "Hide details" : "Show details"}
        {gpt && (
          <span className="ml-1.5 px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
            GPT
          </span>
        )}
        {pipeline.intent && (
          <span className="ml-auto text-[10px] opacity-60">
            {Math.round(pipeline.intent.confidence * 100)}% confidence
          </span>
        )}
      </button>
      {open && (
        <div className="px-4 pb-3 text-xs text-[var(--muted)] leading-relaxed">
          <MarkdownLite text={details} />
        </div>
      )}
    </div>
  );
}

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
