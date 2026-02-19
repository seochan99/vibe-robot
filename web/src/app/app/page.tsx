"use client";

import Link from "next/link";
import { useCallback, useEffect, useState, useRef } from "react";
import type {
  AIModel,
  ExecutionResponse,
  PipelineResponse,
  SimCameraState,
  SceneObjectInfo,
} from "@/lib/api";
import {
  approveExecution,
  rejectExecution,
  fetchModels,
  sendCommandStream,
  initScene,
  fetchSceneObjects,
  addSceneObject,
  moveSceneObject,
  removeSceneObject,
  fetchSimCamera,
  pickSimObject,
  projectSimToWorld,
  updateSimCamera,
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
import { formatExecution, formatSummary } from "./chat-format";
import { MarkdownLite } from "./markdown-lite";
import { formatElapsed, clamp, msgId, projectViewToPlane } from "./page-helpers";
import { PipelineDetails } from "./pipeline-details";
import { SimulationPanel } from "./simulation-panel";
import { CHATGPT_MODELS, FALLBACK_MODELS, SCENES } from "./ui-constants";
import type { SimTool } from "./ui-constants";

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

type DragState =
  | {
      mode: "camera_orbit" | "camera_pan";
      lastX: number;
      lastY: number;
    }
  | {
      mode: "move";
      lastX: number;
      lastY: number;
      objectName: string;
      planeZ: number;
      dragStartWorld: [number, number, number];
      objectStartPos: [number, number, number];
      objectPos: [number, number, number];
    };

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
  const [simTool, setSimTool] = useState<SimTool>("camera");
  const [selectedObject, setSelectedObject] = useState<string | null>(null);
  const [simCamera, setSimCamera] = useState<SimCameraState | null>(null);
  const [pipelineStage, setPipelineStage] = useState<string>("idle");
  const [requestStartedAt, setRequestStartedAt] = useState<number | null>(null);
  const [lastStreamAt, setLastStreamAt] = useState<number | null>(null);
  const [clockMs, setClockMs] = useState<number>(Date.now());
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recognitionRef = useRef<any>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const simImgRef = useRef<HTMLImageElement>(null);
  const streamUiTickRef = useRef(0);
  const dragRef = useRef<DragState | null>(null);
  const simCameraRef = useRef<SimCameraState | null>(null);
  const cameraPendingRef = useRef<SimCameraState | null>(null);
  const cameraSyncingRef = useRef(false);
  const moveSendAtRef = useRef(0);

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
    setEditorMode(null);
    setSimTool("camera");
    setSelectedObject(null);
    initScene(scene)
      .then(async () => {
        setSimConnected(true);
        refreshSceneObjects();
        try {
          const camera = await fetchSimCamera();
          setSimCamera(camera);
        } catch {
          setSimCamera(null);
        }
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

  // Live timer while command is running
  useEffect(() => {
    if (!loading) return;
    const id = window.setInterval(() => setClockMs(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [loading]);

  useEffect(() => {
    simCameraRef.current = simCamera;
  }, [simCamera]);

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

  // Update the last assistant message in-place (for streaming updates)
  const updateLastAssistant = useCallback((updater: (msg: ChatMessage) => ChatMessage) => {
    setMessages((prev) => {
      const idx = prev.length - 1;
      if (idx < 0 || prev[idx].role !== "assistant") return prev;
      const updated = [...prev];
      updated[idx] = updater(updated[idx]);
      return updated;
    });
  }, []);

  const markStreamActivity = useCallback(() => {
    const now = Date.now();
    if (now - streamUiTickRef.current < 700) return;
    streamUiTickRef.current = now;
    setLastStreamAt(now);
  }, []);

  const flushCameraSync = useCallback(async () => {
    if (cameraSyncingRef.current) return;
    cameraSyncingRef.current = true;
    try {
      while (cameraPendingRef.current) {
        const next = cameraPendingRef.current;
        cameraPendingRef.current = null;
        try {
          const applied = await updateSimCamera(next);
          setSimCamera(applied);
          simCameraRef.current = applied;
        } catch {
          // Keep local state and let user continue interacting.
        }
      }
    } finally {
      cameraSyncingRef.current = false;
    }
  }, []);

  const queueCameraState = useCallback(
    (next: SimCameraState) => {
      setSimCamera(next);
      simCameraRef.current = next;
      cameraPendingRef.current = next;
      void flushCameraSync();
    },
    [flushCameraSync],
  );

  const getImageNormalizedPoint = useCallback(
    (clientX: number, clientY: number) => {
      const img = simImgRef.current;
      if (!img) return null;
      const rect = img.getBoundingClientRect();
      if (rect.width <= 1 || rect.height <= 1) return null;

      const naturalW = img.naturalWidth || 640;
      const naturalH = img.naturalHeight || 480;
      const imageAspect = naturalW / naturalH;
      const boxAspect = rect.width / rect.height;

      let drawW = rect.width;
      let drawH = rect.height;
      let offX = 0;
      let offY = 0;

      if (boxAspect > imageAspect) {
        drawW = rect.height * imageAspect;
        offX = (rect.width - drawW) * 0.5;
      } else {
        drawH = rect.width / imageAspect;
        offY = (rect.height - drawH) * 0.5;
      }

      const px = clientX - rect.left - offX;
      const py = clientY - rect.top - offY;
      if (px < 0 || py < 0 || px > drawW || py > drawH) return null;
      return {
        nx: px / drawW,
        ny: py / drawH,
        aspect: drawW / drawH,
      };
    },
    [],
  );

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

  // Send command (with SSE streaming for progressive updates)
  const handleSend = useCallback(async () => {
    const cmd = input.trim();
    if (!cmd || loading) return;

    setInput("");
    addMessage({ role: "user", content: cmd });
    setLoading(true);
    setAwaitingApproval(false);
    setPipelineStage("intent_inference");
    const startedAt = Date.now();
    setRequestStartedAt(startedAt);
    setLastStreamAt(startedAt);
    streamUiTickRef.current = startedAt;

    // Add a placeholder assistant message that we'll update progressively
    addMessage({ role: "assistant", content: "Starting pipeline...\nRequest sent to backend." });

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

      // Use SSE streaming to get progressive updates
      const res = await sendCommandStream(
        cmd, scene, model,
        (evt) => {
          markStreamActivity();
          if (evt.stage !== "heartbeat") {
            // Update the placeholder message with each stage
            setPipelineStage(evt.stage);
            updateLastAssistant((msg) => ({
              ...msg,
              content: msg.content + "\n" + evt.message,
            }));
          }
        },
        accessToken,
        accountId,
        {
          timeoutMs: 180_000,
          onActivity: markStreamActivity,
        },
      );

      // Replace the streaming message with the final result
      setPipelineStage(res.stage);
      updateLastAssistant((msg) => ({
        ...msg,
        content: formatSummary(res),
        pipeline: res,
        previewImage: res.preview_image,
        previewFormat: res.preview_format,
      }));
      setAwaitingApproval(res.stage === "awaiting_approval");
    } catch (err) {
      updateLastAssistant((msg) => ({
        ...msg,
        content: `**Error:** ${err instanceof Error ? err.message : "Something went wrong"}.\n\nMake sure the backend is running: \`python api/server.py\``,
      }));
    } finally {
      setLoading(false);
      setPipelineStage("idle");
      setRequestStartedAt(null);
      setLastStreamAt(null);
    }
  }, [input, loading, scene, model, addMessage, updateLastAssistant, markStreamActivity]);

  // Approve
  const handleApprove = useCallback(async () => {
    setAwaitingApproval(false);
    setLoading(true);
    setPipelineStage("executing");
    const startedAt = Date.now();
    setRequestStartedAt(startedAt);
    setLastStreamAt(startedAt);
    addMessage({
      role: "assistant",
      content: "Executing approved plan in simulation...",
    });
    try {
      const res = await approveExecution();
      addMessage({
        role: "assistant",
        content: formatExecution(res),
        execution: res,
      });
      refreshSceneObjects();
    } catch (err) {
      addMessage({
        role: "assistant",
        content: `**Execution error:** ${err instanceof Error ? err.message : "Failed"}`,
      });
    } finally {
      setLoading(false);
      setPipelineStage("idle");
      setRequestStartedAt(null);
      setLastStreamAt(null);
    }
  }, [addMessage, refreshSceneObjects]);

  // Reject
  const handleReject = useCallback(async () => {
    setAwaitingApproval(false);
    try {
      await rejectExecution();
      addMessage({
        role: "assistant",
        content: "Plan rejected. Send a new command to try again.",
      });
    } catch (err) {
      addMessage({
        role: "assistant",
        content: `Failed to reject plan: ${err instanceof Error ? err.message : "error"}`,
      });
    }
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

  // Scene editor: add object at clicked image point (camera-aware projection)
  const handleSimClick = useCallback(
    async (e: React.MouseEvent<HTMLImageElement>) => {
      if (!editorMode || loading) return;
      const pt = getImageNormalizedPoint(e.clientX, e.clientY);
      if (!pt) return;

      try {
        const [x, y, z] = await projectSimToWorld(pt.nx, pt.ny, 0.35, pt.aspect);
        const result = await addSceneObject(editorMode, [x, y, z]);
        refreshSceneObjects();
        setEditorMode(null);
        setSimTool("camera");
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
    [editorMode, loading, getImageNormalizedPoint, addMessage, refreshSceneObjects],
  );

  const handleSimPointerDown = useCallback(
    (e: React.PointerEvent<HTMLImageElement>) => {
      if (loading || editorMode) return;
      e.preventDefault();

      const isPan = e.button === 1 || e.button === 2 || e.shiftKey;
      const clientX = e.clientX;
      const clientY = e.clientY;
      const pointerId = e.pointerId;
      const target = e.currentTarget;

      if (simTool !== "move" || isPan) {
        dragRef.current = {
          mode: isPan ? "camera_pan" : "camera_orbit",
          lastX: clientX,
          lastY: clientY,
        };
        target.setPointerCapture(pointerId);
        return;
      }

      // Move tool: pick object directly from cursor, then drag immediately.
      target.setPointerCapture(pointerId);
      const startMove = async () => {
        const cam = simCameraRef.current;
        const pt = getImageNormalizedPoint(clientX, clientY);
        if (!cam || !pt) {
          if (target.hasPointerCapture(pointerId)) target.releasePointerCapture(pointerId);
          return;
        }

        let objectName = selectedObject;
        let objectPos: [number, number, number] | null = null;
        try {
          const picked = await pickSimObject(pt.nx, pt.ny, pt.aspect, 0.09);
          if (picked) {
            objectName = picked.name;
            objectPos = picked.position;
          }
        } catch {
          // Fall back to selected object in list.
        }

        if (!objectName) {
          if (target.hasPointerCapture(pointerId)) target.releasePointerCapture(pointerId);
          return;
        }

        if (!objectPos) {
          const obj = sceneObjects.find((o) => o.name === objectName);
          if (!obj) {
            if (target.hasPointerCapture(pointerId)) target.releasePointerCapture(pointerId);
            return;
          }
          objectPos = [...obj.position] as [number, number, number];
        }

        const dragStartWorld = projectViewToPlane(
          pt.nx,
          pt.ny,
          cam,
          objectPos[2],
          pt.aspect,
        );
        if (!dragStartWorld) {
          if (target.hasPointerCapture(pointerId)) target.releasePointerCapture(pointerId);
          return;
        }

        setSelectedObject(objectName);
        dragRef.current = {
          mode: "move",
          lastX: clientX,
          lastY: clientY,
          objectName,
          planeZ: objectPos[2],
          dragStartWorld,
          objectStartPos: objectPos,
          objectPos,
        };
      };
      void startMove();
    },
    [loading, editorMode, simTool, selectedObject, sceneObjects, getImageNormalizedPoint],
  );

  const handleSimPointerMove = useCallback(
    (e: React.PointerEvent<HTMLImageElement>) => {
      const drag = dragRef.current;
      if (!drag || loading) return;

      const dx = e.clientX - drag.lastX;
      const dy = e.clientY - drag.lastY;
      drag.lastX = e.clientX;
      drag.lastY = e.clientY;

      if (drag.mode === "camera_orbit" || drag.mode === "camera_pan") {
        const cam = simCameraRef.current;
        if (!cam) return;

        if (drag.mode === "camera_orbit") {
          const next: SimCameraState = {
            ...cam,
            azimuth: cam.azimuth + dx * 0.55,
            elevation: clamp(cam.elevation - dy * 0.42, -89, 89),
          };
          queueCameraState(next);
          return;
        }

        const az = (cam.azimuth * Math.PI) / 180;
        const right = [Math.cos(az), Math.sin(az)] as const;
        const forward = [-Math.sin(az), Math.cos(az)] as const;
        const scale = cam.distance * 0.0036;
        const lx = cam.lookat[0] - dx * right[0] * scale + dy * forward[0] * scale;
        const ly = cam.lookat[1] - dx * right[1] * scale + dy * forward[1] * scale;
        const next: SimCameraState = {
          ...cam,
          lookat: [lx, ly, cam.lookat[2]],
        };
        queueCameraState(next);
        return;
      }

      if (drag.mode !== "move") return;

      // Move selected object using true view-plane projection delta.
      const cam = simCameraRef.current;
      if (!cam) return;
      const pt = getImageNormalizedPoint(e.clientX, e.clientY);
      if (!pt) return;
      const world = projectViewToPlane(
        pt.nx,
        pt.ny,
        cam,
        drag.planeZ,
        pt.aspect,
      );
      if (!world) return;

      const nextPos: [number, number, number] = [
        clamp(drag.objectStartPos[0] + (world[0] - drag.dragStartWorld[0]), 0.1, 0.9),
        clamp(drag.objectStartPos[1] + (world[1] - drag.dragStartWorld[1]), -0.5, 0.5),
        drag.objectStartPos[2],
      ];
      drag.objectPos = nextPos;

      setSceneObjects((prev) =>
        prev.map((o) => (o.name === drag.objectName ? { ...o, position: nextPos } : o)),
      );

      const now = Date.now();
      if (now - moveSendAtRef.current > 90) {
        moveSendAtRef.current = now;
        void moveSceneObject(drag.objectName, nextPos);
      }
    },
    [loading, queueCameraState, getImageNormalizedPoint],
  );

  const handleSimPointerUp = useCallback(
    (e: React.PointerEvent<HTMLImageElement>) => {
      const drag = dragRef.current;
      if (drag?.mode === "move") {
        void moveSceneObject(drag.objectName, drag.objectPos).then(() => {
          refreshSceneObjects();
        });
      }
      dragRef.current = null;
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
    },
    [refreshSceneObjects],
  );

  const handleSimContextMenu = useCallback((e: React.MouseEvent<HTMLImageElement>) => {
    e.preventDefault();
  }, []);

  const handleSimWheel = useCallback((e: React.WheelEvent<HTMLImageElement>) => {
    if (editorMode || loading) return;
    e.preventDefault();
    const cam = simCameraRef.current;
    if (!cam) return;
    const zoom = Math.exp(e.deltaY * 0.0015);
    const next: SimCameraState = {
      ...cam,
      distance: clamp(cam.distance * zoom, 0.4, 6.0),
    };
    queueCameraState(next);
  }, [editorMode, loading, queueCameraState]);

  const handleResetCamera = useCallback(async () => {
    try {
      const camera = await updateSimCamera({
        azimuth: 120,
        elevation: -22,
        distance: 2.0,
        lookat: [0.45, 0.0, 0.35],
      });
      setSimCamera(camera);
      simCameraRef.current = camera;
    } catch {
      // keep UI usable even if camera reset fails
    }
  }, []);

  const handleCancelEditor = useCallback(() => {
    setEditorMode(null);
    setSimTool("camera");
  }, []);

  const handleSelectTool = useCallback((tool: SimTool) => {
    setSimTool(tool);
    setEditorMode(null);
  }, []);

  const handleToggleAddMode = useCallback((objType: string) => {
    if (editorMode === objType) {
      setEditorMode(null);
      setSimTool("camera");
      return;
    }
    setEditorMode(objType);
    setSimTool("add");
  }, [editorMode]);

  const handleSelectObject = useCallback((name: string) => {
    setSelectedObject(name);
    setSimTool("move");
    setEditorMode(null);
  }, []);

  // Scene editor: remove object
  const handleRemoveObject = useCallback(
    async (name: string) => {
      try {
        await removeSceneObject(name);
        refreshSceneObjects();
        if (selectedObject === name) {
          setSelectedObject(null);
          setSimTool("camera");
        }
        addMessage({ role: "system", content: `Removed **${name}** from the scene.` });
      } catch (err) {
        addMessage({
          role: "system",
          content: `Failed to remove: ${err instanceof Error ? err.message : "error"}`,
        });
      }
    },
    [addMessage, refreshSceneObjects, selectedObject],
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
      type SpeechResult = { 0?: { transcript?: string } };
      const transcript = Array.from(event.results as ArrayLike<SpeechResult>)
        .map((r) => r[0]?.transcript || "")
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
  const elapsedSec =
    loading && requestStartedAt
      ? Math.max(0, Math.floor((clockMs - requestStartedAt) / 1000))
      : 0;
  const sinceUpdateSec =
    loading && lastStreamAt
      ? Math.max(0, Math.floor((clockMs - lastStreamAt) / 1000))
      : 0;
  const stageLabel = pipelineStage === "idle" ? "starting" : pipelineStage.replace(/_/g, " ");

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
                          /* eslint-disable-next-line @next/next/no-img-element */
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
              {loading && (
                <p className={`mt-2 text-xs text-center ${sinceUpdateSec > 8 ? "text-amber-500" : "text-[var(--muted)]"}`}>
                  Running: {stageLabel} · elapsed {formatElapsed(elapsedSec)} · last update {formatElapsed(sinceUpdateSec)} ago
                </p>
              )}
            </div>
          </div>
        </div>

        <SimulationPanel
          simConnected={simConnected}
          simCamera={simCamera}
          pipelineStage={pipelineStage}
          stageLabel={stageLabel}
          simImgRef={simImgRef}
          editorMode={editorMode}
          loading={loading}
          simTool={simTool}
          selectedObject={selectedObject}
          sceneObjects={sceneObjects}
          simStreamUrl={getSimStreamUrl()}
          onResetCamera={handleResetCamera}
          onSimClick={handleSimClick}
          onSimPointerDown={handleSimPointerDown}
          onSimPointerMove={handleSimPointerMove}
          onSimPointerUp={handleSimPointerUp}
          onSimContextMenu={handleSimContextMenu}
          onSimWheel={handleSimWheel}
          onCancelEditor={handleCancelEditor}
          onSelectTool={handleSelectTool}
          onToggleAddMode={handleToggleAddMode}
          onSelectObject={handleSelectObject}
          onRemoveObject={handleRemoveObject}
        />
      </div>
    </div>
  );
}
