const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export interface PipelineResponse {
  intent?: {
    raw_command: string;
    literal_meaning: string;
    intended_meaning: string;
    immediate_goal: string;
    deep_goal: string;
    target_objects: string[];
    implicit_constraints: string[];
    confidence: number;
    reasoning: string;
  };
  affordance?: {
    recommended_object: string;
    recommended_affordance: string;
    objects: {
      object_name: string;
      object_category: string;
      active_affordances: { name: string; score: number }[];
    }[];
    reasoning: string;
  };
  plan?: {
    plan_description: string;
    steps: {
      action: string;
      target: string;
      params: Record<string, unknown>;
      description: string;
    }[];
    estimated_duration: number;
    risk_level: string;
    reasoning: string;
  };
  safety?: {
    preconditions: { description: string; severity: string }[];
    invariants: { description: string; severity: string }[];
    tripwires: { description: string }[];
    limits: {
      max_velocity_m_s: number;
      max_torque_Nm: number;
    };
  };
  preview_image?: string;  // base64 PNG or GIF
  preview_format?: "png" | "gif";
  stage: string;
  error?: string;
}

export interface AIModel {
  id: string;
  name: string;
  requires_auth: boolean;
}

export interface ExecutionResponse {
  success: boolean;
  results: {
    step: number;
    action: string;
    target: string;
    success: boolean;
    message: string;
  }[];
  error?: string;
}

export async function sendCommand(
  command: string,
  scene: string,
  model: string = "rule_based",
  access_token?: string,
  account_id?: string,
): Promise<PipelineResponse> {
  const body: Record<string, string> = { command, scene, model };
  if (access_token) body.access_token = access_token;
  if (account_id) body.account_id = account_id;

  const res = await fetch(`${API_BASE}/api/command`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export interface StageEvent {
  stage: string;
  message: string;
}

interface StreamOptions {
  timeoutMs?: number;
  onActivity?: () => void;
}

function parseSseBlock(block: string): { eventType: string; data: string } | null {
  const lines = block.trim().split("\n");
  let eventType = "";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event: ")) {
      eventType = line.slice(7).trim();
    } else if (line.startsWith("data: ")) {
      dataLines.push(line.slice(6));
    }
  }

  const data = dataLines.join("\n");
  if (!data) return null;
  return { eventType, data };
}

/**
 * Stream pipeline execution via SSE.
 * Calls onStage for each pipeline stage update, then resolves with the final result.
 */
export async function sendCommandStream(
  command: string,
  scene: string,
  model: string = "rule_based",
  onStage: (evt: StageEvent) => void,
  access_token?: string,
  account_id?: string,
  options?: StreamOptions,
): Promise<PipelineResponse> {
  console.info("[viberobot] stream request", { scene, model, command });
  const body: Record<string, string> = { command, scene, model };
  if (access_token) body.access_token = access_token;
  if (account_id) body.account_id = account_id;

  const timeoutMs = options?.timeoutMs ?? 180_000;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/command/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeout);
    if (err instanceof Error && err.name === "AbortError") {
      throw new Error("Pipeline timed out while waiting for server response");
    }
    throw err;
  }

  clearTimeout(timeout);
  if (!res.ok) throw new Error(`API error: ${res.status}`);

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult: PipelineResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    options?.onActivity?.();

    buffer += decoder.decode(value, { stream: true });

    // Parse SSE events from buffer
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || ""; // keep incomplete part

    for (const part of parts) {
      const parsedBlock = parseSseBlock(part);
      if (!parsedBlock) continue;
      const { eventType, data } = parsedBlock;

      try {
        const parsed = JSON.parse(data);
        if (eventType === "stage") {
          console.debug("[viberobot] stage", parsed);
          onStage(parsed as StageEvent);
        } else if (eventType === "result") {
          console.info("[viberobot] stream result", {
            stage: (parsed as PipelineResponse).stage,
            error: (parsed as PipelineResponse).error,
          });
          finalResult = parsed as PipelineResponse;
        } else if (eventType === "error") {
          console.error("[viberobot] stream error event", parsed);
          throw new Error(parsed.error || "Pipeline error");
        } else if (eventType === "heartbeat") {
          options?.onActivity?.();
        }
      } catch (e) {
        if (e instanceof SyntaxError) continue; // ignore partial JSON
        console.error("[viberobot] stream parse failure", e);
        throw e;
      }
    }
  }

  // Some servers terminate without trailing blank line; parse any remaining block.
  const rest = parseSseBlock(buffer);
  if (rest) {
    try {
      const parsed = JSON.parse(rest.data);
      if (rest.eventType === "stage") {
        onStage(parsed as StageEvent);
      } else if (rest.eventType === "result") {
        finalResult = parsed as PipelineResponse;
      } else if (rest.eventType === "error") {
        throw new Error(parsed.error || "Pipeline error");
      }
    } catch (e) {
      if (!(e instanceof SyntaxError)) throw e;
    }
  }

  if (!finalResult) throw new Error("No result received from pipeline");
  console.info("[viberobot] stream completed", { stage: finalResult.stage });
  return finalResult;
}

export async function fetchModels(): Promise<AIModel[]> {
  try {
    const res = await fetch(`${API_BASE}/api/models`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.models;
  } catch {
    return [];
  }
}

export async function approveExecution(): Promise<ExecutionResponse> {
  const res = await fetch(`${API_BASE}/api/approve`, { method: "POST" });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function rejectExecution(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/api/reject`, { method: "POST" });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function initScene(scene: string): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/api/scene/init`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene }),
  });
  if (!res.ok) throw new Error(`Scene init error: ${res.status}`);
  return res.json();
}

export interface SceneObjectInfo {
  name: string;
  type: string;
  position: [number, number, number];
  mass_kg: number;
  [key: string]: unknown;
}

export interface SimCameraState {
  azimuth: number;
  elevation: number;
  distance: number;
  lookat: [number, number, number];
}

export interface PickedSimObject {
  name: string;
  position: [number, number, number];
  screen: [number, number];
  distance: number;
  depth: number;
}

export async function fetchSceneObjects(): Promise<SceneObjectInfo[]> {
  try {
    const res = await fetch(`${API_BASE}/api/scene/objects`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.objects;
  } catch {
    return [];
  }
}

export async function addSceneObject(
  obj_type: string,
  position: [number, number, number],
  name?: string,
): Promise<{ name: string }> {
  const res = await fetch(`${API_BASE}/api/scene/add-object`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ obj_type, position, name }),
  });
  if (!res.ok) throw new Error(`Add object error: ${res.status}`);
  return res.json();
}

export async function moveSceneObject(
  name: string,
  position: [number, number, number],
): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/api/scene/move-object`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, position }),
  });
  if (!res.ok) throw new Error(`Move object error: ${res.status}`);
  return res.json();
}

export async function removeSceneObject(name: string): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/api/scene/remove-object`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error(`Remove object error: ${res.status}`);
  return res.json();
}

export function getSimStreamUrl(): string {
  return `${API_BASE}/api/sim/stream`;
}

export async function fetchSimCamera(): Promise<SimCameraState> {
  const res = await fetch(`${API_BASE}/api/sim/camera`);
  if (!res.ok) throw new Error(`Camera fetch error: ${res.status}`);
  const data = await res.json();
  return data.camera as SimCameraState;
}

export async function updateSimCamera(
  patch: Partial<SimCameraState>,
): Promise<SimCameraState> {
  const res = await fetch(`${API_BASE}/api/sim/camera`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error(`Camera update error: ${res.status}`);
  const data = await res.json();
  return data.camera as SimCameraState;
}

export async function projectSimToWorld(
  nx: number,
  ny: number,
  plane_z = 0.35,
  aspect = 4 / 3,
): Promise<[number, number, number]> {
  const res = await fetch(`${API_BASE}/api/sim/project`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ nx, ny, plane_z, aspect }),
  });
  if (!res.ok) throw new Error(`Projection error: ${res.status}`);
  const data = await res.json();
  if (data.status !== "ok" || !Array.isArray(data.point)) {
    throw new Error(data.error || "projection_failed");
  }
  return data.point as [number, number, number];
}

export async function pickSimObject(
  nx: number,
  ny: number,
  aspect = 4 / 3,
  radius = 0.08,
): Promise<PickedSimObject | null> {
  const res = await fetch(`${API_BASE}/api/sim/pick-object`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ nx, ny, aspect, radius }),
  });
  if (!res.ok) throw new Error(`Pick object error: ${res.status}`);
  const data = await res.json();
  if (data.status !== "ok") return null;
  return data.object as PickedSimObject;
}

// Auth is now handled client-side via chatgpt-oauth.ts (device code flow)
