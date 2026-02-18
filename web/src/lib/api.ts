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

// Auth is now handled client-side via chatgpt-oauth.ts (device code flow)
