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
  preview_image?: string;  // base64 PNG
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
  model: string = "rule_based"
): Promise<PipelineResponse> {
  const res = await fetch(`${API_BASE}/api/command`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command, scene, model }),
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

// Auth is now handled client-side via chatgpt-oauth.ts (device code flow)
