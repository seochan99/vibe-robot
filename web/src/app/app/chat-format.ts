import type { ExecutionResponse, PipelineResponse } from "@/lib/api";

/** One-line summary shown above the preview */
export function formatSummary(p: PipelineResponse): string {
  if (p.error) return p.error;
  if (p.stage === "awaiting_approval" && p.plan) {
    return (
      `${p.plan.plan_description}\n\n` +
      "Plan ready for approval. Click **Approve & Execute** to run on the right simulation panel."
    );
  }
  if (p.plan) return p.plan.plan_description;
  if (p.intent) return p.intent.intended_meaning;
  return "Processing...";
}

/** Check if the response likely came from GPT (has substantive reasoning text) */
export function isGptResponse(p: PipelineResponse): boolean {
  const intentReasoning = p.intent?.reasoning || "";
  const planReasoning = p.plan?.reasoning || "";
  // Rule-based responses have empty or very short reasoning
  return intentReasoning.length > 30 || planReasoning.length > 30;
}

/** Detailed breakdown (shown in expandable section) */
export function formatDetails(p: PipelineResponse): string {
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

export function formatExecution(e: ExecutionResponse): string {
  const lines: string[] = [];
  lines.push(e.success ? "Execution complete." : `Execution failed: ${e.error || "Unknown error"}`);
  if (e.results?.length) {
    lines.push("");
    for (const r of e.results) {
      lines.push(`- ${r.success ? "[ok]" : "[fail]"} ${r.action}(${r.target}) - ${r.message}`);
    }
  }
  return lines.join("\n");
}
