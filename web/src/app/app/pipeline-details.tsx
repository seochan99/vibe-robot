import { useState } from "react";

import type { PipelineResponse } from "@/lib/api";

import { formatDetails, isGptResponse } from "./chat-format";
import { MarkdownLite } from "./markdown-lite";

export function PipelineDetails({ pipeline }: { pipeline: PipelineResponse }) {
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
