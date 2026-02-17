"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { exchangeCode } from "@/lib/chatgpt-oauth";

function CallbackContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
  const [error, setError] = useState("");

  useEffect(() => {
    async function process() {
      try {
        const code = searchParams.get("code");
        if (!code) {
          // No code — show instructions to copy URL
          setStatus("error");
          setError("Copy this page's full URL and paste it in the VibeRobot app.");
          return;
        }
        await exchangeCode(code);
        setStatus("success");
        // If we're in a popup, close it — the parent detects auth via storage event
        if (window.opener) {
          setTimeout(() => window.close(), 600);
        } else {
          setTimeout(() => router.push("/app"), 1200);
        }
      } catch (e) {
        setStatus("error");
        setError(e instanceof Error ? e.message : "Unknown error");
      }
    }
    process();
  }, [searchParams, router]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--background)]">
      <div className="text-center max-w-md px-6">
        {status === "loading" && (
          <>
            <div className="w-8 h-8 border-2 border-[var(--foreground)] border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="mt-4 text-[var(--muted)]">Connecting to ChatGPT...</p>
          </>
        )}
        {status === "success" && (
          <>
            <svg className="w-12 h-12 text-[#10a37f] mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
            <p className="mt-4 font-medium">Connected to ChatGPT</p>
            <p className="mt-1 text-sm text-[var(--muted)]">Redirecting to app...</p>
          </>
        )}
        {status === "error" && (
          <>
            <p className="text-[var(--accent)] font-medium">Connection issue</p>
            <p className="mt-2 text-sm text-[var(--muted)]">{error}</p>
            <button
              onClick={() => router.push("/app")}
              className="mt-4 px-4 py-2 text-sm border border-[var(--border)] rounded-md hover:bg-[var(--surface)] transition-colors"
            >
              Back to app
            </button>
          </>
        )}
      </div>
    </div>
  );
}

export default function CallbackPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center bg-[var(--background)]">
          <div className="w-8 h-8 border-2 border-[var(--foreground)] border-t-transparent rounded-full animate-spin" />
        </div>
      }
    >
      <CallbackContent />
    </Suspense>
  );
}
