"use client";

import { useState } from "react";
import { collection, addDoc, serverTimestamp } from "firebase/firestore";
import { db } from "@/lib/firebase";

type FormState = "idle" | "loading" | "success" | "error";

export default function SignupForm() {
  const [email, setEmail] = useState("");
  const [state, setState] = useState<FormState>("idle");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = email.trim();
    if (!trimmed) return;

    setState("loading");
    try {
      await addDoc(collection(db, "waitlist"), {
        email: trimmed,
        createdAt: serverTimestamp(),
        source: "landing",
      });
      setState("success");
      setEmail("");
    } catch {
      setState("error");
    }
  };

  if (state === "success") {
    return (
      <div className="flex items-center gap-2 px-4 py-3 rounded-md bg-[var(--tag-soft)] border border-[var(--tag)]/20">
        <svg className="w-4 h-4 text-[var(--tag)] shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
        </svg>
        <p className="text-sm text-[var(--tag)]">You&apos;re on the list. We&apos;ll be in touch.</p>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-2 max-w-md">
      <input
        type="email"
        required
        value={email}
        onChange={(e) => {
          setEmail(e.target.value);
          if (state === "error") setState("idle");
        }}
        placeholder="you@email.com"
        className="flex-1 px-4 py-2.5 rounded-md border border-[var(--border)] bg-[var(--background)] text-sm placeholder:text-[var(--muted)] focus:outline-none focus:border-[var(--accent)]/50 transition-colors"
      />
      <button
        type="submit"
        disabled={state === "loading"}
        className="px-5 py-2.5 text-sm font-medium bg-[var(--foreground)] text-[var(--background)] rounded-md hover:opacity-80 transition-opacity disabled:opacity-50 shrink-0"
      >
        {state === "loading" ? "Sending..." : "Get notified"}
      </button>
      {state === "error" && (
        <p className="text-xs text-[#b44] sm:self-center">Something went wrong. Try again.</p>
      )}
    </form>
  );
}
