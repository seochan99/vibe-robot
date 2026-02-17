"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { motion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";

const RobotArm3D = dynamic(() => import("@/components/RobotArm3D"), {
  ssr: false,
  loading: () => <div className="w-full h-full" />,
});

/* ── Data ───────────────────────────────────────────────────────────────────── */

const STEPS = [
  {
    label: "Speak",
    code: '"Prevent the papers from flying away!"',
    body: "No code, no menus. Tell the robot arm what you want in any language you think in.",
  },
  {
    label: "Infer",
    code: "say ≠ mean → Theory of Mind",
    body: 'You said "block." You meant "secure papers with a heavy object." The system bridges that gap.',
  },
  {
    label: "Discover",
    code: "book.affordance → weight_provider",
    body: "Gibson's ecological psychology: a book isn't for reading here — it's a weight. Context activates hidden capabilities.",
  },
  {
    label: "Verify",
    code: "pick(book) → place(book, on=papers)",
    body: "A safety contract checks workspace limits, velocity caps, and collision avoidance. Then you see it run in simulation.",
  },
  {
    label: "Approve",
    code: "user.approve() → execute()",
    body: "Nothing physical happens without your explicit approval. You always have the final say.",
  },
];

const CAPABILITIES = [
  ["Theory of Mind", "Bridges what you say and what you mean"],
  ["Affordance Discovery", "Finds hidden capabilities in everyday objects"],
  ["Safety Contracts", "Pre/invariant/post conditions with emergency tripwires"],
  ["MuJoCo Simulation", "Franka Panda + XArm at 500 Hz physics"],
  ["Voice & Text Input", "Speak or type commands in any language"],
  ["ChatGPT Auth", "Your existing subscription — no API key needed"],
];

const STACK = [
  "MuJoCo 3.5", "Franka Panda", "GPT-4o", "Python",
  "Next.js", "FastAPI", "TypeScript", "roboticstoolbox",
];

/* ── Animation ──────────────────────────────────────────────────────────────── */

const reveal = {
  hidden: { opacity: 0, y: 24 },
  visible: (d: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, delay: d * 0.1, ease: "easeOut" as const },
  }),
};

/* ── Page ───────────────────────────────────────────────────────────────────── */

export default function LandingPage() {
  const heroRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"],
  });
  const heroY = useTransform(scrollYProgress, [0, 1], [0, -120]);
  const heroOp = useTransform(scrollYProgress, [0, 0.6], [1, 0]);

  return (
    <div className="noise min-h-screen bg-[var(--background)] text-[var(--foreground)]">
      {/* ── Nav ─────────────────────────────────────────────────────────────── */}
      <nav className="fixed top-0 inset-x-0 z-50 backdrop-blur-md bg-[var(--background)]/80 border-b border-[var(--border)]">
        <div className="max-w-[1100px] mx-auto px-6 h-14 flex items-center justify-between">
          <Link href="/" className="font-mono text-sm font-semibold tracking-tight">
            VibeRobot<span className="text-[var(--accent)]">!</span>
          </Link>
          <div className="flex items-center gap-5">
            <Link
              href="#how"
              className="hidden sm:block text-[13px] text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
            >
              How it works
            </Link>
            <a
              href="https://github.com/seochan99/vibe-robot"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:block text-[13px] text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
            >
              Source
            </a>
            <Link
              href="/app"
              className="px-3.5 py-1.5 text-[13px] font-medium bg-[var(--foreground)] text-[var(--background)] rounded-md hover:opacity-80 transition-opacity"
            >
              Open app
            </Link>
          </div>
        </div>
      </nav>

      {/* ── Hero ────────────────────────────────────────────────────────────── */}
      <section ref={heroRef} className="relative min-h-[100svh] overflow-hidden">
        <div className="max-w-[1100px] mx-auto px-6 min-h-[100svh] grid grid-cols-1 lg:grid-cols-2 gap-8 items-end lg:items-center pt-20 pb-24 sm:pb-32 lg:pb-0">
          {/* Left — text */}
          <motion.div style={{ y: heroY, opacity: heroOp }} className="relative z-10">
            <motion.p
              initial="hidden" animate="visible" variants={reveal} custom={0}
              className="font-mono text-xs tracking-wider text-[var(--muted)] mb-6 uppercase"
            >
              Research Project
            </motion.p>

            <motion.h1
              initial="hidden" animate="visible" variants={reveal} custom={1}
              className="text-[clamp(2.5rem,7vw,5.5rem)] font-bold leading-[1.05] tracking-tight"
            >
              Talk to a robot arm.
              <br />
              <span className="text-[var(--muted)]">
                It verifies before it moves.
              </span>
            </motion.h1>

            <motion.p
              initial="hidden" animate="visible" variants={reveal} custom={3}
              className="mt-8 text-lg sm:text-xl text-[var(--muted)] max-w-xl leading-relaxed"
            >
              VibeRobot! lets you command a robot with natural language.
              The system infers your intent, discovers object affordances,
              and runs a full simulation — all before anything physical happens.
            </motion.p>

            <motion.div
              initial="hidden" animate="visible" variants={reveal} custom={5}
              className="mt-10 flex flex-wrap gap-3"
            >
              <Link
                href="/app"
                className="group inline-flex items-center gap-2 px-5 py-3 bg-[var(--foreground)] text-[var(--background)] text-sm font-medium rounded-md hover:opacity-80 transition-opacity"
              >
                Try the demo
                <span className="inline-block group-hover:translate-x-0.5 transition-transform">&rarr;</span>
              </Link>
              <a
                href="https://github.com/seochan99/vibe-robot"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-5 py-3 text-sm font-medium border border-[var(--border)] rounded-md hover:bg-[var(--surface)] transition-colors"
              >
                View on GitHub
              </a>
            </motion.div>
          </motion.div>

          {/* Right — 3D robot arm */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1.2, delay: 0.5 }}
            className="relative h-[400px] sm:h-[500px] lg:h-[600px] -mr-6 lg:mr-0"
          >
            <RobotArm3D />
          </motion.div>
        </div>
      </section>

      {/* ── The tension ─────────────────────────────────────────────────────── */}
      <section className="py-24 sm:py-32 px-6 border-t border-[var(--border)]">
        <div className="max-w-[1100px] mx-auto">
          <motion.div
            initial="hidden" whileInView="visible"
            viewport={{ once: true, margin: "-80px" }}
            variants={reveal}
            className="max-w-2xl"
          >
            <h2 className="text-3xl sm:text-4xl font-bold leading-tight tracking-tight">
              Vibe coding works for apps.
              <br />
              Robots are different.
            </h2>
            <p className="mt-5 text-[var(--muted)] text-lg leading-relaxed">
              A crashed web app is an undo away. A robot arm that swings into a table
              causes real damage. VibeRobot! adds a{" "}
              <strong className="text-[var(--foreground)] font-medium">Vibe-to-Verify</strong>{" "}
              layer — natural language in, simulation-checked motion out.
            </p>
          </motion.div>

          <motion.div
            initial="hidden" whileInView="visible"
            viewport={{ once: true, margin: "-40px" }}
            variants={{ visible: { transition: { staggerChildren: 0.08 } } }}
            className="mt-14 grid sm:grid-cols-2 gap-4"
          >
            {/* Without */}
            <motion.div variants={reveal} className="p-6 rounded-lg border border-[var(--border)] bg-[var(--surface)]">
              <p className="font-mono text-xs uppercase tracking-wider text-[var(--muted)] mb-4">Without verification</p>
              <div className="font-mono text-[13px] leading-relaxed space-y-1.5">
                <p className="text-[var(--muted)]">&gt; &quot;Move that thing over there&quot;</p>
                <p className="text-[#b44]">Arm swings immediately</p>
                <p className="text-[#b44]">Coffee mug knocked over</p>
                <p className="text-[#b44]">Physical damage, no undo</p>
              </div>
            </motion.div>

            {/* With */}
            <motion.div variants={reveal} className="p-6 rounded-lg border border-[var(--border)] bg-[var(--surface)]">
              <p className="font-mono text-xs uppercase tracking-wider text-[var(--muted)] mb-4">With VibeRobot!</p>
              <div className="font-mono text-[13px] leading-relaxed space-y-1.5">
                <p className="text-[var(--muted)]">&gt; &quot;Move that thing over there&quot;</p>
                <p className="text-[var(--tag)]">Intent: move mug to shelf</p>
                <p className="text-[var(--tag)]">Safety: fragile object → slow speed</p>
                <p className="text-[var(--tag)]">Sim preview → approve → safe move</p>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ── Pipeline steps ──────────────────────────────────────────────────── */}
      <section id="how" className="py-24 sm:py-32 px-6 border-t border-[var(--border)]">
        <div className="max-w-[1100px] mx-auto">
          <motion.div
            initial="hidden" whileInView="visible"
            viewport={{ once: true }}
            variants={reveal}
            className="mb-16"
          >
            <p className="font-mono text-xs uppercase tracking-wider text-[var(--muted)] mb-3">Pipeline</p>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Five steps, one loop</h2>
          </motion.div>

          <div className="space-y-0">
            {STEPS.map((step, i) => (
              <motion.div
                key={step.label}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-60px" }}
                variants={reveal}
                custom={0}
                className="step-line grid grid-cols-[48px_1fr] gap-6 pb-16"
              >
                {/* Number circle */}
                <div className="relative z-10 flex items-start">
                  <div className="w-12 h-12 rounded-full border-2 border-[var(--border)] bg-[var(--background)] flex items-center justify-center font-mono text-sm font-semibold text-[var(--muted)]">
                    {String(i + 1).padStart(2, "0")}
                  </div>
                </div>

                {/* Content */}
                <div className="pt-1.5">
                  <h3 className="text-xl font-bold mb-1">{step.label}</h3>
                  <code className="inline-block px-2.5 py-1 rounded text-[13px] font-mono bg-[var(--code-bg)] text-[var(--code-fg)] mb-3">
                    {step.code}
                  </code>
                  <p className="text-[var(--muted)] leading-relaxed max-w-lg">{step.body}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Live demo mockup ────────────────────────────────────────────────── */}
      <section className="py-24 sm:py-32 px-6 bg-[var(--surface)] border-t border-[var(--border)]">
        <div className="max-w-[1100px] mx-auto">
          <motion.div
            initial="hidden" whileInView="visible"
            viewport={{ once: true }}
            variants={reveal}
            className="mb-12"
          >
            <p className="font-mono text-xs uppercase tracking-wider text-[var(--muted)] mb-3">Example</p>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">The wind &amp; paper scenario</h2>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="rounded-lg border border-[var(--border)] bg-[var(--background)] overflow-hidden max-w-2xl"
          >
            {/* Terminal header */}
            <div className="flex items-center gap-1.5 px-4 py-2.5 border-b border-[var(--border)]">
              <div className="w-2.5 h-2.5 rounded-full bg-[var(--border)]" />
              <div className="w-2.5 h-2.5 rounded-full bg-[var(--border)]" />
              <div className="w-2.5 h-2.5 rounded-full bg-[var(--border)]" />
              <span className="ml-2 font-mono text-[11px] text-[var(--muted)]">viberobot — wind_paper scene</span>
            </div>

            <div className="p-5 sm:p-6 space-y-5 text-sm">
              {/* User */}
              <motion.div
                initial={{ opacity: 0, x: 12 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.15 }}
                className="flex justify-end"
              >
                <div className="px-4 py-2.5 rounded-lg rounded-br-sm bg-[var(--foreground)] text-[var(--background)] max-w-[280px]">
                  Prevent the papers from flying away!
                </div>
              </motion.div>

              {/* System response */}
              <motion.div
                initial={{ opacity: 0, x: -12 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.4 }}
                className="flex justify-start"
              >
                <div className="px-4 py-3 rounded-lg rounded-bl-sm border border-[var(--border)] max-w-sm space-y-2.5 font-mono text-[13px]">
                  <p>
                    <span className="text-[var(--muted)]">intent:</span>{" "}
                    Secure loose papers against wind
                  </p>
                  <p>
                    <span className="text-[var(--muted)]">affordance:</span>{" "}
                    book &rarr;{" "}
                    <span className="px-1.5 py-0.5 rounded text-xs bg-[var(--accent-soft)] text-[var(--accent)]">
                      weight_provider
                    </span>
                  </p>
                  <p>
                    <span className="text-[var(--muted)]">plan:</span>{" "}
                    <span className="text-[var(--code-fg)] bg-[var(--code-bg)] px-1.5 py-0.5 rounded text-xs">pick(book)</span>
                    {" "}&rarr;{" "}
                    <span className="text-[var(--code-fg)] bg-[var(--code-bg)] px-1.5 py-0.5 rounded text-xs">place(book, on=papers)</span>
                  </p>
                  <div className="flex gap-2 pt-1">
                    <span className="px-3 py-1 rounded bg-[var(--tag)] text-white text-xs font-medium">
                      Approve
                    </span>
                    <span className="px-3 py-1 rounded border border-[var(--border)] text-xs text-[var(--muted)]">
                      Reject
                    </span>
                  </div>
                </div>
              </motion.div>

              {/* Result */}
              <motion.div
                initial={{ opacity: 0, x: -12 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.7 }}
                className="flex justify-start"
              >
                <div className="px-4 py-2.5 rounded-lg rounded-bl-sm bg-[var(--tag-soft)] border border-[var(--tag)]/20 font-mono text-[13px] text-[var(--tag)]">
                  Done. Book placed on papers. Papers secured.
                </div>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── Capabilities ────────────────────────────────────────────────────── */}
      <section className="py-24 sm:py-32 px-6 border-t border-[var(--border)]">
        <div className="max-w-[1100px] mx-auto">
          <motion.div
            initial="hidden" whileInView="visible"
            viewport={{ once: true }}
            variants={reveal}
            className="mb-14"
          >
            <p className="font-mono text-xs uppercase tracking-wider text-[var(--muted)] mb-3">Capabilities</p>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">What&apos;s under the hood</h2>
          </motion.div>

          <motion.div
            initial="hidden" whileInView="visible"
            viewport={{ once: true }}
            variants={{ visible: { transition: { staggerChildren: 0.06 } } }}
            className="grid sm:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-6"
          >
            {CAPABILITIES.map(([title, desc]) => (
              <motion.div
                key={title}
                variants={reveal}
                className="py-4 border-t border-[var(--border)]"
              >
                <h3 className="font-medium text-[15px] mb-1">{title}</h3>
                <p className="text-sm text-[var(--muted)] leading-relaxed">{desc}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── Stack ───────────────────────────────────────────────────────────── */}
      <section className="py-20 px-6 border-t border-[var(--border)] bg-[var(--surface)]">
        <div className="max-w-[1100px] mx-auto">
          <motion.p
            initial="hidden" whileInView="visible"
            viewport={{ once: true }}
            variants={reveal}
            className="font-mono text-xs uppercase tracking-wider text-[var(--muted)] mb-5"
          >
            Stack
          </motion.p>
          <motion.div
            initial="hidden" whileInView="visible"
            viewport={{ once: true }}
            variants={{ visible: { transition: { staggerChildren: 0.04 } } }}
            className="flex flex-wrap gap-2"
          >
            {STACK.map((t) => (
              <motion.span
                key={t}
                variants={reveal}
                className="px-3 py-1.5 rounded font-mono text-[13px] border border-[var(--border)] bg-[var(--background)] text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
              >
                {t}
              </motion.span>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── CTA ─────────────────────────────────────────────────────────────── */}
      <section className="py-24 sm:py-32 px-6 border-t border-[var(--border)]">
        <motion.div
          initial="hidden" whileInView="visible"
          viewport={{ once: true }}
          variants={{ visible: { transition: { staggerChildren: 0.1 } } }}
          className="max-w-[1100px] mx-auto"
        >
          <motion.h2
            variants={reveal}
            className="text-4xl sm:text-5xl font-bold tracking-tight mb-5"
          >
            See it yourself.
          </motion.h2>
          <motion.p
            variants={reveal}
            className="text-lg text-[var(--muted)] mb-8 max-w-md"
          >
            Open the app, explore the demo scene, connect ChatGPT when you&apos;re ready for full AI analysis.
          </motion.p>
          <motion.div variants={reveal} className="flex flex-wrap gap-3">
            <Link
              href="/app"
              className="group inline-flex items-center gap-2 px-6 py-3 bg-[var(--foreground)] text-[var(--background)] font-medium rounded-md hover:opacity-80 transition-opacity"
            >
              Launch VibeRobot!
              <span className="inline-block group-hover:translate-x-0.5 transition-transform">&rarr;</span>
            </Link>
          </motion.div>
        </motion.div>
      </section>

      {/* ── Footer ──────────────────────────────────────────────────────────── */}
      <footer className="border-t border-[var(--border)] py-6 px-6">
        <div className="max-w-[1100px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <p className="font-mono text-xs text-[var(--muted)]">
            VibeRobot<span className="text-[var(--accent)]">!</span> &mdash; Vibe-to-Verify for Safe Robotic Manipulation
          </p>
          <div className="flex items-center gap-5 font-mono text-xs text-[var(--muted)]">
            <a
              href="https://github.com/seochan99/vibe-robot"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-[var(--foreground)] transition-colors"
            >
              GitHub
            </a>
            <span>Research</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
