import Link from "next/link";

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--background)] text-[var(--foreground)] px-6">
      <div className="max-w-md text-center">
        <p className="font-mono text-[120px] font-bold leading-none text-[var(--border)] select-none">
          404
        </p>
        <h1 className="mt-4 text-2xl font-bold tracking-tight">
          Page not found
        </h1>
        <p className="mt-3 text-[var(--muted)] leading-relaxed">
          The robot arm looked everywhere, but this page doesn&apos;t exist.
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <Link
            href="/"
            className="px-5 py-2.5 text-sm font-medium bg-[var(--foreground)] text-[var(--background)] rounded-md hover:opacity-80 transition-opacity"
          >
            Back to home
          </Link>
          <Link
            href="/app"
            className="px-5 py-2.5 text-sm font-medium border border-[var(--border)] rounded-md hover:bg-[var(--surface)] transition-colors"
          >
            Open app
          </Link>
        </div>
      </div>
    </div>
  );
}
