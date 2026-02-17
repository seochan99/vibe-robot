#!/usr/bin/env node
/**
 * Auth relay server — runs on port 1455.
 *
 * OpenAI's Codex CLI OAuth client only allows redirect_uri = localhost:1455.
 * This tiny server catches that redirect and forwards the auth code
 * to the Next.js app on port 3000.
 */
import { createServer } from "node:http";

const NEXT_PORT = process.env.NEXT_PORT || 3000;
const RELAY_PORT = 1455;

const server = createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${RELAY_PORT}`);

  if (url.pathname === "/auth/callback") {
    // Forward all query params to the Next.js callback page
    const target = new URL(`http://localhost:${NEXT_PORT}/callback`);
    for (const [k, v] of url.searchParams) {
      target.searchParams.set(k, v);
    }
    res.writeHead(302, { Location: target.toString() });
    res.end();
    return;
  }

  res.writeHead(404, { "Content-Type": "text/plain" });
  res.end("VibeRobot auth relay — nothing here.");
});

server.listen(RELAY_PORT, () => {
  console.log(`Auth relay listening on http://localhost:${RELAY_PORT}`);
});
