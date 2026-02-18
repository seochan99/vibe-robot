import type { SimCameraState } from "@/lib/api";

export function msgId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

export function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

export function projectViewToPlane(
  nx: number,
  ny: number,
  camera: SimCameraState,
  planeZ: number,
  aspect: number,
  fovYDeg = 45,
): [number, number, number] | null {
  const az = (camera.azimuth * Math.PI) / 180;
  const el = (camera.elevation * Math.PI) / 180;
  const dist = Math.max(0.4, camera.distance);
  const lookat = camera.lookat;

  const fx = Math.cos(el) * Math.cos(az);
  const fy = Math.cos(el) * Math.sin(az);
  const fz = Math.sin(el);
  const forward: [number, number, number] = [fx, fy, fz];
  const camPos: [number, number, number] = [
    lookat[0] - forward[0] * dist,
    lookat[1] - forward[1] * dist,
    lookat[2] - forward[2] * dist,
  ];

  // right = forward x worldUp
  let rx = forward[1];
  let ry = -forward[0];
  let rz = 0;
  const rNorm = Math.hypot(rx, ry, rz);
  if (rNorm < 1e-8) {
    rx = 1;
    ry = 0;
    rz = 0;
  } else {
    rx /= rNorm;
    ry /= rNorm;
    rz /= rNorm;
  }

  // up = right x forward
  let ux = ry * forward[2] - rz * forward[1];
  let uy = rz * forward[0] - rx * forward[2];
  let uz = rx * forward[1] - ry * forward[0];
  const uNorm = Math.hypot(ux, uy, uz);
  if (uNorm > 1e-8) {
    ux /= uNorm;
    uy /= uNorm;
    uz /= uNorm;
  }

  const xNdc = (nx - 0.5) * 2;
  const yNdc = (0.5 - ny) * 2;
  const tanHalf = Math.tan((fovYDeg * Math.PI / 180) * 0.5);

  let dx = forward[0] + rx * xNdc * tanHalf * aspect + ux * yNdc * tanHalf;
  let dy = forward[1] + ry * xNdc * tanHalf * aspect + uy * yNdc * tanHalf;
  let dz = forward[2] + rz * xNdc * tanHalf * aspect + uz * yNdc * tanHalf;
  const dNorm = Math.hypot(dx, dy, dz);
  if (dNorm < 1e-8) return null;
  dx /= dNorm;
  dy /= dNorm;
  dz /= dNorm;

  if (Math.abs(dz) < 1e-8) return null;
  const t = (planeZ - camPos[2]) / dz;
  if (t <= 0) return null;

  return [camPos[0] + dx * t, camPos[1] + dy * t, planeZ];
}
