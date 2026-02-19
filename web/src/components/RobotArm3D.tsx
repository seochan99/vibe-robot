"use client";

import { useRef, useState, useMemo } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Line } from "@react-three/drei";
import * as THREE from "three";

/* ── Wireframe robot arm ──────────────────────────────────────────────────── */

const LINE_COLOR = "#6b6b6b";
const ACCENT = "#c8441e";

function WireArm() {
  const groupRef = useRef<THREE.Group>(null);
  const shoulderRef = useRef<THREE.Group>(null);
  const elbowRef = useRef<THREE.Group>(null);
  const wristRef = useRef<THREE.Group>(null);
  const [grip, setGrip] = useState(0.25);
  const gripVal = useRef(0.25);

  useFrame((state) => {
    const t = state.clock.elapsedTime;

    if (groupRef.current) groupRef.current.rotation.y = Math.sin(t * 0.25) * 0.4;
    if (shoulderRef.current) shoulderRef.current.rotation.z = Math.sin(t * 0.35 + 0.5) * 0.12 - 0.15;
    if (elbowRef.current) elbowRef.current.rotation.z = Math.sin(t * 0.4 + 1.5) * 0.18 + 0.35;
    if (wristRef.current) wristRef.current.rotation.z = Math.sin(t * 0.5 + 2.5) * 0.1;

    const g = 0.2 + Math.sin(t * 0.55) * 0.12;
    if (Math.abs(gripVal.current - g) > 0.005) {
      gripVal.current = g;
      setGrip(g);
    }
  });

  return (
    <group ref={groupRef} position={[0, -1.6, 0]}>
      {/* Base rings */}
      <Ring y={0} radius={0.35} opacity={0.4} />
      <Ring y={0.04} radius={0.32} opacity={0.3} />

      {/* Base stem */}
      <Segment from={[0, 0.04, 0]} to={[0, 0.2, 0]} opacity={0.3} />
      <Joint position={[0, 0.2, 0]} radius={0.08} />

      {/* Link 1 */}
      <Segment from={[0, 0.2, 0]} to={[0, 0.85, 0]} />
      <WireBox position={[0, 0.52, 0]} size={[0.07, 0.55, 0.07]} />

      <group ref={shoulderRef} position={[0, 0.85, 0]}>
        <Joint position={[0, 0, 0]} radius={0.07} />

        {/* Link 2 */}
        <Segment from={[0, 0, 0]} to={[0, 0.7, 0]} />
        <WireBox position={[0, 0.35, 0]} size={[0.06, 0.6, 0.06]} />

        <group ref={elbowRef} position={[0, 0.7, 0]}>
          <Joint position={[0, 0, 0]} radius={0.06} />

          {/* Link 3 */}
          <Segment from={[0, 0, 0]} to={[0, 0.55, 0]} />
          <WireBox position={[0, 0.275, 0]} size={[0.05, 0.45, 0.05]} />

          <group ref={wristRef} position={[0, 0.55, 0]}>
            <Joint position={[0, 0, 0]} radius={0.05} />
            <Gripper angle={grip} />
          </group>
        </group>
      </group>
    </group>
  );
}

/* ── Primitives ───────────────────────────────────────────────────────────── */

function Ring({ y, radius, opacity = 0.4 }: { y: number; radius: number; opacity?: number }) {
  const pts = useMemo(() => {
    const arr: [number, number, number][] = [];
    for (let i = 0; i <= 48; i++) {
      const a = (i / 48) * Math.PI * 2;
      arr.push([Math.cos(a) * radius, y, Math.sin(a) * radius]);
    }
    return arr;
  }, [y, radius]);

  return <Line points={pts} color={LINE_COLOR} lineWidth={1} transparent opacity={opacity} />;
}

function Joint({ position, radius }: { position: [number, number, number]; radius: number }) {
  const rings = useMemo(() => {
    const result: [number, number, number][][] = [];
    const axes: ((a: number) => [number, number, number])[] = [
      (a) => [Math.cos(a) * radius, Math.sin(a) * radius, 0],
      (a) => [0, Math.cos(a) * radius, Math.sin(a) * radius],
      (a) => [Math.cos(a) * radius, 0, Math.sin(a) * radius],
    ];
    for (const fn of axes) {
      const pts: [number, number, number][] = [];
      for (let i = 0; i <= 32; i++) {
        const a = (i / 32) * Math.PI * 2;
        pts.push(fn(a));
      }
      result.push(pts);
    }
    return result;
  }, [radius]);

  return (
    <group position={position}>
      {rings.map((pts, i) => (
        <Line key={i} points={pts} color={ACCENT} lineWidth={1.2} transparent opacity={0.6} />
      ))}
    </group>
  );
}

function Segment({ from, to, opacity = 0.35 }: {
  from: [number, number, number]; to: [number, number, number]; opacity?: number;
}) {
  return <Line points={[from, to]} color={LINE_COLOR} lineWidth={1} transparent opacity={opacity} />;
}

function WireBox({ position, size }: {
  position: [number, number, number]; size: [number, number, number];
}) {
  return (
    <mesh position={position}>
      <boxGeometry args={size} />
      <meshBasicMaterial color={LINE_COLOR} wireframe transparent opacity={0.12} />
    </mesh>
  );
}

function Gripper({ angle }: { angle: number }) {
  const fingerLen = 0.18;
  return (
    <group position={[0, 0.03, 0]}>
      {/* Mount */}
      <mesh>
        <boxGeometry args={[0.1, 0.015, 0.05]} />
        <meshBasicMaterial color={LINE_COLOR} wireframe transparent opacity={0.2} />
      </mesh>

      {/* Left */}
      <group rotation={[0, 0, angle]} position={[-0.02, 0.01, 0]}>
        <mesh position={[0, fingerLen / 2, 0]}>
          <boxGeometry args={[0.012, fingerLen, 0.025]} />
          <meshBasicMaterial color={ACCENT} wireframe transparent opacity={0.4} />
        </mesh>
      </group>

      {/* Right */}
      <group rotation={[0, 0, -angle]} position={[0.02, 0.01, 0]}>
        <mesh position={[0, fingerLen / 2, 0]}>
          <boxGeometry args={[0.012, fingerLen, 0.025]} />
          <meshBasicMaterial color={ACCENT} wireframe transparent opacity={0.4} />
        </mesh>
      </group>
    </group>
  );
}

/* ── Grid floor ───────────────────────────────────────────────────────────── */

function GridFloor() {
  const lines = useMemo(() => {
    const result: [number, number, number][][] = [];
    const size = 2;
    const div = 10;
    for (let i = -div; i <= div; i++) {
      const p = (i / div) * size;
      result.push([
        [p, 0, -size],
        [p, 0, size],
      ]);
      result.push([
        [-size, 0, p],
        [size, 0, p],
      ]);
    }
    return result;
  }, []);

  return (
    <group position={[0, -1.6, 0]}>
      {lines.map((pts, i) => (
        <Line key={i} points={pts} color={LINE_COLOR} lineWidth={0.5} transparent opacity={0.06} />
      ))}
    </group>
  );
}

/* ── Camera rig ───────────────────────────────────────────────────────────── */

function CameraRig() {
  const { size } = useThree();
  useFrame((state) => {
    const targetZ = size.width / size.height < 1 ? 6 : 4.5;
    const cam = state.camera;
    const nextZ = cam.position.z + (targetZ - cam.position.z) * 0.05;
    cam.position.set(cam.position.x, cam.position.y, nextZ);
  });
  return null;
}

/* ── Export ────────────────────────────────────────────────────────────────── */

export default function RobotArm3D() {
  return (
    <div className="w-full h-full">
      <Canvas
        camera={{ position: [2, 1.2, 4.5], fov: 30 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
        dpr={[1, 2]}
      >
        <CameraRig />
        <GridFloor />
        <WireArm />
      </Canvas>
    </div>
  );
}
