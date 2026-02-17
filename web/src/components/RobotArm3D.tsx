"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float } from "@react-three/drei";
import * as THREE from "three";

/* ── Joint (sphere at each articulation) ──────────────────────────────────── */

function Joint({ position, size = 0.12 }: { position: [number, number, number]; size?: number }) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[size, 16, 16]} />
      <meshStandardMaterial
        color="#c8441e"
        roughness={0.6}
        metalness={0.3}
      />
    </mesh>
  );
}

/* ── Link (cylinder between two joints) ───────────────────────────────────── */

function Link({
  start,
  end,
  thickness = 0.06,
}: {
  start: [number, number, number];
  end: [number, number, number];
  thickness?: number;
}) {
  const ref = useRef<THREE.Mesh>(null);

  const { position, quaternion, length } = useMemo(() => {
    const s = new THREE.Vector3(...start);
    const e = new THREE.Vector3(...end);
    const mid = s.clone().add(e).multiplyScalar(0.5);
    const dir = e.clone().sub(s);
    const len = dir.length();
    dir.normalize();

    const q = new THREE.Quaternion();
    q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);

    return {
      position: [mid.x, mid.y, mid.z] as [number, number, number],
      quaternion: q,
      length: len,
    };
  }, [start, end]);

  return (
    <mesh ref={ref} position={position} quaternion={quaternion}>
      <cylinderGeometry args={[thickness, thickness, length, 12]} />
      <meshStandardMaterial
        color="#e8e4df"
        roughness={0.8}
        metalness={0.1}
      />
    </mesh>
  );
}

/* ── Gripper fingers ──────────────────────────────────────────────────────── */

function Gripper({ position, openAngle }: { position: [number, number, number]; openAngle: number }) {
  const len = 0.25;
  return (
    <group position={position}>
      {/* Left finger */}
      <group rotation={[0, 0, openAngle]}>
        <mesh position={[0, len / 2, 0]}>
          <boxGeometry args={[0.03, len, 0.05]} />
          <meshStandardMaterial color="#c8441e" roughness={0.6} metalness={0.3} />
        </mesh>
      </group>
      {/* Right finger */}
      <group rotation={[0, 0, -openAngle]}>
        <mesh position={[0, len / 2, 0]}>
          <boxGeometry args={[0.03, len, 0.05]} />
          <meshStandardMaterial color="#c8441e" roughness={0.6} metalness={0.3} />
        </mesh>
      </group>
    </group>
  );
}

/* ── Animated arm assembly ────────────────────────────────────────────────── */

function Arm() {
  const groupRef = useRef<THREE.Group>(null);
  const j2Ref = useRef<THREE.Group>(null);
  const j3Ref = useRef<THREE.Group>(null);
  const gripRef = useRef<number>(0);
  const [gripAngle, setGripAngle] = useState(0.2);

  useFrame((state) => {
    const t = state.clock.elapsedTime;

    // Gentle base rotation
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(t * 0.3) * 0.3 + 0.2;
    }

    // Shoulder sway
    if (j2Ref.current) {
      j2Ref.current.rotation.z = Math.sin(t * 0.4 + 1) * 0.15 - 0.2;
    }

    // Elbow flex
    if (j3Ref.current) {
      j3Ref.current.rotation.z = Math.sin(t * 0.5 + 2) * 0.2 + 0.4;
    }

    // Gripper open/close
    const g = 0.15 + Math.sin(t * 0.6) * 0.1;
    if (gripRef.current !== g) {
      gripRef.current = g;
      setGripAngle(g);
    }
  });

  return (
    <group ref={groupRef} position={[0, -1.2, 0]}>
      {/* Base platform */}
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.35, 0.4, 0.15, 24]} />
        <meshStandardMaterial color="#2a2a28" roughness={0.9} metalness={0.1} />
      </mesh>

      {/* Base column */}
      <Joint position={[0, 0.15, 0]} size={0.14} />
      <Link start={[0, 0.15, 0]} end={[0, 0.7, 0]} thickness={0.08} />

      {/* Shoulder joint */}
      <group ref={j2Ref} position={[0, 0.7, 0]}>
        <Joint position={[0, 0, 0]} size={0.13} />
        <Link start={[0, 0, 0]} end={[0, 0.65, 0]} thickness={0.065} />

        {/* Elbow joint */}
        <group ref={j3Ref} position={[0, 0.65, 0]}>
          <Joint position={[0, 0, 0]} size={0.11} />
          <Link start={[0, 0, 0]} end={[0, 0.5, 0]} thickness={0.05} />

          {/* Wrist */}
          <Joint position={[0, 0.5, 0]} size={0.09} />

          {/* Gripper */}
          <Gripper position={[0, 0.55, 0]} openAngle={gripAngle} />
        </group>
      </group>
    </group>
  );
}

/* We need useState for the gripper */
import { useState } from "react";

/* ── Exported canvas component ────────────────────────────────────────────── */

export default function RobotArm3D() {
  return (
    <div className="w-full h-full">
      <Canvas
        camera={{ position: [2.5, 1.5, 3], fov: 35 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 8, 5]} intensity={0.8} color="#faf9f7" />
        <directionalLight position={[-3, 4, -2]} intensity={0.3} color="#e8e4df" />

        <Float
          speed={1.2}
          rotationIntensity={0.1}
          floatIntensity={0.3}
          floatingRange={[-0.05, 0.05]}
        >
          <Arm />
        </Float>
      </Canvas>
    </div>
  );
}
