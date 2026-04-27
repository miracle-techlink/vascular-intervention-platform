import React, { useRef, useState, useEffect } from "react";
import { Canvas, useLoader, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader";

const COLORS = {
  aorta:                       "#e03232",
  common_carotid_artery_left:  "#29b6f6",
  common_carotid_artery_right: "#4db6ac",
  brachiocephalic_trunk:       "#ffb300",
  subclavian_artery_left:      "#ab47bc",
  subclavian_artery_right:     "#ce93d8",
};

// Fit camera + orbit target to scene bounding box after vessels load
function SceneSetup({ orbitRef }) {
  const { camera, scene } = useThree();
  const fitted = useRef(false);

  useFrame(() => {
    if (fitted.current) return;
    const box = new THREE.Box3();
    scene.traverse(obj => { if (obj.isMesh) box.expandByObject(obj); });
    if (box.isEmpty()) return;

    const center = box.getCenter(new THREE.Vector3());
    const size   = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    camera.position.set(center.x + maxDim, center.y, center.z + maxDim * 0.8);
    camera.lookAt(center);
    camera.near = 1;
    camera.far  = maxDim * 20;
    camera.updateProjectionMatrix();

    if (orbitRef.current) {
      orbitRef.current.target.copy(center);
      orbitRef.current.update();
    }
    fitted.current = true;
  });
  return null;
}

function VesselMesh({ caseId, vessel }) {
  const obj = useLoader(OBJLoader, `/cases/${caseId}/mesh/${vessel}`);
  useEffect(() => {
    obj.traverse(child => {
      if (!child.isMesh) return;
      child.material = new THREE.MeshStandardMaterial({
        color: COLORS[vessel] ?? "#888",
        transparent: true, opacity: 0.32,
        side: THREE.DoubleSide, depthWrite: false,
      });
    });
  }, [obj]);
  return <primitive object={obj} dispose={null} />;
}

function PulsingMarker({ position, color, label }) {
  const ref = useRef();
  useFrame(({ clock }) => {
    if (ref.current)
      ref.current.scale.setScalar(1 + 0.25 * Math.sin(clock.getElapsedTime() * 3));
  });
  const [x, y, z] = position;
  return (
    <group position={[x, y, z]}>
      <mesh ref={ref}>
        <sphereGeometry args={[5, 16, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.5} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[9, 1, 8, 32]} />
        <meshStandardMaterial color={color} transparent opacity={0.5} />
      </mesh>
      <Html distanceFactor={400} style={{ pointerEvents: "none" }}>
        <div style={{
          background: "rgba(0,0,0,0.75)", color, border: `1px solid ${color}`,
          padding: "2px 8px", borderRadius: 4, fontSize: 11,
          fontFamily: "monospace", whiteSpace: "nowrap",
        }}>{label}</div>
      </Html>
    </group>
  );
}

// Animated catheter — supports external seek via seekStep prop
function Catheter({ allPoints, playing, speed, onDone, resetKey, seekStep, onStepChange }) {
  const tubeRef = useRef();
  const tipRef  = useRef();
  const step    = useRef(0);
  const clock   = useRef(0);

  useEffect(() => { step.current = 0; clock.current = 0; }, [resetKey]);

  // External seek: jump to specific step
  useEffect(() => {
    if (seekStep == null) return;
    step.current = Math.max(0, Math.min(seekStep, (allPoints?.length ?? 1) - 1));
    clock.current = 0;
    _render(step.current);
  }, [seekStep]);

  function _render(s) {
    if (!allPoints || allPoints.length < 2) return;
    const pts = allPoints.slice(0, s + 1);
    if (tipRef.current) {
      const [x, y, z] = pts[pts.length - 1];
      tipRef.current.position.set(x, y, z);
    }
    if (tubeRef.current && pts.length >= 2) {
      const curve = new THREE.CatmullRomCurve3(
        pts.map(([x, y, z]) => new THREE.Vector3(x, y, z))
      );
      const geo = new THREE.TubeGeometry(curve, Math.min(pts.length * 2, 400), 2.8, 8, false);
      tubeRef.current.geometry.dispose();
      tubeRef.current.geometry = geo;
    }
  }

  useFrame((_, delta) => {
    if (!playing || !allPoints || allPoints.length < 2) return;
    clock.current += delta;
    if (clock.current < speed) return;
    clock.current = 0;

    step.current = Math.min(step.current + 1, allPoints.length - 1);
    _render(step.current);
    onStepChange?.(step.current);
    if (step.current >= allPoints.length - 1) onDone?.();
  });

  if (!allPoints || allPoints.length < 2) return null;
  const [sx, sy, sz] = allPoints[0];

  return (
    <group>
      <mesh ref={tubeRef}>
        <tubeGeometry args={[
          new THREE.CatmullRomCurve3([
            new THREE.Vector3(sx, sy, sz),
            new THREE.Vector3(sx, sy, sz + 0.1),
          ]), 2, 2.8, 8, false
        ]} />
        <meshStandardMaterial color="#00e5ff" emissive="#00bcd4" emissiveIntensity={0.9} />
      </mesh>
      <mesh ref={tipRef} position={[sx, sy, sz]}>
        <sphereGeometry args={[4.5, 12, 12]} />
        <meshStandardMaterial color="#fff" emissive="#00e5ff" emissiveIntensity={2.5} />
      </mesh>
    </group>
  );
}

export default function Viewer3D({ caseId, vessels, allPathPoints, nav }) {
  const orbitRef   = useRef();
  const [playing,  setPlaying]  = useState(false);
  const [done,     setDone]     = useState(false);
  const [speed,    setSpeed]    = useState(0.04);
  const [resetKey, setResetKey] = useState(0);
  const [curStep,  setCurStep]  = useState(0);
  const [seekStep, setSeekStep] = useState(null);
  const seekRef = useRef(null);

  const hasPath = allPathPoints && allPathPoints.length >= 2;
  const total   = allPathPoints?.length ?? 1;

  // auto-play when path arrives
  useEffect(() => {
    if (!hasPath) return;
    setDone(false); setCurStep(0); setSeekStep(null);
    setResetKey(k => k + 1);
    setPlaying(true);
  }, [allPathPoints]);

  const replay = () => {
    setDone(false); setCurStep(0);
    setResetKey(k => k + 1); setPlaying(true);
  };
  const pause = () => setPlaying(false);

  // scrub: pause and jump
  function handleScrub(val) {
    setPlaying(false);
    const s = parseInt(val);
    setCurStep(s);
    seekRef.current = s;
    setSeekStep(s);
  }

  return (
    <div style={{ width: "100%", height: "100%", position: "relative", background: "#0d1117" }}>

      <Canvas
        style={{ position: "absolute", inset: 0 }}
        camera={{ fov: 45, near: 1, far: 50000 }}
        gl={{ antialias: true }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[300, 400, 300]} intensity={1.5} />
        <directionalLight position={[-200, -100, -300]} intensity={0.4} />

        {vessels.map(v => <VesselMesh key={v} caseId={caseId} vessel={v} />)}

        {nav?.target_lcca && (
          <PulsingMarker position={nav.target_lcca} color="#ff4444" label="LCCA Target" />
        )}
        {nav?.insertion_position && (
          <PulsingMarker position={nav.insertion_position} color="#00ff88" label="Insertion" />
        )}

        {hasPath && (
          <Catheter
            allPoints={allPathPoints}
            playing={playing}
            speed={speed}
            resetKey={resetKey}
            seekStep={seekStep}
            onStepChange={setCurStep}
            onDone={() => { setPlaying(false); setDone(true); setCurStep(total - 1); }}
          />
        )}

        <SceneSetup orbitRef={orbitRef} />
        <OrbitControls ref={orbitRef} makeDefault enableDamping dampingFactor={0.08} />
      </Canvas>

      {/* ── controls overlay ── */}
      {hasPath && (
        <div style={{
          position: "absolute", bottom: 16, left: "50%", transform: "translateX(-50%)",
          display: "flex", flexDirection: "column", gap: 6, alignItems: "stretch",
          background: "rgba(0,0,0,0.85)", padding: "10px 16px", borderRadius: 10,
          border: "1px solid #374151", zIndex: 10, pointerEvents: "auto",
          userSelect: "none", minWidth: 340,
        }}>
          {/* 进度条 scrubber */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 10, color: "#6b7280", minWidth: 28, textAlign: "right" }}>
              {curStep}
            </span>
            <input
              type="range" min="0" max={total - 1} step="1" value={curStep}
              onChange={e => handleScrub(e.target.value)}
              style={{ flex: 1, cursor: "pointer", accentColor: "#3b82f6" }}
            />
            <span style={{ fontSize: 10, color: "#6b7280", minWidth: 28 }}>
              {total - 1}
            </span>
          </div>

          {/* 控制按钮 + 速度 */}
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button
              onClick={() => { setDone(false); setCurStep(0); setResetKey(k => k + 1); setPlaying(false); }}
              style={btn("#374151")}>⏮
            </button>
            {playing
              ? <button onClick={pause} style={btn("#1e40af")}>⏸ 暂停</button>
              : <button onClick={replay} style={btn(done ? "#065f46" : "#1e40af")}>
                  {done ? "↺ 重播" : "▶ 播放"}
                </button>
            }
            <div style={{ flex: 1 }} />
            <span style={{ fontSize: 11, color: "#6b7280" }}>速度</span>
            <input type="range" min="0.01" max="0.2" step="0.01" value={speed}
                   onChange={e => setSpeed(+e.target.value)}
                   style={{ width: 80, cursor: "pointer", accentColor: "#60a5fa" }} />
            <span style={{ fontSize: 10, color: "#4b5563", minWidth: 30 }}>
              {(1 / speed).toFixed(0)}x
            </span>
          </div>
        </div>
      )}

      {!hasPath && vessels.length > 0 && (
        <div style={{
          position: "absolute", bottom: 16, left: "50%", transform: "translateX(-50%)",
          fontSize: 11, color: "#374151", zIndex: 10, pointerEvents: "none",
        }}>
          点击左侧「运行推理」查看导管路径回放
        </div>
      )}
    </div>
  );
}

const btn = bg => ({
  padding: "5px 12px", background: bg, color: "#fff",
  border: "none", borderRadius: 4, cursor: "pointer", fontSize: 12,
});
