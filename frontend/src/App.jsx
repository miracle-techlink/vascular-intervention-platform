import React, { useState, useRef, useEffect, Suspense } from "react";
import Viewer3D from "./Viewer3D";

const S = {
  root: { display: "flex", height: "100vh", overflow: "hidden", fontFamily: "monospace" },
  sidebar: { width: 280, background: "#111827", display: "flex", flexDirection: "column",
             padding: 14, gap: 10, overflowY: "auto", borderRight: "1px solid #1f2937" },
  main: { flex: 1 },
  sec: { borderTop: "1px solid #1f2937", paddingTop: 10 },
  label: { fontSize: 11, color: "#6b7280", marginBottom: 6 },
  title: { fontSize: 13, fontWeight: "bold", color: "#60a5fa", marginBottom: 10 },
  btn: (c) => ({
    padding: "6px 10px", background: c || "#1e40af", color: "#fff",
    border: "none", borderRadius: 4, cursor: "pointer", fontSize: 11, width: "100%",
  }),
  btnDis: { padding: "6px 10px", background: "#1f2937", color: "#4b5563",
            border: "none", borderRadius: 4, fontSize: 11, width: "100%", cursor: "not-allowed" },
  log: { background: "#0d1117", border: "1px solid #1f2937", borderRadius: 4,
         padding: 8, fontSize: 10, color: "#9ca3af", height: 100, overflowY: "auto",
         whiteSpace: "pre-wrap" },
  dot: (s) => ({
    display: "inline-block", width: 7, height: 7, borderRadius: "50%", marginRight: 5,
    background: s === "ready" ? "#22c55e" : s === "running" ? "#f59e0b" : "#374151",
  }),
  dbItem: (sel) => ({
    padding: "5px 8px", fontSize: 11, cursor: "pointer", borderRadius: 3,
    background: sel ? "#1e3a5f" : "transparent", color: sel ? "#93c5fd" : "#d1d5db",
    display: "flex", justifyContent: "space-between", alignItems: "center",
  }),
  dbWrap: { maxHeight: 200, overflowY: "auto", border: "1px solid #1f2937",
            borderRadius: 4, background: "#0d1117" },
  badge: (ok) => ({
    fontSize: 9, padding: "1px 5px", borderRadius: 3,
    background: ok ? "#064e3b" : "#1f2937", color: ok ? "#6ee7b7" : "#4b5563",
  }),
  info: { fontSize: 10, color: "#6b7280" },
  dropzone: { border: "2px dashed #1f2937", borderRadius: 6, padding: 14,
              textAlign: "center", fontSize: 11, color: "#4b5563", cursor: "pointer" },
};

const PHASES = ["preprocess", "segment", "mesh", "nav"];

export default function App() {
  const [caseId, setCaseId] = useState(null);
  const [phase, setPhase] = useState({});
  const [vessels, setVessels] = useState([]);
  const [nav, setNav] = useState(null);
  const [log, setLog] = useState("");
  const [pathPoints, setPathPoints] = useState(null);
  const [inferStats, setInferStats] = useState(null);
  const [inferring, setInferring] = useState(false);
  const [inferStep, setInferStep] = useState(0);
  const [running, setRunning] = useState(false);
  const [library, setLibrary] = useState([]);
  const [selectedLib, setSelectedLib] = useState(null);
  const [planning, setPlanning] = useState(false);
  const [maxSteps, setMaxSteps] = useState(800);
  const [threshMm, setThreshMm] = useState(15);
  const [friction, setFriction] = useState(0.01);
  const logRef = useRef(null);

  useEffect(() => {
    fetch("/library").then(r => r.json()).then(setLibrary).catch(() => {});
  }, []);

  function appendLog(msg) {
    setLog(l => l + msg + "\n");
    setTimeout(() => logRef.current && (logRef.current.scrollTop = 99999), 10);
  }

  function resetCase() {
    setPhase({}); setVessels([]); setNav(null);
    setPathPoints(null); setInferStats(null); setLog("");
  }

  async function onUpload(e) {
    const file = e.target.files?.[0] || e.dataTransfer?.files?.[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/cases/upload", { method: "POST", body: fd });
    const { case_id } = await res.json();
    resetCase();
    setCaseId(case_id);
    setSelectedLib(null);
    appendLog(`[upload] ${case_id}  ${file.name}`);
  }

  async function loadFromLibrary(entry) {
    setSelectedLib(entry.name);
    resetCase();
    const res = await fetch(`/library/${entry.name}/load`, { method: "POST" });
    const data = await res.json();
    setCaseId(data.case_id);
    appendLog(`[library] loaded ${entry.name}`);
    if (data.cached) {
      // already built — fetch status directly
      const st = await fetch(`/cases/${data.case_id}/status`).then(r => r.json());
      setVessels(st.vessels || []);
      const nv = await fetch(`/cases/${data.case_id}/nav`).then(r => r.json()).catch(() => null);
      setNav(nv);
      setPhase({ preprocess: "done", segment: "done", mesh: "done", nav: "done" });
      appendLog(`[library] cache hit — meshes ready`);
    }
  }

  async function planPath() {
    if (!caseId) return;
    setPlanning(true);
    setPathPoints(null);
    appendLog("[path] 计算血管中心线最短路径...");
    try {
      const res = await fetch(`/cases/${caseId}/planned_path`);
      const data = await res.json();
      setPathPoints(data.path_points);
      appendLog(`[path] 完成  ${data.n_points} 个路径点`);
    } catch (e) {
      appendLog(`[path] 错误: ${e.message}`);
    }
    setPlanning(false);
  }

  function runPipeline() {
    if (!caseId) return;
    setRunning(true);
    const es = new EventSource(`/cases/${caseId}/run`);
    es.onmessage = (e) => {
      const d = JSON.parse(e.data);
      appendLog(`[${d.phase}] ${d.msg}`);
      setPhase(p => ({ ...p, [d.phase]: d.msg }));
      if (d.phase === "ready") {
        const info = JSON.parse(d.msg);
        setVessels(info.vessels);
        setNav(info.nav);
        setRunning(false);
        es.close();
        setLibrary(prev => prev.map(e => e.name === caseId ? { ...e, status: "ready" } : e));
      }
      if (d.phase === "error") { setRunning(false); es.close(); }
    };
    es.onerror = () => { setRunning(false); es.close(); };
  }

  function runInfer() {
    if (!caseId || !nav) return;
    setPathPoints(null);
    setInferStats(null);
    setInferring(true);
    setInferStep(0);
    appendLog("\n[infer] 初始化 SOFA 环境 (~20s)...");
    const es = new EventSource(`/cases/${caseId}/infer?max_steps=${maxSteps}&threshold_mm=${threshMm}&friction=${friction}`);
    es.onmessage = (e) => {
      const d = JSON.parse(e.data);
      if (d.log !== undefined) {
        // stderr line from SOFA — show in log panel
        appendLog(`[sofa] ${d.log}`);
      } else if (d.error !== undefined) {
        appendLog(`[error] ${d.error}`);
        setInferring(false);
        es.close();
      } else if (d.final) {
        setInferStats(d);
        setPathPoints(d.path_points);
        setInferring(false);
        appendLog(`[infer] 完成  dist=${d.dist_to_lcca_mm}mm  reached=${d.target_reached}  steps=${d.n_steps}`);
        es.close();
      } else if (d.step !== undefined) {
        setInferStep(d.step);
        setPathPoints(prev => [...(prev || []), d.tip]);
        if (d.step === 1) appendLog("[infer] SOFA 已启动，推理中...");
      }
    };
    es.onerror = () => { setInferring(false); es.close(); };
  }

  const pipelineDone = PHASES.every(p => phase[p]);

  return (
    <div style={S.root}>
      <div style={S.sidebar}>
        <div style={S.title}>血管介入仿真平台</div>

        {/* ── 内置数据库 ── */}
        <div>
          <div style={S.label}>内置数据库 ({library.length} 例)</div>
          <div style={S.dbWrap}>
            {library.length === 0 && (
              <div style={{ padding: 8, fontSize: 11, color: "#4b5563" }}>加载中...</div>
            )}
            {library.map(entry => (
              <div key={entry.name}
                   style={S.dbItem(selectedLib === entry.name)}
                   onClick={() => loadFromLibrary(entry)}>
                <span>{entry.name}</span>
                <span style={S.badge(entry.status === "ready")}>
                  {entry.status === "ready" ? "已建模" : "未建模"}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* ── 上传 ── */}
        <div style={S.sec}>
          <div style={S.label}>或上传 CT 文件</div>
          <label>
            <div style={S.dropzone}
                 onDragOver={e => e.preventDefault()}
                 onDrop={e => { e.preventDefault(); onUpload(e); }}>
              {caseId && !selectedLib ? `✓ ${caseId}` : "拖拽 .nrrd / .nii.gz"}
            </div>
            <input type="file" accept=".nrrd,.nii,.gz" style={{ display: "none" }} onChange={onUpload} />
          </label>
        </div>

        {/* ── 流水线 ── */}
        <div style={S.sec}>
          <div style={S.label}>处理流水线</div>
          {PHASES.map(p => (
            <div key={p} style={{ fontSize: 11, marginBottom: 3 }}>
              <span style={S.dot(phase[p] ? "ready" : running ? "running" : "")} />
              {p}
            </div>
          ))}
          <button
            style={caseId && !running && !pipelineDone ? S.btn() : S.btnDis}
            onClick={runPipeline}
            disabled={!caseId || running || pipelineDone}>
            {running ? "运行中..." : pipelineDone ? "✓ 已完成" : "▶ 运行流水线"}
          </button>
        </div>

        {/* ── 导航点 + 路径规划 ── */}
        {nav && (
          <div style={S.sec}>
            <div style={S.label}>导航参数</div>
            <div style={S.info}>插入点: [{nav.insertion_position?.map(v => v.toFixed(0)).join(", ")}]</div>
            <div style={S.info}>LCCA:  [{nav.target_lcca?.map(v => v.toFixed(0)).join(", ")}]</div>
            <div style={S.info}>血管: {vessels.length} 个</div>
            <button
              style={!planning ? S.btn("#7c3aed") : S.btnDis}
              onClick={planPath}
              disabled={planning}>
              {planning ? "⏳ 路径规划中..." : "🔍 最短路径规划"}
            </button>
            {pathPoints && (
              <div style={{...S.info, color: "#a78bfa", marginTop: 4}}>
                ✓ 路径 {pathPoints.length} 点 — 点击播放查看动画
              </div>
            )}
          </div>
        )}

        {/* ── SAC 推理 ── */}
        <div style={S.sec}>
          <div style={S.label}>SAC 路径推理（SOFA 物理仿真）</div>

          {/* 参数面板 */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 8px", marginBottom: 8 }}>
            <div style={S.label}>最大步数</div>
            <div style={S.label}>到达阈值 (mm)</div>
            <input type="number" min="100" max="3000" step="100" value={maxSteps}
              onChange={e => setMaxSteps(+e.target.value)} disabled={inferring}
              style={{ background: "#0d1117", border: "1px solid #374151", borderRadius: 3,
                       color: "#d1d5db", fontSize: 11, padding: "2px 5px", width: "100%" }} />
            <input type="number" min="5" max="50" step="1" value={threshMm}
              onChange={e => setThreshMm(+e.target.value)} disabled={inferring}
              style={{ background: "#0d1117", border: "1px solid #374151", borderRadius: 3,
                       color: "#d1d5db", fontSize: 11, padding: "2px 5px", width: "100%" }} />
            <div style={{ ...S.label, gridColumn: "span 2" }}>摩擦系数</div>
            <div style={{ gridColumn: "span 2", display: "flex", alignItems: "center", gap: 6 }}>
              <input type="range" min="0.001" max="0.1" step="0.001" value={friction}
                onChange={e => setFriction(+e.target.value)} disabled={inferring}
                style={{ flex: 1, cursor: inferring ? "not-allowed" : "pointer" }} />
              <span style={{ ...S.info, minWidth: 38, textAlign: "right" }}>{friction.toFixed(3)}</span>
            </div>
          </div>

          <button
            style={pipelineDone && !inferring ? S.btn("#065f46") : S.btnDis}
            onClick={runInfer}
            disabled={!pipelineDone || inferring}>
            {inferring ? "推理中..." : "▶ 运行推理"}
          </button>
          {inferring && (
            <div style={{ marginTop: 6 }}>
              {inferStep === 0
                ? <div style={{ ...S.info, color: "#f59e0b" }}>⏳ SOFA 初始化中 (~20s)...</div>
                : <div style={{ ...S.info, color: "#60a5fa" }}>⚡ 步数: {inferStep} / {maxSteps}</div>
              }
              <div style={{
                marginTop: 4, height: 3, background: "#1f2937", borderRadius: 2
              }}>
                <div style={{
                  height: "100%", borderRadius: 2, background: "#3b82f6",
                  width: `${(inferStep / maxSteps) * 100}%`, transition: "width 0.3s",
                }} />
              </div>
            </div>
          )}
          {inferStats && (
            <div style={{ marginTop: 6 }}>
              <div style={S.info}>到达目标: {inferStats.target_reached ? "✓ 是" : "✗ 否"}</div>
              <div style={S.info}>距 LCCA: {inferStats.dist_to_lcca_mm} mm</div>
              <div style={S.info}>步数: {inferStats.n_steps}  奖励: {inferStats.total_reward}</div>
            </div>
          )}
        </div>

        {/* ── 日志 ── */}
        <div style={{ marginTop: "auto" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
            <div style={S.label}>日志</div>
            <div style={{ display: "flex", gap: 4 }}>
              <button onClick={() => navigator.clipboard.writeText(log)}
                style={{ padding: "2px 7px", fontSize: 10, background: "#1f2937", color: "#9ca3af",
                         border: "1px solid #374151", borderRadius: 3, cursor: "pointer" }}>复制</button>
              <button onClick={() => setLog("")}
                style={{ padding: "2px 7px", fontSize: 10, background: "#1f2937", color: "#9ca3af",
                         border: "1px solid #374151", borderRadius: 3, cursor: "pointer" }}>清空</button>
            </div>
          </div>
          <div style={S.log} ref={logRef}>{log || "等待操作..."}</div>
        </div>
      </div>

      {/* ── 3D 视图 ── */}
      <div style={S.main}>
        <Suspense fallback={<div style={{ color: "#333", padding: 20, fontSize: 12 }}>加载中...</div>}>
          {caseId && vessels.length > 0 ? (
            <Viewer3D caseId={caseId} vessels={vessels} allPathPoints={pathPoints} nav={nav} />
          ) : (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center",
                          height: "100%", color: "#1f2937", fontSize: 13 }}>
              从数据库选择病例 或 上传 CT 文件
            </div>
          )}
        </Suspense>
      </div>
    </div>
  );
}
