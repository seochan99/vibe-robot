import type { MouseEvent, PointerEvent, RefObject, WheelEvent } from "react";

import type { SceneObjectInfo, SimCameraState } from "@/lib/api";

import type { SimTool } from "./ui-constants";
import { OBJECT_PALETTE } from "./ui-constants";

type SimulationPanelProps = {
  simConnected: boolean;
  simCamera: SimCameraState | null;
  pipelineStage: string;
  stageLabel: string;
  simImgRef: RefObject<HTMLImageElement | null>;
  editorMode: string | null;
  loading: boolean;
  simTool: SimTool;
  selectedObject: string | null;
  sceneObjects: SceneObjectInfo[];
  simStreamUrl: string;
  onResetCamera: () => void;
  onSimClick: (e: MouseEvent<HTMLImageElement>) => void;
  onSimPointerDown: (e: PointerEvent<HTMLImageElement>) => void;
  onSimPointerMove: (e: PointerEvent<HTMLImageElement>) => void;
  onSimPointerUp: (e: PointerEvent<HTMLImageElement>) => void;
  onSimContextMenu: (e: MouseEvent<HTMLImageElement>) => void;
  onSimWheel: (e: WheelEvent<HTMLImageElement>) => void;
  onCancelEditor: () => void;
  onSelectTool: (tool: SimTool) => void;
  onToggleAddMode: (objType: string) => void;
  onSelectObject: (name: string) => void;
  onRemoveObject: (name: string) => void;
};

export function SimulationPanel({
  simConnected,
  simCamera,
  pipelineStage,
  stageLabel,
  simImgRef,
  editorMode,
  loading,
  simTool,
  selectedObject,
  sceneObjects,
  simStreamUrl,
  onResetCamera,
  onSimClick,
  onSimPointerDown,
  onSimPointerMove,
  onSimPointerUp,
  onSimContextMenu,
  onSimWheel,
  onCancelEditor,
  onSelectTool,
  onToggleAddMode,
  onSelectObject,
  onRemoveObject,
}: SimulationPanelProps) {
  return (
    <div className="hidden lg:flex flex-col w-[480px] shrink-0 border-l border-[var(--border)] bg-[var(--surface)]">
      {/* Sim header */}
      <div className="shrink-0 px-4 py-2 border-b border-[var(--border)] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${simConnected ? "bg-emerald-500" : "bg-red-500"}`} />
          <span className="text-xs font-medium">Simulation View</span>
        </div>
        <div className="flex items-center gap-2">
          {simCamera && (
            <span className="hidden xl:inline text-[10px] text-[var(--muted)] font-mono">
              az {Math.round(simCamera.azimuth)} / el {Math.round(simCamera.elevation)} / d {simCamera.distance.toFixed(2)}
            </span>
          )}
          <button
            onClick={onResetCamera}
            className="text-[10px] px-2 py-0.5 rounded border border-[var(--border)] hover:border-[var(--accent)]/40 text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
          >
            reset cam
          </button>
          {pipelineStage !== "idle" && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--accent)]/10 text-[var(--accent)] font-medium">
              {pipelineStage.replace(/_/g, " ")}
            </span>
          )}
        </div>
      </div>

      {/* MJPEG stream */}
      <div className="relative flex-1 min-h-0 bg-black flex items-center justify-center">
        {simConnected ? (
          <>
            <img
              ref={simImgRef}
              src={simStreamUrl}
              alt="MuJoCo simulation"
              draggable={false}
              className={`w-full h-full object-contain ${
                editorMode && !loading
                  ? "cursor-crosshair"
                  : simTool === "move" && selectedObject
                    ? "cursor-grab"
                    : "cursor-move"
              } select-none`}
              style={{ touchAction: "none" }}
              onDragStart={(e) => e.preventDefault()}
              onClick={loading ? undefined : onSimClick}
              onPointerDown={onSimPointerDown}
              onPointerMove={onSimPointerMove}
              onPointerUp={onSimPointerUp}
              onPointerCancel={onSimPointerUp}
              onContextMenu={onSimContextMenu}
              onWheel={onSimWheel}
            />
            {loading && (
              <div className="absolute bottom-3 left-3 right-3 rounded-lg border border-white/20 bg-black/60 backdrop-blur-sm px-3 py-2 text-xs text-white/90">
                Live simulation view · planning/safety checks in progress ({stageLabel})
              </div>
            )}
          </>
        ) : (
          <div className="text-center text-[var(--muted)] text-sm p-8">
            <p className="font-medium mb-2">Simulation offline</p>
            <p className="text-xs">Start the backend: <code className="px-1.5 py-0.5 rounded bg-[var(--border)] text-xs font-mono">python api/server.py</code></p>
          </div>
        )}

        {/* Editor mode indicator */}
        {editorMode && (
          <div className="absolute top-2 left-2 right-2 flex items-center justify-between">
            <span className="px-3 py-1.5 rounded-lg bg-blue-600/90 text-white text-xs font-medium backdrop-blur-sm">
              Click to place: {editorMode}
            </span>
            <button
              onClick={onCancelEditor}
              className="px-2 py-1 rounded-lg bg-black/50 text-white text-xs backdrop-blur-sm hover:bg-black/70"
            >
              Cancel
            </button>
          </div>
        )}
        {!editorMode && simTool === "move" && selectedObject && (
          <div className="absolute top-2 left-2 right-2">
            <span className="inline-block px-3 py-1.5 rounded-lg bg-emerald-600/85 text-white text-xs font-medium backdrop-blur-sm">
              Move mode: click any object and drag · selected {selectedObject}
            </span>
          </div>
        )}
        {!editorMode && simTool === "move" && !selectedObject && (
          <div className="absolute top-2 left-2 right-2">
            <span className="inline-block px-3 py-1.5 rounded-lg bg-emerald-600/85 text-white text-xs font-medium backdrop-blur-sm">
              Move mode: click object then drag to reposition
            </span>
          </div>
        )}
      </div>

      {/* Object palette / scene editor */}
      <div className="shrink-0 border-t border-[var(--border)]">
        <div className="px-3 py-2 border-b border-[var(--border)]">
          <p className="text-[10px] text-[var(--muted)] uppercase tracking-wider font-medium mb-1.5">Tools</p>
          <div className="flex gap-1.5">
            <button
              onClick={() => onSelectTool("camera")}
              className={`px-2.5 py-1 rounded-lg text-xs border transition-colors ${
                simTool === "camera" && !editorMode
                  ? "bg-[var(--accent-soft)] border-[var(--accent)]/40 text-[var(--foreground)]"
                  : "border-[var(--border)] text-[var(--muted)] hover:text-[var(--foreground)]"
              }`}
            >
              Camera
            </button>
            <button
              onClick={() => onSelectTool("move")}
              className={`px-2.5 py-1 rounded-lg text-xs border transition-colors ${
                simTool === "move" && !editorMode
                  ? "bg-[var(--accent-soft)] border-[var(--accent)]/40 text-[var(--foreground)]"
                  : "border-[var(--border)] text-[var(--muted)] hover:text-[var(--foreground)]"
              }`}
            >
              Move
            </button>
          </div>
          <p className="mt-1 text-[10px] text-[var(--muted)]">
            Left-drag: orbit · right-drag/Shift-drag: pan · wheel: zoom
          </p>
          <p className="mt-1 text-[10px] text-[var(--muted)]">
            Move tool: click object in view to pick, then drag freely
          </p>
        </div>

        {/* Palette */}
        <div className="px-3 py-2">
          <p className="text-[10px] text-[var(--muted)] uppercase tracking-wider font-medium mb-1.5">Add Objects</p>
          <div className="flex flex-wrap gap-1">
            {OBJECT_PALETTE.map((obj) => (
              <button
                key={obj.type}
                onClick={() => onToggleAddMode(obj.type)}
                className={`px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  editorMode === obj.type
                    ? "bg-blue-600 text-white ring-2 ring-blue-400/50"
                    : "bg-[var(--background)] border border-[var(--border)] hover:border-[var(--accent)]/30"
                }`}
              >
                <span className="mr-1">{obj.icon}</span>{obj.label}
              </button>
            ))}
          </div>
        </div>

        {/* Scene objects list */}
        <div className="px-3 py-2 border-t border-[var(--border)] max-h-40 overflow-y-auto">
          <p className="text-[10px] text-[var(--muted)] uppercase tracking-wider font-medium mb-1.5">
            Scene Objects ({sceneObjects.length})
          </p>
          {sceneObjects.length === 0 ? (
            <p className="text-xs text-[var(--muted)]">No objects loaded</p>
          ) : (
            <div className="space-y-1">
              {sceneObjects.map((obj) => (
                <div
                  key={obj.name}
                  className={`flex items-center justify-between text-xs group rounded px-1 ${
                    selectedObject === obj.name ? "bg-blue-500/10 border border-blue-500/25" : "border border-transparent"
                  }`}
                >
                  <span className="truncate">
                    <button
                      onClick={() => onSelectObject(obj.name)}
                      className="font-medium hover:text-[var(--foreground)] text-left"
                      title="Select for move tool"
                    >
                      {obj.name}
                    </button>
                    <span className="text-[var(--muted)] ml-1">
                      ({obj.position[0].toFixed(2)}, {obj.position[1].toFixed(2)}, {obj.position[2].toFixed(2)})
                    </span>
                  </span>
                  <button
                    onClick={() => onRemoveObject(obj.name)}
                    className="opacity-0 group-hover:opacity-100 text-red-500 hover:text-red-400 transition-opacity px-1"
                    title="Remove"
                  >
                    &times;
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
