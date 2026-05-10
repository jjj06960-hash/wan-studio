import { AlertCircle, CheckCircle2, Cloud, Cpu, ExternalLink, FolderOpen, FolderSync, RefreshCcw, TerminalSquare } from "lucide-react";
import { useState } from "react";
import type { ColabDriveStatus, RuntimeKind } from "../lib/types";

interface RuntimePanelProps {
  runtime: RuntimeKind;
  colabDrive: ColabDriveStatus;
  onRuntimeChange: (runtime: RuntimeKind) => void;
  onColabRootChange: (rootPath: string) => void;
  onColabRefresh: (rootPath?: string) => Promise<ColabDriveStatus>;
  onChooseColabFolder: () => Promise<void>;
}

export function RuntimePanel({ runtime, colabDrive, onRuntimeChange, onColabRootChange, onColabRefresh, onChooseColabFolder }: RuntimePanelProps) {
  const [checking, setChecking] = useState(false);

  async function refresh() {
    setChecking(true);
    try {
      await onColabRefresh(colabDrive.rootPath);
    } finally {
      setChecking(false);
    }
  }

  async function chooseFolder() {
    setChecking(true);
    try {
      await onChooseColabFolder();
    } finally {
      setChecking(false);
    }
  }

  return (
    <div className="runtime-stack">
      <div className="runtime-layout">
        <section className={`runtime-option ${runtime === "local" ? "selected" : ""}`}>
          <div className="runtime-icon">
            <Cpu size={24} />
          </div>
          <div>
            <h2>Local Python worker</h2>
            <p>Uses your machine, Python venv, and connected Wan folders.</p>
          </div>
          <button type="button" className="primary-button" onClick={() => onRuntimeChange("local")}>
            <TerminalSquare size={17} />
            Use local
          </button>
          <dl className="runtime-facts">
            <div>
              <dt>Target</dt>
              <dd>Windows NVIDIA first</dd>
            </div>
            <div>
              <dt>Worker API</dt>
              <dd>127.0.0.1:8765</dd>
            </div>
            <div>
              <dt>Install</dt>
              <dd>Python venv</dd>
            </div>
          </dl>
        </section>

        <section className={`runtime-option ${runtime === "colab-drive" ? "selected" : ""}`}>
          <div className="runtime-icon">
            <Cloud size={24} />
          </div>
          <div>
            <h2>Colab + Drive worker</h2>
            <p>Syncs job JSON and results through a WanStudio folder.</p>
          </div>
          <button type="button" className="primary-button" onClick={() => onRuntimeChange("colab-drive")}>
            <FolderSync size={17} />
            Use Colab
          </button>
          <dl className="runtime-facts">
            <div>
              <dt>Notebook</dt>
              <dd>worker only</dd>
            </div>
            <div>
              <dt>Jobs</dt>
              <dd>WanStudio/jobs</dd>
            </div>
            <div>
              <dt>Results</dt>
              <dd>WanStudio/results</dd>
            </div>
          </dl>
        </section>
      </div>

      {runtime === "colab-drive" && (
        <section className={`panel colab-control ${colabDrive.readyToPrompt ? "ready" : ""}`}>
          <div className="section-heading">
            <div>
              <h2>Colab control panel</h2>
              <p>Use the app as the prompt surface. Colab only needs to keep the worker notebook running.</p>
            </div>
            {colabDrive.readyToPrompt ? <CheckCircle2 size={20} /> : <AlertCircle size={20} />}
          </div>

          <div className="colab-status-banner">
            <strong>{colabDrive.readyToPrompt ? "Ready to prompt!" : "Not ready yet"}</strong>
            <span>{colabDrive.message}</span>
          </div>

          <div className="colab-connect-grid">
            <label className="wide-field">
              Synced Drive WanStudio folder
              <div className="folder-field-inline">
                <input
                  value={colabDrive.rootPath}
                  onChange={(event) => onColabRootChange(event.target.value)}
                  placeholder="/Users/you/Library/CloudStorage/GoogleDrive-.../My Drive/WanStudio"
                />
                <button type="button" className="icon-text-button" onClick={chooseFolder} disabled={checking}>
                  <FolderOpen size={16} />
                  Choose
                </button>
              </div>
            </label>
            <button type="button" className="primary-button" onClick={refresh} disabled={checking || !colabDrive.rootPath.trim()}>
              <RefreshCcw size={17} />
              {checking ? "Checking" : "Check status"}
            </button>
            <a className="secondary-link-button" href="https://colab.research.google.com/" target="_blank" rel="noreferrer">
              <ExternalLink size={17} />
              Open Colab
            </a>
          </div>

          <div className="colab-checklist">
            <StatusPill label="Colab worker" ready={colabDrive.runtimeReady} />
            <StatusPill label="Model verified" ready={colabDrive.modelReady} />
            <StatusPill label="Prompting" ready={colabDrive.readyToPrompt} />
          </div>

          <dl className="runtime-facts colab-facts">
            <div>
              <dt>Status files</dt>
              <dd>{colabDrive.statusCount}</dd>
            </div>
            <div>
              <dt>Results</dt>
              <dd>{colabDrive.resultCount}</dd>
            </div>
            <div>
              <dt>Model path</dt>
              <dd>{colabDrive.modelPath || "Waiting"}</dd>
            </div>
          </dl>

          {colabDrive.error && <p className="connection-error">{colabDrive.error}</p>}
        </section>
      )}
    </div>
  );
}

function StatusPill({ label, ready }: { label: string; ready: boolean }) {
  return (
    <span className={`status-pill ${ready ? "ready" : ""}`}>
      {ready ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
      {label}
    </span>
  );
}
