import { useCallback, useEffect, useMemo, useState } from "react";
import { Cpu, Gauge, HardDrive, ListVideo, MessageSquare, Settings2 } from "lucide-react";
import { ModelPanel } from "./components/ModelPanel";
import { RuntimePanel } from "./components/RuntimePanel";
import { StudioPanel } from "./components/StudioPanel";
import { JobsPanel } from "./components/JobsPanel";
import { makeStatus } from "./lib/jobs";
import { chooseBrowserDriveFolder, inspectColabDrive, mergeColabStatuses, queueColabDriveJob, readColabJobStatuses } from "./lib/colabDrive";
import {
  loadColabRootPath,
  loadJobs,
  loadModels,
  loadRuntime,
  saveColabRootPath,
  saveJobs,
  saveModels,
  saveRuntime,
} from "./lib/storage";
import type { ColabDriveStatus, GenerationJob, ModelInstall, RuntimeKind } from "./lib/types";

type View = "studio" | "models" | "runtime" | "jobs";

const views: Array<{ id: View; label: string; icon: typeof MessageSquare }> = [
  { id: "studio", label: "Studio", icon: MessageSquare },
  { id: "models", label: "Models", icon: HardDrive },
  { id: "runtime", label: "Runtime", icon: Cpu },
  { id: "jobs", label: "Jobs", icon: ListVideo },
];

export function App() {
  const [view, setView] = useState<View>("studio");
  const [runtime, setRuntime] = useState<RuntimeKind>(() => loadRuntime());
  const [models, setModels] = useState<ModelInstall[]>(() => loadModels());
  const [jobs, setJobs] = useState<GenerationJob[]>(() => loadJobs());
  const [colabDrive, setColabDrive] = useState<ColabDriveStatus>(() => initialColabDriveStatus(loadColabRootPath()));

  const readyModels = useMemo(
    () =>
      [...models]
        .filter((model) => model.status === "ready")
        .sort((a, b) => Number(Boolean(b.capabilities.optimized)) - Number(Boolean(a.capabilities.optimized))),
    [models],
  );
  const activeJobCount = jobs.filter((job) => job.status.state === "queued" || job.status.state === "running").length;

  useEffect(() => {
    saveModels(models);
  }, [models]);

  useEffect(() => {
    saveJobs(jobs);
  }, [jobs]);

  useEffect(() => {
    saveRuntime(runtime);
  }, [runtime]);

  useEffect(() => {
    saveColabRootPath(colabDrive.rootPath);
  }, [colabDrive.rootPath]);

  const refreshColabDrive = useCallback(
    async (rootPath = colabDrive.rootPath) => {
      const status = await inspectColabDrive(rootPath);
      setColabDrive(status);
      const statuses = await readColabJobStatuses(status.rootPath);
      setJobs((current) => mergeColabStatuses(current, statuses));
      return status;
    },
    [colabDrive.rootPath],
  );

  async function chooseColabDriveFolder() {
    const status = await chooseBrowserDriveFolder();
    setColabDrive(status);
    const statuses = await readColabJobStatuses(status.rootPath);
    setJobs((current) => mergeColabStatuses(current, statuses));
  }

  async function createJob(job: GenerationJob) {
    if (job.request.runtime !== "colab-drive") {
      setJobs((items) => [job, ...items]);
      return;
    }

    const result = await queueColabDriveJob(colabDrive.rootPath, job);
    if (!result.ok) {
      setJobs((items) => [
        {
          ...job,
          status: makeStatus(job.request.id, "failed", 0, {
            error: result.message,
          }),
        },
        ...items,
      ]);
      return;
    }

    setJobs((items) => [
      {
        ...job,
        status: makeStatus(job.request.id, "queued", 0, {
          outputPath: result.path,
        }),
      },
      ...items,
    ]);
    void refreshColabDrive();
  }

  useEffect(() => {
    const timer = window.setInterval(() => {
      setJobs((current) =>
        current.map((job) => {
          if (job.request.runtime === "colab-drive") {
            return job;
          }
          if (job.status.state === "queued") {
            return { ...job, status: makeStatus(job.request.id, "running", 12) };
          }
          if (job.status.state !== "running") {
            return job;
          }
          const nextProgress = Math.min(100, job.status.progress + (job.request.runtime === "local" ? 17 : 11));
          if (nextProgress >= 100) {
            return {
              ...job,
              status: makeStatus(job.request.id, "succeeded", 100, {
                outputPath: `outputs/${job.request.id}.mp4`,
              }),
            };
          }
          return { ...job, status: makeStatus(job.request.id, "running", nextProgress) };
        }),
      );
    }, 1400);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (runtime !== "colab-drive" || !colabDrive.rootPath.trim()) {
      return undefined;
    }
    void refreshColabDrive();
    const timer = window.setInterval(() => {
      void refreshColabDrive();
    }, 3500);
    return () => window.clearInterval(timer);
  }, [runtime, colabDrive.rootPath, refreshColabDrive]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Gauge size={22} strokeWidth={2.4} />
          </div>
          <div>
            <strong>Wan Studio</strong>
            <span>Open model desktop</span>
          </div>
        </div>

        <nav className="nav-list" aria-label="Primary">
          {views.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                type="button"
                aria-label={item.label}
                className={`nav-item ${view === item.id ? "active" : ""}`}
                onClick={() => setView(item.id)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-status">
          <div className="status-row">
            <span>Runtime</span>
            <strong>{runtime === "local" ? "Local GPU" : "Colab Drive"}</strong>
          </div>
          <div className="status-row">
            <span>Ready models</span>
            <strong>{readyModels.length}</strong>
          </div>
          <div className="status-row">
            <span>Active jobs</span>
            <strong>{activeJobCount}</strong>
          </div>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>{viewTitle(view)}</h1>
            <p>{viewSubtitle(view)}</p>
          </div>
          <button className="icon-button" type="button" aria-label="Runtime settings" onClick={() => setView("runtime")}>
            <Settings2 size={18} />
          </button>
        </header>

        {view === "studio" && (
          <StudioPanel models={readyModels} runtime={runtime} jobs={jobs} colabDrive={colabDrive} onCreateJob={createJob} />
        )}
        {view === "models" && <ModelPanel models={models} onModelsChange={setModels} />}
        {view === "runtime" && (
          <RuntimePanel
            runtime={runtime}
            colabDrive={colabDrive}
            onRuntimeChange={setRuntime}
            onColabRootChange={(rootPath) => setColabDrive((current) => ({ ...current, rootPath }))}
            onColabRefresh={refreshColabDrive}
            onChooseColabFolder={chooseColabDriveFolder}
          />
        )}
        {view === "jobs" && <JobsPanel jobs={jobs} onJobsChange={setJobs} />}
      </section>
    </main>
  );
}

function initialColabDriveStatus(rootPath: string): ColabDriveStatus {
  return {
    rootPath,
    connected: false,
    runtimeReady: false,
    modelReady: false,
    readyToPrompt: false,
    message: rootPath ? "Check the WanStudio Drive folder to confirm readiness." : "Connect your WanStudio Drive folder.",
    mode: "manual",
    statusCount: 0,
    resultCount: 0,
  };
}

function viewTitle(view: View): string {
  switch (view) {
    case "studio":
      return "Conversation studio";
    case "models":
      return "Wan model library";
    case "runtime":
      return "Runtime control";
    case "jobs":
      return "Generation queue";
  }
}

function viewSubtitle(view: View): string {
  switch (view) {
    case "studio":
      return "Prompt, choose a Wan model, and send a local or Colab job.";
    case "models":
      return "Connect downloaded Wan folders and keep the optimized preset visible.";
    case "runtime":
      return "Choose local Python worker or Colab Drive synchronization.";
    case "jobs":
      return "Track requests, status files, and generated outputs.";
  }
}
