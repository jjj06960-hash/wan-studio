import { useEffect, useMemo, useState } from "react";
import { CheckCircle2, Cloud, Image, Play, Send, SlidersHorizontal } from "lucide-react";
import { makeJob, serializeDriveJob } from "../lib/jobs";
import { WAN_TASK_LABELS } from "../lib/modelCatalog";
import type { ColabDriveStatus, GenerationJob, ModelInstall, RuntimeKind, WanTask } from "../lib/types";

interface StudioPanelProps {
  models: ModelInstall[];
  runtime: RuntimeKind;
  jobs: GenerationJob[];
  colabDrive: ColabDriveStatus;
  onCreateJob: (job: GenerationJob) => void | Promise<void>;
}

export function StudioPanel({ models, runtime, jobs, colabDrive, onCreateJob }: StudioPanelProps) {
  const preferredModel = useMemo(() => models.find((model) => model.localPath) ?? models[0], [models]);
  const [modelId, setModelId] = useState(preferredModel?.modelId ?? "");
  const [prompt, setPrompt] = useState("A cinematic camera move across a quiet neon street after rain.");
  const [image, setImage] = useState("");
  const selectedModel = useMemo(() => models.find((model) => model.modelId === modelId) ?? preferredModel, [modelId, models, preferredModel]);
  const availableTasks = selectedModel?.capabilities.tasks ?? [];
  const [task, setTask] = useState<WanTask>("ti2v");
  const resolvedTask = availableTasks.includes(task) ? task : availableTasks[0] ?? "t2v";
  const latestJob = jobs[0];
  const canSubmit =
    Boolean(selectedModel && prompt.trim()) && (runtime === "local" || (colabDrive.readyToPrompt && Boolean(selectedModel?.localPath)));

  useEffect(() => {
    if (selectedModel) {
      return;
    }
    setModelId(preferredModel?.modelId ?? "");
  }, [preferredModel, selectedModel]);

  function submitJob() {
    if (!selectedModel || !prompt.trim() || !canSubmit) {
      return;
    }
    const job = makeJob({
      prompt: prompt.trim(),
      image: image.trim() || undefined,
      model: selectedModel,
      runtime,
      task: resolvedTask,
    });
    onCreateJob(job);
  }

  return (
    <div className="studio-layout">
      <section className="conversation-panel">
        <div className="prompt-card">
          <label>
            Prompt
            <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={8} />
          </label>
          <label>
            Image reference
            <div className="input-with-icon">
              <Image size={17} />
              <input value={image} onChange={(event) => setImage(event.target.value)} placeholder="/path/to/reference.png" />
            </div>
          </label>
          <button type="button" className="run-button" onClick={submitJob} disabled={!canSubmit}>
            <Send size={18} />
            {runtime === "colab-drive" ? "Send to Colab" : "Queue generation"}
          </button>
          {!selectedModel && <p className="helper-note">Connect a Wan model in Models before sending prompts.</p>}
          {runtime === "colab-drive" && !canSubmit && (
            <p className="helper-note">
              Colab must show Ready to prompt and the selected model needs a Drive model path before jobs can be sent.
            </p>
          )}
        </div>
      </section>

      <aside className="control-panel">
        <section className="panel">
          <div className="section-heading">
            <div>
              <h2>Model</h2>
              <p>{selectedModel?.capabilities.optimized ? "Optimized Wan2.2 profile" : "Custom Wan profile"}</p>
            </div>
          </div>
          <select value={selectedModel?.modelId ?? ""} onChange={(event) => setModelId(event.target.value)}>
            {models.length === 0 && <option value="">No connected models</option>}
            {models.map((model) => (
              <option key={model.modelId} value={model.modelId}>
                {model.displayName}
              </option>
            ))}
          </select>
          <div className="task-row">
            {availableTasks.map((item) => (
              <button key={item} type="button" className={resolvedTask === item ? "task-button active" : "task-button"} onClick={() => setTask(item)}>
                {WAN_TASK_LABELS[item]}
              </button>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <h2>Settings</h2>
              <p>{runtime === "local" ? "Local worker request" : "Drive job request"}</p>
            </div>
            <SlidersHorizontal size={18} />
          </div>
          <div className="settings-list">
            <div>
              <span>Runtime</span>
              <strong>{runtime === "local" ? "Local GPU" : "Colab Drive"}</strong>
            </div>
            <div>
              <span>Task</span>
              <strong>{resolvedTask.toUpperCase()}</strong>
            </div>
            <div>
              <span>Preset</span>
              <strong>{selectedModel?.capabilities.optimized ? "24 GB" : "Safe"}</strong>
            </div>
            {runtime === "colab-drive" && (
              <div>
                <span>Colab</span>
                <strong>{colabDrive.readyToPrompt ? "Ready" : "Not ready"}</strong>
              </div>
            )}
            {runtime === "colab-drive" && selectedModel?.localPath && (
              <div>
                <span>Model path</span>
                <strong>{selectedModel.localPath}</strong>
              </div>
            )}
          </div>
        </section>

        {runtime === "colab-drive" && (
          <section className={`panel colab-studio-status ${colabDrive.readyToPrompt ? "ready" : ""}`}>
            <div className="section-heading">
              <div>
                <h2>{colabDrive.readyToPrompt ? "Ready to prompt!" : "Connect Colab first"}</h2>
                <p>{colabDrive.message}</p>
              </div>
              {colabDrive.readyToPrompt ? <CheckCircle2 size={18} /> : <Cloud size={18} />}
            </div>
            <div className="settings-list">
              <div>
                <span>Runtime status</span>
                <strong>{colabDrive.runtimeReady ? "Connected" : "Waiting"}</strong>
              </div>
              <div>
                <span>Model status</span>
                <strong>{colabDrive.modelReady ? "Verified" : "Waiting"}</strong>
              </div>
              <div>
                <span>Results</span>
                <strong>{colabDrive.resultCount}</strong>
              </div>
            </div>
          </section>
        )}

        {latestJob && (
          <section className="panel output-panel">
            <div className="section-heading">
              <div>
                <h2>Latest job</h2>
                <p>{latestJob.request.id}</p>
              </div>
              <Play size={18} />
            </div>
            <div className="progress-bar" aria-label={`Progress ${latestJob.status.progress}%`}>
              <span style={{ width: `${latestJob.status.progress}%` }} />
            </div>
            <pre>{runtime === "colab-drive" ? serializeDriveJob(latestJob) : latestJob.status.outputPath ?? latestJob.status.state}</pre>
          </section>
        )}
      </aside>
    </div>
  );
}
