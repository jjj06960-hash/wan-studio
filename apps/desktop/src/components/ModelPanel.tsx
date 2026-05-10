import { useMemo, useState } from "react";
import { Check, Clipboard, ExternalLink, FolderCheck, Plus, ShieldCheck, Trash2, TriangleAlert } from "lucide-react";
import {
  buildHuggingFaceCommand,
  buildModelScopeCommand,
  createCustomModel,
  RECOMMENDED_WAN_MODEL,
  WAN_TASK_LABELS,
} from "../lib/modelCatalog";
import type { ModelInstall, ModelSource } from "../lib/types";

interface ModelPanelProps {
  models: ModelInstall[];
  onModelsChange: (models: ModelInstall[]) => void;
}

export function ModelPanel({ models, onModelsChange }: ModelPanelProps) {
  const [repoId, setRepoId] = useState(RECOMMENDED_WAN_MODEL.repoId ?? "Wan-AI/Wan2.2-I2V-A14B");
  const [targetPath, setTargetPath] = useState("./models/Wan2.2-I2V-A14B");
  const [source, setSource] = useState<ModelSource>("huggingface");
  const [localPath, setLocalPath] = useState("");
  const [copied, setCopied] = useState(false);

  const command = useMemo(
    () => (source === "modelscope" ? buildModelScopeCommand(repoId, targetPath) : buildHuggingFaceCommand(repoId, targetPath)),
    [repoId, source, targetPath],
  );

  function addModel() {
    const model = createCustomModel({ repoId, localPath: localPath || targetPath, source });
    const withoutDuplicate = models.filter((item) => item.modelId !== model.modelId);
    onModelsChange([model, ...withoutDuplicate]);
  }

  async function copyCommand() {
    await navigator.clipboard?.writeText(command);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <div className="panel-grid model-grid">
      <section className="panel recommended-panel">
        <div className="section-heading">
          <div>
            <h2>Optimized recommendation</h2>
            <p>Default A14B base for the 8GB/16GB/24GB builds.</p>
          </div>
          <span className="quality-chip">
            <ShieldCheck size={15} />
            8GB ready
          </span>
        </div>
        <div className="model-hero">
          <div>
            <h3>{RECOMMENDED_WAN_MODEL.displayName}</h3>
            <p>{RECOMMENDED_WAN_MODEL.repoId}</p>
          </div>
          <a className="text-link" href="https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B" target="_blank" rel="noreferrer">
            <ExternalLink size={16} />
            Hugging Face
          </a>
        </div>
        <div className="task-row">
          {RECOMMENDED_WAN_MODEL.capabilities.tasks.map((task) => (
            <span key={task}>{WAN_TASK_LABELS[task]}</span>
          ))}
        </div>
        <ul className="feature-list">
          {RECOMMENDED_WAN_MODEL.capabilities.notes.map((note) => (
            <li key={note}>
              <Check size={15} />
              {note}
            </li>
          ))}
        </ul>
      </section>

      <section className="panel guide-panel">
        <div className="section-heading">
          <div>
            <h2>Download guide</h2>
            <p>Bring your own Wan-family model folder.</p>
          </div>
        </div>

        <div className="form-grid">
          <label>
            Source
            <select value={source} onChange={(event) => setSource(event.target.value as ModelSource)}>
              <option value="huggingface">Hugging Face</option>
              <option value="modelscope">ModelScope</option>
              <option value="local">Local only</option>
            </select>
          </label>
          <label>
            Repo id
            <input value={repoId} onChange={(event) => setRepoId(event.target.value)} placeholder="Wan-AI/Wan2.2-I2V-A14B" />
          </label>
          <label className="wide-field">
            Download folder
            <input value={targetPath} onChange={(event) => setTargetPath(event.target.value)} placeholder="./models/my-wan-model" />
          </label>
        </div>

        <div className="command-box">
          <code>{source === "local" ? "Choose an existing Wan model folder below." : command}</code>
          {source !== "local" && (
            <button type="button" className="icon-text-button" onClick={copyCommand}>
              {copied ? <Check size={16} /> : <Clipboard size={16} />}
              {copied ? "Copied" : "Copy"}
            </button>
          )}
        </div>

        <label className="folder-field">
          Local model folder
          <div>
            <input value={localPath} onChange={(event) => setLocalPath(event.target.value)} placeholder="/path/to/Wan2.2-I2V-A14B" />
            <button type="button" className="primary-button" onClick={addModel}>
              <Plus size={17} />
              Add
            </button>
          </div>
        </label>
      </section>

      <section className="panel library-panel">
        <div className="section-heading">
          <div>
            <h2>Connected models</h2>
            <p>{models.length === 0 ? "No custom folders connected yet." : `${models.length} custom folder${models.length > 1 ? "s" : ""}`}</p>
          </div>
        </div>

        <div className="model-list">
          {models.map((model) => (
            <article className="model-row" key={model.modelId}>
              <div className={`status-dot ${model.status}`} />
              <div>
                <h3>{model.displayName}</h3>
                <p>{model.localPath}</p>
                <div className="task-row compact">
                  {model.capabilities.tasks.map((task) => (
                    <span key={task}>{task}</span>
                  ))}
                  {model.capabilities.tasks.length === 0 && <span>needs review</span>}
                </div>
              </div>
              <div className="row-actions">
                {model.status === "ready" ? <FolderCheck size={18} /> : <TriangleAlert size={18} />}
                <button type="button" className="icon-button" onClick={() => onModelsChange(models.filter((item) => item.modelId !== model.modelId))} aria-label={`Remove ${model.displayName}`}>
                  <Trash2 size={17} />
                </button>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
