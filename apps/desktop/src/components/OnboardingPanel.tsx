import { useMemo, useState } from "react";
import {
  ArrowLeft,
  Check,
  Clipboard,
  Cloud,
  Cpu,
  ExternalLink,
  FolderCheck,
  HardDrive,
  Play,
  Plus,
  ReceiptText,
  TerminalSquare,
} from "lucide-react";
import {
  buildHuggingFaceCommand,
  buildModelScopeCommand,
  createCustomModel,
  RECOMMENDED_WAN_MODEL,
  WAN_TASK_LABELS,
} from "../lib/modelCatalog";
import type { ModelInstall, ModelSource, RuntimeKind } from "../lib/types";

interface OnboardingPanelProps {
  runtime: RuntimeKind;
  models: ModelInstall[];
  onRuntimeChange: (runtime: RuntimeKind) => void;
  onModelsChange: (models: ModelInstall[]) => void;
  onComplete: () => void;
}

export function OnboardingPanel({ runtime, models, onRuntimeChange, onModelsChange, onComplete }: OnboardingPanelProps) {
  const [setupPage, setSetupPage] = useState<"choose" | "colab" | "local">("choose");
  const [source, setSource] = useState<ModelSource>("huggingface");
  const [repoId, setRepoId] = useState(RECOMMENDED_WAN_MODEL.repoId ?? "lkzd7/WAN2.2_LoraSet_NSFW");
  const [targetPath, setTargetPath] = useState("./models/WAN2.2_LoraSet_NSFW");
  const [localPath, setLocalPath] = useState("");
  const [copied, setCopied] = useState(false);
  const [addedModelId, setAddedModelId] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState("");

  const command = useMemo(
    () => (source === "modelscope" ? buildModelScopeCommand(repoId, targetPath) : buildHuggingFaceCommand(repoId, targetPath)),
    [repoId, source, targetPath],
  );
  const hasConnectedModel = models.some((model) => model.localPath.trim().length > 0);

  async function copyCommand() {
    try {
      await navigator.clipboard?.writeText(command);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  }

  function addModel() {
    if (!localPath.trim()) {
      setConnectionError("Paste the model folder path after downloading or running the Colab notebook.");
      return;
    }
    const model = createCustomModel({ repoId, localPath: localPath.trim(), source });
    onModelsChange([model, ...models.filter((item) => item.modelId !== model.modelId)]);
    setAddedModelId(model.modelId);
    setConnectionError("");
  }

  function chooseRuntime(nextRuntime: RuntimeKind) {
    onRuntimeChange(nextRuntime);
    setTargetPath(nextRuntime === "colab-drive" ? "/content/drive/MyDrive/WanStudio/models/WAN2.2_LoraSet_NSFW" : "./models/WAN2.2_LoraSet_NSFW");
    setLocalPath(nextRuntime === "colab-drive" ? "/content/drive/MyDrive/WanStudio/models/WAN2.2_LoraSet_NSFW" : "");
    setSetupPage(nextRuntime === "colab-drive" ? "colab" : "local");
  }

  return (
    <main className="onboarding-shell">
      <section className="onboarding-hero">
        <div className="brand onboarding-brand">
          <div className="brand-mark">
            <Play size={21} />
          </div>
          <div>
            <strong>Wan Studio</strong>
            <span>First run setup</span>
          </div>
        </div>
        <div className="onboarding-copy">
          <h1>Set up your Wan runtime first.</h1>
          <p>Choose where generation will run, connect a Wan-family model folder, then open the conversation studio.</p>
        </div>
      </section>

      <section className="onboarding-board">
        {setupPage !== "choose" && (
          <button type="button" className="back-button" onClick={() => setSetupPage("choose")}>
            <ArrowLeft size={16} />
            Runtime options
          </button>
        )}

        {setupPage === "choose" && (
          <>
            <div className="setup-step">
              <div className="step-number">1</div>
              <div>
                <h2>Choose runtime</h2>
                <p>Pick the beginner path first. Each option opens a setup page with exact next steps.</p>
              </div>
            </div>

            <div className="runtime-choice-grid">
              <button
                type="button"
                className={`runtime-choice ${runtime === "colab-drive" ? "selected" : ""}`}
                onClick={() => chooseRuntime("colab-drive")}
              >
                <Cloud size={22} />
                <span>
                  <strong>Google Colab + Drive</strong>
                  <small>Open the Colab setup guide and sync jobs through Drive.</small>
                </span>
              </button>
              <button type="button" className={`runtime-choice ${runtime === "local" ? "selected" : ""}`} onClick={() => chooseRuntime("local")}>
                <Cpu size={22} />
                <span>
                  <strong>Desktop local worker</strong>
                  <small>Install Python worker and use your local GPU.</small>
                </span>
              </button>
            </div>
          </>
        )}

        {setupPage === "colab" && (
          <section className="runtime-guide">
            <div className="setup-step">
              <div className="step-number">1</div>
              <div>
                <h2>Connect Google Colab</h2>
                <p>Colab hosts the same Wan Studio Web UI when local GPU access is not available.</p>
              </div>
            </div>
            <ol className="guide-list">
              <li>
                <strong>Choose payment</strong>
                <span>For a connection test, free Colab is enough. For real Wan video generation, use Colab Pro, Pro+, or Pay As You Go and choose a GPU runtime.</span>
              </li>
              <li>
                <strong>Open Colab</strong>
                <span>Open Google Colab, then upload or open `notebooks/wan_colab_worker.ipynb` from this project.</span>
              </li>
              <li>
                <strong>Install Wan Studio</strong>
                <span>Run the clone/install cell. It installs the package in the Colab runtime.</span>
              </li>
              <li>
                <strong>Download model files</strong>
                <span>Keep the default LoRA/adapters repo or change it to another Wan-family folder before downloading.</span>
              </li>
              <li>
                <strong>Launch the Web UI</strong>
                <span>Leave the final cell running and use the Colab browser window for prompts, status, and results.</span>
              </li>
            </ol>
            <div className="guide-actions">
              <a className="secondary-link-button" href="https://colab.research.google.com/signup" target="_blank" rel="noreferrer">
                <ReceiptText size={17} />
                Colab pricing
              </a>
              <a className="primary-link-button" href="https://colab.research.google.com/" target="_blank" rel="noreferrer">
                <ExternalLink size={17} />
                Open Google Colab
              </a>
              <code>notebooks/wan_colab_worker.ipynb</code>
            </div>
            <div className="payment-guide">
              <strong>Recommended first test</strong>
              <span>Run the notebook's smoke job on a free GPU first. If Drive/results work, upgrade only when you are ready to run the real Wan model.</span>
            </div>
            <div className="code-recipe">
              <div>
                <strong>Runtime menu</strong>
                <code>Runtime → Change runtime type → GPU → Save</code>
              </div>
              <div>
                <strong>Cell 1</strong>
                <code>git clone + python install.py --accelerator cuda --system</code>
              </div>
              <div>
                <strong>Cell 2</strong>
                <code>hf download lkzd7/WAN2.2_LoraSet_NSFW --local-dir /content/wan-studio/models/WAN2.2_LoraSet_NSFW</code>
              </div>
              <div>
                <strong>Cell 3</strong>
                <code>python wan_studio.py run --host 0.0.0.0 --port 7860 --share</code>
              </div>
            </div>
          </section>
        )}

        {setupPage === "local" && (
          <section className="runtime-guide">
            <div className="setup-step">
              <div className="step-number">1</div>
              <div>
                <h2>Connect local worker</h2>
                <p>Use this path when the desktop has a local NVIDIA GPU and Python can access CUDA.</p>
              </div>
            </div>
            <ol className="guide-list">
              <li>
                <strong>Install Wan Studio</strong>
                <span>Run `python install.py --accelerator cuda` from the project root.</span>
              </li>
              <li>
                <strong>Start the Web UI</strong>
                <span>Run `.venv/bin/python wan_studio.py run --open-browser`.</span>
              </li>
              <li>
                <strong>Download a Wan model</strong>
                <span>Use the download command below, then paste the model folder path.</span>
              </li>
            </ol>
            <div className="guide-actions">
              <span className="primary-link-button local-only-button">
                <TerminalSquare size={17} />
                Local Web UI: 127.0.0.1:7860
              </span>
            </div>
          </section>
        )}

        <div className="setup-step">
          <div className="step-number">{setupPage === "choose" ? "2" : "2"}</div>
          <div>
            <h2>Prepare a Wan model</h2>
            <p>The default repo is a Wan2.2 LoRA/adapters set, but any Wan-family model folder can be connected.</p>
          </div>
        </div>

        <div className="onboarding-model-card">
          <div className="model-hero">
            <div>
              <h3>{RECOMMENDED_WAN_MODEL.displayName}</h3>
              <p>{RECOMMENDED_WAN_MODEL.repoId}</p>
            </div>
            <a className="text-link" href="https://huggingface.co/lkzd7/WAN2.2_LoraSet_NSFW" target="_blank" rel="noreferrer">
              <ExternalLink size={16} />
              Model page
            </a>
          </div>
          <div className="task-row">
            {RECOMMENDED_WAN_MODEL.capabilities.tasks.map((task) => (
              <span key={task}>{WAN_TASK_LABELS[task]}</span>
            ))}
          </div>
        </div>

        <div className="form-grid onboarding-form">
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
            <input value={repoId} onChange={(event) => setRepoId(event.target.value)} placeholder="lkzd7/WAN2.2_LoraSet_NSFW" />
          </label>
          <label className="wide-field">
            Download folder
            <input value={targetPath} onChange={(event) => setTargetPath(event.target.value)} placeholder="./models/my-wan-model" />
          </label>
        </div>

        <div className="command-box onboarding-command">
          <code>{source === "local" ? "Use an existing Wan model folder and connect it below." : command}</code>
          {source !== "local" && (
            <button type="button" className="icon-text-button" onClick={copyCommand}>
              {copied ? <Check size={16} /> : <Clipboard size={16} />}
              {copied ? "Copied" : "Copy"}
            </button>
          )}
        </div>

        {runtime === "colab-drive" && setupPage !== "choose" && (
          <div className="colab-note">
            <HardDrive size={18} />
            <span>After the Colab model check succeeds, keep this path connected here: `/content/drive/MyDrive/WanStudio/models/WAN2.2_LoraSet_NSFW`.</span>
          </div>
        )}

        <label className="folder-field">
          {runtime === "colab-drive" ? "Colab model folder path" : "Local model folder"}
          <div>
            <input
              value={localPath}
              onChange={(event) => setLocalPath(event.target.value)}
              placeholder={runtime === "colab-drive" ? "/content/drive/MyDrive/WanStudio/models/WAN2.2_LoraSet_NSFW" : "/path/to/WAN2.2_LoraSet_NSFW"}
            />
            <button type="button" className="primary-button" onClick={addModel}>
              <Plus size={17} />
              Connect
            </button>
          </div>
        </label>
        {connectionError && <p className="connection-error">{connectionError}</p>}

        <footer className="onboarding-footer">
          <div>
            <strong>{hasConnectedModel ? "Ready to prompt!" : "Waiting for model connection"}</strong>
            <span>
              {addedModelId
                ? "The connected Wan model is now available in Studio."
                : hasConnectedModel
                  ? "A connected Wan model is available in Studio."
                  : "Connect a downloaded Wan model folder before opening Studio."}
            </span>
          </div>
          <button type="button" className="run-button onboarding-open" onClick={onComplete} disabled={!hasConnectedModel}>
            <FolderCheck size={18} />
            Open Studio
          </button>
        </footer>
      </section>
    </main>
  );
}
