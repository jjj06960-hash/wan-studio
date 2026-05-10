from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .model_registry import build_hf_command, build_modelscope_command, create_model_install
from .runner import FakeWanRunner, SubprocessWanRunner, WanRunner
from .schemas import GenerationJob, GenerationRequest, GenerationStatus, JobState, ModelInstall, RuntimeKind, WanTask, status_for
from .wan_lora_generate import LORA_SUFFIXES


DEFAULT_REPO_ID = "lkzd7/WAN2.2_LoraSet_NSFW"
DEFAULT_MODEL_DIR = "models/WAN2.2_LoraSet_NSFW"
DEFAULT_MODEL_NOTE = "Default download is a Wan2.2 LoRA/adapters set; real Wan inference still needs a compatible Wan2.2 base model runner."
WAN_BASE_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN_BASE_MODEL_DIR = "models/Wan2.2-TI2V-5B"


class ConnectModelInput(BaseModel):
    repo_id: str = DEFAULT_REPO_ID
    local_path: str = DEFAULT_MODEL_DIR
    source: Literal["huggingface", "modelscope", "local"] = "huggingface"


class CreateJobInput(BaseModel):
    prompt: str
    model_id: str
    task: WanTask = WanTask.TI2V
    image: str | None = None
    size: str = "1280x704"
    steps: int = 24
    offload_model: bool = True
    t5_cpu: bool = True
    seed: int = 0
    lora_paths: list[str] = Field(default_factory=list)
    lora_scale: float = 1.0


class WebState:
    def __init__(self, *, root: Path, runner: WanRunner) -> None:
        self.root = root.expanduser().resolve()
        self.config_dir = self.root / ".wan-studio"
        self.output_dir = self.root / "outputs"
        self.models_path = self.config_dir / "models.json"
        self.runner = runner
        self.jobs: dict[str, GenerationJob] = {}
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = self._load_models()
        if not self.models:
            self.models = self._discover_models()
            self._save_models()

    def _load_models(self) -> list[ModelInstall]:
        try:
            raw_items = json.loads(self.models_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        models: list[ModelInstall] = []
        for item in raw_items:
            try:
                models.append(ModelInstall.model_validate(item))
            except Exception:
                continue
        return models

    def _save_models(self) -> None:
        payload = [model.model_dump(mode="json", by_alias=True) for model in self.models]
        self.models_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _discover_models(self) -> list[ModelInstall]:
        candidates: list[tuple[str, Path, str]] = [
            (WAN_BASE_REPO_ID, self.root / WAN_BASE_MODEL_DIR, "huggingface"),
            (WAN_BASE_REPO_ID, Path(WAN_BASE_MODEL_DIR), "huggingface"),
            (WAN_BASE_REPO_ID, Path("/content/drive/MyDrive/WanStudio/models/Wan2.2-TI2V-5B"), "huggingface"),
            (DEFAULT_REPO_ID, self.root / DEFAULT_MODEL_DIR, "huggingface"),
            (DEFAULT_REPO_ID, Path(DEFAULT_MODEL_DIR), "huggingface"),
            (DEFAULT_REPO_ID, Path(f"/content/drive/MyDrive/WanStudio/{DEFAULT_MODEL_DIR}"), "huggingface"),
        ]
        discovered: list[ModelInstall] = []
        for repo_id, path, source in candidates:
            if not path.exists():
                continue
            model = create_model_install(repo_id, str(path), source)
            if model.status == "ready":
                discovered.append(model)
        return discovered

    def connect_model(self, data: ConnectModelInput) -> ModelInstall:
        model = create_model_install(data.repo_id, data.local_path, data.source)
        self.models = [model, *[item for item in self.models if item.model_id != model.model_id]]
        self._save_models()
        return model

    def ready_models(self) -> list[ModelInstall]:
        return [model for model in self.models if model.status in {"ready", "recommended"}]


def create_app(*, root: Path | None = None, runner_kind: str = "fake", wan_repo_dir: Path | None = None) -> FastAPI:
    runner: WanRunner
    if runner_kind == "wan":
        if not wan_repo_dir:
            raise ValueError("--wan-repo-dir is required when --runner wan is used")
        runner = SubprocessWanRunner(wan_repo_dir)
    else:
        runner = FakeWanRunner()

    state = WebState(root=root or Path.cwd(), runner=runner)
    app = FastAPI(title="Wan Studio", version="0.1.0")
    app.state.wan_studio = state

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    @app.get("/api/state")
    async def get_state() -> dict[str, object]:
        suggested_real_model_dir = "/content/drive/MyDrive/WanStudio/models/Wan2.2-TI2V-5B" if Path("/content").exists() else str(state.root / WAN_BASE_MODEL_DIR)
        suggested_lora_dir = f"/content/drive/MyDrive/WanStudio/{DEFAULT_MODEL_DIR}" if Path("/content").exists() else str(state.root / DEFAULT_MODEL_DIR)
        return {
            "root": str(state.root),
            "runner": runner_kind,
            "models": [model.model_dump(mode="json", by_alias=True) for model in state.models],
            "jobs": [job.model_dump(mode="json", by_alias=True) for job in state.jobs.values()],
            "downloadCommand": build_hf_command(DEFAULT_REPO_ID, str(state.root / DEFAULT_MODEL_DIR)),
            "defaultRepoId": DEFAULT_REPO_ID,
            "defaultModelDir": DEFAULT_MODEL_DIR,
            "defaultModelNote": DEFAULT_MODEL_NOTE,
            "wanBaseRepoId": WAN_BASE_REPO_ID,
            "wanBaseModelDir": suggested_real_model_dir,
            "defaultLoraRepoId": DEFAULT_REPO_ID,
            "defaultLoraModelDir": suggested_lora_dir,
        }

    @app.get("/api/loras")
    async def list_loras(path: str | None = None) -> dict[str, object]:
        root = Path(path or state.root / DEFAULT_MODEL_DIR).expanduser()
        if not root.exists():
            return {"loras": []}
        if root.is_file():
            loras = [root] if root.suffix.lower() in LORA_SUFFIXES else []
        else:
            loras = sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in LORA_SUFFIXES)
        return {"loras": [str(item) for item in loras[:200]], "truncated": len(loras) > 200}

    @app.post("/api/models/connect")
    async def connect_model(data: ConnectModelInput) -> dict[str, object]:
        model = state.connect_model(data)
        return {"model": model.model_dump(mode="json", by_alias=True)}

    @app.post("/api/jobs")
    async def create_job(data: CreateJobInput) -> dict[str, object]:
        model = next((item for item in state.models if item.model_id == data.model_id), None)
        if not model:
            raise HTTPException(status_code=404, detail="model not connected")
        if model.status != "ready":
            raise HTTPException(status_code=400, detail=f"model is {model.status}, not ready")
        request = GenerationRequest(
            prompt=data.prompt,
            image=data.image or None,
            model_id=model.model_id,
            model_path=model.local_path,
            size=data.size,
            seed=data.seed,
            steps=data.steps,
            offload_model=data.offload_model,
            t5_cpu=data.t5_cpu,
            lora_paths=data.lora_paths,
            lora_scale=data.lora_scale,
            runtime=RuntimeKind.LOCAL,
            task=data.task,
        )
        job = GenerationJob(request=request, status=status_for(request.id, JobState.QUEUED, 0))
        state.jobs[request.id] = job
        state.tasks[request.id] = asyncio.create_task(run_job(state, request))
        return {"job": job.model_dump(mode="json", by_alias=True)}

    @app.get("/api/jobs")
    async def list_jobs() -> dict[str, object]:
        jobs = sorted(state.jobs.values(), key=lambda job: job.request.created_at, reverse=True)
        return {"jobs": [job.model_dump(mode="json", by_alias=True) for job in jobs]}

    @app.get("/outputs/{file_name}")
    async def output(file_name: str) -> FileResponse:
        safe_name = Path(file_name).name
        path = state.output_dir / safe_name
        if not path.exists():
            raise HTTPException(status_code=404, detail="output not found")
        return FileResponse(path)

    return app


async def run_job(state: WebState, request: GenerationRequest) -> None:
    async def on_progress(status: GenerationStatus) -> None:
        if request.id in state.jobs:
            state.jobs[request.id].status = status

    try:
        state.jobs[request.id].status = status_for(request.id, JobState.RUNNING, 1)
        state.jobs[request.id].status = await state.runner.run(request, state.output_dir, on_progress)
    except asyncio.CancelledError:
        state.jobs[request.id].status = status_for(request.id, JobState.CANCELLED, state.jobs[request.id].status.progress)
    except Exception as exc:
        current = state.jobs[request.id].status.progress
        state.jobs[request.id].status = status_for(request.id, JobState.FAILED, current, error=str(exc))


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Wan Studio</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --bg: #eef2f5;
      --surface: #ffffff;
      --surface-strong: #f7fafc;
      --ink: #17202a;
      --muted: #657382;
      --border: #d8e0e7;
      --accent: #0f8f86;
      --accent-strong: #096a65;
      --accent-soft: #dff4ef;
      --danger: #bd2d3a;
      --warn: #c86b24;
      --shadow: 0 18px 50px rgba(32, 50, 62, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(140deg, rgba(15, 143, 134, 0.11), transparent 34%),
        radial-gradient(circle at 88% 10%, rgba(40, 85, 126, 0.12), transparent 30%),
        var(--bg);
    }
    button, input, select, textarea { font: inherit; }
    button { border: 0; cursor: pointer; }
    .shell { display: grid; grid-template-columns: 260px minmax(0, 1fr); min-height: 100vh; }
    .sidebar {
      padding: 26px 18px;
      border-right: 1px solid rgba(156, 171, 183, 0.38);
      background: rgba(255,255,255,0.74);
    }
    .brand { display: flex; gap: 12px; align-items: center; margin-bottom: 30px; }
    .mark {
      width: 42px; height: 42px; display: grid; place-items: center; color: white;
      border-radius: 12px; background: linear-gradient(145deg, #112833, #0f8f86);
      box-shadow: 0 12px 24px rgba(15, 143, 134, 0.24);
      font-weight: 800;
    }
    .brand strong { display: block; font-size: 15px; }
    .brand span { display: block; color: var(--muted); font-size: 12px; margin-top: 2px; }
    .nav { display: grid; gap: 8px; }
    .nav a { color: #40505e; text-decoration: none; padding: 11px 12px; border-radius: 8px; font-weight: 700; font-size: 14px; }
    .nav a:hover { background: white; color: var(--ink); }
    .workspace { padding: 30px; }
    .topbar { display: flex; justify-content: space-between; align-items: start; gap: 20px; margin-bottom: 22px; }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: 34px; line-height: 1.05; }
    h2 { font-size: 17px; }
    h3 { font-size: 15px; }
    .topbar p, .section p, .muted { color: var(--muted); font-size: 13px; line-height: 1.45; }
    .badge { padding: 8px 10px; border-radius: 999px; color: var(--accent-strong); background: var(--accent-soft); font-size: 12px; font-weight: 800; }
    .grid { display: grid; grid-template-columns: minmax(320px, 0.95fr) minmax(380px, 1.05fr); gap: 18px; align-items: start; }
    .section {
      padding: 20px;
      border: 1px solid rgba(198, 210, 219, 0.8);
      border-radius: 12px;
      background: rgba(255,255,255,0.88);
      box-shadow: var(--shadow);
    }
    .section-head { display: flex; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
    label { display: grid; gap: 8px; color: #31414d; font-size: 12px; font-weight: 760; }
    input, select, textarea {
      width: 100%; min-width: 0; border: 1px solid var(--border); border-radius: 8px;
      background: #fbfdfe; color: var(--ink); font-size: 13px; line-height: 1.35; outline: 0;
    }
    input, select { height: 42px; padding: 0 12px; }
    textarea { min-height: 150px; resize: vertical; padding: 13px; }
    input:focus, select:focus, textarea:focus { border-color: rgba(15,143,134,0.72); box-shadow: 0 0 0 4px rgba(15,143,134,0.12); }
    .form { display: grid; gap: 14px; }
    .row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .command {
      display: grid; gap: 8px; margin-top: 14px; padding: 12px; border-radius: 10px;
      background: #16232d; color: #e7f5f3; font: 12px/1.55 "SFMono-Regular", Consolas, monospace;
      white-space: pre-wrap; word-break: break-word;
    }
    .button {
      min-height: 42px; display: inline-flex; align-items: center; justify-content: center; gap: 8px;
      padding: 0 14px; border-radius: 8px; color: white; background: var(--accent); font-size: 13px; font-weight: 800;
    }
    .button:hover { background: var(--accent-strong); }
    .button.secondary { color: #324350; background: var(--surface); border: 1px solid var(--border); }
    .button:disabled { opacity: 0.5; cursor: not-allowed; }
    .model-list, .jobs { display: grid; gap: 10px; }
    .card-row {
      display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 12px; align-items: center;
      padding: 14px; border: 1px solid #dce5ec; border-radius: 10px; background: var(--surface-strong);
    }
    .card-row p { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .state { font-size: 12px; font-weight: 800; color: var(--accent-strong); }
    .state.failed { color: var(--danger); }
    .state.running { color: var(--warn); }
    .progress { height: 8px; overflow: hidden; border-radius: 999px; background: #d9e3e9; margin-top: 10px; }
    .progress span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), #79c89e); }
    .wide { grid-column: 1 / -1; }
    .notice { padding: 12px 14px; border: 1px solid #d8e9e5; border-radius: 10px; background: #f0faf7; color: var(--accent-strong); font-size: 13px; line-height: 1.45; font-weight: 760; }
    .error { color: var(--danger); font-size: 12px; line-height: 1.45; font-weight: 760; }
    .loading-layer {
      position: fixed; inset: 0; z-index: 20; display: none; place-items: center;
      background: rgba(238, 242, 245, 0.68); backdrop-filter: blur(7px);
    }
    .loading-layer.active { display: grid; }
    .loading-box {
      width: min(360px, calc(100vw - 36px)); padding: 18px; border: 1px solid rgba(198, 210, 219, 0.9);
      border-radius: 12px; background: rgba(255,255,255,0.94); box-shadow: var(--shadow);
    }
    .spinner {
      width: 28px; height: 28px; border-radius: 999px; border: 3px solid #d8e9e5;
      border-top-color: var(--accent); animation: spin 0.86s linear infinite; margin-bottom: 12px;
    }
    .loading-box strong { display: block; font-size: 14px; margin-bottom: 4px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--border); }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="loading-layer" id="loadingLayer" aria-live="polite" aria-busy="true">
    <div class="loading-box">
      <div class="spinner"></div>
      <strong id="loadingTitle">Working</strong>
      <p class="muted" id="loadingText">Please keep this tab open.</p>
    </div>
  </div>
  <main class="shell">
    <aside class="sidebar">
      <div class="brand"><div class="mark">W</div><div><strong>Wan Studio</strong><span>Installable Web UI</span></div></div>
      <nav class="nav">
        <a href="#models">Models</a>
        <a href="#prompt">Prompt</a>
        <a href="#jobs">Jobs</a>
      </nav>
    </aside>
    <section class="workspace">
      <header class="topbar">
        <div>
          <h1>Wan Studio</h1>
          <p>Download the default Wan adapter set, connect the folder, then continue from this Web UI.</p>
        </div>
        <span class="badge" id="runnerBadge">Loading</span>
      </header>

      <div class="grid">
        <section class="section" id="models">
          <div class="section-head">
            <div><h2>Model setup</h2><p>Bring a Hugging Face, ModelScope, or existing Wan-compatible folder.</p></div>
          </div>
          <div class="form">
            <div class="row">
              <label>Source
                <select id="source"><option value="huggingface">Hugging Face</option><option value="modelscope">ModelScope</option><option value="local">Local only</option></select>
              </label>
              <label>Repo id
                <input id="repoId" value="lkzd7/WAN2.2_LoraSet_NSFW" />
              </label>
            </div>
            <label>Model folder
              <input id="modelPath" value="models/WAN2.2_LoraSet_NSFW" />
            </label>
            <div class="notice">Default repo is a Wan2.2 LoRA/adapters set. For real Wan inference, pair it with a compatible Wan2.2 base model runner.</div>
            <div class="command" id="downloadCommand"></div>
            <button class="button" id="connectModel">Connect model</button>
            <div id="modelMessage" class="muted"></div>
          </div>
          <div class="section-head" style="margin-top:22px"><div><h2>Connected models</h2><p id="modelCount">No models connected.</p></div></div>
          <div class="model-list" id="modelsList"></div>
        </section>

        <section class="section" id="prompt">
          <div class="section-head">
            <div><h2>Prompt</h2><p>Send a job only after a model shows Ready to prompt.</p></div>
          </div>
          <div class="form">
            <div id="readyBanner" class="notice">Connect a ready Wan-compatible folder first.</div>
            <label>Model
              <select id="modelSelect"></select>
            </label>
            <div class="row">
              <label>Task
                <select id="task"><option value="t2v">Text to video</option><option value="i2v">Image to video</option><option value="ti2v">Text + image to video</option></select>
              </label>
              <label>Size
                <input id="size" value="832x480" />
              </label>
            </div>
            <label>Prompt text
              <textarea id="promptText">A cinematic camera move across a quiet neon street after rain.</textarea>
            </label>
            <label>Image reference path
              <input id="imagePath" placeholder="/path/to/reference.png" />
            </label>
            <label>LoRA adapter folder or file
              <input id="loraPath" placeholder="/content/drive/MyDrive/WanStudio/models/WAN2.2_LoraSet_NSFW or a .safetensors file" />
            </label>
            <div class="row">
              <label>LoRA file
                <select id="loraSelect"><option value="">No LoRA layer</option></select>
              </label>
              <label>LoRA scale
                <input id="loraScale" type="number" min="0" max="2" step="0.05" value="1" />
              </label>
            </div>
            <button class="button secondary" id="scanLoras">Scan LoRA files</button>
            <div class="row">
              <label>Steps
                <input id="steps" type="number" min="1" max="80" value="18" />
              </label>
              <label>Seed
                <input id="seed" type="number" value="0" />
              </label>
            </div>
            <button class="button" id="runJob">Run generation</button>
            <div id="jobMessage" class="muted"></div>
          </div>
        </section>

        <section class="section wide" id="jobs">
          <div class="section-head">
            <div><h2>Jobs</h2><p>Recent requests and generated outputs.</p></div>
            <button class="button secondary" id="refreshJobs">Refresh</button>
          </div>
          <div class="jobs" id="jobsList"></div>
        </section>
      </div>
    </section>
  </main>
  <script>
    const state = { models: [], jobs: [], root: "", runner: "fake" };
    const $ = (id) => document.getElementById(id);
    const appBase = () => new URL(".", window.location.href);
    const appUrl = (path) => new URL(path.replace(/^\/+/, ""), appBase()).toString();

    function setBusy(active, title = "Working", text = "Please keep this tab open.") {
      $("loadingTitle").textContent = title;
      $("loadingText").textContent = text;
      $("loadingLayer").classList.toggle("active", active);
    }

    async function withBusy(title, text, task) {
      setBusy(true, title, text);
      try {
        return await task();
      } finally {
        setBusy(false);
      }
    }

    async function api(path, options = {}) {
      const res = await fetch(appUrl(path), { headers: { "content-type": "application/json" }, ...options });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }
      return res.json();
    }

    function commandFor() {
      const source = $("source").value;
      const repo = $("repoId").value.trim() || "<repo-id>";
      const path = $("modelPath").value.trim() || "<model-folder>";
      if (source === "local") return "Use an existing Wan model folder and connect it below.";
      if (source === "modelscope") return `modelscope download ${repo} --local_dir ${path}`;
      return `hf download ${repo} --local-dir ${path}`;
    }

    function renderModels() {
      $("downloadCommand").textContent = commandFor();
      $("modelCount").textContent = state.models.length ? `${state.models.length} connected model(s)` : "No models connected.";
      $("modelsList").innerHTML = state.models.map((model) => `
        <article class="card-row">
          <div><h3>${model.displayName}</h3><p>${model.localPath}</p><p class="muted">${model.repoId || model.source}</p><p class="muted">${(model.capabilities?.notes || []).join(" · ")}</p></div>
          <strong class="state ${model.status === "ready" ? "" : "failed"}">${model.status}</strong>
        </article>
      `).join("");
      const ready = state.models.filter((model) => model.status === "ready");
      $("modelSelect").innerHTML = ready.length
        ? ready.map((model) => `<option value="${model.modelId}">${model.displayName}</option>`).join("")
        : `<option value="">Connect a ready model first</option>`;
      $("runJob").disabled = ready.length === 0;
      $("readyBanner").textContent = ready.length
        ? "Ready to prompt! A connected folder is available in this Web UI."
        : state.runner === "wan"
          ? "Real Wan runner mode. Connect Wan-AI/Wan2.2-TI2V-5B or another compatible base checkpoint folder first."
          : "Connect a ready Wan-compatible folder first.";
    }

    function renderJobs() {
      $("jobsList").innerHTML = state.jobs.length ? state.jobs.map((job) => {
        const status = job.status;
        const outputName = status.outputPath ? status.outputPath.split("/").pop() : "";
        const output = outputName && status.state === "succeeded" ? `<p><a href="${appUrl("outputs/" + outputName)}" target="_blank">${status.outputPath}</a></p>` : "";
        const error = status.error ? `<p class="error">${status.error}</p>` : "";
        const loras = job.request.loraPaths?.length ? ` · LoRA ${job.request.loraPaths.length}` : "";
        return `<article class="card-row">
          <div>
            <h3>${job.request.prompt}</h3>
            <p>${job.request.modelId} · ${job.request.task} · ${job.request.size}${loras}</p>
            <div class="progress"><span style="width:${status.progress}%"></span></div>
            ${output}${error}
          </div>
          <strong class="state ${status.state}">${status.state} ${status.progress}%</strong>
        </article>`;
      }).join("") : `<div class="notice">No jobs yet. Connect a model and run your first prompt.</div>`;
    }

    async function loadState() {
      const payload = await api("api/state");
      state.root = payload.root;
      state.runner = payload.runner;
      state.models = payload.models || [];
      state.jobs = payload.jobs || [];
      $("runnerBadge").textContent = `${state.runner} runner · ${state.root}`;
      if (state.runner === "wan") {
        if ($("repoId").value === payload.defaultRepoId) $("repoId").value = payload.wanBaseRepoId || "Wan-AI/Wan2.2-TI2V-5B";
        if ($("modelPath").value.includes("WAN2.2_LoraSet_NSFW")) $("modelPath").value = payload.wanBaseModelDir || "models/Wan2.2-TI2V-5B";
        if (!$("loraPath").value) $("loraPath").value = payload.defaultLoraModelDir || "";
        if ($("size").value === "832x480") $("size").value = "1280x704";
        if ($("steps").value === "18") $("steps").value = "24";
        $("readyBanner").textContent = "Real Wan runner mode. Connect Wan-AI/Wan2.2-TI2V-5B or another compatible base checkpoint folder.";
      }
      renderModels();
      renderJobs();
    }

    async function connectModel() {
      const payload = await withBusy("Checking model", "Validating the selected Wan folder.", () => api("api/models/connect", {
          method: "POST",
          body: JSON.stringify({ repo_id: $("repoId").value, local_path: $("modelPath").value, source: $("source").value })
        })
      );
      $("modelMessage").textContent = `Connected: ${payload.model.displayName} (${payload.model.status})`;
      await loadState();
    }

    function selectedLoraPaths() {
      const selected = $("loraSelect").value;
      const rawPath = $("loraPath").value.trim();
      if (selected) return [selected];
      if (rawPath.match(/\.(safetensors|pt|pth|ckpt)$/i)) return [rawPath];
      return [];
    }

    async function scanLoras() {
      const rawPath = $("loraPath").value.trim();
      const payload = await withBusy("Scanning LoRA files", "Reading adapter files from the selected folder.", () => api(`api/loras?path=${encodeURIComponent(rawPath)}`));
      const loras = payload.loras || [];
      $("loraSelect").innerHTML = loras.length
        ? [`<option value="">No LoRA layer</option>`, ...loras.map((path) => `<option value="${path}">${path.split("/").pop()}</option>`)].join("")
        : `<option value="">No LoRA files found</option>`;
      $("jobMessage").textContent = loras.length
        ? `Found ${loras.length} LoRA file(s). Pick one before running if you want an adapter layer.`
        : "No LoRA files found at that path.";
    }

    async function runJob() {
      $("jobMessage").textContent = "Queueing job...";
      const loraPaths = selectedLoraPaths();
      await withBusy("Queueing generation", loraPaths.length ? "Attaching the selected LoRA layer to this job." : "Sending the prompt to the runner.", () => api("api/jobs", {
          method: "POST",
          body: JSON.stringify({
            prompt: $("promptText").value,
            model_id: $("modelSelect").value,
            task: $("task").value,
            image: $("imagePath").value,
            size: $("size").value,
            steps: Number($("steps").value),
            seed: Number($("seed").value),
            offload_model: true,
            t5_cpu: true,
            lora_paths: loraPaths,
            lora_scale: Number($("loraScale").value) || 1
          })
        })
      );
      $("jobMessage").textContent = loraPaths.length
        ? "Job queued with LoRA layer. Watch the Jobs panel for compatibility errors or output."
        : "Job queued.";
      await refreshJobs();
    }

    async function refreshJobs() {
      const payload = await api("api/jobs");
      state.jobs = payload.jobs || [];
      renderJobs();
    }

    ["source", "repoId", "modelPath"].forEach((id) => $(id).addEventListener("input", renderModels));
    $("connectModel").addEventListener("click", () => connectModel().catch((error) => $("modelMessage").textContent = error.message));
    $("scanLoras").addEventListener("click", () => scanLoras().catch((error) => $("jobMessage").textContent = error.message));
    $("runJob").addEventListener("click", () => runJob().catch((error) => $("jobMessage").textContent = error.message));
    $("refreshJobs").addEventListener("click", () => refreshJobs());
    setInterval(() => refreshJobs().catch(() => {}), 1800);
    loadState().catch((error) => {
      $("runnerBadge").textContent = "API not connected";
      $("modelMessage").textContent = error.message;
    });
  </script>
</body>
</html>"""
