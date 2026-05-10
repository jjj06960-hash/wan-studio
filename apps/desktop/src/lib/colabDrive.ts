import { invoke, isTauri } from "@tauri-apps/api/core";
import { driveJobFileName, serializeDriveJob } from "./jobs";
import type { ColabDriveStatus, GenerationJob, GenerationStatus } from "./types";

type BrowserDirectoryHandle = FileSystemDirectoryHandle & {
  values?: () => AsyncIterable<FileSystemHandle>;
};

declare global {
  interface Window {
    showDirectoryPicker?: (options?: { id?: string; mode?: "read" | "readwrite" }) => Promise<FileSystemDirectoryHandle>;
  }
}

interface TauriStatusFile {
  fileName: string;
  payload: unknown;
}

interface QueueResult {
  ok: boolean;
  path?: string;
  message: string;
}

let browserDriveHandle: BrowserDirectoryHandle | null = null;

const EMPTY_STATUS: ColabDriveStatus = {
  rootPath: "",
  connected: false,
  runtimeReady: false,
  modelReady: false,
  readyToPrompt: false,
  message: "Connect your WanStudio Drive folder.",
  mode: "manual",
  statusCount: 0,
  resultCount: 0,
};

export function supportsBrowserDirectoryPicker(): boolean {
  return typeof window !== "undefined" && typeof window.showDirectoryPicker === "function";
}

export async function chooseBrowserDriveFolder(): Promise<ColabDriveStatus> {
  if (!supportsBrowserDirectoryPicker()) {
    return {
      ...EMPTY_STATUS,
      message: "This preview browser cannot directly choose a Drive folder. Use the desktop build or paste a synced folder path.",
      error: "Directory picker unavailable",
    };
  }

  const picker = window.showDirectoryPicker;
  if (!picker) {
    return {
      ...EMPTY_STATUS,
      message: "This preview browser cannot directly choose a Drive folder. Use the desktop build or paste a synced folder path.",
      error: "Directory picker unavailable",
    };
  }

  browserDriveHandle = (await picker({
    id: "wan-studio-colab-drive",
    mode: "readwrite",
  })) as BrowserDirectoryHandle;

  await ensureBrowserTree(browserDriveHandle);
  return inspectColabDrive(browserDriveHandle.name);
}

export async function inspectColabDrive(rootPath: string): Promise<ColabDriveStatus> {
  const trimmedRoot = rootPath.trim();

  if (runningInTauri() && trimmedRoot) {
    try {
      return await invoke<ColabDriveStatus>("read_colab_drive", { rootPath: trimmedRoot });
    } catch (error) {
      return {
        ...EMPTY_STATUS,
        rootPath: trimmedRoot,
        mode: "tauri",
        message: "Could not read the WanStudio Drive folder.",
        error: errorMessage(error),
      };
    }
  }

  if (browserDriveHandle) {
    try {
      return await inspectBrowserDrive(browserDriveHandle, trimmedRoot || browserDriveHandle.name);
    } catch (error) {
      return {
        ...EMPTY_STATUS,
        rootPath: trimmedRoot || browserDriveHandle.name,
        mode: "browser-directory",
        message: "Could not read the selected WanStudio folder.",
        error: errorMessage(error),
      };
    }
  }

  if (trimmedRoot) {
    return {
      ...EMPTY_STATUS,
      rootPath: trimmedRoot,
      mode: "manual",
      message: "Folder path saved. Open the desktop build or choose a synced Drive folder to read status files.",
    };
  }

  return EMPTY_STATUS;
}

export async function queueColabDriveJob(rootPath: string, job: GenerationJob): Promise<QueueResult> {
  const fileName = driveJobFileName(job);
  const payload = serializeDriveJob(job);
  const trimmedRoot = rootPath.trim();

  if (runningInTauri() && trimmedRoot) {
    try {
      const path = await invoke<string>("write_colab_job", { rootPath: trimmedRoot, fileName, payload });
      return { ok: true, path, message: "Job request written to Drive." };
    } catch (error) {
      return { ok: false, message: errorMessage(error) };
    }
  }

  if (browserDriveHandle) {
    try {
      const jobsDir = await browserDriveHandle.getDirectoryHandle("jobs", { create: true });
      const file = await jobsDir.getFileHandle(fileName, { create: true });
      const writer = await file.createWritable();
      await writer.write(payload);
      await writer.close();
      return { ok: true, path: `${browserDriveHandle.name}/jobs/${fileName}`, message: "Job request written to the selected Drive folder." };
    } catch (error) {
      return { ok: false, message: errorMessage(error) };
    }
  }

  downloadTextFile(fileName, payload);
  return {
    ok: true,
    path: fileName,
    message: "Job JSON downloaded. Put it in WanStudio/jobs if direct Drive access is not connected.",
  };
}

export async function readColabJobStatuses(rootPath: string): Promise<GenerationStatus[]> {
  const trimmedRoot = rootPath.trim();

  if (runningInTauri() && trimmedRoot) {
    try {
      const files = await invoke<TauriStatusFile[]>("read_colab_status_files", { rootPath: trimmedRoot });
      return files.map((file) => normalizeGenerationStatus(file.payload)).filter((status): status is GenerationStatus => Boolean(status));
    } catch {
      return [];
    }
  }

  if (!browserDriveHandle) {
    return [];
  }

  try {
    const statusDir = await browserDriveHandle.getDirectoryHandle("status", { create: true });
    const statuses: GenerationStatus[] = [];
    for await (const handle of directoryValues(statusDir as BrowserDirectoryHandle)) {
      if (handle.kind !== "file" || !handle.name.endsWith(".status.json")) {
        continue;
      }
      const file = await (handle as FileSystemFileHandle).getFile();
      const status = normalizeGenerationStatus(JSON.parse(await file.text()));
      if (status) {
        statuses.push(status);
      }
    }
    return statuses;
  } catch {
    return [];
  }
}

export function mergeColabStatuses(jobs: GenerationJob[], statuses: GenerationStatus[]): GenerationJob[] {
  if (statuses.length === 0) {
    return jobs;
  }

  const byId = new Map(statuses.map((status) => [status.id, status]));
  return jobs.map((job) => {
    if (job.request.runtime !== "colab-drive") {
      return job;
    }
    const status = byId.get(job.request.id);
    return status ? { ...job, status } : job;
  });
}

async function inspectBrowserDrive(handle: BrowserDirectoryHandle, rootPath: string): Promise<ColabDriveStatus> {
  await ensureBrowserTree(handle);
  const statusDir = await handle.getDirectoryHandle("status", { create: true });
  const resultsDir = await handle.getDirectoryHandle("results", { create: true });
  const runtime = await readJsonFile(statusDir, "runtime.json");
  const model = await readFirstJsonFile(statusDir, "model-", ".json");
  const statusCount = await countFiles(statusDir as BrowserDirectoryHandle, ".status.json");
  const resultCount = await countFiles(resultsDir as BrowserDirectoryHandle, ".mp4");

  return normalizeDriveStatus({
    rootPath,
    mode: "browser-directory",
    runtime,
    model,
    statusCount,
    resultCount,
  });
}

function normalizeDriveStatus(input: {
  rootPath: string;
  mode: ColabDriveStatus["mode"];
  runtime: Record<string, unknown> | null;
  model: Record<string, unknown> | null;
  statusCount: number;
  resultCount: number;
}): ColabDriveStatus {
  const runtimeReady = input.runtime?.ready === true || input.runtime?.message === "Ready to prompt!";
  const modelReady = input.model?.status === "ready";
  const readyToPrompt = runtimeReady && modelReady;
  const modelPath = typeof input.model?.localPath === "string" ? input.model.localPath : undefined;
  const lastCheckedAt =
    (typeof input.model?.checkedAt === "string" && input.model.checkedAt) ||
    (typeof input.runtime?.checkedAt === "string" && input.runtime.checkedAt) ||
    undefined;

  return {
    rootPath: input.rootPath,
    connected: true,
    runtimeReady,
    modelReady,
    readyToPrompt,
    message: readyToPrompt ? "Ready to prompt!" : runtimeReady ? "Runtime connected. Waiting for model verification." : "Waiting for Colab worker.",
    mode: input.mode,
    modelPath,
    lastCheckedAt,
    statusCount: input.statusCount,
    resultCount: input.resultCount,
  };
}

function normalizeGenerationStatus(value: unknown): GenerationStatus | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const item = value as Partial<GenerationStatus>;
  if (!item.id || !item.state || typeof item.progress !== "number") {
    return null;
  }
  return {
    id: item.id,
    state: item.state,
    progress: item.progress,
    outputPath: item.outputPath,
    error: item.error,
    updatedAt: item.updatedAt ?? new Date().toISOString(),
  };
}

async function ensureBrowserTree(handle: FileSystemDirectoryHandle): Promise<void> {
  await handle.getDirectoryHandle("jobs", { create: true });
  await handle.getDirectoryHandle("results", { create: true });
  await handle.getDirectoryHandle("status", { create: true });
}

async function readJsonFile(handle: FileSystemDirectoryHandle, name: string): Promise<Record<string, unknown> | null> {
  try {
    const fileHandle = await handle.getFileHandle(name);
    const file = await fileHandle.getFile();
    return JSON.parse(await file.text()) as Record<string, unknown>;
  } catch {
    return null;
  }
}

async function readFirstJsonFile(handle: BrowserDirectoryHandle, prefix: string, suffix: string): Promise<Record<string, unknown> | null> {
  for await (const item of directoryValues(handle)) {
    if (item.kind === "file" && item.name.startsWith(prefix) && item.name.endsWith(suffix)) {
      const file = await (item as FileSystemFileHandle).getFile();
      return JSON.parse(await file.text()) as Record<string, unknown>;
    }
  }
  return null;
}

async function countFiles(handle: BrowserDirectoryHandle, suffix: string): Promise<number> {
  let count = 0;
  for await (const item of directoryValues(handle)) {
    if (item.kind === "file" && item.name.endsWith(suffix)) {
      count += 1;
    }
  }
  return count;
}

async function* directoryValues(handle: BrowserDirectoryHandle): AsyncIterable<FileSystemHandle> {
  if (handle.values) {
    yield* handle.values();
  }
}

function runningInTauri(): boolean {
  try {
    return isTauri();
  } catch {
    return false;
  }
}

function downloadTextFile(fileName: string, text: string): void {
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
