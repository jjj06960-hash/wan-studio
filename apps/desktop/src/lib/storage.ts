import type { GenerationJob, ModelInstall, RuntimeKind } from "./types";

const MODEL_KEY = "wan-studio.models.v1";
const JOB_KEY = "wan-studio.jobs.v1";
const RUNTIME_KEY = "wan-studio.runtime.v1";
const ONBOARDING_KEY = "wan-studio.onboarding-complete.v1";
const COLAB_ROOT_KEY = "wan-studio.colab-root.v1";

export function loadModels(): ModelInstall[] {
  return load<ModelInstall[]>(MODEL_KEY, []);
}

export function saveModels(models: ModelInstall[]): void {
  save(MODEL_KEY, models);
}

export function loadJobs(): GenerationJob[] {
  return load<GenerationJob[]>(JOB_KEY, []);
}

export function saveJobs(jobs: GenerationJob[]): void {
  save(JOB_KEY, jobs.slice(0, 20));
}

export function loadRuntime(): RuntimeKind {
  return load<RuntimeKind>(RUNTIME_KEY, "local");
}

export function saveRuntime(runtime: RuntimeKind): void {
  save(RUNTIME_KEY, runtime);
}

export function loadOnboardingComplete(): boolean {
  return load<boolean>(ONBOARDING_KEY, false);
}

export function saveOnboardingComplete(complete: boolean): void {
  save(ONBOARDING_KEY, complete);
}

export function loadColabRootPath(): string {
  return load<string>(COLAB_ROOT_KEY, "");
}

export function saveColabRootPath(rootPath: string): void {
  save(COLAB_ROOT_KEY, rootPath);
}

function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function save<T>(key: string, value: T): void {
  localStorage.setItem(key, JSON.stringify(value));
}
