export type WanTask = "t2v" | "i2v" | "ti2v" | "s2v" | "animate";

export type ModelSource = "huggingface" | "modelscope" | "local";

export type ModelStatus = "recommended" | "ready" | "missing" | "invalid" | "unchecked";

export type RuntimeKind = "local" | "colab-drive";

export interface ModelCapabilities {
  tasks: WanTask[];
  optimized: boolean;
  minVramGb?: number;
  notes: string[];
}

export interface ModelInstall {
  modelId: string;
  displayName: string;
  family: "wan";
  task: WanTask | "multi";
  localPath: string;
  source: ModelSource;
  status: ModelStatus;
  repoId?: string;
  capabilities: ModelCapabilities;
  lastVerifiedAt?: string;
}

export interface GenerationRequest {
  id: string;
  prompt: string;
  image?: string;
  modelId: string;
  modelPath?: string;
  size: string;
  seed: number;
  steps: number;
  offloadModel: boolean;
  t5Cpu: boolean;
  runtime: RuntimeKind;
  task: WanTask;
  createdAt: string;
}

export interface GenerationStatus {
  id: string;
  state: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  progress: number;
  outputPath?: string;
  error?: string;
  updatedAt: string;
}

export interface GenerationJob {
  request: GenerationRequest;
  status: GenerationStatus;
}

export interface PresetSettings {
  size: string;
  steps: number;
  offloadModel: boolean;
  t5Cpu: boolean;
}

export type ColabDriveMode = "tauri" | "browser-directory" | "manual";

export interface ColabDriveStatus {
  rootPath: string;
  connected: boolean;
  runtimeReady: boolean;
  modelReady: boolean;
  readyToPrompt: boolean;
  message: string;
  mode: ColabDriveMode;
  modelPath?: string;
  lastCheckedAt?: string;
  statusCount: number;
  resultCount: number;
  error?: string;
}
