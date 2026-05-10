import type { ModelCapabilities, ModelInstall, PresetSettings, WanTask } from "./types";

export const WAN_TASK_LABELS: Record<WanTask, string> = {
  t2v: "Text to video",
  i2v: "Image to video",
  ti2v: "Text + image to video",
  s2v: "Speech to video",
  animate: "Animate",
};

export const RECOMMENDED_WAN_MODEL: ModelInstall = {
  modelId: "wan2.2-loraset-nsfw",
  displayName: "WAN2.2 LoRA Set",
  family: "wan",
  task: "multi",
  localPath: "",
  source: "huggingface",
  status: "recommended",
  repoId: "lkzd7/WAN2.2_LoraSet_NSFW",
  capabilities: {
    tasks: ["i2v", "ti2v"],
    optimized: false,
    notes: [
      "Default Wan2.2 LoRA/adapters download target",
      "Requires a compatible Wan2.2 base model runner for real inference",
      "Use as an add-on model folder, not as a standalone base checkpoint",
    ],
  },
};

export const OPTIMIZED_WAN_5B_PRESET: PresetSettings = {
  size: "1280x704",
  steps: 24,
  offloadModel: true,
  t5Cpu: true,
};

export const SAFE_CUSTOM_WAN_PRESET: PresetSettings = {
  size: "832x480",
  steps: 18,
  offloadModel: true,
  t5Cpu: true,
};

const WEIGHT_FILE_PATTERN = /\.(safetensors|bin|pt|pth|ckpt)$/i;

export function buildHuggingFaceCommand(repoId: string, targetPath: string): string {
  const safeRepo = repoId.trim() || "<repo-id>";
  const safePath = targetPath.trim() || "<model-folder>";
  return `hf download ${safeRepo} --local-dir ${safePath}`;
}

export function buildModelScopeCommand(repoId: string, targetPath: string): string {
  const safeRepo = repoId.trim() || "<repo-id>";
  const safePath = targetPath.trim() || "<model-folder>";
  return `modelscope download ${safeRepo} --local_dir ${safePath}`;
}

export function inferWanCapabilities(input: string): ModelCapabilities {
  const normalized = input.toLowerCase();
  const tasks = new Set<WanTask>();
  const notes: string[] = [];

  if (normalized.includes("ti2v")) {
    tasks.add("t2v");
    tasks.add("i2v");
    tasks.add("ti2v");
    notes.push("Hybrid TI2V naming detected");
  }
  if (normalized.includes("t2v")) {
    tasks.add("t2v");
  }
  if (normalized.includes("i2v")) {
    tasks.add("i2v");
  }
  if (normalized.includes("s2v") || normalized.includes("speech")) {
    tasks.add("s2v");
  }
  if (normalized.includes("animate")) {
    tasks.add("animate");
  }
  if (tasks.size === 0 && normalized.includes("wan")) {
    tasks.add("t2v");
    notes.push("Wan model detected; defaulting to text-to-video until verified");
  }

  const optimized = normalized.includes("wan2.2") && normalized.includes("ti2v") && normalized.includes("5b");
  if (optimized) {
    notes.push("Matches the Wan2.2 TI2V 5B optimized profile");
  }

  return {
    tasks: Array.from(tasks),
    optimized,
    minVramGb: optimized ? 24 : undefined,
    notes,
  };
}

export function inspectWanFolder(pathOrRepo: string, fileNames: string[] = []): Pick<ModelInstall, "status" | "capabilities"> {
  const capabilities = inferWanCapabilities([pathOrRepo, ...fileNames].join(" "));
  const hasWanSignal = pathOrRepo.toLowerCase().includes("wan") || fileNames.some((name) => name.toLowerCase().includes("wan"));
  const hasWeights = fileNames.length === 0 || fileNames.some((name) => WEIGHT_FILE_PATTERN.test(name));

  if (!pathOrRepo.trim()) {
    return {
      status: "missing",
      capabilities: { ...capabilities, notes: ["Choose a local model folder first"] },
    };
  }

  if (!hasWanSignal) {
    return {
      status: "invalid",
      capabilities: { ...capabilities, notes: ["Folder does not look like a Wan-family model"] },
    };
  }

  if (!hasWeights) {
    return {
      status: "unchecked",
      capabilities: { ...capabilities, notes: ["Wan naming detected; model weights were not found in the sampled files"] },
    };
  }

  return {
    status: capabilities.tasks.length > 0 ? "ready" : "unchecked",
    capabilities,
  };
}

export function createCustomModel(input: {
  repoId: string;
  localPath: string;
  source: "huggingface" | "modelscope" | "local";
}): ModelInstall {
  const nameSeed = input.localPath || input.repoId;
  const displayName = nameSeed.split(/[\\/]/).filter(Boolean).pop() || "Custom Wan model";
  const inspected = inspectWanFolder(input.localPath || input.repoId);
  const primaryTask = inspected.capabilities.tasks[0] ?? "t2v";

  return {
    modelId: `custom-${displayName.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "wan-model"}`,
    displayName,
    family: "wan",
    task: inspected.capabilities.tasks.length > 1 ? "multi" : primaryTask,
    localPath: input.localPath,
    source: input.source,
    repoId: input.repoId || undefined,
    status: inspected.status,
    capabilities: inspected.capabilities,
    lastVerifiedAt: new Date().toISOString(),
  };
}

export function getPresetForModel(model: ModelInstall): PresetSettings {
  return model.capabilities.optimized ? OPTIMIZED_WAN_5B_PRESET : SAFE_CUSTOM_WAN_PRESET;
}
