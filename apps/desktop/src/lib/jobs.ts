import type { GenerationJob, GenerationRequest, GenerationStatus, ModelInstall, RuntimeKind, WanTask } from "./types";
import { getPresetForModel } from "./modelCatalog";

export function makeJob(input: {
  prompt: string;
  image?: string;
  model: ModelInstall;
  runtime: RuntimeKind;
  task: WanTask;
}): GenerationJob {
  const id = crypto.randomUUID();
  const preset = getPresetForModel(input.model);
  const request: GenerationRequest = {
    id,
    prompt: input.prompt,
    image: input.image || undefined,
    modelId: input.model.modelId,
    modelPath: input.model.localPath || undefined,
    runtime: input.runtime,
    task: input.task,
    seed: Math.floor(Math.random() * 2_147_483_647),
    size: preset.size,
    steps: preset.steps,
    offloadModel: preset.offloadModel,
    t5Cpu: preset.t5Cpu,
    createdAt: new Date().toISOString(),
  };

  return {
    request,
    status: makeStatus(id, "queued", 0),
  };
}

export function makeStatus(
  id: string,
  state: GenerationStatus["state"],
  progress: number,
  extras: Partial<GenerationStatus> = {},
): GenerationStatus {
  return {
    id,
    state,
    progress,
    updatedAt: new Date().toISOString(),
    ...extras,
  };
}

export function driveJobFileName(job: GenerationJob): string {
  return `${job.request.createdAt.replace(/[:.]/g, "-")}-${job.request.id}.request.json`;
}

export function serializeDriveJob(job: GenerationJob): string {
  return JSON.stringify(
    {
      schemaVersion: 1,
      request: job.request,
      model: {
        modelId: job.request.modelId,
      },
    },
    null,
    2,
  );
}
