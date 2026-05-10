import { describe, expect, it } from "vitest";
import { RECOMMENDED_WAN_MODEL } from "./modelCatalog";
import { driveJobFileName, makeJob, serializeDriveJob } from "./jobs";

describe("generation jobs", () => {
  it("creates safe local jobs for the default LoRA set", () => {
    const job = makeJob({
      prompt: "A quiet neon street",
      model: RECOMMENDED_WAN_MODEL,
      runtime: "local",
      task: "ti2v",
    });

    expect(job.request.size).toBe("832x480");
    expect(job.request.offloadModel).toBe(true);
    expect(job.status.state).toBe("queued");
  });

  it("serializes Drive job requests", () => {
    const job = makeJob({
      prompt: "A mountain at dawn",
      model: RECOMMENDED_WAN_MODEL,
      runtime: "colab-drive",
      task: "t2v",
    });
    const payload = JSON.parse(serializeDriveJob(job));

    expect(payload.schemaVersion).toBe(1);
    expect(payload.request.runtime).toBe("colab-drive");
    expect(driveJobFileName(job)).toContain(job.request.id);
  });
});
