import { describe, expect, it } from "vitest";
import { RECOMMENDED_WAN_MODEL } from "./modelCatalog";
import { driveJobFileName, makeJob, serializeDriveJob } from "./jobs";

describe("generation jobs", () => {
  it("creates low-VRAM local jobs for the default A14B base", () => {
    const job = makeJob({
      prompt: "A quiet neon street",
      image: "/tmp/source.png",
      model: RECOMMENDED_WAN_MODEL,
      runtime: "local",
      task: "i2v",
    });

    expect(job.request.vramTierGb).toBe(8);
    expect(job.request.size).toBe("832x480");
    expect(job.request.steps).toBe(4);
    expect(job.request.offloadModel).toBe(true);
    expect(job.status.state).toBe("queued");
  });

  it("serializes Drive job requests", () => {
    const job = makeJob({
      prompt: "A mountain at dawn",
      model: RECOMMENDED_WAN_MODEL,
      runtime: "colab-drive",
      task: "t2v",
      vramTierGb: 24,
    });
    const payload = JSON.parse(serializeDriveJob(job));

    expect(payload.schemaVersion).toBe(1);
    expect(payload.request.runtime).toBe("colab-drive");
    expect(payload.request.vramTierGb).toBe(24);
    expect(payload.request.size).toBe("1280x720");
    expect(driveJobFileName(job)).toContain(job.request.id);
  });
});
