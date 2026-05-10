import { describe, expect, it } from "vitest";
import {
  buildHuggingFaceCommand,
  buildModelScopeCommand,
  createCustomModel,
  inferWanCapabilities,
  inspectWanFolder,
  RECOMMENDED_WAN_MODEL,
} from "./modelCatalog";

describe("model download guides", () => {
  it("uses Wan2.2 I2V A14B as the recommended quality base", () => {
    expect(RECOMMENDED_WAN_MODEL.repoId).toBe("Wan-AI/Wan2.2-I2V-A14B");
    expect(RECOMMENDED_WAN_MODEL.capabilities.tasks).toEqual(["i2v"]);
  });

  it("builds Hugging Face download commands", () => {
    expect(buildHuggingFaceCommand("Wan-AI/Wan2.2-TI2V-5B", "./models/Wan2.2-TI2V-5B")).toBe(
      "hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./models/Wan2.2-TI2V-5B",
    );
  });

  it("builds ModelScope download commands", () => {
    expect(buildModelScopeCommand("Wan-AI/Wan2.2-TI2V-5B", "./models/Wan2.2-TI2V-5B")).toBe(
      "modelscope download Wan-AI/Wan2.2-TI2V-5B --local_dir ./models/Wan2.2-TI2V-5B",
    );
  });
});

describe("Wan model capability inference", () => {
  it("treats Wan2.2 TI2V 5B as optimized and multi-capable", () => {
    const capabilities = inferWanCapabilities("Wan-AI/Wan2.2-TI2V-5B");
    expect(capabilities.optimized).toBe(true);
    expect(capabilities.tasks).toEqual(["t2v", "i2v", "ti2v"]);
    expect(capabilities.minVramGb).toBe(24);
  });

  it("treats Wan2.2 I2V A14B as the quality optimized profile", () => {
    const capabilities = inferWanCapabilities("Wan-AI/Wan2.2-I2V-A14B");
    expect(capabilities.optimized).toBe(true);
    expect(capabilities.tasks).toEqual(["i2v"]);
    expect(capabilities.minVramGb).toBe(80);
  });

  it("treats Wan2.2 A14B GGUF as the low-VRAM optimized profile", () => {
    const capabilities = inferWanCapabilities("QuantStack/Wan2.2-I2V-A14B-GGUF Q4_K_M");
    expect(capabilities.optimized).toBe(true);
    expect(capabilities.tasks).toEqual(["i2v"]);
    expect(capabilities.minVramGb).toBe(8);
    expect(capabilities.notes.some((note) => note.includes("low-VRAM"))).toBe(true);
  });

  it("detects speech and animate variants", () => {
    expect(inferWanCapabilities("Wan2.2-S2V-14B").tasks).toEqual(["s2v"]);
    expect(inferWanCapabilities("Wan2.2-Animate-14B").tasks).toEqual(["animate"]);
  });

  it("rejects folders without a Wan signal", () => {
    const result = inspectWanFolder("/models/flux", ["model.safetensors"]);
    expect(result.status).toBe("invalid");
  });

  it("registers custom Wan models from repo and path", () => {
    const model = createCustomModel({
      repoId: "somebody/Wan2.1-I2V-custom",
      localPath: "/models/Wan2.1-I2V-custom",
      source: "huggingface",
    });
    expect(model.status).toBe("ready");
    expect(model.capabilities.tasks).toContain("i2v");
  });
});
