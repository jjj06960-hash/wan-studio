# Model Support

Wan Studio v1 supports Wan-family models first.

## Default base model

- `Wan-AI/Wan2.2-I2V-A14B`
- Type: Wan2.2 I2V A14B base
- Purpose: default quality base for the MVP Web UI
- User-facing runtime: choose `8GB`, `16GB`, or `24GB`; Wan Studio maps that to an optimized build
- Real inference note: official BF16-style generation is the heavy reference path, not the beginner default

## Default low-VRAM build assets

- `lightx2v/Wan2.2-Distill-Models`
- Type: Wan2.2 A14B 4-step distilled high/low-noise weights
- Purpose: product default for consumer GPUs
- Presets:
  - `8GB`: `832x480`, 4 steps, aggressive CPU/offload behavior
  - `16GB`: `960x544`, 4 steps, balanced CPU/offload behavior
  - `24GB`: `1280x720`, 4 steps, quality-focused consumer GPU behavior

## Default adapter target

- `lkzd7/WAN2.2_LoraSet_NSFW`
- Type: Wan2.2 LoRA/adapters set
- Purpose: default optional adapter download target for the MVP Web UI
- Real inference note: pair one selected `.safetensors` file with a compatible Wan2.2 base model runner
- Compatibility note: many adapter files in this repo are A14B high/low-noise LoRAs and will not apply to the TI2V-5B base model

## Fast base-model preset

- `Wan-AI/Wan2.2-TI2V-5B`
- Tasks: `t2v`, `i2v`, `ti2v`
- Recommended local target: lower than A14B, especially with offload/optimized backends
- Safe preset: `1280x704`, 24 steps, model offload enabled, T5 on CPU

## Quality LoRA preset

- Base: `Wan-AI/Wan2.2-I2V-A14B` for I2V adapter sets
- Adapter: selected LOW/HIGH `.safetensors` pair from `lkzd7/WAN2.2_LoraSet_NSFW`
- Why: most files in that adapter set are A14B I2V LoRAs with a 5120 hidden dimension
- Web UI behavior: `Scan LoRA files` groups LOW/HIGH pairs and disables incompatible options for the currently selected base model

## Lower-spec direction

- Official Wan2.2 A14B inference is the quality baseline, not the consumer-GPU baseline.
- The product default should be LightX2V 4-step distillation and FP8/offload builds.
- GGUF workflows can reduce VRAM further and should be exposed as an advanced low-VRAM build once Wan Studio has a stable non-Comfy GGUF runner.

## Custom Wan models

Users can connect a local folder for Wan2.1, Wan2.2, and Wan-derived checkpoints. The app inspects the folder name and model files to infer capabilities:

- `t2v`: text-to-video
- `i2v`: image-to-video
- `ti2v`: text+image-to-video hybrid
- `s2v`: speech-to-video
- `animate`: character animation/replacement

Wan-derived models without a known optimization profile start with the safe low-VRAM preset.

## LoRA adapter loading

The Web UI exposes a LoRA adapter layer in the Prompt panel:

- Paste a `.safetensors` file path directly, or paste a folder and click `Scan LoRA files`.
- Select an embedded preset before running a job.
- For Wan2.2 A14B models, LOW/HIGH adapter filenames are grouped and routed to the low-noise and high-noise models together.
- Shape mismatches are detected before queueing and point the user to the compatible A14B base preset.
