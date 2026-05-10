# Workflow Presets

Wan Studio should feel like a packaged product, not a blank ComfyUI canvas. The app embeds common Wan workflows as presets so users choose an intent, connect/download the required files, and prompt.

## Embedded Now

### A14B I2V Consumer Builds

- Base model: `Wan-AI/Wan2.2-I2V-A14B`
- Optimized assets: `lightx2v/Wan2.2-Distill-Models`
- Task: `i2v-A14B`
- User control: only choose `8GB`, `16GB`, or `24GB`
- Preset mapping:
  - `8GB`: `832x480`, 4 steps, heavy offload
  - `16GB`: `960x544`, 4 steps, balanced offload
  - `24GB`: `1280x720`, 4 steps, quality-oriented consumer GPU build
- Runner: LightX2V low-VRAM bridge

### A14B I2V LoRA Quality

- Base model: `Wan-AI/Wan2.2-I2V-A14B`
- Task: `i2v-A14B`
- Adapter behavior: scan a folder, group LOW/HIGH `.safetensors` files into one selectable preset, and send both to the runner
- Use when: the adapter folder contains Wan2.2 I2V A14B LoRAs, such as the current default adapter set
- UX rule: if the selected base is `Wan2.2-TI2V-5B`, the app disables these LoRA presets and recommends `Wan-AI/Wan2.2-I2V-A14B`

### Fast 5B Starter

- Base model: `Wan-AI/Wan2.2-TI2V-5B`
- Task: `ti2v-5B`
- Use when: user wants a lower-load official Wan2.2 path without the default A14B LoRA set
- Defaults: `1280x704`, 24 steps, model offload enabled, T5 on CPU
- LoRA: only compatible 5B/TI2V LoRAs should be attached

## Research Backlog

### A14B T2V LoRA Quality

- Base model: `Wan-AI/Wan2.2-T2V-A14B`
- Task: `t2v-A14B`
- Adapter behavior: same LOW/HIGH pairing as I2V, but only for T2V-labeled LoRA files

### Low-VRAM A14B

- Goal: hide FP8/GGUF/offload complexity behind a "quality on lower VRAM" preset
- Status: LightX2V 8GB/16GB/24GB runner bridge is embedded
- Next runner gap: GGUF community workflows are often ComfyUI-wrapper-specific today, so Wan Studio needs a stable non-Comfy GGUF path before embedding them as another build option

## Source Notes

- Official Wan2.2 lists T2V-A14B, I2V-A14B, and TI2V-5B as separate model downloads/tasks.
- Official Wan2.2 describes A14B as a high-noise/low-noise two-expert setup, while TI2V-5B is a separate dense 5B model.
- Hugging Face configs show `Wan2.2-I2V-A14B` uses `dim=5120`, while `Wan2.2-TI2V-5B` uses `dim=3072`; LoRA presets must match this hidden dimension.
- ComfyUI and community workflows consistently represent Wan2.2 A14B LoRA usage as HIGH/LOW pairs, which Wan Studio now scans and groups automatically.
