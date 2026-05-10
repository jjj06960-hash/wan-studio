# Workflow Presets

Wan Studio should feel like a packaged product, not a blank ComfyUI canvas. The app embeds common Wan workflows as presets so users choose an intent, connect/download the required files, and prompt.

## Embedded Now

### Fast 5B Starter

- Base model: `Wan-AI/Wan2.2-TI2V-5B`
- Task: `ti2v-5B`
- Use when: user wants the fastest official Wan2.2 path on a 24 GB class GPU
- Defaults: `1280x704`, 24 steps, model offload enabled, T5 on CPU
- LoRA: only compatible 5B/TI2V LoRAs should be attached

### A14B I2V LoRA Quality

- Base model: `Wan-AI/Wan2.2-I2V-A14B`
- Task: `i2v-A14B`
- Adapter behavior: scan a folder, group LOW/HIGH `.safetensors` files into one selectable preset, and send both to the runner
- Use when: the adapter folder contains Wan2.2 I2V A14B LoRAs, such as the current default adapter set
- UX rule: if the selected base is `Wan2.2-TI2V-5B`, the app disables these LoRA presets and recommends `Wan-AI/Wan2.2-I2V-A14B`

## Research Backlog

### A14B T2V LoRA Quality

- Base model: `Wan-AI/Wan2.2-T2V-A14B`
- Task: `t2v-A14B`
- Adapter behavior: same LOW/HIGH pairing as I2V, but only for T2V-labeled LoRA files

### Lightning / LightX2V Fast A14B

- Goal: embed a 4-step A14B preset for much faster generation
- Pattern found in community workflows: use paired HIGH/LOW Lightning or LightX2V LoRAs and reduce sampling steps
- Runner gap: the current official `generate.py` bridge can launch A14B tasks, but more tuning is needed before Wan Studio should advertise this as a one-click preset

### Low-VRAM A14B

- Goal: hide FP8/GGUF/offload complexity behind a "quality on lower VRAM" preset
- Runner gap: GGUF and some FP8 community workflows are ComfyUI-wrapper-specific today, so Wan Studio needs a non-Comfy backend path before embedding them

## Source Notes

- Official Wan2.2 lists T2V-A14B, I2V-A14B, and TI2V-5B as separate model downloads/tasks.
- Official Wan2.2 describes A14B as a high-noise/low-noise two-expert setup, while TI2V-5B is a separate dense 5B model.
- Hugging Face configs show `Wan2.2-I2V-A14B` uses `dim=5120`, while `Wan2.2-TI2V-5B` uses `dim=3072`; LoRA presets must match this hidden dimension.
- ComfyUI and community workflows consistently represent Wan2.2 A14B LoRA usage as HIGH/LOW pairs, which Wan Studio now scans and groups automatically.
