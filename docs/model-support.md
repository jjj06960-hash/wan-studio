# Model Support

Wan Studio v1 supports Wan-family models first.

## Default download target

- `lkzd7/WAN2.2_LoraSet_NSFW`
- Type: Wan2.2 LoRA/adapters set
- Purpose: default Hugging Face download target for the MVP Web UI
- Real inference note: pair with a compatible Wan2.2 base model runner

## Optimized base-model preset

- `Wan-AI/Wan2.2-TI2V-5B`
- Tasks: `t2v`, `i2v`, `ti2v`
- Recommended local target: 24 GB VRAM class GPU
- Safe preset: `1280x704`, 24 steps, model offload enabled, T5 on CPU

## Custom Wan models

Users can connect a local folder for Wan2.1, Wan2.2, and Wan-derived checkpoints. The app inspects the folder name and model files to infer capabilities:

- `t2v`: text-to-video
- `i2v`: image-to-video
- `ti2v`: text+image-to-video hybrid
- `s2v`: speech-to-video
- `animate`: character animation/replacement

Wan-derived models without a known optimization profile start with the safe low-VRAM preset.
