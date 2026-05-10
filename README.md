# Wan Studio

Wan Studio is an experimental installable Web UI for running Wan-family open-source video models without ComfyUI. The intended distribution flow is FaceFusion-style: clone the GitHub repo, run the installer, then open the local browser UI.

The product default is now the A14B image-to-video quality path with user-facing `8GB`, `16GB`, and `24GB` optimized builds. Internally those builds target LightX2V/GGUF/FP8-style low-VRAM workflows; the official Wan runner remains a quality reference, not the beginner default. The default base model is [`Wan-AI/Wan2.2-I2V-A14B`](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B), and the optional adapter target is [`lkzd7/WAN2.2_LoraSet_NSFW`](https://huggingface.co/lkzd7/WAN2.2_LoraSet_NSFW).

## Quick Start

```bash
python install.py --accelerator cuda
python wan_studio.py run --open-browser
```

Open the Web UI, download/connect a Wan-compatible folder, choose `8GB`, `16GB`, or `24GB`, wait for `Ready to prompt!`, then run a prompt.

## Default Download Command

```bash
pip install -U "huggingface_hub[cli]"
hf download Wan-AI/Wan2.2-I2V-A14B --local-dir ./models/Wan2.2-I2V-A14B
hf download lightx2v/Wan2.2-Distill-Models --local-dir ./models/Wan2.2-LightX2V --include "wan2.2_i2v_A14b_*_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"
hf download lkzd7/WAN2.2_LoraSet_NSFW --local-dir ./models/WAN2.2_LoraSet_NSFW
```

Users can replace the repo id and folder with any Wan-family model or adapter folder they want to test.

For the workflow philosophy and embedded preset map, see [`docs/workflow-presets.md`](docs/workflow-presets.md).

## Colab

For GPU-less machines, use the Colab launch flow in [`docs/getting-started.md`](docs/getting-started.md). The notebook installs the package and opens the same Web UI from Colab, so prompting happens in the browser instead of by editing code cells.

## What Is Included

- `install.py` and `wan_studio.py` for package-style local Web UI launch
- Python FastAPI Web UI and worker code in `workers/python`
- Fake runner for UI/job testing without a GPU
- LightX2V low-VRAM runner bridge for 8GB/16GB/24GB A14B presets
- Optional subprocess bridge for an official Wan `generate.py` checkout
- Legacy Tauri/React prototype in `apps/desktop`
- Colab notebook scaffold in `notebooks/wan_colab_worker.ipynb`

## Development

```bash
npm install
npm test
npm run build
```

Python worker tests:

```bash
python install.py --dev
.venv/bin/python -m pytest workers/python/tests
```

Run the Web UI:

```bash
.venv/bin/python wan_studio.py run --host 127.0.0.1 --port 7860 --open-browser
```
