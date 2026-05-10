# Wan Studio

Wan Studio is an experimental installable Web UI for running Wan-family open-source video models without ComfyUI. The intended distribution flow is FaceFusion-style: clone the GitHub repo, run the installer, then open the local browser UI.

The current default download target is [`lkzd7/WAN2.2_LoraSet_NSFW`](https://huggingface.co/lkzd7/WAN2.2_LoraSet_NSFW). It is treated as a Wan2.2 LoRA/adapters set, not a locked product default or a standalone base checkpoint. For real Wan inference, pair it with a compatible Wan2.2 base model runner.

## Quick Start

```bash
python install.py --accelerator cuda
python wan_studio.py run --open-browser
```

Open the Web UI, download/connect a Wan-compatible folder, wait for `Ready to prompt!`, then run a prompt.

## Default Download Command

```bash
pip install -U "huggingface_hub[cli]"
hf download lkzd7/WAN2.2_LoraSet_NSFW --local-dir ./models/WAN2.2_LoraSet_NSFW
```

Users can replace the repo id and folder with any Wan-family model or adapter folder they want to test.

## Colab

For GPU-less machines, use the Colab launch flow in [`docs/getting-started.md`](docs/getting-started.md). The notebook installs the package and opens the same Web UI from Colab, so prompting happens in the browser instead of by editing code cells.

## What Is Included

- `install.py` and `wan_studio.py` for package-style local Web UI launch
- Python FastAPI Web UI and worker code in `workers/python`
- Fake runner for UI/job testing without a GPU
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
