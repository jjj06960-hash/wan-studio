# Getting Started

Wan Studio now follows an install-then-Web-UI workflow:

1. Clone or download the repo.
2. Run the installer.
3. Download or connect a Wan-compatible model folder.
4. Open the Web UI and prompt from the browser.

This is closer to FaceFusion than to a separate desktop app: setup happens once, and day-to-day use happens in the local Web UI.

## Local Install

Use this when the machine has a local GPU, especially a Windows or Linux NVIDIA setup.

```bash
git clone https://github.com/jjj06960-hash/wan-studio.git
cd wan-studio
python install.py --accelerator cuda
python wan_studio.py run --open-browser
```

For macOS or CPU-only testing:

```bash
python install.py --accelerator auto
python wan_studio.py run --open-browser
```

## Default Model Download

The default download target is:

```text
lkzd7/WAN2.2_LoraSet_NSFW
```

Download it with the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
hf download lkzd7/WAN2.2_LoraSet_NSFW --local-dir ./models/WAN2.2_LoraSet_NSFW
```

Then open the Web UI and connect:

```text
models/WAN2.2_LoraSet_NSFW
```

Important: this repo is handled as a Wan2.2 LoRA/adapters set. For real generation with the Wan subprocess runner, you still need a compatible Wan2.2 base model/official Wan checkout that knows how to use that adapter set. The fake runner is enough to test the app flow without a GPU.

## Custom Wan Folder

Users are not locked to the default repo. In the Web UI:

1. Go to `Models`.
2. Set `Source` to Hugging Face, ModelScope, or Local only.
3. Paste a repo id and folder path, or point to an existing folder.
4. Click `Connect model`.
5. Continue only after the prompt panel shows `Ready to prompt!`.

The app inspects the folder name and sampled files, then marks supported tasks such as `t2v`, `i2v`, or `ti2v` when it can infer them.

## Colab Web UI

Use this when the local machine has no suitable GPU.

In Colab:

```python
!git clone https://github.com/jjj06960-hash/wan-studio.git /content/wan-studio
%cd /content/wan-studio
!python install.py --accelerator cuda --system
```

Optional default download:

```python
!pip install -U "huggingface_hub[cli]"
!hf download lkzd7/WAN2.2_LoraSet_NSFW --local-dir /content/wan-studio/models/WAN2.2_LoraSet_NSFW
```

Start the Web UI:

```python
!python wan_studio.py run --host 0.0.0.0 --port 7860 --share
```

Colab will expose a browser window for port `7860`. Keep that cell running while using the UI.

## Runner Modes

Fake runner, useful for UI and job-flow testing:

```bash
python wan_studio.py run --runner fake --open-browser
```

Wan subprocess runner, useful after installing an official Wan repo checkout:

```bash
python wan_studio.py run --runner wan --wan-repo-dir /path/to/Wan2.2 --open-browser
```

The subprocess runner builds a `generate.py` command and imports the newest `.mp4` result into `outputs/`.
