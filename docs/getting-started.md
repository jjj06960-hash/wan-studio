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

### 1. Clone and install Wan Studio

```python
!git clone https://github.com/jjj06960-hash/wan-studio.git /content/wan-studio
%cd /content/wan-studio
!python install.py --accelerator cuda --system
```

### 2. Mount Drive and download the real base model once

Store the real Wan2.2 base model in Drive so you do not download it every time the Colab runtime resets. The default LoRA/adapters repo is useful for future adapter support, but it is not enough for video generation by itself.

```python
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

BASE_REPO_ID = 'Wan-AI/Wan2.2-TI2V-5B'
BASE_MODEL_DIR = Path('/content/drive/MyDrive/WanStudio/models/Wan2.2-TI2V-5B')
WEIGHT_SUFFIXES = {'.safetensors', '.bin', '.pt', '.pth', '.ckpt'}

!pip install -U "huggingface_hub[cli]"

has_weights = BASE_MODEL_DIR.exists() and any(
    path.suffix in WEIGHT_SUFFIXES for path in BASE_MODEL_DIR.rglob('*') if path.is_file()
)

if has_weights:
    print('Base model already exists in Drive:', BASE_MODEL_DIR)
else:
    !hf download {BASE_REPO_ID} --local-dir {BASE_MODEL_DIR}
```

### 3. Install the official Wan runner

```python
WAN_REPO_DIR = '/content/Wan2.2'

!rm -rf {WAN_REPO_DIR}
!git clone https://github.com/Wan-Video/Wan2.2.git {WAN_REPO_DIR}
%cd {WAN_REPO_DIR}
!pip install -r requirements.txt
```

### 4. Start the Web UI in real generation mode

```python
%cd /content/wan-studio
!python wan_studio.py run --host 127.0.0.1 --port 7860 --share --runner wan --wan-repo-dir /content/Wan2.2
```

Colab will show an iframe and print an `Open Wan Studio Web UI:` proxy link for port `7860`. Use that proxy link, not a `0.0.0.0` or `127.0.0.1` link. Keep that cell running while using the UI.

In the Web UI, connect this model folder if it is not detected automatically:

```text
/content/drive/MyDrive/WanStudio/models/Wan2.2-TI2V-5B
```

Real generation needs a 24 GB+ GPU. If Colab assigns a T4, the UI can open but generation may fail with an out-of-memory error.

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
