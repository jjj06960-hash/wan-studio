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

Then open the Web UI and connect it as a LoRA/adapters folder, not as the base model:

```text
models/WAN2.2_LoraSet_NSFW
```

Important: this repo is handled as a Wan2.2 LoRA/adapters set. For real generation with the Wan subprocess runner, connect a compatible Wan2.2 base checkpoint first, then use the Prompt panel's `LoRA adapter folder or file` field to scan and attach one `.safetensors` adapter file. Many files in this set target Wan2.2 A14B/high-low noise models, so they will fail clearly if attached to an incompatible 5B base model.

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

### 2. Mount Drive and download the real base model and LoRA set once

Store the real Wan2.2 base model and optional LoRA set in Drive so you do not download them every time the Colab runtime resets. The LoRA/adapters repo is not enough for video generation by itself.

```python
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

BASE_REPO_ID = 'Wan-AI/Wan2.2-TI2V-5B'
BASE_MODEL_DIR = Path('/content/drive/MyDrive/WanStudio/models/Wan2.2-TI2V-5B')
LORA_REPO_ID = 'lkzd7/WAN2.2_LoraSet_NSFW'
LORA_MODEL_DIR = Path('/content/drive/MyDrive/WanStudio/models/WAN2.2_LoraSet_NSFW')
WEIGHT_SUFFIXES = {'.safetensors', '.bin', '.pt', '.pth', '.ckpt'}

!pip install -U "huggingface_hub[cli]"

def has_weights(model_dir):
    return model_dir.exists() and any(
        path.suffix in WEIGHT_SUFFIXES for path in model_dir.rglob('*') if path.is_file()
    )

if has_weights(BASE_MODEL_DIR):
    print('Base model already exists in Drive:', BASE_MODEL_DIR)
else:
    !hf download {BASE_REPO_ID} --local-dir {BASE_MODEL_DIR}

if has_weights(LORA_MODEL_DIR):
    print('LoRA set already exists in Drive:', LORA_MODEL_DIR)
else:
    !hf download {LORA_REPO_ID} --local-dir {LORA_MODEL_DIR}
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

To attach the optional LoRA set, paste this folder in `LoRA adapter folder or file`, click `Scan LoRA files`, then select one adapter file:

```text
/content/drive/MyDrive/WanStudio/models/WAN2.2_LoraSet_NSFW
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
