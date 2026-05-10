# Getting Started

Wan Studio now follows an install-then-Web-UI workflow:

1. Clone or download the repo.
2. Run the installer.
3. Download or connect a Wan-compatible model folder.
4. Open the Web UI, choose an `8GB`, `16GB`, or `24GB` build, and prompt from the browser.

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

The default A14B base for the low-VRAM builds is:

```text
Wan-AI/Wan2.2-I2V-A14B
```

The low-VRAM build assets are:

```text
lightx2v/Wan2.2-Distill-Models
```

The default LoRA/adapters folder is:

```text
lkzd7/WAN2.2_LoraSet_NSFW
```

Download both with the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
hf download Wan-AI/Wan2.2-I2V-A14B --local-dir ./models/Wan2.2-I2V-A14B
hf download lightx2v/Wan2.2-Distill-Models --local-dir ./models/Wan2.2-LightX2V --include "wan2.2_i2v_A14b_*_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"
hf download lkzd7/WAN2.2_LoraSet_NSFW --local-dir ./models/WAN2.2_LoraSet_NSFW
```

Then open the Web UI and connect the base model:

```text
models/Wan2.2-I2V-A14B
```

Use this as the LoRA/adapters folder in the Prompt panel:

```text
models/WAN2.2_LoraSet_NSFW
```

Important: `WAN2.2_LoraSet_NSFW` is handled as a Wan2.2 LoRA/adapters set. It is not a standalone model. Many files in this set target Wan2.2 A14B high/low-noise models, so the Web UI groups LOW/HIGH pairs and disables incompatible base-model combinations.

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

### 2. Mount Drive and download the base model and LoRA set once

Store the Wan2.2 I2V A14B base, LightX2V low-VRAM files, and LoRA set in Drive so you do not download them every time the Colab runtime resets.

```python
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

A14B_I2V_REPO_ID = 'Wan-AI/Wan2.2-I2V-A14B'
A14B_I2V_MODEL_DIR = Path('/content/drive/MyDrive/WanStudio/models/Wan2.2-I2V-A14B')
LORA_REPO_ID = 'lkzd7/WAN2.2_LoraSet_NSFW'
LORA_MODEL_DIR = Path('/content/drive/MyDrive/WanStudio/models/WAN2.2_LoraSet_NSFW')
LIGHTX2V_REPO_ID = 'lightx2v/Wan2.2-Distill-Models'
LIGHTX2V_MODEL_DIR = Path('/content/drive/MyDrive/WanStudio/models/Wan2.2-LightX2V')
WEIGHT_SUFFIXES = {'.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.gguf'}

!pip install -U "huggingface_hub[cli]"

def has_weights(model_dir):
    return model_dir.exists() and any(
        path.suffix in WEIGHT_SUFFIXES for path in model_dir.rglob('*') if path.is_file()
    )

if has_weights(A14B_I2V_MODEL_DIR):
    print('A14B I2V base model already exists in Drive:', A14B_I2V_MODEL_DIR)
else:
    !hf download {A14B_I2V_REPO_ID} --local-dir {A14B_I2V_MODEL_DIR}

if has_weights(LIGHTX2V_MODEL_DIR):
    print('LightX2V low-VRAM files already exist in Drive:', LIGHTX2V_MODEL_DIR)
else:
    !hf download {LIGHTX2V_REPO_ID} --local-dir {LIGHTX2V_MODEL_DIR} --include "wan2.2_i2v_A14b_*_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"

if has_weights(LORA_MODEL_DIR):
    print('LoRA set already exists in Drive:', LORA_MODEL_DIR)
else:
    !hf download {LORA_REPO_ID} --local-dir {LORA_MODEL_DIR}
```

### 3. Install the LightX2V low-VRAM runner

```python
WAN_REPO_DIR = '/content/Wan2.2'
LIGHTX2V_REPO_DIR = '/content/LightX2V'

!rm -rf {WAN_REPO_DIR}
!git clone https://github.com/Wan-Video/Wan2.2.git {WAN_REPO_DIR}
%cd {WAN_REPO_DIR}
!pip install -r requirements.txt

!rm -rf {LIGHTX2V_REPO_DIR}
!git clone https://github.com/ModelTC/LightX2V.git {LIGHTX2V_REPO_DIR}
%cd {LIGHTX2V_REPO_DIR}
!pip install -v .
```

### 4. Start the Web UI in real generation mode

```python
%cd /content/wan-studio
!python wan_studio.py run --host 127.0.0.1 --port 7860 --share --runner lightx2v
```

Colab will show an iframe and print an `Open Wan Studio Web UI:` proxy link for port `7860`. Use that proxy link, not a `0.0.0.0` or `127.0.0.1` link. Keep that cell running while using the UI.

In the Web UI, connect this base model folder if it is not detected automatically:

```text
/content/drive/MyDrive/WanStudio/models/Wan2.2-I2V-A14B
```

Choose `8GB`, `16GB`, or `24GB` in the Web UI. Wan Studio maps that single choice to resolution, steps, offload behavior, and the low-VRAM backend.

To attach the optional LoRA set, paste this folder in `LoRA adapter folder or file`, click `Scan LoRA files`, then select a compatible embedded preset. LOW/HIGH pairs are grouped automatically:

```text
/content/drive/MyDrive/WanStudio/models/WAN2.2_LoraSet_NSFW
```

The official Wan runner is still available as a heavy quality reference, but the beginner product path is the LightX2V low-VRAM runner.

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

LightX2V low-VRAM runner, useful for the 8GB/16GB/24GB builds:

```bash
python wan_studio.py run --runner lightx2v --open-browser
```
