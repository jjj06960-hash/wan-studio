# Wan Studio Worker

This worker owns model-folder inspection, job schemas, a local HTTP API, a fake runner for development, and Drive-style job folder processing for the Colab workflow.

Run locally:

```bash
python3 -m pip install -e ".[dev]"
python3 -m wan_studio_worker.server --host 127.0.0.1 --port 8765
```

The real Wan integration should replace `FakeWanRunner` with a runner that calls the official Wan2.2 `generate.py` or a Diffusers pipeline. ComfyUI is intentionally not used.
