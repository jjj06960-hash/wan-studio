from pathlib import Path

from fastapi.testclient import TestClient

from wan_studio_worker.web import create_app
from test_lora_compat import write_model_config, write_safetensors_header


def test_state_exposes_default_lora_folder(tmp_path: Path) -> None:
    client = TestClient(create_app(root=tmp_path, runner_kind="fake"))

    payload = client.get("/api/state").json()

    assert payload["defaultLoraRepoId"] == "lkzd7/WAN2.2_LoraSet_NSFW"
    assert payload["defaultLoraModelDir"].endswith("models/WAN2.2_LoraSet_NSFW")


def test_lora_scan_lists_adapter_files(tmp_path: Path) -> None:
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    adapter = lora_dir / "scene_high.safetensors"
    adapter.write_bytes(b"fake")
    (lora_dir / "notes.txt").write_text("ignored", encoding="utf-8")
    client = TestClient(create_app(root=tmp_path, runner_kind="fake"))

    payload = client.get("/api/loras", params={"path": str(lora_dir)}).json()

    assert payload["loras"] == [str(adapter)]


def test_lora_scan_marks_incompatible_adapter_for_selected_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "Wan2.2-TI2V-5B"
    write_model_config(model_dir, dim=3072, model_type="ti2v")
    (model_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"fake")
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    adapter = lora_dir / "scene_i2v_HIGH.safetensors"
    write_safetensors_header(adapter, {"lora_unet_blocks_0_self_attn_q.lora_down.weight": [16, 5120]})
    client = TestClient(create_app(root=tmp_path, runner_kind="fake"))

    payload = client.get("/api/loras", params={"path": str(lora_dir), "model_path": str(model_dir)}).json()

    assert payload["items"][0]["compatible"] is False
    assert payload["items"][0]["suggestedBaseRepo"] == "Wan-AI/Wan2.2-I2V-A14B"


def test_create_job_rejects_incompatible_lora_before_queueing(tmp_path: Path) -> None:
    model_dir = tmp_path / "Wan2.2-TI2V-5B"
    write_model_config(model_dir, dim=3072, model_type="ti2v")
    (model_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"fake")
    adapter = tmp_path / "scene_i2v_HIGH.safetensors"
    write_safetensors_header(adapter, {"lora_unet_blocks_0_self_attn_q.lora_down.weight": [16, 5120]})
    client = TestClient(create_app(root=tmp_path, runner_kind="fake"))
    model = client.post(
        "/api/models/connect",
        json={"repo_id": "Wan-AI/Wan2.2-TI2V-5B", "local_path": str(model_dir), "source": "huggingface"},
    ).json()["model"]

    response = client.post(
        "/api/jobs",
        json={
            "prompt": "test",
            "model_id": model["modelId"],
            "task": "i2v",
            "image": "/tmp/input.png",
            "lora_paths": [str(adapter)],
        },
    )

    assert response.status_code == 400
    assert "Wan-AI/Wan2.2-I2V-A14B" in response.text
