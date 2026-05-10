from pathlib import Path

from fastapi.testclient import TestClient

from wan_studio_worker.web import create_app


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
