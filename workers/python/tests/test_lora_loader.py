from pathlib import Path

import pytest

from wan_studio_worker.wan_lora_generate import classify_lora_paths, resolve_lora_files


def test_resolve_lora_files_accepts_files_and_small_folders(tmp_path: Path) -> None:
    first = tmp_path / "first_high.safetensors"
    second = tmp_path / "second_low.safetensors"
    ignored = tmp_path / "notes.txt"
    first.write_bytes(b"fake")
    second.write_bytes(b"fake")
    ignored.write_text("ignore me", encoding="utf-8")

    assert resolve_lora_files([str(tmp_path)]) == [second, first]


def test_resolve_lora_files_rejects_large_adapter_folders(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"adapter_{index}.safetensors").write_bytes(b"fake")

    with pytest.raises(ValueError, match="Choose one or two"):
        resolve_lora_files([str(tmp_path)])


def test_classify_lora_paths_detects_high_and_low_noise_files() -> None:
    low = Path("/models/lora_scene_low_noise.safetensors")
    high = Path("/models/lora_scene_HIGH.safetensors")

    classified = classify_lora_paths([low, high])

    assert classified.low == [low]
    assert classified.high == [high]
    assert classified.generic == []
