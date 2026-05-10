from pathlib import Path

from wan_studio_worker.model_registry import (
    build_hf_command,
    build_modelscope_command,
    create_model_install,
    infer_capabilities,
    inspect_model_folder,
)
from wan_studio_worker.schemas import WanTask


def test_download_commands() -> None:
    assert build_hf_command("Wan-AI/Wan2.2-TI2V-5B", "./models/Wan2.2-TI2V-5B") == (
        "hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./models/Wan2.2-TI2V-5B"
    )
    assert build_modelscope_command("Wan-AI/Wan2.2-TI2V-5B", "./models/Wan2.2-TI2V-5B") == (
        "modelscope download Wan-AI/Wan2.2-TI2V-5B --local_dir ./models/Wan2.2-TI2V-5B"
    )


def test_recommended_model_capabilities() -> None:
    capabilities = infer_capabilities("Wan-AI/Wan2.2-TI2V-5B")
    assert capabilities.optimized is True
    assert capabilities.min_vram_gb == 24
    assert capabilities.tasks == [WanTask.T2V, WanTask.I2V, WanTask.TI2V]


def test_default_quality_base_capabilities() -> None:
    capabilities = infer_capabilities("Wan-AI/Wan2.2-I2V-A14B")
    assert capabilities.optimized is True
    assert capabilities.min_vram_gb == 80
    assert capabilities.tasks == [WanTask.I2V]


def test_quantized_a14b_gguf_capabilities_target_low_vram() -> None:
    capabilities = infer_capabilities("QuantStack/Wan2.2-I2V-A14B-GGUF HighNoise Q4_K_M")

    assert capabilities.optimized is True
    assert capabilities.min_vram_gb == 8
    assert capabilities.tasks == [WanTask.I2V]
    assert any("low-VRAM" in note for note in capabilities.notes)


def test_gguf_folder_inspection(tmp_path: Path) -> None:
    model_dir = tmp_path / "Wan2.2-I2V-A14B-GGUF"
    model_dir.mkdir()
    (model_dir / "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf").write_text("fake", encoding="utf-8")
    (model_dir / "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf").write_text("fake", encoding="utf-8")

    inspection = inspect_model_folder(str(model_dir))

    assert inspection.status == "ready"
    assert inspection.capabilities.min_vram_gb == 8


def test_folder_inspection(tmp_path: Path) -> None:
    model_dir = tmp_path / "Wan2.1-I2V-custom"
    model_dir.mkdir()
    (model_dir / "diffusion_pytorch_model.safetensors").write_text("fake", encoding="utf-8")

    inspection = inspect_model_folder(str(model_dir))

    assert inspection.status == "ready"
    assert WanTask.I2V in inspection.capabilities.tasks


def test_folder_rejection(tmp_path: Path) -> None:
    model_dir = tmp_path / "flux"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_text("fake", encoding="utf-8")

    inspection = inspect_model_folder(str(model_dir))

    assert inspection.status == "invalid"


def test_create_model_install(tmp_path: Path) -> None:
    model_dir = tmp_path / "Wan2.2-S2V-14B"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_text("fake", encoding="utf-8")

    install = create_model_install("Wan-AI/Wan2.2-S2V-14B", str(model_dir), "huggingface")

    assert install.status == "ready"
    assert install.capabilities.tasks == [WanTask.S2V]
