import json
from pathlib import Path

from wan_studio_worker.lora_compat import check_lora_compatibility, group_lora_files, infer_lora_profile


def write_safetensors_header(path: Path, tensors: dict[str, list[int]]) -> None:
    header = {
        name: {"dtype": "F16", "shape": shape, "data_offsets": [0, 0]}
        for name, shape in tensors.items()
    }
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(len(raw).to_bytes(8, "little") + raw)


def write_model_config(model_dir: Path, *, dim: int, model_type: str, subfolder: str | None = None) -> None:
    target = model_dir / subfolder if subfolder else model_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(json.dumps({"dim": dim, "model_type": model_type}), encoding="utf-8")


def test_infer_lora_profile_reads_hidden_dim_and_task(tmp_path: Path) -> None:
    lora = tmp_path / "example_i2v_HIGH.safetensors"
    write_safetensors_header(
        lora,
        {
            "lora_unet_blocks_0_self_attn_q.lora_down.weight": [16, 5120],
            "lora_unet_blocks_0_self_attn_q.lora_up.weight": [5120, 16],
            "lora_unet_blocks_0_ffn_0.lora_up.weight": [13824, 16],
        },
    )

    profile = infer_lora_profile(lora)

    assert profile.hidden_dim == 5120
    assert profile.task == "i2v"
    assert profile.noise == "high"


def test_a14b_lora_is_incompatible_with_ti2v_5b_base(tmp_path: Path) -> None:
    lora = tmp_path / "example_i2v_HIGH.safetensors"
    write_safetensors_header(lora, {"lora_unet_blocks_0_self_attn_q.lora_down.weight": [16, 5120]})
    model_dir = tmp_path / "Wan2.2-TI2V-5B"
    write_model_config(model_dir, dim=3072, model_type="ti2v")

    result = check_lora_compatibility(lora, model_dir)

    assert result.compatible is False
    assert "5120" in result.reason
    assert "3072" in result.reason
    assert result.suggested_base_repo == "Wan-AI/Wan2.2-I2V-A14B"


def test_a14b_lora_is_compatible_with_i2v_a14b_base(tmp_path: Path) -> None:
    lora = tmp_path / "example_i2v_HIGH.safetensors"
    write_safetensors_header(lora, {"lora_unet_blocks_0_self_attn_q.lora_down.weight": [16, 5120]})
    model_dir = tmp_path / "Wan2.2-I2V-A14B"
    write_model_config(model_dir, dim=5120, model_type="i2v", subfolder="low_noise_model")

    result = check_lora_compatibility(lora, model_dir)

    assert result.compatible is True


def test_group_lora_files_pairs_high_and_low_noise(tmp_path: Path) -> None:
    low = tmp_path / "scene_i2v_LOW.safetensors"
    high = tmp_path / "scene_i2v_HIGH.safetensors"
    other = tmp_path / "other_t2v_HIGH.safetensors"

    groups = group_lora_files([low, high, other])

    first = groups[0]
    assert first.paths == [low, high]
    assert first.label == "scene_i2v"
    assert groups[1].paths == [other]
