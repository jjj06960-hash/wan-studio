import asyncio
import json
from pathlib import Path

from wan_studio_worker.drive_worker import ensure_drive_tree, process_once
from wan_studio_worker.runner import FakeWanRunner, build_lightx2v_command, build_wan_generate_command
from wan_studio_worker.schemas import DriveJobEnvelope, GenerationRequest, JobState, RuntimeKind, VramTier, WanTask


def test_fake_runner_writes_output(tmp_path: Path) -> None:
    request = GenerationRequest(prompt="A neon street", model_id="wan2.2-ti2v-5b", task=WanTask.TI2V)
    status = asyncio.run(FakeWanRunner().run(request, tmp_path))

    assert status.state == JobState.SUCCEEDED
    assert status.output_path is not None
    assert Path(status.output_path).exists()


def test_build_wan_command_shape() -> None:
    request = GenerationRequest(
        prompt="A mountain at dawn",
        image="/tmp/input.png",
        model_id="custom-wan",
        model_path="/models/Wan2.2-TI2V-5B",
        size="1280x704",
        steps=24,
        task=WanTask.TI2V,
    )

    command = build_wan_generate_command(request, Path("/opt/Wan2.2"), save_file=Path("/outputs/test.mp4"))

    assert command[:3] == ["python", "/opt/Wan2.2/generate.py", "--task"]
    assert "ti2v-5B" in command
    assert "/models/Wan2.2-TI2V-5B" in command
    assert "--image" in command
    assert "--base_seed" in command
    assert "--save_file" in command
    assert "/outputs/test.mp4" in command


def test_build_wan_command_with_lora_uses_loader_wrapper() -> None:
    request = GenerationRequest(
        prompt="A mountain at dawn",
        model_id="custom-wan",
        model_path="/models/Wan2.2-I2V-A14B",
        size="1280x704",
        steps=24,
        task=WanTask.I2V,
        lora_paths=["/models/WAN2.2_LoraSet_NSFW/example_high.safetensors"],
        lora_scale=0.75,
    )

    command = build_wan_generate_command(request, Path("/opt/Wan2.2"), save_file=Path("/outputs/test.mp4"))

    assert command[:3] == ["python", "-m", "wan_studio_worker.wan_lora_generate"]
    assert "--wan_repo_dir" in command
    assert "/opt/Wan2.2" in command
    assert "--lora_path" in command
    assert "/models/WAN2.2_LoraSet_NSFW/example_high.safetensors" in command
    assert "--lora_scale" in command
    assert "0.75" in command
    assert "--ckpt_dir" in command
    assert "/models/Wan2.2-I2V-A14B" in command
    assert "i2v-A14B" in command
    assert "--save_file" in command
    assert "/outputs/test.mp4" in command


def test_build_lightx2v_command_shape_for_low_vram_i2v() -> None:
    request = GenerationRequest(
        prompt="A guitar player smiles",
        image="/tmp/reference.png",
        model_id="wan2.2-i2v-a14b",
        model_path="/models/Wan2.2-I2V-A14B",
        size="832x480",
        steps=4,
        task=WanTask.I2V,
        vram_tier_gb=VramTier.GB8,
    )

    command = build_lightx2v_command(request, save_file=Path("/outputs/test.mp4"))

    assert command[:2] == ["python", "-c"]
    script = command[2]
    assert "LightX2VPipeline" in script
    assert '"model_path": "/models/Wan2.2-I2V-A14B"' in script
    assert '"steps": 4' in script
    assert '"width": 832' in script
    assert '"height": 480' in script
    assert 'save_result_path=data["save_file"]' in script


def test_drive_worker_processes_camel_case_job(tmp_path: Path) -> None:
    jobs_dir, results_dir, status_dir = ensure_drive_tree(tmp_path)
    request = GenerationRequest(
        prompt="A river in fog",
        model_id="wan2.2-ti2v-5b",
        runtime=RuntimeKind.COLAB_DRIVE,
        task=WanTask.T2V,
    )
    envelope = DriveJobEnvelope(request=request)
    job_path = jobs_dir / f"{request.id}.request.json"
    job_path.write_text(json.dumps(envelope.model_dump(mode="json", by_alias=True), indent=2), encoding="utf-8")

    statuses = asyncio.run(process_once(tmp_path))

    assert statuses[0].state == JobState.SUCCEEDED
    assert (results_dir / f"{request.id}.mp4").exists()
    assert (status_dir / f"{request.id}.status.json").exists()
