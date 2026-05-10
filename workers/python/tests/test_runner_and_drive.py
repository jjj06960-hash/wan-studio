import asyncio
import json
from pathlib import Path

from wan_studio_worker.drive_worker import ensure_drive_tree, process_once
from wan_studio_worker.runner import FakeWanRunner, build_wan_generate_command
from wan_studio_worker.schemas import DriveJobEnvelope, GenerationRequest, JobState, RuntimeKind, WanTask


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
