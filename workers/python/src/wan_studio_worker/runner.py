from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from .schemas import GenerationRequest, GenerationStatus, JobState, status_for

ProgressCallback = Callable[[GenerationStatus], Awaitable[None]]


class WanRunner(Protocol):
    async def run(self, request: GenerationRequest, output_dir: Path, progress: ProgressCallback | None = None) -> GenerationStatus:
        ...


class FakeWanRunner:
    """Development runner that exercises queue/status behavior without a GPU."""

    async def run(self, request: GenerationRequest, output_dir: Path, progress: ProgressCallback | None = None) -> GenerationStatus:
        output_dir.mkdir(parents=True, exist_ok=True)
        for value in (8, 24, 46, 72, 91):
            await asyncio.sleep(0.02)
            if progress:
                await progress(status_for(request.id, JobState.RUNNING, value))

        output_path = output_dir / f"{request.id}.mp4"
        sidecar_path = output_dir / f"{request.id}.json"
        output_path.write_bytes(b"WAN_STUDIO_FAKE_MP4\n")
        sidecar_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
        return status_for(request.id, JobState.SUCCEEDED, 100, output_path=output_path)


class SubprocessWanRunner:
    """Runs the official Wan generate.py script and imports the newest MP4 result."""

    def __init__(self, wan_repo_dir: Path) -> None:
        self.wan_repo_dir = wan_repo_dir.expanduser().resolve()

    async def run(self, request: GenerationRequest, output_dir: Path, progress: ProgressCallback | None = None) -> GenerationStatus:
        generate_py = self.wan_repo_dir / "generate.py"
        if not generate_py.exists():
            return status_for(request.id, JobState.FAILED, 0, error=f"generate.py not found: {generate_py}")
        if not request.model_path:
            return status_for(request.id, JobState.FAILED, 0, error="Model path is required for the Wan runner")

        output_dir.mkdir(parents=True, exist_ok=True)
        before = {path.resolve() for path in self.wan_repo_dir.rglob("*.mp4")}
        command = build_wan_generate_command(request, self.wan_repo_dir)
        if progress:
            await progress(status_for(request.id, JobState.RUNNING, 5))

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.wan_repo_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output_lines: list[str] = []
        if process.stdout:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                output_lines.append(line.decode("utf-8", errors="replace").rstrip())
        return_code = await process.wait()
        if return_code != 0:
            return status_for(request.id, JobState.FAILED, 0, error="\n".join(output_lines[-40:]) or f"Wan exited with {return_code}")

        if progress:
            await progress(status_for(request.id, JobState.RUNNING, 92))

        candidates = sorted(
            [path for path in self.wan_repo_dir.rglob("*.mp4") if path.resolve() not in before],
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            candidates = sorted(self.wan_repo_dir.rglob("*.mp4"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            return status_for(request.id, JobState.FAILED, 0, error="Wan finished but no MP4 output was found")

        output_path = output_dir / f"{request.id}.mp4"
        shutil.copy2(candidates[-1], output_path)
        (output_dir / f"{request.id}.log").write_text("\n".join(output_lines), encoding="utf-8")
        return status_for(request.id, JobState.SUCCEEDED, 100, output_path=output_path)


def build_wan_generate_command(request: GenerationRequest, wan_repo_dir: Path) -> list[str]:
    """Builds the official Wan generate.py command shape without executing it."""
    command = [
        "python",
        str(wan_repo_dir / "generate.py"),
        "--task",
        request.task.value if request.task.value != "ti2v" else "ti2v-5B",
        "--size",
        request.size.replace("x", "*"),
        "--ckpt_dir",
        request.model_path,
        "--prompt",
        request.prompt,
        "--sample_steps",
        str(request.steps),
        "--convert_model_dtype",
    ]
    if request.image:
        command.extend(["--image", request.image])
    if request.offload_model:
        command.extend(["--offload_model", "True"])
    if request.t5_cpu:
        command.append("--t5_cpu")
    return command


def write_status(path: Path, status: GenerationStatus) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status.model_dump(mode="json", by_alias=True), indent=2), encoding="utf-8")
