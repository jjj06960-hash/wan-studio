from __future__ import annotations

import asyncio
import json
import shutil
import textwrap
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
        output_path = output_dir / f"{request.id}.mp4"
        before = {path.resolve() for path in self.wan_repo_dir.rglob("*.mp4")}
        command = build_wan_generate_command(request, self.wan_repo_dir, save_file=output_path)
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

        (output_dir / f"{request.id}.log").write_text("\n".join(output_lines), encoding="utf-8")
        if output_path.exists():
            return status_for(request.id, JobState.SUCCEEDED, 100, output_path=output_path)

        candidates = sorted(
            [path for path in self.wan_repo_dir.rglob("*.mp4") if path.resolve() not in before],
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            candidates = sorted(self.wan_repo_dir.rglob("*.mp4"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            return status_for(request.id, JobState.FAILED, 0, error="Wan finished but no MP4 output was found")

        shutil.copy2(candidates[-1], output_path)
        return status_for(request.id, JobState.SUCCEEDED, 100, output_path=output_path)


class LightX2VRunner:
    """Runs the LightX2V low-VRAM Wan backend for 8/16/24 GB presets."""

    async def run(self, request: GenerationRequest, output_dir: Path, progress: ProgressCallback | None = None) -> GenerationStatus:
        if not request.model_path:
            return status_for(request.id, JobState.FAILED, 0, error="Model path is required for the LightX2V runner")
        if request.task.value == "i2v" and not request.image:
            return status_for(request.id, JobState.FAILED, 0, error="I2V generation requires an image reference path")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{request.id}.mp4"
        command = build_lightx2v_command(request, save_file=output_path)
        if progress:
            await progress(status_for(request.id, JobState.RUNNING, 5))

        process = await asyncio.create_subprocess_exec(
            *command,
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
        (output_dir / f"{request.id}.log").write_text("\n".join(output_lines), encoding="utf-8")
        if return_code != 0:
            return status_for(request.id, JobState.FAILED, 0, error="\n".join(output_lines[-40:]) or f"LightX2V exited with {return_code}")
        if not output_path.exists():
            return status_for(request.id, JobState.FAILED, 0, error="LightX2V finished but no MP4 output was found")
        return status_for(request.id, JobState.SUCCEEDED, 100, output_path=output_path)


def build_wan_generate_command(request: GenerationRequest, wan_repo_dir: Path, *, save_file: Path | None = None) -> list[str]:
    """Builds the official Wan generate.py command shape without executing it."""
    generate_args = [
        "--task",
        wan_task_name(request),
        "--size",
        request.size.replace("x", "*"),
        "--ckpt_dir",
        request.model_path,
        "--prompt",
        request.prompt,
        "--sample_steps",
        str(request.steps),
        "--base_seed",
        str(request.seed),
        "--convert_model_dtype",
    ]
    if save_file:
        generate_args.extend(["--save_file", str(save_file)])
    if request.image:
        generate_args.extend(["--image", request.image])
    if request.offload_model:
        generate_args.extend(["--offload_model", "True"])
    if request.t5_cpu:
        generate_args.append("--t5_cpu")

    if request.lora_paths:
        command = [
            "python",
            "-m",
            "wan_studio_worker.wan_lora_generate",
            "--wan_repo_dir",
            str(wan_repo_dir),
            "--lora_scale",
            str(request.lora_scale),
        ]
        for path in request.lora_paths:
            command.extend(["--lora_path", path])
        command.extend(generate_args)
        return command

    return ["python", str(wan_repo_dir / "generate.py"), *generate_args]


def build_lightx2v_command(request: GenerationRequest, *, save_file: Path) -> list[str]:
    """Builds a LightX2V script command for the low-VRAM 8/16/24 GB presets."""
    width, height = parse_size(request.size)
    data = {
        "model_path": request.model_path,
        "task": request.task.value,
        "prompt": request.prompt,
        "image": request.image,
        "save_file": str(save_file),
        "steps": request.steps,
        "width": width,
        "height": height,
        "seed": request.seed,
        "vram_tier_gb": int(request.vram_tier_gb),
    }
    payload = json.dumps(data)
    script = textwrap.dedent(
        f"""
        import json
        data = json.loads({payload!r})
        from lightx2v import LightX2VPipeline

        pipe = LightX2VPipeline(
            model_path=data["model_path"],
            model_cls="wan2.2_moe",
            task=data["task"],
        )
        pipe.enable_offload(
            cpu_offload=True,
            offload_granularity="block" if data["vram_tier_gb"] <= 16 else "phase",
            text_encoder_offload=True,
            image_encoder_offload=data["vram_tier_gb"] <= 8,
            vae_offload=data["vram_tier_gb"] <= 16,
        )
        pipe.create_generator(
            attn_mode="sage_attn2",
            infer_steps=data["steps"],
            height=data["height"],
            width=data["width"],
            num_frames=81,
            guidance_scale=[3.5, 3.5],
            sample_shift=5.0,
        )
        pipe.generate(
            seed=data["seed"],
            image_path=data["image"],
            prompt=data["prompt"],
            negative_prompt="overexposed, low quality, blurry, distorted, bad anatomy, watermark, text",
            save_result_path=data["save_file"],
        )
        """
    ).strip()
    return ["python", "-c", script]


def parse_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("*", "x")
    width, height = normalized.split("x", 1)
    return int(width), int(height)


def wan_task_name(request: GenerationRequest) -> str:
    value = request.task.value
    model_hint = f"{request.model_id} {request.model_path}".lower()
    if value == "ti2v":
        return "ti2v-5B"
    if "a14b" in model_hint:
        if value == "t2v":
            return "t2v-A14B"
        if value == "i2v":
            return "i2v-A14B"
    return value


def write_status(path: Path, status: GenerationStatus) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status.model_dump(mode="json", by_alias=True), indent=2), encoding="utf-8")
