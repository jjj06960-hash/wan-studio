from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .schemas import ModelCapabilities, ModelInstall, WanTask

WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"}


@dataclass(frozen=True)
class ModelInspection:
    status: str
    capabilities: ModelCapabilities
    sampled_files: list[str]


def build_hf_command(repo_id: str, local_dir: str) -> str:
    return f"hf download {repo_id.strip() or '<repo-id>'} --local-dir {local_dir.strip() or '<model-folder>'}"


def build_modelscope_command(repo_id: str, local_dir: str) -> str:
    return f"modelscope download {repo_id.strip() or '<repo-id>'} --local_dir {local_dir.strip() or '<model-folder>'}"


def infer_capabilities(value: str) -> ModelCapabilities:
    normalized = value.lower()
    tasks: list[WanTask] = []
    notes: list[str] = []

    def add(task: WanTask) -> None:
        if task not in tasks:
            tasks.append(task)

    if "ti2v" in normalized:
        add(WanTask.T2V)
        add(WanTask.I2V)
        add(WanTask.TI2V)
        notes.append("Hybrid TI2V naming detected")
    if "t2v" in normalized:
        add(WanTask.T2V)
    if "i2v" in normalized:
        add(WanTask.I2V)
    if "s2v" in normalized or "speech" in normalized:
        add(WanTask.S2V)
    if "animate" in normalized:
        add(WanTask.ANIMATE)
    if not tasks and "wan" in normalized:
        add(WanTask.T2V)
        notes.append("Wan model detected; defaulting to text-to-video until verified")
    if "lora" in normalized or "adapter" in normalized:
        notes.append("LoRA/adapters detected; pair with a compatible Wan base model for real inference")

    low_vram_signal = any(token in normalized for token in ("gguf", "lightx2v", "fp8", "int8", "q2_", "q3_", "q4_", "q5_", "q6_", "q8_"))
    optimized_5b = "wan2.2" in normalized and "ti2v" in normalized and "5b" in normalized
    optimized_a14b = "wan2.2" in normalized and "a14b" in normalized and ("i2v" in normalized or "t2v" in normalized)
    optimized_low_vram = optimized_a14b and low_vram_signal
    optimized = optimized_5b or optimized_a14b
    if optimized_5b:
        notes.append("Matches the Wan2.2 TI2V 5B optimized profile")
    if optimized_low_vram:
        notes.append("Matches the Wan2.2 A14B low-VRAM optimized profile")
    if optimized_a14b:
        notes.append("Matches the Wan2.2 A14B quality workflow profile")

    min_vram = 8 if optimized_low_vram else (80 if optimized_a14b else (24 if optimized_5b else None))
    return ModelCapabilities(tasks=tasks, optimized=optimized, min_vram_gb=min_vram, notes=notes)


def inspect_model_folder(path: str, *, max_files: int = 80) -> ModelInspection:
    target = Path(path).expanduser()
    if not path.strip() or not target.exists():
        capabilities = infer_capabilities(path)
        capabilities.notes.append("Model folder does not exist")
        return ModelInspection("missing", capabilities, [])

    sampled_files = _sample_files(target, max_files=max_files)
    signal = " ".join([str(target), *sampled_files])
    capabilities = infer_capabilities(signal)
    has_wan_signal = "wan" in signal.lower()
    has_weight = any(Path(name).suffix.lower() in WEIGHT_SUFFIXES for name in sampled_files)

    if not has_wan_signal:
        capabilities.notes.append("Folder does not look like a Wan-family model")
        return ModelInspection("invalid", capabilities, sampled_files)
    if not has_weight:
        capabilities.notes.append("Wan naming detected; no sampled weight files found")
        return ModelInspection("unchecked", capabilities, sampled_files)

    return ModelInspection("ready" if capabilities.tasks else "unchecked", capabilities, sampled_files)


def create_model_install(repo_id: str, local_path: str, source: str) -> ModelInstall:
    inspection = inspect_model_folder(local_path)
    display_name = _display_name(repo_id or local_path)
    task = "multi" if len(inspection.capabilities.tasks) > 1 else (inspection.capabilities.tasks[0] if inspection.capabilities.tasks else WanTask.T2V)
    return ModelInstall(
        model_id=f"custom-{_slug(display_name)}",
        display_name=display_name,
        task=task,
        local_path=local_path,
        source=source,  # type: ignore[arg-type]
        status=inspection.status,  # type: ignore[arg-type]
        capabilities=inspection.capabilities,
        repo_id=repo_id or None,
    )


def _sample_files(path: Path, *, max_files: int) -> list[str]:
    if path.is_file():
        return [path.name]
    files: list[str] = []
    for item in path.rglob("*"):
        if item.is_file():
            files.append(str(item.relative_to(path)))
        if len(files) >= max_files:
            break
    return files


def _display_name(value: str) -> str:
    parts = re.split(r"[\\/]", value.strip())
    return next((part for part in reversed(parts) if part), "Custom Wan model")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "wan-model"
