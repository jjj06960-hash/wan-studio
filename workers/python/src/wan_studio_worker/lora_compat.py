from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


LORA_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt"}


@dataclass(frozen=True)
class LoraProfile:
    path: Path
    hidden_dim: int | None
    task: Literal["t2v", "i2v"] | None
    noise: Literal["low", "high"] | None


@dataclass(frozen=True)
class ModelProfile:
    path: Path
    dim: int | None
    model_type: str | None


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool | None
    reason: str
    suggested_base_repo: str | None = None


@dataclass(frozen=True)
class LoraGroup:
    label: str
    paths: list[Path]


def infer_lora_profile(path: Path) -> LoraProfile:
    name = path.name.lower()
    task: Literal["t2v", "i2v"] | None = None
    if "i2v" in name:
        task = "i2v"
    elif "t2v" in name:
        task = "t2v"

    noise: Literal["low", "high"] | None = None
    if re.search(r"(^|[_ .-])(low|ln)([_ .-]|$)", name) or "low_noise" in name:
        noise = "low"
    elif re.search(r"(^|[_ .-])(high|hn)([_ .-]|$)", name) or "high_noise" in name:
        noise = "high"

    hidden_dim = _infer_hidden_dim(path)
    return LoraProfile(path=path, hidden_dim=hidden_dim, task=task, noise=noise)


def infer_model_profile(path: Path) -> ModelProfile:
    candidates = [
        path / "config.json",
        path / "low_noise_model" / "config.json",
        path / "high_noise_model" / "config.json",
    ]
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        dim = payload.get("dim")
        model_type = payload.get("model_type")
        return ModelProfile(
            path=path,
            dim=int(dim) if isinstance(dim, int | float) else None,
            model_type=str(model_type).lower() if model_type else None,
        )
    return ModelProfile(path=path, dim=None, model_type=None)


def check_lora_compatibility(lora_path: Path, model_path: Path) -> CompatibilityResult:
    lora = infer_lora_profile(lora_path)
    model = infer_model_profile(model_path)
    suggested = suggest_base_repo(lora)

    if lora.hidden_dim and model.dim and lora.hidden_dim != model.dim:
        return CompatibilityResult(
            compatible=False,
            reason=f"LoRA hidden dim {lora.hidden_dim} does not match base model dim {model.dim}. Use {suggested}.",
            suggested_base_repo=suggested,
        )

    if lora.task and model.model_type and model.model_type != "ti2v" and lora.task != model.model_type:
        return CompatibilityResult(
            compatible=False,
            reason=f"{lora.task.upper()} LoRA does not match {model.model_type.upper()} base model. Use {suggested}.",
            suggested_base_repo=suggested,
        )

    if lora.hidden_dim is None or model.dim is None:
        return CompatibilityResult(
            compatible=None,
            reason="Compatibility could not be fully verified from metadata.",
            suggested_base_repo=suggested,
        )

    return CompatibilityResult(compatible=True, reason="Compatible", suggested_base_repo=suggested)


def suggest_base_repo(lora: LoraProfile) -> str:
    if lora.task == "t2v":
        return "Wan-AI/Wan2.2-T2V-A14B"
    return "Wan-AI/Wan2.2-I2V-A14B"


def group_lora_files(paths: list[Path]) -> list[LoraGroup]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(paths):
        grouped.setdefault(_pair_key(path), []).append(path)

    groups = [LoraGroup(label=key, paths=_sort_lora_pair(items)) for key, items in grouped.items()]
    return sorted(groups, key=lambda group: (-len(group.paths), group.label))


def lora_item_payload(path: Path, *, model_path: Path | None = None) -> dict[str, Any]:
    profile = infer_lora_profile(path)
    result = check_lora_compatibility(path, model_path) if model_path else CompatibilityResult(
        compatible=None,
        reason="Select a base model to verify compatibility.",
        suggested_base_repo=suggest_base_repo(profile),
    )
    return {
        "path": str(path),
        "name": path.name,
        "compatible": result.compatible,
        "reason": result.reason,
        "suggestedBaseRepo": result.suggested_base_repo,
        "profile": {
            "hiddenDim": profile.hidden_dim,
            "task": profile.task,
            "noise": profile.noise,
        },
    }


def _infer_hidden_dim(path: Path) -> int | None:
    if path.suffix.lower() != ".safetensors":
        return None
    try:
        header = _read_safetensors_header(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    counter: Counter[int] = Counter()
    for key, value in header.items():
        if key == "__metadata__" or not isinstance(value, dict):
            continue
        shape = value.get("shape")
        if not isinstance(shape, list):
            continue
        for dim in shape:
            if isinstance(dim, int) and 512 <= dim <= 8192:
                counter[dim] += 1
    return counter.most_common(1)[0][0] if counter else None


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        header_size = int.from_bytes(file.read(8), "little")
        if header_size <= 0 or header_size > 100_000_000:
            raise ValueError(f"Invalid safetensors header size: {path}")
        return json.loads(file.read(header_size).decode("utf-8"))


def _pair_key(path: Path) -> str:
    value = path.stem
    value = re.sub(r"(?i)(^|[_ .-])(high|low|hn|ln)([_ .-]|$)", "_", value)
    value = re.sub(r"(?i)(high|low)_noise", "", value)
    value = re.sub(r"[_ .-]+", "_", value).strip("_")
    return value or path.stem


def _sort_lora_pair(paths: list[Path]) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, str]:
        noise = infer_lora_profile(path).noise
        order = {"low": 0, None: 1, "high": 2}
        return order[noise], path.name

    return sorted(paths, key=sort_key)
