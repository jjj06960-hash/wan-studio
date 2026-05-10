from __future__ import annotations

import argparse
import logging
import re
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .lora_compat import LORA_SUFFIXES


@dataclass(frozen=True)
class ClassifiedLoras:
    low: list[Path]
    high: list[Path]
    generic: list[Path]


def _split_lora_paths(raw_paths: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for raw_path in raw_paths:
        for part in re.split(r"[,;\n]", raw_path):
            value = part.strip()
            if value:
                paths.append(value)
    return paths


def _sort_lora_paths(paths: Iterable[Path]) -> list[Path]:
    classified = classify_lora_paths(paths)
    return [*classified.low, *classified.generic, *classified.high]


def resolve_lora_files(raw_paths: Iterable[str], *, max_folder_files: int = 2) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in _split_lora_paths(raw_paths):
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise ValueError(f"LoRA path does not exist: {path}")
        if path.is_file():
            if path.suffix.lower() not in LORA_SUFFIXES:
                raise ValueError(f"LoRA file must be one of {sorted(LORA_SUFFIXES)}: {path}")
            resolved.append(path)
            continue

        files = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in LORA_SUFFIXES)
        if len(files) > max_folder_files:
            raise ValueError(
                f"LoRA folder contains {len(files)} adapter files. Choose one or two .safetensors files instead: {path}"
            )
        resolved.extend(files)

    unique = list(dict.fromkeys(resolved))
    return _sort_lora_paths(unique)


def classify_lora_paths(paths: Iterable[Path]) -> ClassifiedLoras:
    low: list[Path] = []
    high: list[Path] = []
    generic: list[Path] = []
    for path in paths:
        name = path.name.lower()
        if re.search(r"(^|[_ .-])(low|ln)([_ .-]|$)", name) or "low_noise" in name:
            low.append(path)
        elif re.search(r"(^|[_ .-])(high|hn)([_ .-]|$)", name) or "high_noise" in name:
            high.append(path)
        else:
            generic.append(path)
    return ClassifiedLoras(low=sorted(low), high=sorted(high), generic=sorted(generic))


def install_lora_hooks(lora_files: list[Path], scale: float) -> None:
    classified = classify_lora_paths(lora_files)
    counters: dict[type[Any], int] = {}

    def select_paths(role: str) -> list[Path]:
        if role == "low":
            return classified.low or classified.generic
        if role == "high":
            return classified.high or classified.generic
        return classified.generic or [*classified.low, *classified.high]

    def patch_class(module_name: str, class_name: str, roles: list[str]) -> None:
        module = __import__(module_name, fromlist=[class_name])
        pipeline_class = getattr(module, class_name)
        original = pipeline_class._configure_model

        def wrapped(self: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
            configured = original(self, model, *args, **kwargs)
            index = counters.get(pipeline_class, 0)
            counters[pipeline_class] = index + 1
            role = roles[min(index, len(roles) - 1)]
            for lora_file in select_paths(role):
                applied = apply_lora_file(configured, lora_file, scale)
                logging.info("Applied %s LoRA layers from %s to %s model", applied, lora_file, role)
            return configured

        pipeline_class._configure_model = wrapped

    patch_class("wan.text2video", "WanT2V", ["low", "high"])
    patch_class("wan.image2video", "WanI2V", ["low", "high"])
    patch_class("wan.textimage2video", "WanTI2V", ["generic"])


def apply_lora_file(model: Any, path: Path, scale: float) -> int:
    state_dict = load_lora_state_dict(path)
    return apply_lora_state_dict(model, state_dict, scale=scale, source_name=str(path))


def load_lora_state_dict(path: Path) -> dict[str, Any]:
    import torch

    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")

    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict):
        for key in ("state_dict", "module", "model"):
            nested = loaded.get(key)
            if isinstance(nested, dict):
                return nested
        return loaded
    raise ValueError(f"Unsupported LoRA checkpoint payload: {path}")


def apply_lora_state_dict(model: Any, state_dict: dict[str, Any], *, scale: float, source_name: str = "LoRA") -> int:
    import torch

    module_lookup = dict(model.named_modules())
    grouped = _group_lora_tensors(state_dict)
    applied = 0
    errors: list[str] = []
    for raw_name, tensors in grouped.items():
        down = tensors.get("down")
        up = tensors.get("up")
        if down is None or up is None:
            continue

        target_name = _resolve_target_name(raw_name, module_lookup)
        if not target_name:
            continue
        target = module_lookup[target_name]
        weight = getattr(target, "weight", None)
        if weight is None:
            continue

        try:
            rank = int(down.shape[0])
            alpha_tensor = tensors.get("alpha")
            alpha = float(alpha_tensor.item()) if alpha_tensor is not None and hasattr(alpha_tensor, "item") else float(rank)
            multiplier = scale * (alpha / max(rank, 1))

            if weight.ndim != 2 or up.ndim != 2 or down.ndim != 2:
                raise ValueError(f"only Linear LoRA tensors are supported for {target_name}")
            expected = tuple(weight.shape)
            actual = (int(up.shape[0]), int(down.shape[1]))
            if expected != actual:
                raise ValueError(f"{target_name} expects {expected}, LoRA provides {actual}")

            delta = torch.mm(up.float(), down.float()).to(device=weight.device, dtype=weight.dtype)
            with torch.no_grad():
                weight.add_(delta * multiplier)
            applied += 1
        except Exception as exc:
            errors.append(f"{raw_name}: {exc}")
            break

    if errors:
        raise ValueError(f"{source_name} is not compatible with this Wan model. {errors[0]}")
    if applied == 0:
        samples = ", ".join(list(state_dict.keys())[:5])
        raise ValueError(f"No compatible LoRA layers were applied from {source_name}. Sample keys: {samples}")
    return applied


def _group_lora_tensors(state_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    suffixes = {
        ".lora_down.weight": "down",
        ".lora_up.weight": "up",
        ".lora_A.weight": "down",
        ".lora_B.weight": "up",
        ".alpha": "alpha",
    }
    for key, tensor in state_dict.items():
        for suffix, part in suffixes.items():
            if key.endswith(suffix):
                stem = key[: -len(suffix)]
                groups.setdefault(stem, {})[part] = tensor
                break
    return groups


def _resolve_target_name(raw_name: str, module_lookup: dict[str, Any]) -> str | None:
    candidates = _target_name_candidates(raw_name)
    for candidate in candidates:
        if candidate in module_lookup:
            return candidate
    for candidate in candidates:
        suffix = f".{candidate}"
        matches = [name for name in module_lookup if name.endswith(suffix)]
        if matches:
            return matches[0]
    return None


def _target_name_candidates(raw_name: str) -> list[str]:
    value = raw_name
    prefixes = (
        "model.diffusion_model.",
        "diffusion_model.",
        "base_model.model.",
        "model.",
        "transformer.",
    )
    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break

    candidates = [value]
    underscore = value
    for prefix in ("lora_unet_", "diffusion_model_"):
        if underscore.startswith(prefix):
            underscore = underscore[len(prefix) :]

    converted = underscore
    converted = re.sub(r"blocks_(\d+)_", r"blocks.\1.", converted)
    converted = converted.replace("_self_attn_", ".self_attn.")
    converted = converted.replace("_cross_attn_", ".cross_attn.")
    converted = re.sub(r"_ffn_(\d+)$", r".ffn.\1", converted)
    converted = converted.replace("_patch_embedding", ".patch_embedding")
    converted = converted.replace("_text_embedding_", ".text_embedding.")
    converted = converted.replace("_time_embedding_", ".time_embedding.")
    converted = converted.replace("_time_projection_", ".time_projection.")
    candidates.append(converted)
    candidates.append(underscore.replace("_", "."))

    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run official Wan generation with Wan Studio LoRA injection", add_help=True)
    parser.add_argument("--wan_repo_dir", required=True)
    parser.add_argument("--lora_path", action="append", default=[])
    parser.add_argument("--lora_scale", type=float, default=1.0)
    known, generate_args = parser.parse_known_args(argv)

    wan_repo_dir = Path(known.wan_repo_dir).expanduser().resolve()
    generate_py = wan_repo_dir / "generate.py"
    if not generate_py.exists():
        raise FileNotFoundError(f"generate.py not found: {generate_py}")

    lora_files = resolve_lora_files(known.lora_path)
    sys.path.insert(0, str(wan_repo_dir))
    install_lora_hooks(lora_files, known.lora_scale)
    sys.argv = [str(generate_py), *generate_args]
    runpy.run_path(str(generate_py), run_name="__main__")


if __name__ == "__main__":
    main()
