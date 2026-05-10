from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class WanBaseModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class WanTask(StrEnum):
    T2V = "t2v"
    I2V = "i2v"
    TI2V = "ti2v"
    S2V = "s2v"
    ANIMATE = "animate"


class RuntimeKind(StrEnum):
    LOCAL = "local"
    COLAB_DRIVE = "colab-drive"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelCapabilities(WanBaseModel):
    tasks: list[WanTask] = Field(default_factory=list)
    optimized: bool = False
    min_vram_gb: int | None = None
    notes: list[str] = Field(default_factory=list)


class ModelInstall(WanBaseModel):
    model_id: str
    display_name: str
    family: Literal["wan"] = "wan"
    task: WanTask | Literal["multi"]
    local_path: str
    source: Literal["huggingface", "modelscope", "local"]
    status: Literal["recommended", "ready", "missing", "invalid", "unchecked"]
    capabilities: ModelCapabilities
    repo_id: str | None = None
    last_verified_at: datetime | None = None


class GenerationRequest(WanBaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    prompt: str
    image: str | None = None
    model_id: str
    model_path: str = ""
    size: str = "832x480"
    seed: int = 0
    steps: int = 18
    offload_model: bool = True
    t5_cpu: bool = True
    runtime: RuntimeKind = RuntimeKind.LOCAL
    task: WanTask = WanTask.T2V
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GenerationStatus(WanBaseModel):
    id: str
    state: JobState
    progress: int = Field(ge=0, le=100)
    output_path: str | None = None
    error: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GenerationJob(WanBaseModel):
    request: GenerationRequest
    status: GenerationStatus


class DriveJobEnvelope(WanBaseModel):
    schema_version: int = 1
    request: GenerationRequest


def status_for(request_id: str, state: JobState, progress: int, *, output_path: Path | None = None, error: str | None = None) -> GenerationStatus:
    return GenerationStatus(
        id=request_id,
        state=state,
        progress=progress,
        output_path=str(output_path) if output_path else None,
        error=error,
    )
