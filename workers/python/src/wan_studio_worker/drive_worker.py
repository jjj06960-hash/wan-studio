from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .runner import FakeWanRunner, write_status
from .schemas import DriveJobEnvelope, GenerationStatus, JobState, status_for


def ensure_drive_tree(root: Path) -> tuple[Path, Path, Path]:
    jobs_dir = root / "jobs"
    results_dir = root / "results"
    status_dir = root / "status"
    for directory in (jobs_dir, results_dir, status_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return jobs_dir, results_dir, status_dir


def list_pending_jobs(root: Path) -> list[Path]:
    jobs_dir, _, _ = ensure_drive_tree(root)
    return sorted(path for path in jobs_dir.glob("*.request.json") if not (jobs_dir / path.name.replace(".request.json", ".claimed")).exists())


def claim_job(job_path: Path) -> Path:
    claim_path = job_path.with_name(job_path.name.replace(".request.json", ".claimed"))
    claim_path.write_text("claimed\n", encoding="utf-8")
    return claim_path


async def process_once(root: Path, runner: FakeWanRunner | None = None) -> list[GenerationStatus]:
    runner = runner or FakeWanRunner()
    jobs_dir, results_dir, status_dir = ensure_drive_tree(root)
    statuses: list[GenerationStatus] = []
    for job_path in list_pending_jobs(root):
        try:
            envelope = DriveJobEnvelope.model_validate_json(job_path.read_text(encoding="utf-8"))
        except (ValidationError, json.JSONDecodeError) as exc:
            failed = status_for(job_path.stem, JobState.FAILED, 0, error=str(exc))
            write_status(status_dir / f"{job_path.stem}.status.json", failed)
            statuses.append(failed)
            continue

        claim_job(job_path)
        running = status_for(envelope.request.id, JobState.RUNNING, 1)
        write_status(status_dir / f"{envelope.request.id}.status.json", running)
        status = await runner.run(envelope.request, results_dir)
        write_status(status_dir / f"{envelope.request.id}.status.json", status)
        statuses.append(status)

    return statuses
