from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .runner import FakeWanRunner
from .schemas import GenerationJob, GenerationRequest, GenerationStatus, JobState, status_for

app = FastAPI(title="Wan Studio Worker", version="0.1.0")
runner = FakeWanRunner()
output_dir = Path("outputs")
jobs: dict[str, GenerationJob] = {}
tasks: dict[str, asyncio.Task[None]] = {}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "runner": "fake"}


@app.post("/v1/generations", response_model=GenerationJob)
async def create_generation(request: GenerationRequest) -> GenerationJob:
    job = GenerationJob(request=request, status=status_for(request.id, JobState.QUEUED, 0))
    jobs[request.id] = job
    tasks[request.id] = asyncio.create_task(_run_job(request))
    return job


@app.get("/v1/generations/{job_id}", response_model=GenerationJob)
async def get_generation(job_id: str) -> GenerationJob:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="generation not found")
    return jobs[job_id]


@app.post("/v1/generations/{job_id}/cancel", response_model=GenerationStatus)
async def cancel_generation(job_id: str) -> GenerationStatus:
    task = tasks.get(job_id)
    if task and not task.done():
        task.cancel()
    status = status_for(job_id, JobState.CANCELLED, jobs.get(job_id).status.progress if job_id in jobs else 0)
    if job_id in jobs:
        jobs[job_id].status = status
    return status


async def _run_job(request: GenerationRequest) -> None:
    async def on_progress(status: GenerationStatus) -> None:
        if request.id in jobs:
            jobs[request.id].status = status

    try:
        jobs[request.id].status = status_for(request.id, JobState.RUNNING, 1)
        jobs[request.id].status = await runner.run(request, output_dir, on_progress)
    except asyncio.CancelledError:
        jobs[request.id].status = status_for(request.id, JobState.CANCELLED, jobs[request.id].status.progress)
    except Exception as exc:  # pragma: no cover - defensive API boundary
        jobs[request.id].status = status_for(request.id, JobState.FAILED, jobs[request.id].status.progress, error=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
