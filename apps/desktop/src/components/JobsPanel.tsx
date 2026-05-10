import { CheckCircle2, Clock3, Copy, ExternalLink, RotateCcw, Trash2, XCircle } from "lucide-react";
import type { GenerationJob } from "../lib/types";

interface JobsPanelProps {
  jobs: GenerationJob[];
  onJobsChange: (jobs: GenerationJob[]) => void;
}

export function JobsPanel({ jobs, onJobsChange }: JobsPanelProps) {
  return (
    <section className="panel jobs-panel">
      <div className="section-heading">
        <div>
          <h2>Queue</h2>
          <p>{jobs.length === 0 ? "No generations queued yet." : `${jobs.length} recent request${jobs.length > 1 ? "s" : ""}`}</p>
        </div>
        {jobs.length > 0 && (
          <button className="icon-text-button" type="button" onClick={() => onJobsChange([])}>
            <Trash2 size={16} />
            Clear
          </button>
        )}
      </div>

      <div className="job-list">
        {jobs.map((job) => (
          <article key={job.request.id} className="job-row">
            <StateIcon state={job.status.state} />
            <div>
              <h3>{job.request.prompt}</h3>
              <p>
                {job.request.modelId} · {job.request.task.toUpperCase()} · {job.request.runtime}
              </p>
              <div className="progress-bar compact" aria-label={`Progress ${job.status.progress}%`}>
                <span style={{ width: `${job.status.progress}%` }} />
              </div>
            </div>
            <div className="job-actions">
              <strong>{job.status.progress}%</strong>
              <button type="button" className="icon-button" aria-label="Copy job id" onClick={() => navigator.clipboard?.writeText(job.request.id)}>
                <Copy size={16} />
              </button>
            </div>
            {(job.status.outputPath || job.status.error) && (
              <div className={job.status.error ? "job-detail error" : "job-detail"}>
                {job.status.error ? (
                  <span>{job.status.error}</span>
                ) : (
                  <>
                    <ExternalLink size={15} />
                    <span>{job.status.outputPath}</span>
                  </>
                )}
              </div>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}

function StateIcon({ state }: { state: GenerationJob["status"]["state"] }) {
  if (state === "succeeded") {
    return <CheckCircle2 className="state-icon succeeded" size={22} />;
  }
  if (state === "failed" || state === "cancelled") {
    return <XCircle className="state-icon failed" size={22} />;
  }
  if (state === "running") {
    return <RotateCcw className="state-icon running" size={22} />;
  }
  return <Clock3 className="state-icon queued" size={22} />;
}
