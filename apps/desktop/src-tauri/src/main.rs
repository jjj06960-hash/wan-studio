use serde::Serialize;
use serde_json::Value;
use std::{fs, path::PathBuf};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ColabDriveStatus {
    root_path: String,
    connected: bool,
    runtime_ready: bool,
    model_ready: bool,
    ready_to_prompt: bool,
    message: String,
    mode: String,
    model_path: Option<String>,
    last_checked_at: Option<String>,
    status_count: usize,
    result_count: usize,
    error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StatusFile {
    file_name: String,
    payload: Value,
}

#[tauri::command]
fn read_colab_drive(root_path: String) -> Result<ColabDriveStatus, String> {
    let root = PathBuf::from(root_path.trim());
    if !root.exists() {
        return Ok(disconnected_status(
            root_path,
            "WanStudio folder does not exist yet.".to_string(),
            Some("Choose the synced Google Drive/WanStudio folder after running the Colab setup cell.".to_string()),
        ));
    }

    let jobs_dir = root.join("jobs");
    let results_dir = root.join("results");
    let status_dir = root.join("status");
    for directory in [&jobs_dir, &results_dir, &status_dir] {
        fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    }

    let runtime = read_json(status_dir.join("runtime.json"));
    let model = read_first_model_status(&status_dir);
    let runtime_ready = runtime
        .as_ref()
        .and_then(|value| value.get("ready"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || runtime
            .as_ref()
            .and_then(|value| value.get("message"))
            .and_then(Value::as_str)
            == Some("Ready to prompt!");
    let model_ready = model
        .as_ref()
        .and_then(|value| value.get("status"))
        .and_then(Value::as_str)
        == Some("ready");
    let ready_to_prompt = runtime_ready && model_ready;
    let model_path = model
        .as_ref()
        .and_then(|value| value.get("localPath"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let last_checked_at = model
        .as_ref()
        .and_then(|value| value.get("checkedAt"))
        .and_then(Value::as_str)
        .or_else(|| {
            runtime
                .as_ref()
                .and_then(|value| value.get("checkedAt"))
                .and_then(Value::as_str)
        })
        .map(str::to_string);

    Ok(ColabDriveStatus {
        root_path,
        connected: true,
        runtime_ready,
        model_ready,
        ready_to_prompt,
        message: if ready_to_prompt {
            "Ready to prompt!".to_string()
        } else if runtime_ready {
            "Runtime connected. Waiting for model verification.".to_string()
        } else {
            "Waiting for Colab worker.".to_string()
        },
        mode: "tauri".to_string(),
        model_path,
        last_checked_at,
        status_count: count_matching(&status_dir, ".status.json"),
        result_count: count_matching(&results_dir, ".mp4"),
        error: None,
    })
}

#[tauri::command]
fn write_colab_job(root_path: String, file_name: String, payload: String) -> Result<String, String> {
    let root = PathBuf::from(root_path.trim());
    let jobs_dir = root.join("jobs");
    fs::create_dir_all(&jobs_dir).map_err(|error| error.to_string())?;

    let safe_file_name = PathBuf::from(file_name)
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "Invalid job file name".to_string())?
        .to_string();
    let job_path = jobs_dir.join(safe_file_name);
    fs::write(&job_path, payload).map_err(|error| error.to_string())?;
    Ok(job_path.to_string_lossy().to_string())
}

#[tauri::command]
fn read_colab_status_files(root_path: String) -> Result<Vec<StatusFile>, String> {
    let status_dir = PathBuf::from(root_path.trim()).join("status");
    if !status_dir.exists() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    for entry in fs::read_dir(status_dir).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !file_name.ends_with(".status.json") {
            continue;
        }
        if let Some(payload) = read_json(path) {
            files.push(StatusFile {
                file_name: file_name.to_string(),
                payload,
            });
        }
    }
    Ok(files)
}

fn disconnected_status(root_path: String, message: String, error: Option<String>) -> ColabDriveStatus {
    ColabDriveStatus {
        root_path,
        connected: false,
        runtime_ready: false,
        model_ready: false,
        ready_to_prompt: false,
        message,
        mode: "tauri".to_string(),
        model_path: None,
        last_checked_at: None,
        status_count: 0,
        result_count: 0,
        error,
    }
}

fn read_json(path: PathBuf) -> Option<Value> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn read_first_model_status(status_dir: &PathBuf) -> Option<Value> {
    let entries = fs::read_dir(status_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if file_name.starts_with("model-") && file_name.ends_with(".json") {
            return read_json(path);
        }
    }
    None
}

fn count_matching(directory: &PathBuf, suffix: &str) -> usize {
    fs::read_dir(directory)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| {
                    entry
                        .path()
                        .file_name()
                        .and_then(|value| value.to_str())
                        .map(|file_name| file_name.ends_with(suffix))
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            read_colab_drive,
            write_colab_job,
            read_colab_status_files
        ])
        .run(tauri::generate_context!())
        .expect("error while running Wan Studio");
}

fn main() {
    run()
}
