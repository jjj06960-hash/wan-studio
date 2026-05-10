import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { join } from "node:path";

const candidates = [
  join(process.cwd(), ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python"),
  "python3",
  "python",
];

const python = candidates.find((candidate) => candidate.includes("/") || candidate.includes("\\") ? existsSync(candidate) : true);
const result = spawnSync(python, ["-m", "pytest", "workers/python/tests"], {
  stdio: "inherit",
  cwd: process.cwd(),
});

process.exit(result.status ?? 1);
