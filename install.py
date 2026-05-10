from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(command: list[str | Path]) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"> {printable}")
    subprocess.run([str(part) for part in command], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Wan Studio Web UI")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dev", action="store_true", help="Install test dependencies")
    parser.add_argument("--system", action="store_true", help="Install into the current Python instead of .venv")
    args = parser.parse_args()

    if args.system:
        python = Path(sys.executable)
    else:
        if not venv_python().exists():
            run([sys.executable, "-m", "venv", VENV])
        python = venv_python()

    run([python, "-m", "pip", "install", "-U", "pip"])
    package_target = str(ROOT / "workers" / "python")
    if args.dev:
        package_target = f"{package_target}[dev]"
    run([python, "-m", "pip", "install", "-e", package_target])

    print()
    print("Wan Studio installed.")
    print(f"Accelerator selected: {args.accelerator}")
    if args.accelerator == "cuda":
        print("Install the PyTorch CUDA build and Wan2.2 dependencies required by the official Wan repository before using --runner wan.")
    print()
    print("Run the Web UI:")
    print(f"  {python} {ROOT / 'wan_studio.py'} run --open-browser")
    print("Run the low-VRAM 8GB/16GB/24GB backend after installing LightX2V:")
    print(f"  {python} {ROOT / 'wan_studio.py'} run --runner lightx2v --open-browser")


if __name__ == "__main__":
    main()
