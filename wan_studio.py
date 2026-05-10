from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKER_SRC = ROOT / "workers" / "python" / "src"
if str(WORKER_SRC) not in sys.path:
    sys.path.insert(0, str(WORKER_SRC))

from wan_studio_worker.web import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan Studio")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Start the Wan Studio Web UI")
    run_parser.add_argument("--host", default="127.0.0.1")
    run_parser.add_argument("--port", type=int, default=7860)
    run_parser.add_argument("--root", default=".", help="Workspace root for config, models, and outputs")
    run_parser.add_argument("--open-browser", action="store_true")
    run_parser.add_argument("--share", action="store_true", help="Expose the port in Google Colab when available")
    run_parser.add_argument("--runner", choices=["fake", "wan"], default="fake")
    run_parser.add_argument("--wan-repo-dir", default="", help="Path to the official Wan2.2 repository when --runner wan is used")

    args = parser.parse_args()
    if args.command in {None, "run"}:
        run_web(args)


def run_web(args: argparse.Namespace) -> None:
    import uvicorn

    app = create_app(
        root=Path(args.root),
        runner_kind=args.runner,
        wan_repo_dir=Path(args.wan_repo_dir) if args.wan_repo_dir else None,
    )
    url = f"http://{args.host}:{args.port}"

    if args.open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    if args.share and in_colab():
        config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        time.sleep(2)
        from google.colab import output  # type: ignore

        output.serve_kernel_port_as_window(args.port)
        print(f"Wan Studio is running on Colab port {args.port}. Keep this cell alive.")
        try:
            while thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            server.should_exit = True
        return

    print(f"Wan Studio Web UI: {url}")
    uvicorn.run(app, host=args.host, port=args.port)


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
