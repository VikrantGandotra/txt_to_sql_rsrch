"""Persist matrix status and optionally send minimal ntfy failure/completion alerts."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from src.evaluation_checkpoint import atomic_write_json


def publish_status(root, status, configuration, exit_code=0):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    event = {"status": status, "configuration": configuration, "exit_code": exit_code,
             "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    atomic_write_json(root / "STATUS.json", event)
    if status == "running":
        # These two markers describe the current attempt, not previous attempts.
        for name in ("FAILED.json", "COMPLETED.json"):
            (root / name).unlink(missing_ok=True)
    if status in {"failed", "completed"}:
        atomic_write_json(root / f"{status.upper()}.json", event)
    print(f"MARS MATRIX {status.upper()}: {configuration} (exit={exit_code})", flush=True)
    if status not in {"failed", "completed"}:
        return event
    # Do not source .env as shell code or put notification secrets in commands.
    from dotenv import dotenv_values
    values = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    url = os.environ.get("MARS_NOTIFY_URL", values.get("MARS_NOTIFY_URL", "")) or ""
    token = os.environ.get("MARS_NOTIFY_TOKEN", values.get("MARS_NOTIFY_TOKEN", "")) or ""
    if not url:
        print("Push notification not configured; status is saved locally.", flush=True)
        return event
    try:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Notification URL must be an HTTPS topic URL without credentials/query parameters")
        headers = {"Title": f"MARS ablation {status.upper()}", "Content-Type": "text/plain; charset=utf-8",
                   "Priority": "high" if status == "failed" else "default"}
        if token:
            headers["Authorization"] = "Bearer " + token
        # Only these non-sensitive fields leave the container: no SQL, error
        # payloads, keys, base URLs, dataset samples, or absolute file paths.
        message = f"Matrix {status}. Configuration: {configuration}. Exit code: {exit_code}. UTC: {event['timestamp_utc']}"
        with urlopen(Request(url, data=message.encode(), headers=headers, method="POST"), timeout=10) as response:
            response.read(1024)
    except Exception as error:
        # Never disclose URL, authorization headers, or response bodies.
        print(f"Push notification failed ({type(error).__name__}); consult STATUS.json.", flush=True)
    return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--status", required=True, choices=("running", "failed", "completed"))
    parser.add_argument("--configuration", default="preflight")
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()
    publish_status(args.root, args.status, args.configuration, args.exit_code)


if __name__ == "__main__":
    main()
