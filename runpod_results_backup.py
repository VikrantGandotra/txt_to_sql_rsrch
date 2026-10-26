"""Independent, read-only snapshots of a running experiment, uploaded by rclone.

No imports from the experiment. Deploy outside its source tree. The last
verified Drive slot is never overwritten while a new backup is in progress.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def utc():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def copy_prefix(source, destination):
    """Copy an open file's initial length, without waiting for future appends."""
    with source.open("rb") as reader, destination.open("wb") as writer:
        remaining = os.fstat(reader.fileno()).st_size
        while remaining:
            block = reader.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("A source file shrank during snapshot; retrying is required")
            writer.write(block)
            remaining -= len(block)


def copy_checkpoint(source, destination):
    """Copy only complete, valid question commits visible at file-open time."""
    count = 0
    with source.open("rb") as reader, destination.open("wb") as writer:
        remaining = os.fstat(reader.fileno()).st_size
        header = reader.readline(remaining)
        remaining -= len(header)
        if not header.endswith(b"\n"):
            raise ValueError("Incomplete checkpoint header")
        metadata = json.loads(header)
        if metadata.get("version") != 1:
            raise ValueError("Unsupported checkpoint version")
        ids = metadata["manifest"]["question_ids"]
        writer.write(header)
        while remaining:
            line = reader.readline(remaining)
            remaining -= len(line)
            if not line:
                raise ValueError("Checkpoint shrank during snapshot")
            if not line.endswith(b"\n"):
                break  # An in-flight final line is not a committed record.
            event = json.loads(line)
            record = event["record"]
            if event.get("sha256") != hashlib.sha256(encoded(record)).hexdigest():
                raise ValueError("Checkpoint checksum failed")
            if count >= len(ids) or any(record[part]["question_id"] != ids[count] for part in ("metrics", "result")) or record["question_id"] != ids[count]:
                raise ValueError("Checkpoint question ordering/identity failed")
            writer.write(line)
            count += 1
    return count


def create_snapshot(project, destination):
    """Copy outputs and small reproducibility artifacts; never credentials/data."""
    report = {"created_utc": utc(), "checkpoint_questions": {}, "files": 0, "bytes": 0}
    roots = [project / name for name in ("results", "artifacts", "src")
             if (project / name).is_dir() and not (project / name).is_symlink()]
    sources = [p for root in roots for p in root.rglob("*") if p.is_file()]
    names = ("run_ablation.py", "run_full_ablation.sh", "requirements.txt", "requirements-runpod.txt",
             "METRICS.md", "compiled_dspy_model.json", "compiled_dspy_verifier.json",
             "compiled_dspy_model.compilation.metrics.json", "compiled_dspy_verifier.compilation.metrics.json")
    sources.extend(project / name for name in names if (project / name).is_file())
    for source in sorted(set(sources)):
        relative = source.relative_to(project)
        if source.is_symlink() or any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if source.suffix not in {".json", ".jsonl", ".log", ".txt", ".csv", ".md", ".py", ".sh"}:
            continue
        if "rclone" in source.name.lower() or source.name.endswith(".lock"):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if source.name.endswith(".checkpoint.jsonl"):
                report["checkpoint_questions"][relative.as_posix()] = copy_checkpoint(source, target)
            else:
                copy_prefix(source, target)
        except FileNotFoundError:
            # Only transient runner status markers may disappear between scans.
            if source.name not in {"FAILED.json", "COMPLETED.json"}:
                raise
            target.unlink(missing_ok=True)
            continue
        report["files"] += 1
        report["bytes"] += target.stat().st_size
    if not report["checkpoint_questions"]:
        raise ValueError("No checkpoint files found; check --project before relying on backup")
    save_json(destination / "BACKUP_MANIFEST.json", report)
    return report


def rclone(args, *command):
    process = subprocess.run([args.rclone, "--config", str(args.config), *command],
                             text=True, capture_output=True, timeout=7200)
    if process.returncode:
        # Provider responses may contain private account details. Keep them out
        # of notifications and routine logs; use rclone manually to diagnose.
        raise RuntimeError(f"rclone {command[0]} failed (exit {process.returncode})")
    return process.stdout


def backup_once(args):
    remote = args.remote.rstrip("/")
    rclone(args, "mkdir", remote)
    entries = json.loads(rclone(args, "lsjson", remote, "--files-only"))
    slot = 0
    if any(entry["Name"] == "LAST_SUCCESS.json" for entry in entries):
        previous = json.loads(rclone(args, "cat", remote + "/LAST_SUCCESS.json"))
        if previous.get("slot") not in (0, 1):
            raise ValueError("Invalid remote recovery pointer")
        slot = 1 - previous["slot"]
    with tempfile.TemporaryDirectory(prefix="snapshot_", dir=args.state) as directory:
        base = Path(directory)
        stage = base / "files"
        stage.mkdir()
        report = create_snapshot(args.project, stage)
        upload = base / "upload"
        upload.mkdir()
        archive = upload / f"slot-{slot}.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as handle:
            handle.add(stage, arcname="experiment", recursive=True)
        with archive.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        success = {**report, "verified_utc": None, "slot": slot,
                   "archive": archive.name, "archive_sha256": digest,
                   "archive_bytes": archive.stat().st_size}
        save_json(upload / f"slot-{slot}.manifest.json", success)
        rclone(args, "copy", str(upload), remote, "--checksum", "--transfers", "1", "--checkers", "2")
        rclone(args, "check", str(upload), remote, "--one-way", "--checkers", "2")
        success["verified_utc"] = utc()
        pointer = base / "LAST_SUCCESS.json"
        save_json(pointer, success)
        rclone(args, "copyto", str(pointer), remote + "/LAST_SUCCESS.json", "--checksum")
        save_json(args.state / "LAST_SUCCESS.json", success)
        save_json(args.state / "STATUS.json", {"status": "verified", **success})
        return success


def notify(args, message):
    if not args.notify_file or not args.notify_file.is_file():
        return
    try:
        url = args.notify_file.read_text().strip()
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use an HTTPS notification topic URL")
        with urlopen(Request(url, data=message.encode(), headers={"Title": "MARS BACKUP status"}, method="POST"), timeout=10):
            pass
    except Exception:
        print(f"{utc()} Backup notification could not be delivered.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--remote", default="marsdrive:MARS-SQL-Backups/runpod_full_01")
    parser.add_argument("--rclone", default="rclone")
    parser.add_argument("--interval", type=int, default=900)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--notify-file", type=Path)
    args = parser.parse_args()
    args.project = args.project.resolve()
    args.state = args.state.resolve()
    if not (args.project / "run_ablation.py").is_file() or not args.config.is_file():
        parser.error("Project or rclone configuration missing")
    if args.state == args.project or args.project in args.state.parents:
        parser.error("--state must be outside the experiment directory")
    if args.interval < 60:
        parser.error("--interval must be at least 60 seconds")
    os.umask(0o077)
    args.state.mkdir(parents=True, exist_ok=True)
    # Linux advisory lock is separate from all experiment/checkpoint locks.
    import fcntl
    with (args.state / "backup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failed = False
        while True:
            if not args.once and (args.state / "STOP").exists():
                print(f"{utc()} Backup stopped cleanly; experiment was not changed.", flush=True)
                return
            started = time.monotonic()
            try:
                success = backup_once(args)
                print(f"{utc()} VERIFIED: slot={success['slot']}, files={success['files']}, committed questions={sum(success['checkpoint_questions'].values())}", flush=True)
                if failed:
                    notify(args, "Backup recovered. A new verified Drive copy is available. Ablation was not changed.")
                failed = False
                if args.once:
                    return
            except Exception as error:
                message = f"{type(error).__name__}: {error}" if not isinstance(error, subprocess.TimeoutExpired) else "rclone timed out"
                save_json(args.state / "STATUS.json", {"status": "failed", "time_utc": utc(), "error": message})
                print(f"{utc()} BACKUP FAILED: {message}. Existing verified Drive slot retained.", flush=True)
                if not failed:
                    notify(args, "Backup failed; prior verified Drive copy retained. Check backup STATUS.json. Ablation was not stopped.")
                failed = True
                if args.once:
                    raise SystemExit(1) from None
            deadline = started + args.interval
            while time.monotonic() < deadline and not (args.state / "STOP").exists():
                time.sleep(max(0, min(5, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
