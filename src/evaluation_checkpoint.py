"""Durable question-level commits and a single-writer lock for batch runs."""

from contextlib import contextmanager
import hashlib
import json
import os
import sys
from pathlib import Path
import tempfile

from src.telemetry_errors import redact_payload


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_write_json(path, value):
    """Publish complete derived artifacts; a crash cannot truncate the old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(redact_payload(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _writer_lock(path):
    """OS locks release on process exit/reboot; the lock file may safely remain."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another process is writing this evaluation: {path}") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class EvaluationCheckpoint:
    """Append/fsync a checksummed record after generation, selection and EX.

    Only a final line without a newline may be discarded after interruption.
    Corrupt committed records or changed experiment controls fail closed.
    Failed questions are completed attempts too; resume never cherry-picks them.
    """

    def __init__(self, path, manifest, *, resume=False, outputs=()):
        self.path = Path(path)
        self.manifest = manifest
        self.resume = resume
        self.outputs = outputs
        self.records = {}

    def __enter__(self):
        self._lock = _writer_lock(self.path.with_suffix(self.path.suffix + ".lock"))
        self._lock.__enter__()
        try:
            if self.resume:
                self._load()
            else:
                if self.path.exists() or any(Path(path).exists() for path in self.outputs):
                    raise FileExistsError("Evaluation outputs already exist; use --resume with their checkpoint or choose a new --output")
                with self.path.open("xb") as handle:
                    handle.write(_encoded({"version": 1, "manifest": self.manifest}) + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                _sync_directory(self.path.parent)
            return self
        except BaseException:
            self._lock.__exit__(*sys.exc_info())
            raise

    def _load(self):
        if not self.path.is_file():
            raise FileNotFoundError(f"No resumable checkpoint: {self.path}; older end-of-run JSON files cannot recover an interrupted run")
        with self.path.open("r+b") as handle:
            first = handle.readline()
            if not first.endswith(b"\n"):
                raise ValueError("Checkpoint has an incomplete header")
            header = json.loads(first)
            if header.get("version") != 1:
                raise ValueError("Unsupported checkpoint version")
            saved_manifest = header["manifest"]
            fields = set(self.manifest) | set(saved_manifest)
            changed = sorted(key for key in fields if key != "created_at"
                             and saved_manifest.get(key) != self.manifest.get(key))
            if changed:
                raise ValueError(f"Resume configuration does not match checkpoint: {', '.join(changed)}")
            self.manifest = saved_manifest
            ids = self.manifest["question_ids"]
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    handle.seek(offset)
                    handle.truncate()
                    handle.flush()
                    os.fsync(handle.fileno())
                    break
                event = json.loads(line)
                record = event["record"]
                if hashlib.sha256(_encoded(record)).hexdigest() != event.get("sha256"):
                    raise ValueError("Checkpoint record checksum failed")
                index = len(self.records)
                if index >= len(ids) or record.get("question_id") != ids[index]:
                    raise ValueError("Checkpoint question IDs are duplicated or out of dataset order")
                if (record["metrics"]["question_id"] != ids[index]
                        or record["result"]["question_id"] != ids[index]):
                    raise ValueError("Checkpoint metrics/results do not match question ID")
                self.records[ids[index]] = record

    def append(self, record):
        record = redact_payload(record)
        expected = self.manifest["question_ids"][len(self.records)]
        if record["question_id"] != expected:
            raise ValueError("Checkpoint must commit in dataset order")
        event = {"record": record, "sha256": hashlib.sha256(_encoded(record)).hexdigest()}
        with self.path.open("ab") as handle:
            handle.write(_encoded(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records[expected] = record

    def __exit__(self, *args):
        return self._lock.__exit__(*args)
