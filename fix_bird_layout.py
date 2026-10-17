"""Extract BIRD archives, validate outputs, then remove their source folders."""

import json
from pathlib import Path, PurePosixPath
import shutil
import time
import zipfile

DATA = Path(__file__).resolve().parent / "data"
GIB = 2**30


def checked_path(path):
    root = DATA.resolve()
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise RuntimeError(f"Unsafe target outside data: {path}")
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise RuntimeError(f"Refusing a linked target: {path}")
    return resolved


def validate(split, records):
    database_dir = DATA / f"{split}_databases"
    ids = {record["db_id"] for record in records}
    for db_id in ids:
        if not isinstance(db_id, str) or any(c in db_id for c in "/\\:") or db_id in (".", "..", ""):
            raise RuntimeError(f"Unsafe db_id: {db_id!r}")
        database = checked_path(database_dir / db_id / f"{db_id}.sqlite")
        with database.open("rb") as handle:
            if handle.read(16) != b"SQLite format 3\x00":
                raise RuntimeError(f"Not a SQLite database: {database}")
    print(f"[{split}] Validated {len(records):,} questions and {len(ids)} SQLite databases.", flush=True)


def process(split, source_name):
    source = checked_path(DATA / source_name)
    dataset = source / f"{split}.json"
    destination_json = DATA / f"{split}.json"
    if not source.exists():
        records = json.loads(destination_json.read_text(encoding="utf-8"))
        validate(split, records)
        print(f"[{split}] Already complete.", flush=True)
        return
    records = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Empty or invalid dataset: {dataset}")
    archive_path = source / f"{split}_databases.zip"
    target = checked_path(DATA / f"{split}_databases")
    with zipfile.ZipFile(archive_path) as archive:
        files = []
        for entry in archive.infolist():
            parts = PurePosixPath(entry.filename).parts
            if "__MACOSX" in parts or any(p.startswith("._") or p == ".DS_Store" for p in parts):
                continue
            if entry.is_dir():
                continue
            if (not parts or parts[0] != f"{split}_databases" or
                len(parts) < 2 or ".." in parts or "\\" in entry.filename or ":" in entry.filename):
                raise RuntimeError(f"Unexpected archive path: {entry.filename}")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError(f"Archive contains a symbolic link: {entry.filename}")
            output = checked_path(target.joinpath(*parts[1:]))
            files.append((entry, output))
        total = sum(entry.file_size for entry, _ in files)
        existing = sum(output.stat().st_size for _, output in files if output.is_file())
        required = max(0, total - existing)
        available = shutil.disk_usage(DATA).free
        print(f"[{split}] {len(files)} files, {total / GIB:.2f} GiB expanded; "
              f"{available / GIB:.2f} GiB free.", flush=True)
        if available < required + GIB:
            raise RuntimeError(
                f"INSUFFICIENT SPACE for {split}: free another "
                f"{(required + GIB - available) / GIB:.2f} GiB on C:. "
                f"Archive and source folder retained. Rerun this script after freeing space."
            )
        done = 0
        started = last_print = time.monotonic()
        for index, (entry, output) in enumerate(files, 1):
            output.parent.mkdir(parents=True, exist_ok=True)
            # Re-extract on rerun: size alone is insufficient to validate CRC.
            if output.exists():
                output.unlink()
            partial = checked_path(output.with_name(output.name + ".extracting"))
            with archive.open(entry) as reader, partial.open("wb") as writer:
                while chunk := reader.read(8 * 2**20):
                    writer.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if now - last_print >= 3:
                        print(f"[{split}] {done / GIB:.2f}/{total / GIB:.2f} GiB "
                              f"({100 * done / total:.1f}%), file {index}/{len(files)}: "
                              f"{output.name}; elapsed {now - started:.0f}s", flush=True)
                        last_print = now
            partial.replace(output)
            print(f"[{split}] Completed {index}/{len(files)}: {output.relative_to(DATA)}", flush=True)
    validate(split, records)
    for item in source.iterdir():
        if item.suffix.lower() in {".json", ".sql"}:
            destination = checked_path(DATA / item.name)
            if destination.exists():
                if destination.read_bytes() != item.read_bytes():
                    raise RuntimeError(f"Refusing to replace different file: {destination}")
                item.unlink()
            else:
                item.replace(destination)
    leftovers = [item.name for item in source.iterdir() if item != archive_path]
    if leftovers:
        raise RuntimeError(f"Unexpected files retained in {source}: {leftovers}")
    checked_path(source)
    shutil.rmtree(source)
    print(f"[{split}] JSON moved to {destination_json}; source folder and ZIP removed.", flush=True)


if __name__ == "__main__":
    for split, source in (("dev", "dev_20240627"), ("train", "train")):
        process(split, source)
    macos = DATA / "__MACOSX"
    if macos.exists():
        shutil.rmtree(checked_path(macos))
    print("COMPLETE: BIRD dev and training layouts validated; residual folders removed.", flush=True)
