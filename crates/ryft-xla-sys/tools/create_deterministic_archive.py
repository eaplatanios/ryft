#!/usr/bin/env python3

"""Creates a byte-reproducible gzip-compressed tar archive from a JSON manifest."""

from __future__ import annotations

import gzip
import json
import pathlib
import sys
import tarfile

from typing import TypedDict


class ArchiveEntry(TypedDict):
    """Describes one regular file admitted to the archive."""

    mode: int
    path: str
    source: str


def load_manifest(path: pathlib.Path) -> list[ArchiveEntry]:
    """Loads and validates archive entries from `path`."""
    raw_entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_entries, list):
        raise ValueError("archive manifest must contain a list")

    entries: list[ArchiveEntry] = []
    seen_paths: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("archive manifest entries must be objects")
        source = raw_entry.get("source")
        archive_path = raw_entry.get("path")
        mode = raw_entry.get("mode")
        if not isinstance(source, str) or not isinstance(archive_path, str) or not isinstance(mode, int):
            raise ValueError("archive manifest entries require string `source`/`path` and integer `mode` fields")

        canonical_path = pathlib.PurePosixPath(archive_path)
        if canonical_path.is_absolute() or ".." in canonical_path.parts or canonical_path.as_posix() != archive_path:
            raise ValueError(f"archive path is not canonical and relative: {archive_path}")
        if archive_path in seen_paths:
            raise ValueError(f"duplicate archive path: {archive_path}")
        seen_paths.add(archive_path)
        entries.append({"mode": mode, "path": archive_path, "source": source})
    return sorted(entries, key=lambda entry: entry["path"])


def create_archive(manifest_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """Creates a deterministic archive described by `manifest_path` at `output_path`."""
    entries = load_manifest(manifest_path)
    with output_path.open("wb") as output:
        # Suppress the original filename and fix the gzip timestamp so the wrapper is reproducible.
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as archive:
                for entry in entries:
                    source_path = pathlib.Path(entry["source"])
                    archive_info = tarfile.TarInfo(entry["path"])
                    archive_info.size = source_path.stat().st_size
                    archive_info.mode = entry["mode"]
                    archive_info.mtime = 0
                    archive_info.uid = 0
                    archive_info.gid = 0
                    archive_info.uname = "root"
                    archive_info.gname = "root"
                    with source_path.open("rb") as source:
                        archive.addfile(archive_info, source)


def main() -> int:
    """Runs the deterministic archive creator command-line interface."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: create_deterministic_archive.py <manifest.json> <archive.tar.gz>")
    create_archive(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
