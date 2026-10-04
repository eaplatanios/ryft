#!/usr/bin/env python3
"""Validates byte reproducibility and normalized metadata of native archives."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile


def run_archive_tool(tool: pathlib.Path, manifest: pathlib.Path, output: pathlib.Path) -> None:
    """Runs the archive tool for one test manifest."""
    subprocess.run([str(tool), str(manifest), str(output)], check=True)


def main() -> int:
    """Creates two equivalent archives and requires identical bytes and metadata."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: archive_reproducibility_test.py <archive-tool>")

    tool = pathlib.Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = pathlib.Path(temporary_directory)
        header = root / "header.h"
        library = root / "library.a"
        header.write_bytes(b"header\n")
        library.write_bytes(b"library\n")

        first_manifest = root / "first.json"
        second_manifest = root / "second.json"
        first_manifest.write_text(
            json.dumps(
                [
                    {"mode": 0o755, "path": "lib/library.a", "source": str(library)},
                    {"mode": 0o644, "path": "include/header.h", "source": str(header)},
                ]
            ),
            encoding="utf-8",
        )
        second_manifest.write_text(
            json.dumps(
                [
                    {"mode": 0o644, "path": "include/header.h", "source": str(header)},
                    {"mode": 0o755, "path": "lib/library.a", "source": str(library)},
                ]
            ),
            encoding="utf-8",
        )

        first_archive = root / "first.tar.gz"
        second_archive = root / "second.tar.gz"
        run_archive_tool(tool, first_manifest, first_archive)
        os.utime(header, (1_234_567_890, 1_234_567_890))
        os.utime(library, (1_987_654_321, 1_987_654_321))
        run_archive_tool(tool, second_manifest, second_archive)

        first_bytes = first_archive.read_bytes()
        second_bytes = second_archive.read_bytes()
        if first_bytes != second_bytes:
            raise AssertionError(
                "equivalent manifests produced different archives: "
                f"{hashlib.sha256(first_bytes).hexdigest()} != {hashlib.sha256(second_bytes).hexdigest()}"
            )
        if first_bytes[4:8] != b"\0\0\0\0":
            raise AssertionError("gzip header contains a non-zero modification timestamp")

        with tarfile.open(first_archive, "r:gz") as archive:
            members = archive.getmembers()
        if [member.name for member in members] != ["include/header.h", "lib/library.a"]:
            raise AssertionError(f"archive members are not sorted: {[member.name for member in members]}")
        if [(member.mode, member.mtime) for member in members] != [(0o644, 0), (0o755, 0)]:
            raise AssertionError("archive permissions or timestamps are not normalized")
        if any(
            member.uid != 0 or member.gid != 0 or member.uname != "root" or member.gname != "root"
            for member in members
        ):
            raise AssertionError("archive ownership metadata is not normalized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
