"""Snapshot tests for the curated preserved JAX dump programs."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from ryft.jax.preserved_dump_cases import (
    PreservedDumpProgramCase,
    preserved_dump_program_cases,
)


def case_id(case: PreservedDumpProgramCase) -> str:
    """Returns the stable pytest ID for one preserved dump program case."""

    return case.case_id


@pytest.mark.parametrize(
    "case",
    preserved_dump_program_cases(),
    ids=case_id,
)
def test_preserved_dump_program_snapshots(
    case: PreservedDumpProgramCase,
    snapshot: Any,
) -> None:
    """Verifies the curated preserved dump programs against committed snapshots."""

    actual = {
        "description": case.description,
        "programs": dict(case.programs),
    }
    assert actual == snapshot


def test_preserved_dump_program_corpus_keeps_only_unique_texts() -> None:
    """Ensures the curated preserved corpus does not keep duplicate program texts."""

    program_digests: dict[str, list[str]] = {}
    for case in preserved_dump_program_cases():
        for program_name, program_text in case.programs.items():
            digest = hashlib.sha256(program_text.encode("utf-8")).hexdigest()
            program_digests.setdefault(digest, []).append(
                f"{case.case_id}:{program_name}"
            )

    duplicate_programs = {
        digest: entries
        for digest, entries in program_digests.items()
        if len(entries) > 1
    }
    assert duplicate_programs == {}
