#!/usr/bin/env python3
"""Compare clean pinned Mosaic GPU sources; every byte change needs an exact reviewed decision."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath


# Include implementation, dialect, runtime and FFI changes, not merely exported symbol names.
SCOPES = {
    "jax": ["jax/experimental/mosaic/gpu", "jaxlib/mosaic/dialect/gpu", "jaxlib/mosaic/gpu"],
    "xla": ["xla/ffi/api", "xla/backends/gpu/runtime"],
}


def git(root, *arguments):
    """Run a bounded read-only Git query against an explicit source checkout."""
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True, capture_output=True, timeout=30
    ).stdout


def pins(workspace):
    """Read exact existing upstream commit pins without evaluating the Bazel file."""
    source = workspace.read_text()
    result = {}
    for owner in SCOPES:
        matches = re.findall(rf'^{owner.upper()}_COMMIT = "([0-9a-f]{{40}})"$', source, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {owner.upper()}_COMMIT pin")
        result[owner] = matches[0]
    return result


def scoped(path):
    """Reject noncanonical or out-of-scope paths before interpreting any review entry."""
    if not isinstance(path, str) or "\\" in path:
        return False
    parts = PurePosixPath(path).parts
    if not parts or any(part in (".", "..") for part in parts) or str(PurePosixPath(path)) != path:
        return False
    return any(path.startswith(f"{owner}/{scope}/") for owner, scopes in SCOPES.items() for scope in scopes)


def snapshot(roots, expected_pins):
    """Hash all tracked scope files in clean checkouts at the declared commits; never follow symlinks."""
    files = {}
    for owner, scopes in SCOPES.items():
        root = roots[owner].resolve(strict=True)
        if git(root, "rev-parse", "HEAD").decode().strip() != expected_pins[owner]:
            raise ValueError(f"{owner} checkout does not match its declared pin")
        if git(root, "status", "--porcelain", "--untracked-files=all", "--", *scopes):
            raise ValueError(f"{owner} GPU surface has uncommitted changes")
        for scope in scopes:
            paths = git(root, "ls-files", "-z", "--", scope).decode().split("\0")
            paths = [path for path in paths if path]
            if not paths:
                raise ValueError(f"missing tracked scope {owner}/{scope}")
            for path in paths:
                name = f"{owner}/{path}"
                source = root / path
                if not scoped(name) or source.is_symlink() or source.resolve() != source or not source.is_file():
                    raise ValueError(f"invalid source path {name}")
                files[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        # Refuse a concurrent checkout or dirty edit during collection rather than approving a mixed snapshot.
        if git(root, "rev-parse", "HEAD").decode().strip() != expected_pins[owner] or git(
            root, "status", "--porcelain", "--untracked-files=all", "--", *scopes
        ):
            raise ValueError(f"{owner} checkout changed while collecting its surface")
    return {"schema": 1, "pins": expected_pins, "scopes": SCOPES, "files": dict(sorted(files.items()))}


def validate(document):
    """Validate a snapshot before comparing identities or accepting decisions."""
    if not isinstance(document, dict) or set(document) != {"schema", "pins", "scopes", "files"}:
        raise ValueError("invalid surface snapshot fields")
    if type(document["schema"]) is not int or document["schema"] != 1 or document["scopes"] != SCOPES:
        raise ValueError("unsupported surface schema or scope")
    if not isinstance(document["pins"], dict) or set(document["pins"]) != set(SCOPES):
        raise ValueError("invalid source pins")
    for value in document["pins"].values():
        if not isinstance(value, str) or re.fullmatch("[0-9a-f]{40}", value) is None:
            raise ValueError("invalid source pin digest")
    if not isinstance(document["files"], dict):
        raise ValueError("invalid source files")
    for path, digest in document["files"].items():
        if not scoped(path) or not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError("invalid source path or digest")
    for owner, scopes in SCOPES.items():
        for scope in scopes:
            if not any(path.startswith(f"{owner}/{scope}/") for path in document["files"]):
                raise ValueError(f"snapshot omits required scope {owner}/{scope}")


def changes(before, after):
    """Return exact additions, removals and content changes, with no automatic acceptance."""
    validate(before)
    validate(after)
    return {
        path: {"before": before["files"].get(path), "after": after["files"].get(path)}
        for path in sorted(before["files"].keys() | after["files"].keys())
        if before["files"].get(path) != after["files"].get(path)
    }


def check(before, after, decisions):
    """Require pin-bound review reasons for exactly the changed paths and byte identities."""
    delta = changes(before, after)
    if not isinstance(decisions, dict) or set(decisions) != {"before_pins", "after_pins", "reason", "files"}:
        raise ValueError("invalid review fields")
    if decisions["before_pins"] != before["pins"] or decisions["after_pins"] != after["pins"]:
        raise ValueError("review pins differ from the compared snapshots")
    if not isinstance(decisions["reason"], str) or not decisions["reason"].strip():
        raise ValueError("review requires an explicit upgrade reason")
    if not isinstance(decisions["files"], dict) or set(decisions["files"]) != set(delta):
        raise ValueError("review must cover exactly the changed paths")
    for path, expected in delta.items():
        decision = decisions["files"][path]
        if not isinstance(decision, dict) or set(decision) != {"before", "after", "reason"}:
            raise ValueError(f"invalid review entry {path}")
        if decision["before"] != expected["before"] or decision["after"] != expected["after"]:
            raise ValueError(f"stale review digest {path}")
        if not isinstance(decision["reason"], str) or not decision["reason"].strip():
            raise ValueError(f"missing review reason {path}")
    return len(delta)


def load(path):
    """Read JSON while rejecting duplicate keys that could conceal contradictory decisions."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def main():
    """Print a snapshot/diff or validate explicit review decisions; never modify upstream checkouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("snapshot")
    create.add_argument("--workspace", type=Path, required=True)
    create.add_argument("--jax", type=Path, required=True)
    create.add_argument("--xla", type=Path, required=True)
    for name in ("diff", "check"):
        compare = commands.add_parser(name)
        compare.add_argument("before", type=Path)
        compare.add_argument("after", type=Path)
        if name == "check":
            compare.add_argument("decisions", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.command == "snapshot":
            result = snapshot({owner: getattr(arguments, owner) for owner in SCOPES}, pins(arguments.workspace))
        elif arguments.command == "diff":
            result = changes(load(arguments.before), load(arguments.after))
        else:
            result = {"reviewed_changes": check(load(arguments.before), load(arguments.after), load(arguments.decisions))}
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"surface check failed: {error}\n")


if __name__ == "__main__":
    main()
