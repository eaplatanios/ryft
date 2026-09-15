"""Deterministic surface-change, decision validation and clean-checkout tests."""

import copy
import tempfile
import unittest
from pathlib import Path

import check_surface as surface


class SurfaceTests(unittest.TestCase):
    """Exercise source identity and review decisions without network access or native compilation."""

    def snapshot(self):
        """Construct a complete explicit scope inventory for comparison tests."""
        return {
            "schema": 1,
            "pins": {owner: "a" * 40 for owner in surface.SCOPES},
            "scopes": surface.SCOPES,
            "files": {f"{owner}/{scope}/file.cc": "b" * 64
                      for owner, scopes in surface.SCOPES.items() for scope in scopes},
        }

    def test_changes(self):
        before = self.snapshot()
        after = copy.deepcopy(before)
        changed = "jax/jaxlib/mosaic/gpu/file.cc"
        removed = "jax/jaxlib/mosaic/gpu/removed.cc"
        added = "xla/xla/ffi/api/new.h"
        before["files"][removed] = "c" * 64
        after["files"][changed] = "d" * 64
        after["files"][added] = "e" * 64
        self.assertEqual(surface.changes(before, after), {
            changed: {"before": "b" * 64, "after": "d" * 64},
            removed: {"before": "c" * 64, "after": None},
            added: {"before": None, "after": "e" * 64},
        })

    def test_check(self):
        before = self.snapshot()
        after = copy.deepcopy(before)
        after["pins"]["jax"] = "c" * 40
        path = "jax/jaxlib/mosaic/gpu/file.cc"
        after["files"][path] = "d" * 64
        decisions = {
            "before_pins": before["pins"], "after_pins": after["pins"],
            "reason": "review pinned runtime upgrade and rerun serialization/native gates",
            "files": {path: {"before": "b" * 64, "after": "d" * 64, "reason": "update runtime decoder"}},
        }
        self.assertEqual(surface.check(before, after, decisions), 1)
        for replacement in ({}, {"../../escape": decisions["files"][path]}):
            invalid = copy.deepcopy(decisions)
            invalid["files"] = replacement
            with self.assertRaisesRegex(ValueError, "review must cover exactly the changed paths"):
                surface.check(before, after, invalid)
        invalid = copy.deepcopy(decisions)
        invalid["files"][path]["after"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "stale review digest"):
            surface.check(before, after, invalid)
        invalid = copy.deepcopy(decisions)
        invalid["files"][path]["reason"] = "  "
        with self.assertRaisesRegex(ValueError, "missing review reason"):
            surface.check(before, after, invalid)
        invalid = copy.deepcopy(decisions)
        invalid["after_pins"]["jax"] = "e" * 40
        with self.assertRaisesRegex(ValueError, "review pins differ"):
            surface.check(before, after, invalid)

    def test_validate(self):
        for path in ("/etc/passwd", "jax/jaxlib/mosaic/gpu/../outside", "jax/README.md",
                     "jax/jaxlib/mosaic/gpu//file.cc", "jax\\jaxlib\\mosaic\\gpu\\file.cc"):
            document = self.snapshot()
            document["files"][path] = "b" * 64
            with self.assertRaisesRegex(ValueError, "invalid source path or digest"):
                surface.validate(document)
        document = self.snapshot()
        document["files"] = {}
        with self.assertRaisesRegex(ValueError, "snapshot omits required scope"):
            surface.validate(document)

    def test_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            path.write_text('{"reason":"first","reason":"second"}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key reason"):
                surface.load(path)

    def test_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "WORKSPACE"
            path.write_text(f'JAX_COMMIT = "{"a" * 40}"\nXLA_COMMIT = "{"b" * 40}"\n')
            self.assertEqual(surface.pins(path), {"jax": "a" * 40, "xla": "b" * 40})
            path.write_text('JAX_COMMIT = "branch"\n')
            with self.assertRaisesRegex(ValueError, "expected exactly one JAX_COMMIT pin"):
                surface.pins(path)

    def test_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = {owner: Path(directory) / owner for owner in surface.SCOPES}
            pins = {}
            for owner, root in roots.items():
                root.mkdir()
                surface.git(root, "init", "--quiet")
                for scope in surface.SCOPES[owner]:
                    folder = root / scope
                    folder.mkdir(parents=True)
                    (folder / "file.cc").write_text("original\n")
                surface.git(root, "add", ".")
                surface.git(root, "-c", "user.name=Surface Test", "-c", "user.email=test@example.invalid",
                            "commit", "--quiet", "-m", "fixture")
                pins[owner] = surface.git(root, "rev-parse", "HEAD").decode().strip()
            first = surface.snapshot(roots, pins)
            # Untracked and ignored files outside the explicitly named surfaces cannot affect the inventory.
            (roots["jax"] / "unrelated.txt").write_text("untracked\n")
            (roots["jax"] / ".git/info/exclude").write_text("ignored.txt\n")
            (roots["jax"] / "ignored.txt").write_text("ignored\n")
            self.assertEqual(surface.snapshot(roots, pins), first)
            with self.assertRaisesRegex(ValueError, "checkout does not match"):
                surface.snapshot(roots, {**pins, "jax": "a" * 40})
            path = roots["jax"] / surface.SCOPES["jax"][0] / "file.cc"
            path.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "GPU surface has uncommitted changes"):
                surface.snapshot(roots, pins)


if __name__ == "__main__":
    unittest.main()
