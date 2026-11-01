# Restoring Files in a Staging Worktree

This is the sole exception to the root prohibition on `git restore`. It is not authorization to stage, commit, push,
or perform an extraction. Use it only when an authorized, reviewed extraction plan already requires the procedure.

All of these conditions must hold:

1. The complete source tree has been committed and pushed to an immutable archive branch.
2. The destination is a clean, dedicated staging worktree used by the reviewed extraction plan. It is never the owner
   checkout. Verify that the staging worktree is clean before restoring anything.
3. The plan documents the increment and its explicit paths. Only then may
   `git restore --source=<immutable-archive-or-reviewed-integration-ref> -- <explicit-paths>` restore those paths.
4. Restore a whole path only when the reviewed plan assigns its complete delta to the increment. Use patch mode for
   paths shared with later work.
5. Never target the worktree root, a directory broader than the increment, a glob, or an unresolved variable.

This exception does not permit `git checkout <path>`, `git reset`, `git clean`, or `git stash`. It does not permit
staging or unstaging paths (e.g., `git add`, `git rm`, `git rm --cached`, or equivalents). All other root safeguards
continue to apply.
