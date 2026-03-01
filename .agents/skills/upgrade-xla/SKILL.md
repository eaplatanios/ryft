---
name: upgrade-xla
description: Upgrades the dependency of `ryft` on XLA by switching to a newer / more recent OpenXLA commit,
  and propagating any necessary changes (e.g., in our C API bindings or Protobuf message types).
---

You must look for the latest commit hash in the [OpenXLA GitHub repository](https://github.com/openxla/xla).
You must make all necessary updates in `ryft` to switch us over to this new commit by following the steps described
in the `Contribution` section of the `crates/ryft-xla-sys/README.md` file.

Note that you must always start by creating a new branch called `u/eaplatanios/upgrade-xla-<commit>` and pushing a
change to `crates/ryft-xla-sys/WORKSPACE` updating the XLA commit hash and checksum. Then, you must submit a GitHub
workflow to build the `ryft-xla-sys` binary dependencies for that commit. You can check in on the status of that
workflow every 10 minutes though it can sometimes take a couple hours to complete. While that is happening, you can
start working on the other steps described in `crates/ryft-xla-sys/README.md`. You will eventually need to wait for
that workflow to complete before you can test your changes with the new binaries, but that should not block you from
making all necessary code changes that you will want to test later on.
