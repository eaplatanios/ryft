---
name: upgrade-xla
description: Upgrades the dependency of `ryft` on XLA by switching to a newer / more recent OpenXLA commit,
  and propagating any necessary changes (e.g., in our C API bindings or Protobuf message types).
---

You must look for the latest commit hash in the [OpenXLA GitHub repository](https://github.com/openxla/xla).
You must make all necessary updates in `ryft` to switch us over to this new commit by following the steps described
in the `Contribution` section of the `crates/ryft-xla-sys/README.md` file.
