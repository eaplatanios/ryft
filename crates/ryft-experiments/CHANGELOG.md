# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Initial release with the Inkling 2 mixed precision training experiment.
- Checkpoints of the Inkling 2 training state through the training checkpointer of `ryft-checkpoints`
  (`Inkling2Trainer::save` and `Inkling2Trainer::load`), with leaves that move directly between the device and
  storage, and checkpointing and resuming options in the `inkling2_training` binary.
