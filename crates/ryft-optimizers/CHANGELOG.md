# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Added the `Optimizer` trait, whose states are `Parameterized` structures named by the paths of the parameters,
  together with `UpdateArguments` for the optional arguments of an update (e.g., the value of the objective) and
  `apply_updates`, `incremental_update`, and `periodic_update`.
- Added the gradient transformations of the `transformations` module, which cover the rescalings of the Adam family,
  of root mean squares, of factored statistics, and of signs, together with momentum, norm-based rescalings,
  clipping, constraints, Polyak step sizes, scaling, weight decay, and gradient noise.
- Added ready to use optimizers in the `optimizers` module (e.g., `Sgd`, `Adam`, `AdamW`, `RmsProp`, `Adafactor`,
  `Lamb`, `Lion`, and `Rprop`), with named states.
- Added the combinators `Chain`, `Masked`, and `Partition`, and the wrappers `ApplyIfFinite`, `ConditionallyTransform`,
  `ConditionallyMask`, `MultiSteps`, `ApplyEvery`, `Lookahead`, and `InjectLearningRate`.
- Added the backtracking and zoom line searches, L-BFGS preconditioning, and the `Lbfgs` optimizer.
- Added `Muon`, `Prodigy`, `ReduceOnPlateau`, and schedule-free optimization (`ScheduleFree`, `ScheduleFreeSgd`, and
  `ScheduleFreeAdamW`).
- Added `Dimuon` and `ScaleByDimuon`, an unofficial reconstruction of the LoRA optimizer of Thinking Machines (whose
  formulation is unpublished) after `anpaure/dimuon`.
- Added learning rate schedules that are evaluated from the step count with array operations, including joined,
  piecewise, cosine, exponential, polynomial, one-cycle, and warmup schedules.
