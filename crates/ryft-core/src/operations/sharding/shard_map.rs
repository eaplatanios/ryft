//! Contains the `shard_map` operation: [`ShardMapOperation`], which runs its attached local `body` [`Region`] once per
//! device of a manual SPMD computation, together with its logical boundary metadata [`ShardMap`], its logical error
//! type [`ShardMapError`], and its reference-discharge, interpretation, partial-evaluation, batching, forward-mode
//! differentiation, and transposition rules. This is the analogue of
//! [`jax.shard_map`](https://docs.jax.dev/en/latest/_autosummary/jax.shard_map.html) (refer to the
//! [JAX `shard_map` guide](https://docs.jax.dev/en/latest/notebooks/shard_map.html) for the programming model), and
//! backends lower it, for example, to Shardy's
//! [`sdy.manual_computation`](https://openxla.org/shardy/sdy_dialect#sdymanual_computation-sdymanualcomputationop).
//!
//! Closures construct shard maps through two families of entry points that trace an array-only local body over
//! [`ShardMapTracer`]s: [`shard_map`], [`shard_map_with_options`], and [`shard_map_in_context`] invoke a shard map on
//! input values and bind it (in the context of those values, or in an explicitly provided context whose body context
//! the body closure also receives, so that bodies without inputs can create values), while [`trace_shard_map`],
//! [`trace_shard_map_with_options`], and [`trace_shard_map_with_named_axes`] trace a body for global input types in an
//! explicitly named domain and return the unbound [`TracedShardMap`]. The `_with_options` variants and their
//! counterparts additionally select a subset of the manual mesh axes, and [`trace_shard_map_with_named_axes`] traces a
//! body for the named-axis scope of an enclosing context and passes its body context to the body closure. These
//! closure entry points accept array leaves only, so bodies with reference boundaries are traced explicitly and checked
//! through [`ShardMapOperation::from_program`]. Every boundary position is an array or a reference, so first-class
//! dimensions may appear only inside a body.
//!
//! The operation's inputs and outputs are global values whose placement over the mesh is described by one input or
//! output [`Sharding`] each. The body receives the shard of every global input that its input sharding assigns to the
//! executing device, and the global outputs are assembled from the local body outputs through their output shardings.
//! Inside the body, the active manual mesh axes are bound as named axes, so collectives (e.g., `parallel_reduce`) can
//! communicate along them, and manual variation records along which of those axes each local value may differ.
//! Manual variation is always tracked and checked: unlike JAX, whose `shard_map` can disable these checks with
//! `check_vma=False`, there is deliberately no unchecked mode, because the boundary validation and the transform rules
//! rely on the recorded variation for their correctness (e.g., transposition derives cross-device sums from the
//! adjoints of the variation operations of the body, and the boundary rejects outputs that vary along manual axes that
//! their output shardings do not tile).
//!
//! Batching over an anonymous axis preserves the boundary when every input is unbatched. Otherwise, the batch axis
//! becomes a dimension of every mapped input and output (of its referent, for a reference), placed on the mesh axes
//! that the batching level places the batch axis on (if any), and the local body is batched structurally, so that
//! collectives inside the body over the name of the batch axis are consumed by the batching level that binds it. A
//! batch axis placed on an active manual axis (the analogue of JAX's `spmd_axis_name`) makes that axis free in the
//! batched `shard_map`, which requires that the body does not use it.
//!
//! # Example
//!
//! Invoking `shard_map` on a traced value stages one `shard_map` instruction whose attached body operates on the local
//! shard that each device owns:
//!
//! ```rust
//! # use indoc::indoc;
//! # use pretty_assertions::assert_eq;
//! # use ryft_core::{
//! #     Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType, LogicalMesh, MeshAxis, MeshAxisType,
//! #     ProgramError, ShardMapTracer, Sharding, ShardingDimension, TracingContext, ValueProjection, shard_map,
//! # };
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! type IrContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
//! let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual)?])?;
//! let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])?;
//! let (_, program) = IrContext::trace(
//!     |input| {
//!         let input = ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from)?;
//!         let double = |x: ShardMapTracer<IrContext>| x.clone() + x;
//!         Ok(shard_map(double, input, mesh, sharding.clone(), sharding)?.into_value())
//!     },
//!     ArrayIrType::Array(ArrayType::new_static(DataType::F32, [8])),
//! )?;
//! assert_eq!(
//!     program.to_string(),
//!     indoc! {"
//!         lambda %0:f32[8] .
//!         let %1:f32[8][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
//!             mesh=['x'=2:manual],
//!             in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
//!             out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
//!             manual_axes=['x'],
//!             global_input_types=[f32[8][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
//!             global_output_types=[f32[8][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
//!         ] %0 [
//!             body={
//!                 lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
//!                 let %1:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = add %0 %0
//!                 in (%1)
//!             },
//!         ]
//!         in (%1)
//!     "}
//!     .trim_end(),
//! );
//! # Ok(())
//! # }
//! ```
//!
//! Each of the two devices along the manual axis `x` receives a `f32[4]` shard of the `f32[8]` input, which varies
//! along `x`, doubles it, and contributes it as its shard of the `f32[8]` output, which the output sharding tiles
//! along `x`.
//!
//! # References Under `shard_map`
//!
//! This section is the per-shard ownership contract for mutable references crossing a shard-map boundary. The
//! discharge, forward-mode, partial-evaluation, and reverse-mode rules of [`ShardMapOperation`] implement it, and they
//! check the body against it before doing any work. Construct reference-bearing operations through
//! [`ShardMapOperation::from_program`], which derives output types and reference forwarding from the body, and then
//! attach the checked local body when binding the operation through a tracing context or program builder. Global
//! referent shapes must be static and divisible by the manual partition counts.
//!
//!   - **Ownership follows the input specification.** A reference input `ref<T>` is a global reference whose referent
//!     `T` carries the input's sharding exactly like an array input. The local body receives a local reference
//!     `ref<T_local>` whose referent is the shard of `T` that the input sharding assigns to the executing device, and
//!     that device owns exactly that shard: a write through the local reference mutates only the owning shard, and no
//!     two devices own the same element along an active manual axis that the input sharding shards.
//!   - **Replicated writes are invariant.** Along an active manual axis that the input sharding does not shard, every
//!     device holds its own copy of the whole referent, whose local type is invariant along that axis. Reading it is
//!     allowed, also at a dynamic index that varies along the axis, in which case the value read varies along it as
//!     well. Mutating it is allowed exactly when every device performs the same mutation, so that the copies stay
//!     identical, and type checking enforces this through the variation of the referent: a write, swap, or update
//!     requires a value of the referent's (invariant) view type and dynamic indices that vary along no axis that the
//!     referent is invariant along, a `condition` whose predicate varies along the axis cannot mutate it in a branch
//!     (reference discharge rejects that), and a `while` loop whose predicate varies along the axis cannot carry it.
//!     The final state of such a reference is therefore invariant along the axis, which is what the derivation of its
//!     replicated final-state output requires when the map is discharged (as JAX checks the manual variation of the
//!     discharged final states).
//!   - **Ordering is per device.** Accesses through one local reference follow the body's program order, exactly as on
//!     a single device. Accesses on different devices touch disjoint owned shards or identical copies and have no
//!     defined mutual order. A reference is never an input of a collective; only a value read from it can cross
//!     devices.
//!   - **Captured references are rejected.** A reference must be an explicit shard-map input with an input sharding. A
//!     reference reaching the body as a captured constant has no sharding and therefore no owner.
//!   - **Outputs forward inputs only.** A reference-typed output must forward a reference input by identity, with an
//!     output sharding equal to that input's sharding. A reference allocated inside the body cannot escape as an output
//!     and is rejected. Transform paths belong to the accesses that use them, so a view is never a returnable value.
//!   - **Discharge threads owned shards.** Discharging a shard map with reference inputs turns each reference into a
//!     state carry sharded by its input sharding: the body's local state threads through the local program as local
//!     arrays, the hidden final-state output of a mutated reference carries the input sharding, and the stateful ABI
//!     commits each device's shard into the global referent. Allocations made inside the body follow the caller's
//!     discharge targets like the allocations of any other region, so an allocation that the caller does not select
//!     survives in the rebuilt body.
//!   - **Differentiation preserves local state and residuals.** Tangent references retain their primal input shardings.
//!     Forwarded inactive references keep their primal identity and have no tangent slot. Nonlinear derivatives pass
//!     ordinary residual values between the primal and tangent maps. A residual that is a primal input or output
//!     reaches the tangent map as that boundary value, under its own sharding, and a reference residual that denotes a
//!     reference input is that input, which it forwards by identity. A reference residual allocated inside the body
//!     cannot cross by identity, so it crosses as a snapshot of its final state, from which the tangent body allocates
//!     its own reference. The tangent body thus observes the state that a tangent program observes outside
//!     `shard_map`, since it runs after the whole primal program, and every invocation of it starts from that state.
//!     Other varying residuals are tiled along their varying manual axes (a varying scalar gains a leading dimension
//!     first), which preserves the distinct per-device values, while other replicated residuals keep their shape.
//!   - **Replicated gradients aggregate once.** Reverse mode uses fresh local accumulators for the replicated
//!     reference inputs that the body only reads and adds the resulting value into the caller's cotangent reference
//!     once, so existing destination contents are retained exactly once. Bodies track manual variation, so
//!     cross-device sums come from the adjoints of their variation operations (e.g., an invariant read passed through
//!     `parallel_vary` transposes to a sum over the manual axis), and the boundary performs no output-seed
//!     normalization or reduction of its own. Fully sharded destinations accumulate into each device's owned shard,
//!     and so do the destinations of the replicated reference inputs that the body mutates, whose cotangent references
//!     cross the transposed boundary by identity: the transposed body mutates them with invariant values at invariant
//!     indices only, so every device's copy stays identical, and a mutation that kills the incoming cotangent (e.g.,
//!     the transpose of a write, which zeroes the cotangent of the overwritten state) applies to the caller's
//!     destination itself, which a fresh accumulator could not express. A read of a replicated reference at a
//!     device-varying index transposes into a cross-device sum: each device scatters its cotangent into a local buffer
//!     that varies along the index's axes, the buffer is summed over those axes, and the invariant sum is added into
//!     the replicated cotangent reference (refer to [`ReferenceReadTransposition`](crate::ReferenceReadTransposition)).
//!   - **Rules may assume** that distinct reference inputs are distinct allocations (reference discharge rejects a
//!     repeated allocation, and runtime alias validation checks this at execution boundaries), that each device's
//!     lifecycle over its owned shards is independent of every other device's, and that the reference analysis of the
//!     local body accounts for every access, since the body is an ordinary local program over local references.

// TODO(eaplatanios): Review this module.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashSet};
use std::fmt::Display;
use std::sync::Arc;

use thiserror::Error;

use crate::arrays::sharding::meshes::render_mesh_axis_name;
use crate::arrays::{
    Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayReferenceTransform, ArrayType,
    Dimension, DimensionType, DimensionValue, LogicalMesh, MeshAxisType, Shape, Sharding, ShardingDimension,
    ShardingError,
};
use crate::axes::{NamedAxes, NamedAxis};
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain, ProjectedContext, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, CotangentDestinationKind, CotangentDestinations, DifferentiableOperation, DifferentiableType,
    DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError, DifferentiationPolicy,
    ResidualZeroProvider, TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::arithmetic::AddOperation;
use crate::operations::collectives::parallel_vary::{ParallelVary, ParallelVaryOperation};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::zero::{Zero, ZeroOperation};
use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
use crate::operations::dimensions::dimension_to_scalar::{DIMENSION_DATA_TYPE, DimensionToScalarOperation};
use crate::operations::manipulation::broadcasting::BroadcastOperation;
use crate::operations::manipulation::memory::TransferToMemoryOperation;
use crate::operations::manipulation::reshaping::ReshapeOperation;
use crate::operations::references::{
    ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation,
};
use crate::operations::sharding::reshard::ReshardOperation;
use crate::parameters::{Parameter, ParameterError, ParameterPath, Parameterized, ParameterizedFamily, Placeholder};
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation, ResidualInputSource,
};
use crate::programs::{
    CalleeRegionDriver, EffectClass, EffectClasses, FlatProgram, InputRegionProvenance, MaybeZero, Operation,
    OperationBoundaryPruning, OperationFormatter, OperationPayloadProjection, OperationProjection, OperationProvider,
    OutputRegionProvenance, Program, ProgramBuilder, ProgramError, ProjectedValue, ReferenceDischargeContext,
    ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeRegionBoundary,
    ReferenceDischargeRegionBoundaryInsertion, ReferenceDischargeRegionInput, ReferenceDischargeRegionOutput,
    ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceRoot, ReferenceType, Region, RegionArena,
    RegionDataFlow, RegionInterface, RegionLiveness, RegionRef, RegionSlot, Type, TypeError, Typed, Value,
    ValueProjection, discharge_reference_free_operation,
};
use crate::tracing::{DomainTracer, DomainTracingContext, Tracer, TracingContext};

mod emulation;

/// Logical error type for `shard_map` boundaries, covering mesh and specification validation, global and local
/// boundary derivation, manual variation, and the semantic restrictions on shard-map bodies. It carries no backend
/// state, so backend tracing and lowering errors can wrap it.
///
/// It converts into [`ProgramError`] for operation and transform rules: its [`Program`](ShardMapError::Program) variant
/// passes through unchanged, and every other variant becomes [`ProgramError::custom`], from which callers can recover
/// it with [`ProgramError::downcast_custom`]. The conversion from [`ProgramError`] is its inverse on custom errors: a
/// [`ProgramError::Custom`] that holds a [`ShardMapError`] converts back into that [`ShardMapError`], and every other
/// [`ProgramError`] becomes the [`Program`](ShardMapError::Program) variant. The two conversions therefore form a
/// normalizing cycle, so a shard-map error that crosses a [`ProgramError`] boundary (e.g., one raised while tracing the
/// body of a nested `shard_map`) is reported as itself rather than as a doubly wrapped
/// `ShardMapError::Program(ProgramError::Custom(..))`.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShardMapError {
    /// Underlying error returned by the mesh/sharding layer.
    #[error("{0}")]
    Sharding(#[from] ShardingError),

    /// Error returned when a mesh used for `shard_map` has no mesh axis of type `manual`.
    #[error("`shard_map` requires at least one mesh axis with type `manual`")]
    MeshHasNoManualAxes,

    /// Error returned when every manual axis of the mesh used for `shard_map` is already manual in an enclosing manual
    /// region and no manual axis is selected explicitly, so that no axis is left for the new manual region (refer to
    /// [`ShardMapError::AxisAlreadyManual`] for why an axis cannot be made manual twice).
    #[error(
        "every manual mesh axis is already manual in an enclosing manual region, so `shard_map` has no mesh axis left \
         to make manual"
    )]
    AllManualAxesAlreadyManual,

    /// Error returned when `shard_map` is asked to make a mesh axis manual that an enclosing manual region already
    /// made manual. The enclosing region already runs once per device coordinate along that axis, so making it
    /// manual again would drop the variation that values of the enclosing region carry along it.
    #[error(
        "mesh axis `{axis_name}` is already manual in an enclosing manual region, so `shard_map` cannot make it \
         manual again"
    )]
    AxisAlreadyManual { axis_name: String },

    /// Error returned when an input or output sharding names a mesh axis that an enclosing manual region already made
    /// manual. Inside that region, values are already the per-device shards along the axis, so a sharding over it has
    /// no meaning.
    #[error(
        "{value_kind} sharding #{value_index} names mesh axis `{axis_name}`, which is already manual in an \
         enclosing manual region"
    )]
    SpecificationNamesEnclosingManualAxis { value_kind: &'static str, value_index: usize, axis_name: String },

    /// Error returned when the mesh of a nested `shard_map` gives a mesh axis that an enclosing manual region already
    /// made manual a type other than [`Manual`](MeshAxisType::Manual) (e.g., [`Auto`](MeshAxisType::Auto)). Inside that
    /// region, values are already the per-device shards along the axis, so a mesh that describes the axis as global
    /// cannot place them. A nested map uses the mesh of the enclosing region instead, with the enclosing manual axes
    /// still typed [`Manual`](MeshAxisType::Manual), and selects its own manual axes among the remaining ones (as JAX
    /// requires the mesh of a nested `shard_map` to match its context mesh).
    #[error(
        "mesh axis `{axis_name}` is manual in an enclosing manual region, but the `shard_map` mesh does not type it \
         `manual`"
    )]
    EnclosingManualAxisNotManual { axis_name: String },

    /// Error returned when an enclosing manual region made a mesh axis manual whose name is an axis of the mesh of a
    /// nested `shard_map`, but over a device mesh that differs from the mesh of that `shard_map` (refer to the
    /// documentation of [`InputMeshMismatch`](ShardMapError::InputMeshMismatch) for when two meshes describe the same
    /// device mesh). Names alone cannot tell whether the axis of the nested mesh is the enclosing manual axis, along
    /// which the values of the enclosing region are already per-device shards, or another axis that happens to share
    /// its name, so the nested map is rejected (as JAX requires the mesh of a nested `shard_map` to match its context
    /// mesh). A nested map uses the mesh of the enclosing region instead.
    #[error(
        "mesh axis `{axis_name}` is manual in an enclosing manual region over mesh `{enclosing_mesh}`, which differs \
         from the `shard_map` mesh `{mesh}`"
    )]
    EnclosingManualAxisMeshMismatch { axis_name: String, enclosing_mesh: LogicalMesh, mesh: LogicalMesh },

    /// Error returned when an input of a `shard_map` already varies along one of its manual axes. Only values of a
    /// manual region that already made that axis manual vary along it, so such an input means that the `shard_map`
    /// would make an already manual axis manual again (refer to [`ShardMapError::AxisAlreadyManual`]).
    #[error(
        "input type #{input_index} already varies along manual axis `{axis_name}` of the `shard_map`, which an \
         enclosing manual region therefore already made manual"
    )]
    InputVariesAlongManualAxis { input_index: usize, axis_name: String },

    /// Error returned when an input of a `shard_map` carries a sharding over a mesh whose axes differ from those of the
    /// mesh of the `shard_map` in name, size, or order. The boundary places every input by its input sharding over its
    /// own mesh, and its transposition restores the caller's placement only over that mesh, so an input placed over
    /// another mesh has no counterpart at the boundary. The axis types may differ, because the mesh of a `shard_map`
    /// designates the axes that it makes manual through its axis types (e.g., a value placed along an `Explicit` axis
    /// `x` may enter a `shard_map` over the same mesh axes in which `x` is `Manual`).
    #[error(
        "input type #{input_index} is placed over mesh `{actual}`, whose axes differ from those of the `shard_map` \
         mesh `{expected}`"
    )]
    InputMeshMismatch { input_index: usize, expected: LogicalMesh, actual: LogicalMesh },

    /// Error returned when a partitioned dimension uses a free axis more major than a manual axis.
    #[error(
        "{value_kind} sharding #{value_index} dimension #{dimension} uses free axis `{free_axis_name}` \
         more major than manual axis `{manual_axis_name}`"
    )]
    ManualAxisMustPrecedeFreeAxis {
        value_kind: &'static str,
        value_index: usize,
        dimension: usize,
        free_axis_name: String,
        manual_axis_name: String,
    },

    /// Error returned when the rank of an input or output sharding does not match the rank of the shape that it
    /// places (i.e., the global shape of an input or the local shape of a body output).
    #[error(
        "{value_kind} sharding #{value_index} has rank {sharding_rank}, but the provided shape \
         has rank {shape_rank}"
    )]
    RankMismatch { value_kind: &'static str, value_index: usize, sharding_rank: usize, shape_rank: usize },

    /// Error returned when a manual axis would require padding in the local body shape.
    #[error(
        "{value_kind} sharding #{value_index} dimension #{dimension} has size {dimension_size}, \
         which is not divisible by manual partition count {manual_partition_count}"
    )]
    ManualAxisIntroducesPadding {
        value_kind: &'static str,
        value_index: usize,
        dimension: usize,
        dimension_size: usize,
        manual_partition_count: usize,
    },

    /// Error returned when the number of global input types does not match the number of input shardings.
    #[error("got {actual} global input type(s), but `shard_map` expects {expected}")]
    InputTypeCountMismatch { expected: usize, actual: usize },

    /// Error returned when the number of body output types or declared global output types does not match the number
    /// of outputs of the boundary.
    #[error("got {actual} output type(s), but `shard_map` expects {expected}")]
    OutputTypeCountMismatch { expected: usize, actual: usize },

    /// Error returned when the number of declared output forwardings does not match the number of global outputs.
    #[error("got {actual} output forwarding(s), but `shard_map` expects {expected}")]
    OutputForwardingCountMismatch { expected: usize, actual: usize },

    /// Error returned when a boundary type contains a dynamic dimension, which shard-map boundaries do not support.
    #[error("{value_kind} type #{value_index} dimension #{dimension} must be static at a `shard_map` boundary")]
    DynamicShapeNotSupported { value_kind: &'static str, value_index: usize, dimension: usize },

    /// Error returned when a transform would carry an intermediate array with a dynamic dimension across a shard-map
    /// boundary as a residual. Shard-map boundaries are static (refer to the documentation of
    /// [`ShardMapOperation`]), and the dynamic extent is defined inside the body, so it cannot be named outside it.
    #[error(
        "residual #{residual_index} dimension #{dimension} must be static to cross a `shard_map` boundary; \
         dynamically shaped intermediates cannot be shared between split `shard_map` bodies"
    )]
    DynamicResidualNotSupported { residual_index: usize, dimension: usize },

    /// Error returned when forward-mode differentiation would carry a reference residual between the primal and
    /// tangent maps that neither denotes a reference input of the body nor is allocated inside it. A reference cannot
    /// be carried by a residual edge: a reference residual that denotes a reference input crosses the boundary as that
    /// input, by identity, and one allocated inside the body crosses as a snapshot of its final state, from which the
    /// tangent body allocates its own reference. Any other reference has no owner in the body, so it can cross neither
    /// way. Every reference of a valid body (which captures no reference) is one of the two, so this error guards
    /// against linearizations whose residuals refer to state that the body does not own. No linearization built
    /// through checked construction reaches it: the only other root that the reference analysis of the primal program
    /// can report is an external constant, but program builders reject reference constants, and the analysis rejects
    /// the unbound captures of the explicitly empty capture scope of the primal program. It is kept as a defensive
    /// check for programs assembled without those checks.
    #[error(
        "residual #{residual_index} is a reference that is neither a `shard_map` body input nor allocated in the \
         body; only such references can be passed from the primal `shard_map` to the tangent `shard_map`"
    )]
    ReferenceResidualNotSupported { residual_index: usize },

    /// Error returned when a body output does not vary along a manual axis that its output sharding tiles.
    #[error(
        "`shard_map` body output #{output_index} must vary along tiled manual axis `{axis_name}`; \
         insert `parallel_vary` before returning the output"
    )]
    OutputNotVaryingAlongTiledManualAxis { output_index: usize, axis_name: String },

    /// Error returned when a body output still varies along an active manual axis that its output sharding does not
    /// tile, as under JAX's default `check_vma=True`.
    #[error(
        "output type #{output_index} still varies along manual axis `{axis_name}`, but its output sharding does not \
         mention it"
    )]
    OutputVaryingAlongUntiledManualAxis { output_index: usize, axis_name: String },

    /// Error returned when the unreduced or reduced axes of an input or output type do not match those of its sharding.
    #[error(
        "{value_kind} type #{value_index} has {state_kind} [{}], but `shard_map` expects [{}]",
        .actual.iter().map(|axis_name| format!("`{axis_name}`")).collect::<Vec<_>>().join(", "),
        .expected.iter().map(|axis_name| format!("`{axis_name}`")).collect::<Vec<_>>().join(", "),
    )]
    ShardingStateMismatch {
        value_kind: &'static str,
        value_index: usize,
        state_kind: &'static str,
        expected: Vec<String>,
        actual: Vec<String>,
    },

    /// Error returned when deriving a manual partition count, a global output shape, or the global extent of a
    /// residual edge overflows `usize`.
    #[error("overflow while {context}")]
    Overflow { context: String },

    /// Error returned when [`shard_map`] or [`shard_map_with_options`] is invoked with non-empty outputs but without
    /// input leaves, so that no input supplies the context to bind the operation in. [`shard_map_in_context`] binds
    /// such a `shard_map` in an explicitly provided context instead.
    #[error(
        "`shard_map` with non-empty outputs requires at least one input leaf; use `shard_map_in_context` to provide \
         the context explicitly"
    )]
    MissingTracedInvocationDomain,

    /// Error returned when a shard-map body has the `OrderedIo` effect, whose single order across devices independent
    /// per-device execution cannot provide. Bodies may use `DeviceOrderedIo`, which orders effects on each device.
    #[error("`shard_map` bodies require `DeviceOrderedIo` because `OrderedIo` demands one order across devices")]
    OrderedIoNotSupported,

    /// Error returned when reference discharge finds two reference inputs that denote the same allocation. Each
    /// reference input reaches the body as independently owned per-device shards, so the boundary cannot preserve
    /// aliasing between two inputs, and rules may assume that distinct reference inputs are distinct allocations.
    #[error(
        "`shard_map` reference inputs #{first_input_index} and #{second_input_index} denote the same allocation; \
         pass distinct allocations"
    )]
    RepeatedReferenceInputAllocation { first_input_index: usize, second_input_index: usize },

    /// Error returned when batching a `shard_map` over mapped inputs whose batch extent is not static. The batch axis
    /// becomes a dimension of the boundary types, which are static (refer to the `# Static Boundaries` section of the
    /// documentation of [`ShardMapOperation`]). The rejection is deliberate: padding the batch to a static bound would
    /// run the body on padding items, which is observable for bodies with effects (e.g., `print`) and for bodies that
    /// write through references. JAX rejects this case as well, because its `shard_map` batching rule computes batch
    /// sizes with integer arithmetic (refer to `_shard_map_batch` in `jax/_src/shard_map.py`).
    #[error(
        "batching a `shard_map` over mapped inputs requires a static batch extent, but it has type `{extent_type}`"
    )]
    DynamicBatchExtentNotSupported { extent_type: DimensionType },

    /// Error returned when batching a `shard_map` over mapped inputs places the batch axis on a mesh axis that an input
    /// or output sharding of the `shard_map` names (as JAX rejects a `spmd_axis_name` that its `in_specs` or
    /// `out_specs` mention). The batch dimension would then be partitioned along an axis that already partitions
    /// another dimension of a boundary value or that describes its reduction state.
    #[error(
        "batching a `shard_map` places the batch axis on mesh axis `{axis_name}`, which an input or output sharding \
         of the `shard_map` names"
    )]
    BatchAxisPlacedOnSpecifiedAxis { axis_name: String },

    /// Error returned when batching a `shard_map` over mapped inputs places the batch axis on an active manual axis of
    /// the `shard_map` that its body uses, i.e., along which a value of its body varies (e.g., the result of an
    /// `axis_index` or of a collective over that axis) or over which a collective or an `axis_index` of its body
    /// communicates (e.g., a permutation of invariant values, whose output need not be typed as varying). Placing the
    /// batch axis there makes the axis free in the batched `shard_map`, which preserves the semantics of the body only
    /// when the body computes the same values on every device along that axis and never communicates along it.
    #[error(
        "batching a `shard_map` places the batch axis on manual axis `{axis_name}`, which the `shard_map` body uses"
    )]
    BatchAxisPlacedOnUsedManualAxis { axis_name: String },

    /// Error returned when batching a `shard_map` over mapped inputs places the batch axis on every active manual axis
    /// of the `shard_map`. Placing the batch axis on a manual axis makes that axis free in the batched `shard_map`,
    /// which must keep at least one manual axis. Distributing the batch over every manual axis would instead require
    /// local batch dimensions that vary along those axes (JAX's `spmd_axis_name` semantics), which `shard_map` batching
    /// does not support.
    #[error(
        "batching a `shard_map` places the batch axis on every manual axis of the `shard_map`, so no manual axis would \
         remain"
    )]
    BatchAxisPlacedOnEveryManualAxis,

    /// Error returned when batching a `shard_map` over mapped inputs places the batch axis on an active manual axis of
    /// the `shard_map` whose body has the [`EffectClass::DeviceOrderedIo`] effect. Placing the batch axis there makes
    /// the axis free in the batched `shard_map`, which would change the devices that execute the per-device effects of
    /// the body.
    #[error(
        "batching a `shard_map` places the batch axis on manual axis `{axis_name}`, but the `shard_map` body has the \
         `DeviceOrderedIo` effect, which requires every manual axis to remain manual"
    )]
    BatchAxisPlacedOnManualAxisWithDeviceOrderedIo { axis_name: String },

    /// Error returned when transposition produces a cotangent for input `input_index` whose type differs from the
    /// expected cotangent type in more than the placement and memory kind that transposition reconciles (with
    /// `reshard`, a placement-only `broadcast`, and `transfer_to_memory`). This is a defensive invariant: boundary
    /// validation accepts only inputs whose element type, shape, layout, mesh axes, manual variation, and reduction
    /// state agree with the declared global input types, and the transposed boundary assembles the cotangents of
    /// exactly those declared types, so no well-formed boundary reaches this error.
    #[error(
        "`shard_map` transposition produced cotangent type `{actual}` for input #{input_index}, which cannot be \
         reconciled with the expected cotangent type `{expected}`"
    )]
    CotangentTypeMismatch { input_index: usize, expected: ArrayIrType, actual: ArrayIrType },

    /// Underlying program error returned while staging, validating, or transforming a shard-map body.
    #[error("{0}")]
    Program(ProgramError),

    /// Underlying parameter-structure error returned while reparameterizing traced values.
    #[error("{0}")]
    Parameter(#[from] ParameterError),
}

impl From<ProgramError> for ShardMapError {
    fn from(error: ProgramError) -> Self {
        match error.downcast_custom::<ShardMapError>() {
            Some(error) => error.clone(),
            None => ShardMapError::Program(error),
        }
    }
}

impl From<ShardMapError> for ProgramError {
    fn from(error: ShardMapError) -> Self {
        match error {
            ShardMapError::Program(error) => error,
            error => ProgramError::custom(error),
        }
    }
}

/// Logical boundary metadata of one manual SPMD computation over a mesh, as carried by [`ShardMapOperation`]: the
/// [`LogicalMesh`] of the computation, one validated [`Sharding`] per global input and per global output, and the
/// active manual mesh axes. Manual variation is always tracked, as under JAX's default `check_vma=True`.
///
/// The checked constructor [`ShardMap::new`] projects the provided input and output specifications into their
/// type-level form, so [`Auto`](MeshAxisType::Auto) mesh axes are hidden while the active manual axes drive the body.
/// Every mesh axis that is not an active manual axis (i.e., a free axis) stays global from the body's point of view,
/// and backends may let their compilers propagate placements over free axes across the manual region (e.g., by
/// lowering the dimension shardings of the boundary as open dimension shardings).
///
/// This type is the logical boundary model only and carries no backend state. Backends lower it, for example, to the
/// `in_shardings`, `out_shardings`, and `manual_axes` attributes of Shardy's
/// [`sdy.manual_computation`](https://openxla.org/shardy/sdy_dialect#sdymanual_computation-sdymanualcomputationop)
/// operation. Refer to the [JAX `shard_map` guide](https://docs.jax.dev/en/latest/notebooks/shard_map.html) for more
/// information on the underlying programming model.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShardMap {
    /// Logical mesh that the manual computation is defined over.
    mesh: LogicalMesh,

    /// Validated shardings for each global input leaf.
    in_shardings: Vec<Sharding>,

    /// Validated shardings for each global output leaf.
    out_shardings: Vec<Sharding>,

    /// Active manual mesh axes for this shard-map invocation.
    manual_axes: Vec<String>,
}

impl ShardMap {
    /// Creates a new [`ShardMap`] after validating and normalizing its input and output shardings (i.e., the analogues
    /// of the `in_specs` and `out_specs` of JAX's `shard_map`).
    ///
    /// The active manual axes are the mesh axes named by `manual_axes`, each of which must have type
    /// [`Manual`](MeshAxisType::Manual), or every manual mesh axis when `manual_axes` is empty. They are stored in mesh
    /// order. Every sharding must be defined over `mesh`, and its [`Auto`](MeshAxisType::Auto) axes are removed (refer
    /// to [`Sharding::without_auto_axes`]), as are its varying manual axes (refer to
    /// [`Sharding::varying_manual_axes`]): a boundary sharding describes a placement, while manual variation describes
    /// values and is derived from the boundary types instead. Within each sharded dimension of a sharding, every active
    /// manual axis must precede every free axis (i.e., every axis that is not an active manual axis, including auto
    /// axes). The local and global shape derivations do not depend on that order, but Shardy's
    /// [`sdy.manual_computation`](https://openxla.org/shardy/sdy_dialect#sdymanual_computation-sdymanualcomputationop)
    /// requires manual axes to precede free axes within each dimension sharding, so this constructor rejects other
    /// orders instead of accepting boundaries that a backend cannot represent.
    ///
    /// # Parameters
    ///
    ///   - `mesh`: Logical mesh that the manual computation is defined over.
    ///   - `in_shardings`: Per-input shardings for the global inputs.
    ///   - `out_shardings`: Per-output shardings for the global outputs.
    ///   - `manual_axes`: Active manual mesh axes for this shard map. An empty list means "all manual mesh axes".
    ///
    /// # Errors
    ///
    /// Returns [`ShardMapError::Sharding`] when `manual_axes` names an unknown or non-manual mesh axis or when a
    /// sharding is defined over a different mesh, [`ShardMapError::MeshHasNoManualAxes`] when `mesh` has no manual
    /// axis, and [`ShardMapError::ManualAxisMustPrecedeFreeAxis`] when a free axis precedes an active manual axis.
    pub fn new(
        mesh: LogicalMesh,
        in_shardings: Vec<Sharding>,
        out_shardings: Vec<Sharding>,
        manual_axes: Vec<String>,
    ) -> Result<Self, ShardMapError> {
        Self::new_within(mesh, in_shardings, out_shardings, manual_axes, &HashSet::new())
    }

    /// Creates a new [`ShardMap`] like [`Self::new`] for a manual region nested in enclosing manual regions that
    /// already made the mesh axes in `enclosing_manual_axes` manual. A mesh that does not type such an axis
    /// [`Manual`](MeshAxisType::Manual) is rejected with [`ShardMapError::EnclosingManualAxisNotManual`]. An empty
    /// `manual_axes` selects only the manual mesh axes that are not already manual (and fails with
    /// [`ShardMapError::AllManualAxesAlreadyManual`] when the mesh has manual axes but none remains), an explicitly
    /// requested axis that is already manual is rejected with [`ShardMapError::AxisAlreadyManual`], and a sharding that
    /// names an already manual axis is rejected with [`ShardMapError::SpecificationNamesEnclosingManualAxis`]. Making
    /// an axis manual twice would drop the variation that the values of the enclosing region carry along it, so a
    /// per-device value could leave the nested region typed as replicated, and backends could not represent the nested
    /// manual computation either.
    fn new_within(
        mesh: LogicalMesh,
        in_shardings: Vec<Sharding>,
        out_shardings: Vec<Sharding>,
        manual_axes: Vec<String>,
        enclosing_manual_axes: &HashSet<&str>,
    ) -> Result<Self, ShardMapError> {
        if let Some(axis) = mesh
            .axes()
            .iter()
            .find(|axis| enclosing_manual_axes.contains(axis.name()) && axis.r#type() != MeshAxisType::Manual)
        {
            return Err(ShardMapError::EnclosingManualAxisNotManual { axis_name: axis.name().to_string() });
        }
        let manual_axes = Self::normalize_manual_axes(&mesh, manual_axes, enclosing_manual_axes)?;
        let in_shardings = Self::build_shardings(&mesh, &manual_axes, enclosing_manual_axes, in_shardings, "input")?;
        let out_shardings = Self::build_shardings(&mesh, &manual_axes, enclosing_manual_axes, out_shardings, "output")?;
        Ok(Self { mesh, in_shardings, out_shardings, manual_axes })
    }

    /// Creates a new [`ShardMap`] directly from already validated shardings, skipping every check of [`Self::new`].
    /// Transform rules use it to rebuild a boundary that they derive from one that was already validated.
    pub(crate) fn from_shardings(
        mesh: LogicalMesh,
        in_shardings: Vec<Sharding>,
        out_shardings: Vec<Sharding>,
        manual_axes: Vec<String>,
    ) -> Self {
        Self { mesh, in_shardings, out_shardings, manual_axes }
    }

    /// Returns the logical mesh of this manual computation, which every input and output sharding is defined over.
    #[inline]
    pub fn mesh(&self) -> &LogicalMesh {
        &self.mesh
    }

    /// Returns the per-input shardings, one per global input, in input order. They are defined over [`Self::mesh`],
    /// carry no [`Auto`](MeshAxisType::Auto) axes and no varying manual axes, and list active manual axes before free
    /// axes within each dimension.
    #[inline]
    pub fn in_shardings(&self) -> &[Sharding] {
        self.in_shardings.as_slice()
    }

    /// Returns the per-output shardings, one per global output, in output order, with the same invariants as
    /// [`Self::in_shardings`].
    #[inline]
    pub fn out_shardings(&self) -> &[Sharding] {
        self.out_shardings.as_slice()
    }

    /// Returns the active manual mesh axes of this manual computation, in mesh order. The list is nonempty and names
    /// only mesh axes of type [`Manual`](MeshAxisType::Manual). The body runs once per device coordinate along these
    /// axes and may use them as named axes of collectives.
    #[inline]
    pub fn manual_axes(&self) -> &[String] {
        self.manual_axes.as_slice()
    }

    /// Returns the global boundary type of input `input_index` for the caller's global input type `input_type`: the
    /// data type, shape, and layout of `input_type` placed by the input sharding, together with the manual variation
    /// that `input_type` carries (e.g., the variation of a value of an enclosing manual region along its own manual
    /// axes). The memory kind of `input_type` is not part of the boundary type, so passing an input in another memory
    /// amounts to an implicit transfer into device memory, which transposition undoes. This normalizes the types of the
    /// values that a caller passes to a shard map, as well as the declared global input types that
    /// [`ShardMapOperation::from_program`] receives, into the global input types of the resulting
    /// [`ShardMapOperation`].
    ///
    /// # Parameters
    ///
    ///   - `input_index`: Index of the input sharding to use, which must be less than the number of input shardings.
    ///   - `input_type`: Caller's global type of that input.
    ///
    /// # Errors
    ///
    /// Returns [`ShardMapError::InputMeshMismatch`] when `input_type` carries a sharding over a mesh whose axes differ
    /// from those of [`Self::mesh`], [`ShardMapError::ShardingStateMismatch`] when `input_type` carries unreduced or
    /// reduced axes that differ from those of the input sharding, and the sharding errors of attaching the input
    /// sharding otherwise.
    ///
    /// # Panics
    ///
    /// Panics when `input_index` is not less than the number of input shardings.
    pub(crate) fn global_input_type(
        &self,
        input_index: usize,
        input_type: &ArrayType,
    ) -> Result<ArrayType, ShardMapError> {
        let sharding = &self.in_shardings[input_index];
        self.validate_input_mesh(input_type.sharding(), input_index)?;
        Self::validate_input_sharding_state(input_type.sharding(), sharding, input_index)?;
        let varying_axes =
            input_type.sharding().map(|sharding| sharding.varying_manual_axes().clone()).unwrap_or_default();
        Ok(ArrayType::new(input_type.data_type(), input_type.shape().clone())
            .with_layout(input_type.layout().cloned())
            .with_sharding(sharding.clone().with_varying_manual_axes(varying_axes)?)?)
    }

    /// Returns the local body type of input `input_index` for the provided global input type: the shard of the global
    /// array that the input sharding assigns to the executing device. Dimensions sharded over active manual axes shrink
    /// by their manual partition counts, those placements become manual variation along the corresponding axes, and the
    /// remaining placements and reduction state are preserved, including the variation along the manual axes of
    /// enclosing manual regions. The layout and memory kind of the global input type are preserved. For a reference
    /// input, this is the referent of the local reference that the body receives.
    ///
    /// # Parameters
    ///
    ///   - `input_index`: Index of the input sharding to use, which must be less than the number of input shardings.
    ///   - `global_input_type`: Global input type associated with that input, which must have a static shape.
    ///
    /// # Errors
    ///
    /// Returns [`ShardMapError::ShardingStateMismatch`] when `global_input_type` carries unreduced or reduced axes that
    /// differ from those of the input sharding (the local type takes its reduction state from the input sharding, so
    /// such a global type is not one that the input sharding places), [`ShardMapError::DynamicShapeNotSupported`] for
    /// a dynamic global shape, and the rank, divisibility, and sharding errors of the derivation otherwise.
    ///
    /// # Panics
    ///
    /// Panics when `input_index` is not less than the number of input shardings.
    pub fn local_input_type(
        &self,
        input_index: usize,
        global_input_type: &ArrayType,
    ) -> Result<ArrayType, ShardMapError> {
        Self::validate_input_sharding_state(
            global_input_type.sharding(),
            &self.in_shardings[input_index],
            input_index,
        )?;
        let global_shape = Self::static_dimensions(global_input_type, "input", input_index)?;
        let local_shape = self.local_input_shape(input_index, &global_shape)?;
        let boundary_sharding = &self.in_shardings[input_index];
        let local_sharding = boundary_sharding.local_sharding(self.manual_axes())?;
        let local_varying_axes = global_input_type
            .sharding()
            .map(|sharding| sharding.varying_manual_axes().clone())
            .unwrap_or_default()
            .union(&Self::sharded_manual_axes(boundary_sharding, &self.manual_axis_names()))
            .cloned()
            .collect::<BTreeSet<_>>();
        Ok(ArrayType::new(
            global_input_type.data_type(),
            Shape::new(local_shape.into_iter().map(Dimension::Static).collect()),
        )
        .with_layout(global_input_type.layout().cloned())
        .with_memory(global_input_type.memory())
        .with_sharding(local_sharding.with_varying_manual_axes(local_varying_axes)?)?)
    }

    /// Returns the active manual axes along which input `input_index` is replicated, in mesh order: the manual axes
    /// that its input sharding does not shard over. Every device along such an axis holds its own copy of the whole
    /// referent of a reference input, so a reference input replicated along some manual axis may only be mutated in
    /// ways that keep those copies identical (i.e., with values and indices that are invariant along the axis).
    ///
    /// # Panics
    ///
    /// Panics when `input_index` is not less than the number of input shardings.
    pub(crate) fn input_replicated_manual_axes(&self, input_index: usize) -> Vec<String> {
        let sharded_axes = Self::sharded_manual_axes(&self.in_shardings[input_index], &self.manual_axis_names());
        self.manual_axes.iter().filter(|axis| !sharded_axes.contains(axis.as_str())).cloned().collect()
    }

    /// Returns the active manual axes in mesh order: those named by `manual_axes`, or every manual mesh axis that is
    /// not in `enclosing_manual_axes` (i.e., that no enclosing manual region already made manual) when `manual_axes`
    /// is empty. Explicitly requesting an axis in `enclosing_manual_axes` is rejected.
    fn normalize_manual_axes(
        mesh: &LogicalMesh,
        manual_axes: Vec<String>,
        enclosing_manual_axes: &HashSet<&str>,
    ) -> Result<Vec<String>, ShardMapError> {
        let selected_manual_axes = if manual_axes.is_empty() {
            None
        } else {
            let mut selected_manual_axes = HashSet::new();
            for axis_name in manual_axes {
                if mesh.axis_index(axis_name.as_str()).is_none() {
                    return Err(ShardMapError::Sharding(ShardingError::UnknownMeshAxisName { name: axis_name }));
                }
                if mesh.axis_type(axis_name.as_str()) != Some(MeshAxisType::Manual) {
                    return Err(ShardMapError::Sharding(ShardingError::ExpectedManualMeshAxis { name: axis_name }));
                }
                if enclosing_manual_axes.contains(axis_name.as_str()) {
                    return Err(ShardMapError::AxisAlreadyManual { axis_name });
                }
                selected_manual_axes.insert(axis_name);
            }
            Some(selected_manual_axes)
        };
        let manual_axes = mesh
            .axes()
            .iter()
            .filter_map(|axis| {
                (axis.r#type() == MeshAxisType::Manual
                    && match &selected_manual_axes {
                        None => !enclosing_manual_axes.contains(axis.name()),
                        Some(selected_manual_axes) => selected_manual_axes.contains(axis.name()),
                    })
                .then(|| axis.name().to_string())
            })
            .collect::<Vec<_>>();
        if manual_axes.is_empty() {
            return Err(if mesh.axes().iter().any(|axis| axis.r#type() == MeshAxisType::Manual) {
                ShardMapError::AllManualAxesAlreadyManual
            } else {
                ShardMapError::MeshHasNoManualAxes
            });
        }
        Ok(manual_axes)
    }

    /// Validates the provided shardings against `mesh`, the active manual axes, and the axes that enclosing manual
    /// regions already made manual, and returns their type-level form, with [`Auto`](MeshAxisType::Auto) axes and
    /// varying manual axes removed.
    fn build_shardings(
        mesh: &LogicalMesh,
        manual_axes: &[String],
        enclosing_manual_axes: &HashSet<&str>,
        shardings: Vec<Sharding>,
        value_kind: &'static str,
    ) -> Result<Vec<Sharding>, ShardMapError> {
        let manual_axis_names = manual_axes.iter().map(String::as_str).collect::<HashSet<_>>();
        shardings
            .into_iter()
            .enumerate()
            .map(|(value_index, sharding)| {
                if sharding.mesh() != mesh {
                    return Err(ShardMapError::Sharding(ShardingError::MeshMismatch {
                        expected: mesh.clone(),
                        actual: sharding.mesh().clone(),
                    }));
                }
                let named_axes = sharding
                    .dimensions()
                    .iter()
                    .flat_map(|dimension| match dimension {
                        ShardingDimension::Sharded(axis_names) => axis_names.as_slice(),
                        ShardingDimension::Replicated | ShardingDimension::Unconstrained => &[],
                    })
                    .chain(sharding.unreduced_axes())
                    .chain(sharding.reduced_axes());
                for axis_name in named_axes {
                    if enclosing_manual_axes.contains(axis_name.as_str()) {
                        return Err(ShardMapError::SpecificationNamesEnclosingManualAxis {
                            value_kind,
                            value_index,
                            axis_name: axis_name.clone(),
                        });
                    }
                }
                Self::validate_manual_axis_order(&sharding, &manual_axis_names, value_kind, value_index)?;
                Ok(sharding.without_auto_axes().with_varying_manual_axes(Vec::<String>::new())?)
            })
            .collect()
    }

    /// Validates that every active manual axis precedes every free axis within each sharded dimension of `sharding`
    /// (refer to the documentation of [`ShardMap::new`] for the rationale).
    fn validate_manual_axis_order(
        sharding: &Sharding,
        manual_axes: &HashSet<&str>,
        value_kind: &'static str,
        value_index: usize,
    ) -> Result<(), ShardMapError> {
        for (dimension, sharding_dimension) in sharding.dimensions().iter().enumerate() {
            if let ShardingDimension::Sharded(axis_names) = sharding_dimension {
                let mut first_free_axis: Option<&str> = None;
                for axis_name in axis_names {
                    if manual_axes.contains(axis_name.as_str()) {
                        if let Some(free_axis_name) = first_free_axis {
                            return Err(ShardMapError::ManualAxisMustPrecedeFreeAxis {
                                value_kind,
                                value_index,
                                dimension,
                                free_axis_name: free_axis_name.to_string(),
                                manual_axis_name: axis_name.clone(),
                            });
                        }
                    } else if first_free_axis.is_none() {
                        first_free_axis = Some(axis_name.as_str());
                    }
                }
            }
        }
        Ok(())
    }

    /// Validates that a caller's input, whose sharding is `actual`, is placed over [`Self::mesh`] up to its axis types
    /// (refer to [`ShardMapError::InputMeshMismatch`]). An input without a sharding carries no placement and is
    /// accepted.
    fn validate_input_mesh(&self, actual: Option<&Sharding>, input_index: usize) -> Result<(), ShardMapError> {
        match actual {
            Some(actual) if !same_device_mesh(actual.mesh(), &self.mesh) => Err(ShardMapError::InputMeshMismatch {
                input_index,
                expected: self.mesh.clone(),
                actual: actual.mesh().clone(),
            }),
            _ => Ok(()),
        }
    }

    /// Validates that the reduction state of a caller's input agrees with its input sharding.
    fn validate_input_sharding_state(
        actual: Option<&Sharding>,
        expected: &Sharding,
        input_index: usize,
    ) -> Result<(), ShardMapError> {
        let Some(actual) = actual else {
            return Ok(());
        };
        if actual.unreduced_axes() != expected.unreduced_axes() {
            return Err(ShardMapError::ShardingStateMismatch {
                value_kind: "input",
                value_index: input_index,
                state_kind: "unreduced axes",
                expected: expected.unreduced_axes().iter().cloned().collect(),
                actual: actual.unreduced_axes().iter().cloned().collect(),
            });
        }
        if actual.reduced_axes() != expected.reduced_axes() {
            return Err(ShardMapError::ShardingStateMismatch {
                value_kind: "input",
                value_index: input_index,
                state_kind: "reduced axes",
                expected: expected.reduced_axes().iter().cloned().collect(),
                actual: actual.reduced_axes().iter().cloned().collect(),
            });
        }
        Ok(())
    }

    /// Returns the local body shape for input `input_index`: the shape seen inside the body for the corresponding
    /// global input shape, with each dimension divided by its manual partition count under the input sharding. Only
    /// active manual axes reduce the local shape; free axes remain global from the body's point of view.
    fn local_input_shape(&self, input_index: usize, global_shape: &[usize]) -> Result<Vec<usize>, ShardMapError> {
        let sharding = &self.in_shardings[input_index];
        let manual_axis_names = self.manual_axis_names();
        if sharding.rank() != global_shape.len() {
            return Err(ShardMapError::RankMismatch {
                value_kind: "input",
                value_index: input_index,
                sharding_rank: sharding.rank(),
                shape_rank: global_shape.len(),
            });
        }

        let mut local_shape = Vec::with_capacity(global_shape.len());
        for (dimension, (sharding_dimension, dimension_size)) in
            sharding.dimensions().iter().zip(global_shape.iter().copied()).enumerate()
        {
            let manual_partition_count = match sharding_dimension {
                ShardingDimension::Sharded(axis_names) => axis_names
                    .iter()
                    .filter(|axis_name| manual_axis_names.contains(axis_name.as_str()))
                    .try_fold(1usize, |partition_count, axis_name| -> Result<usize, ShardMapError> {
                        let axis_size = sharding
                            .mesh()
                            .axis_size(axis_name)
                            .ok_or_else(|| ShardingError::UnknownMeshAxisName { name: axis_name.clone() })?;
                        partition_count.checked_mul(axis_size).ok_or_else(|| ShardMapError::Overflow {
                            context: format!(
                                "computing the manual partition count of input #{input_index} dimension #{dimension}",
                            ),
                        })
                    })?,
                ShardingDimension::Replicated | ShardingDimension::Unconstrained => 1,
            };

            if dimension_size % manual_partition_count != 0 {
                return Err(ShardMapError::ManualAxisIntroducesPadding {
                    value_kind: "input",
                    value_index: input_index,
                    dimension,
                    dimension_size,
                    manual_partition_count,
                });
            }

            local_shape.push(dimension_size / manual_partition_count);
        }
        Ok(local_shape)
    }

    /// Returns the global output types for the provided local body output types, one per output sharding, through
    /// [`Self::global_output_type`].
    fn global_output_types(&self, local_output_types: &[ArrayType]) -> Result<Vec<ArrayType>, ShardMapError> {
        if local_output_types.len() != self.out_shardings.len() {
            return Err(ShardMapError::OutputTypeCountMismatch {
                expected: self.out_shardings.len(),
                actual: local_output_types.len(),
            });
        }
        local_output_types
            .iter()
            .enumerate()
            .map(|(output_index, local_output_type)| self.global_output_type(output_index, local_output_type))
            .collect()
    }

    /// Returns the global type of output `output_index` whose body produces `local_output_type`: the local shape scaled
    /// by the manual partition counts of the output sharding, under that sharding with the variation along active
    /// manual axes dropped. The body output must vary along every active manual axis that the output sharding tiles and
    /// along no other active manual axis, and its unreduced and reduced axes must equal those of the output sharding
    /// (as JAX's `_unshard_shaped_array` requires). An output sharding never adopts unreduced axes that the body output
    /// lacks: assembling an invariant body output as a pending sum along an axis would scale it by the size of that
    /// axis, and its cotangent could not be typed in reverse mode.
    fn global_output_type(
        &self,
        output_index: usize,
        local_output_type: &ArrayType,
    ) -> Result<ArrayType, ShardMapError> {
        let manual_axis_names = self.manual_axis_names();
        let local_shape = Self::static_dimensions(local_output_type, "output", output_index)?;
        let output_sharding = &self.out_shardings[output_index];
        let expected_current_varying_axes = Self::sharded_manual_axes(output_sharding, &manual_axis_names);
        let effective_local_varying_axes = local_output_type
            .sharding()
            .map(|sharding| sharding.varying_manual_axes().clone())
            .unwrap_or_default();
        if let Some(axis_name) = expected_current_varying_axes.difference(&effective_local_varying_axes).next() {
            return Err(ShardMapError::OutputNotVaryingAlongTiledManualAxis {
                output_index,
                axis_name: axis_name.clone(),
            });
        }
        let local_unreduced_axes =
            local_output_type.sharding().map(|sharding| sharding.unreduced_axes().clone()).unwrap_or_default();
        if local_unreduced_axes != *output_sharding.unreduced_axes() {
            return Err(ShardMapError::ShardingStateMismatch {
                value_kind: "output",
                value_index: output_index,
                state_kind: "unreduced axes",
                expected: output_sharding.unreduced_axes().iter().cloned().collect(),
                actual: local_unreduced_axes.into_iter().collect(),
            });
        }

        let local_reduced_axes =
            local_output_type.sharding().map(|sharding| sharding.reduced_axes().clone()).unwrap_or_default();
        if local_reduced_axes != *output_sharding.reduced_axes() {
            return Err(ShardMapError::ShardingStateMismatch {
                value_kind: "output",
                value_index: output_index,
                state_kind: "reduced axes",
                expected: output_sharding.reduced_axes().iter().cloned().collect(),
                actual: local_reduced_axes.into_iter().collect(),
            });
        }

        for axis_name in &effective_local_varying_axes {
            if manual_axis_names.contains(axis_name.as_str()) && !expected_current_varying_axes.contains(axis_name) {
                return Err(ShardMapError::OutputVaryingAlongUntiledManualAxis {
                    output_index,
                    axis_name: axis_name.clone(),
                });
            }
        }
        let surviving_varying_axes = effective_local_varying_axes
            .into_iter()
            .filter(|axis_name| !manual_axis_names.contains(axis_name.as_str()))
            .collect::<BTreeSet<_>>();
        let global_shape = self.global_output_shape(output_index, local_shape)?;
        Ok(ArrayType::new(
            local_output_type.data_type(),
            Shape::new(global_shape.into_iter().map(Dimension::Static).collect()),
        )
        .with_layout(local_output_type.layout().cloned())
        .with_sharding(output_sharding.clone().with_varying_manual_axes(surviving_varying_axes)?)?)
    }

    /// Returns the global shape of output `output_index` whose local shape is `local_shape`, scaling each dimension by
    /// its manual partition count under the output sharding.
    fn global_output_shape(&self, output_index: usize, local_shape: Vec<usize>) -> Result<Vec<usize>, ShardMapError> {
        let sharding = &self.out_shardings[output_index];
        let manual_axis_names = self.manual_axis_names();
        if sharding.rank() != local_shape.len() {
            return Err(ShardMapError::RankMismatch {
                value_kind: "output",
                value_index: output_index,
                sharding_rank: sharding.rank(),
                shape_rank: local_shape.len(),
            });
        }

        sharding
            .dimensions()
            .iter()
            .zip(local_shape)
            .enumerate()
            .map(|(dimension, (sharding_dimension, local_dimension_size))| {
                let manual_partition_count = match sharding_dimension {
                    ShardingDimension::Sharded(axis_names) => axis_names
                        .iter()
                        .filter(|axis_name| manual_axis_names.contains(axis_name.as_str()))
                        .try_fold(1usize, |partition_count, axis_name| -> Result<usize, ShardMapError> {
                            let axis_size = sharding
                                .mesh()
                                .axis_size(axis_name)
                                .ok_or_else(|| ShardingError::UnknownMeshAxisName { name: axis_name.clone() })?;
                            partition_count.checked_mul(axis_size).ok_or_else(|| ShardMapError::Overflow {
                                context: format!(
                                    "computing the manual partition count of output #{output_index} dimension \
                                     #{dimension}",
                                ),
                            })
                        })?,
                    ShardingDimension::Replicated | ShardingDimension::Unconstrained => 1,
                };

                local_dimension_size.checked_mul(manual_partition_count).ok_or_else(|| ShardMapError::Overflow {
                    context: format!("computing the global size of output #{output_index} dimension #{dimension}"),
                })
            })
            .collect()
    }

    /// Returns the names of the active manual axes as a set.
    fn manual_axis_names(&self) -> HashSet<&str> {
        self.manual_axes.iter().map(String::as_str).collect()
    }

    /// Returns the active manual axes (i.e., those in `manual_axis_names`) that `sharding` shards some dimension over,
    /// along which a value placed by that sharding varies inside the body.
    fn sharded_manual_axes(sharding: &Sharding, manual_axis_names: &HashSet<&str>) -> BTreeSet<String> {
        let mut sharded_axes = BTreeSet::new();
        for sharding_dimension in sharding.dimensions() {
            if let ShardingDimension::Sharded(axis_names) = sharding_dimension {
                for axis_name in axis_names {
                    if manual_axis_names.contains(axis_name.as_str()) {
                        sharded_axes.insert(axis_name.clone());
                    }
                }
            }
        }
        sharded_axes
    }

    /// Returns the static dimensions of a boundary type, rejecting a dynamic dimension.
    fn static_dimensions(
        array_type: &ArrayType,
        value_kind: &'static str,
        value_index: usize,
    ) -> Result<Vec<usize>, ShardMapError> {
        if let Some(shape) = array_type.static_shape() {
            return Ok(shape.dimensions().to_vec());
        }

        let dimension = array_type
            .shape()
            .dimensions()
            .iter()
            .position(|size| !matches!(size, Dimension::Static(_)))
            .unwrap();
        Err(ShardMapError::DynamicShapeNotSupported { value_kind, value_index, dimension })
    }

    /// Returns the placement of the batch dimension that batching inserts into the mapped boundary positions of this
    /// map, for a batch axis placed on the mesh axes `placement_axes`, together with the active manual axes of the
    /// batched map (refer to the batching rule of [`ShardMapOperation`] for the rationale). Every placement axis must
    /// be an axis of the mesh. [`Auto`](MeshAxisType::Auto) axes are dropped, as the boundary drops them from every
    /// sharding, while free axes (i.e., axes that are not active manual axes) are kept. Active manual axes are kept as
    /// well, but they are no longer active in the batched map, which is only correct when `body` (the local body of
    /// this map) neither varies along them nor communicates along them, when another active manual axis remains, and
    /// when `body` has no [`EffectClass::DeviceOrderedIo`] effect.
    ///
    /// # Errors
    ///
    /// Returns [`ShardMapError::Sharding`] for a placement axis that is not an axis of the mesh,
    /// [`ShardMapError::BatchAxisPlacedOnSpecifiedAxis`] for a placement axis that an input or output sharding names,
    /// [`ShardMapError::BatchAxisPlacedOnEveryManualAxis`] when the placement names every active manual axis,
    /// [`ShardMapError::BatchAxisPlacedOnUsedManualAxis`] for an active manual placement axis that some value of `body`
    /// (including the values of its nested regions) varies along or that a collective or an `axis_index` of `body`
    /// (again including its nested regions) names, and
    /// [`ShardMapError::BatchAxisPlacedOnManualAxisWithDeviceOrderedIo`] for an active manual placement axis when
    /// `body` has the [`EffectClass::DeviceOrderedIo`] effect.
    fn batched_placement<V: Value<Type = ArrayIrType>, O>(
        &self,
        placement_axes: &[String],
        body: RegionRef<'_, V, O>,
    ) -> Result<(ShardingDimension, Vec<String>), ShardMapError>
    where
        O: Operation<Type = ArrayIrType> + OperationPayloadProjection,
    {
        let mut boundary_axes = Vec::new();
        let mut released_axes = Vec::new();
        for axis_name in placement_axes {
            let axis_type = self
                .mesh
                .axis_type(axis_name)
                .ok_or_else(|| ShardingError::UnknownMeshAxisName { name: axis_name.clone() })?;
            if axis_type == MeshAxisType::Auto {
                continue;
            }
            let specified = self.in_shardings.iter().chain(&self.out_shardings).any(|sharding| {
                sharding.unreduced_axes().contains(axis_name)
                    || sharding.reduced_axes().contains(axis_name)
                    || sharding.dimensions().iter().any(|dimension| {
                        matches!(dimension, ShardingDimension::Sharded(axis_names) if axis_names.contains(axis_name))
                    })
            });
            if specified {
                return Err(ShardMapError::BatchAxisPlacedOnSpecifiedAxis { axis_name: axis_name.clone() });
            }
            if self.manual_axes.contains(axis_name) {
                released_axes.push(axis_name.clone());
            }
            boundary_axes.push(axis_name.clone());
        }
        let manual_axes = self
            .manual_axes
            .iter()
            .filter(|axis_name| !released_axes.contains(axis_name))
            .cloned()
            .collect::<Vec<_>>();
        if let Some(released_axis) = released_axes.first() {
            if manual_axes.is_empty() {
                return Err(ShardMapError::BatchAxisPlacedOnEveryManualAxis);
            }
            for region_id in body.region_ids_in_closure() {
                for atom in body.arena()[region_id.index()].atoms() {
                    let atom_type = atom.r#type();
                    let sharding = match atom_type.as_ref() {
                        ArrayIrType::Array(array_type) => array_type.sharding(),
                        ArrayIrType::Reference(reference_type) => reference_type.referent().sharding(),
                        ArrayIrType::Dimension(_) => None,
                    };
                    let varying_axes = sharding.map(Sharding::varying_manual_axes);
                    if let Some(axis_name) = released_axes
                        .iter()
                        .find(|axis_name| varying_axes.is_some_and(|axes| axes.contains(*axis_name)))
                    {
                        return Err(ShardMapError::BatchAxisPlacedOnUsedManualAxis { axis_name: axis_name.clone() });
                    }
                }
            }
            // A collective or an `axis_index` over a released axis communicates along it even when no value varies
            // along it (e.g., a permutation of invariant values in the meshless form, which types its output by its
            // input), so the variation check above cannot detect it.
            for (_, instruction) in body.instructions_in_closure() {
                if let Some(axis_name) = emulation::collective_axis_name(instruction.operation())
                    && released_axes.iter().any(|released_axis| released_axis == axis_name)
                {
                    return Err(ShardMapError::BatchAxisPlacedOnUsedManualAxis { axis_name: axis_name.to_string() });
                }
            }
            if body.effects().classes().contains(EffectClass::DeviceOrderedIo) {
                return Err(ShardMapError::BatchAxisPlacedOnManualAxisWithDeviceOrderedIo {
                    axis_name: released_axis.clone(),
                });
            }
        }
        let placement = if boundary_axes.is_empty() {
            ShardingDimension::Replicated
        } else {
            ShardingDimension::Sharded(boundary_axes)
        };
        Ok((placement, manual_axes))
    }
}

/// Canonical operation name for [`ShardMapOperation`]. Program statistics and their cross-language test cases match
/// this name, so it must not change.
pub const SHARD_MAP_OPERATION_NAME: &str = "shard_map";

/// [`Operation`] that runs its attached local `body` [`Region`] once per device coordinate along the active manual axes
/// of a [`ShardMap`]. The body is not part of this payload: it is the operation's one attached `body` region, which is
/// authoritative for the local boundary types, so this payload carries only the manual SPMD boundary metadata that the
/// body cannot represent (i.e., the [`ShardMap`], the declared global input and output types, and the output
/// forwarding). Refer to the [module documentation](mod@crate::operations::sharding::shard_map) for an overview of the
/// programming model.
///
/// A boundary position may be a reference under the contract in the ``# References Under `shard_map` `` section of the
/// module documentation. A reference input `ref<T>` carries the global referent `T` (sharded like an array input by
/// its input sharding) and reaches the body as `ref<T_local>`, the shard that the input sharding assigns to the
/// executing device. A reference output must forward a reference input by identity, which the operation records in its
/// output forwarding (derived by [`Self::from_program`] from the body's reference analysis) and reports through
/// [`Operation::reference_output_identity_input`]. An output whose forwarding is not declared is rejected by type
/// inference, so a reference output is never accepted on provenance the operation cannot name.
///
/// Batching over an anonymous axis preserves the boundary when all inputs are unbatched. Otherwise, the batch axis
/// becomes a dimension of every mapped input and output (of its referent, for a reference) at its batch axis, with the
/// static batch extent, and the body is batched structurally at the same batching level, as in JAX's `shard_map`
/// batching rule, so that the level that binds a named batch axis consumes the collectives over that name inside the
/// body. The batch dimension is unpartitioned for a replicated batch axis and is otherwise placed on the mesh axes that
/// the batching level places the batch axis on, which must not be named by any input or output sharding. Inside the
/// body, it keeps its global extent, because those axes are free in the batched map: an active manual axis among them
/// (the analogue of JAX's `spmd_axis_name`) stops being manual, which requires that some other manual axis remains,
/// that the body neither varies nor communicates along it, and that the body has no [`EffectClass::DeviceOrderedIo`]
/// effect (refer to the batching errors of [`ShardMapError`]). A forwarded reference output keeps the batch axis of the
/// reference input that it forwards, and an unbatched reference input is shared by every batch item, so writing a
/// batched value into it is rejected. A dynamic batch extent is rejected whenever a boundary position is mapped or the
/// body uses the extent, because boundary types are static (refer to the `# Static Boundaries` section below).
///
/// Interpretation over the [`Array`] values of the reference backend (i.e., in a domain whose values are
/// [`ArrayIrValue<Array>`]) emulates the devices in lockstep: every global input is split into the shard of each device
/// along the active manual axes, the body is replayed once for all devices, binding its ordinary operations once per
/// device and computing its collectives over the active manual axes across the devices, and the global outputs are
/// assembled from the local outputs through the output shardings. Region operations whose regions use such collectives
/// run in lockstep as well, which requires every device to take the same path through them. A reference input reaches
/// each device as a fresh local reference that holds its shard of the current referent, the final local shards are
/// written back into the caller's reference (summing the partial states of the devices along every active manual axis
/// along which the reference is unreduced, and otherwise taking the copy of the first device along every active manual
/// axis along which the reference is replicated, after checking that all copies agree), and a forwarded reference
/// output is the caller's reference input itself. Two reference inputs that denote the same allocation are rejected
/// with [`ShardMapError::RepeatedReferenceInputAllocation`], as in reference discharge. The emulation deliberately
/// supports only the lockstep region operations whose semantics it knows (i.e., `while`, `condition`, `scan` with a
/// static length, `custom_function`, `linear_call`, `rematerialize`, and nested `shard_map`s) and rejects every other
/// region operation whose regions use collectives over the emulated axes, a `scan` of dynamic length that does, and
/// devices that diverge on a predicate around such collectives, with [`ProgramError::UnsupportedOperation`]. These are
/// limitations of the reference emulation, not of `shard_map`. Backends with a device runtime execute the complete
/// boundary through their own contexts instead (e.g., by compiling the whole manual computation for their devices).
///
/// # Boundary Validation
///
/// Every attachment of a body validates it against the boundary through [`Operation::infer_output_types`], not only
/// checked construction through [`Self::from_program`]: each ordinary (non-reference) body input must be the local
/// shard that its input sharding derives from the declared global input type (refer to [`ShardMap::local_input_type`]),
/// and each ordinary body output must derive, through its output sharding, the declared global output type. Every input
/// must also agree with its declared global input type, and an input that carries a sharding must be placed over the
/// mesh of the [`ShardMap`] up to its axis types (otherwise, it is rejected with [`ShardMapError::InputMeshMismatch`]).
/// These checks compare shapes, element types, layouts, and manually varying axes exactly, the latter over every mesh
/// axis. Whenever the derivation carries a sharding, they also require the unreduced and reduced axes of the type to
/// equal those of the derivation, while tolerating other metadata that the type carries beyond the derivation (i.e.,
/// its dimension shardings and its memory kind). The boundary therefore implicitly places a caller's array input by its
/// input sharding and moves it into device memory, which transposition undoes for its cotangent, while a reference
/// input crosses by identity and is only viewed under its input sharding. Since the local body types follow the
/// declared types, the exact variation check is what makes the body inputs the shards of the actual inputs: an input
/// that varies along an axis of an enclosing manual region is accepted only by a boundary that declares that variation,
/// rather than reaching the body typed as invariant along that axis. The output derivation also requires each body
/// output to vary along exactly the active manual axes that its output sharding tiles and to have exactly the unreduced
/// and reduced axes of that sharding. An input that already varies along an active manual axis is rejected with
/// [`ShardMapError::InputVariesAlongManualAxis`]: a value varies along a mesh axis only inside a manual region over
/// that axis, so such an input means that this map makes an axis manual a second time, which would drop the input's
/// variation along it and type a per-device output as replicated. The check rejects no valid boundary, nested or not,
/// and it complements the rejection of already manual axes by the closure entry points (which also covers invariant
/// inputs). A body with the [`EffectClass::OrderedIo`] effect is rejected, because its single order across devices
/// conflicts with independent per-device execution, while [`EffectClass::DeviceOrderedIo`] is supported. Reference
/// positions are checked against the same local derivation of their referents, but whether the body forwards the
/// reference outputs it declares requires reference analysis of the body rather than its interface types, so checked
/// construction and every reference-aware transform rule (i.e., reference discharge, partial evaluation,
/// differentiation, and transposition) validate that property against the body's reference analysis instead.
///
/// A [`ShardMapError`] surfaces through the error type of the function that detects it. Type inference, and therefore
/// every attachment of a body (e.g., when a program builder or a tracing context binds the operation), reports it as
/// [`TypeError::Custom`], from which [`TypeError::downcast_custom`] recovers it, and builders and contexts wrap that
/// [`TypeError`] in [`ProgramError::Type`]. Checked construction through [`Self::from_program`] and the closure entry
/// points return the [`ShardMapError`] itself, while transform rules report it as [`ProgramError::Custom`], from which
/// [`ProgramError::downcast_custom`] recovers it (refer to the documentation of [`ShardMapError`] for the conversions
/// between the two).
///
/// # Static Boundaries
///
/// Every boundary type has a static shape. Checked construction rejects dynamic boundary types with
/// [`ShardMapError::DynamicShapeNotSupported`], and boundary validation rejects them on every attachment. Transform
/// rules that split a body into two maps keep the same invariant for the residuals that cross between them: a
/// first-class dimension residual crosses as its static integer scalar value, partial evaluation keeps the boundary
/// whole instead of splitting it across a dynamically shaped residual, and differentiation rejects such a residual with
/// [`ShardMapError::DynamicResidualNotSupported`]. A dynamically shaped residual would otherwise carry a dimension
/// identity defined inside one body across the boundary, where neither the enclosing program nor the other body can
/// name it. The boundary types therefore never mention dimension identities and need no identity renaming.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShardMapOperation {
    /// Manual SPMD metadata (mesh, boundary shardings, and manual axes) governing the attached body region.
    shard_map: ShardMap,

    /// Global input types declared at the shard-map boundary. A reference position carries the global referent.
    input_types: Vec<ArrayIrType>,

    /// Global output types declared at the shard-map boundary. A reference position carries the global referent.
    output_types: Vec<ArrayIrType>,

    /// Refer to the documentation of [`Self::output_forwarding`].
    output_forwarding: Vec<Option<usize>>,
}

impl ShardMapOperation {
    /// Creates a new [`ShardMapOperation`] for an already traced local `body`, including reference inputs and
    /// forwarded reference outputs. The body is attached when binding this operation. Global output types are derived
    /// from the body's output types through the output shardings, and reference forwarding is derived from the body's
    /// canonical reference analysis. Body outputs must explicitly vary along each manual axis that their output
    /// shardings tile, which may require `parallel_vary` for invariant outputs.
    ///
    /// # Parameters
    ///
    ///   - `body`: Local body whose input types describe the shards seen by each device.
    ///   - `global_input_types`: Global types corresponding positionally to the body's inputs. Each one is normalized
    ///     into its global boundary type through its input sharding, exactly like the types of the values passed to
    ///     [`shard_map_with_options`] (for a reference, its referent is), so the declared types carry the input
    ///     shardings and the manual variation of these types, and the operation accepts only inputs that vary along
    ///     exactly the same axes (refer to the `# Boundary Validation` section of the type documentation).
    ///   - `shard_map`: Checked boundary metadata (refer to [`ShardMap::new`]), whose input shardings place the global
    ///     inputs or reference referents and whose output shardings place the global outputs or forwarded reference
    ///     referents. Its output shardings must cover the active manual axes along which local outputs vary.
    ///
    /// # Errors
    ///
    /// Returns [`ShardMapError::InputTypeCountMismatch`] or [`ShardMapError::OutputTypeCountMismatch`] when the number
    /// of global input types or body outputs differs from the number of input or output shardings,
    /// [`ShardMapError::InputMeshMismatch`] when a global input type carries a sharding over a mesh whose axes differ
    /// from those of the mesh of `shard_map`, [`ShardMapError::ShardingStateMismatch`] when a global input type carries
    /// unreduced or reduced axes that differ from those of its input sharding, [`ShardMapError::Sharding`] when a
    /// global input type cannot be placed by its input sharding, the errors of [`ShardMap::local_input_type`] for the
    /// declared global input types (e.g., [`ShardMapError::DynamicShapeNotSupported`] or
    /// [`ShardMapError::ManualAxisIntroducesPadding`]), the global output derivation errors for the body outputs (i.e.,
    /// [`ShardMapError::OutputNotVaryingAlongTiledManualAxis`], [`ShardMapError::OutputVaryingAlongUntiledManualAxis`],
    /// [`ShardMapError::ShardingStateMismatch`], [`ShardMapError::RankMismatch`],
    /// [`ShardMapError::DynamicShapeNotSupported`], or [`ShardMapError::Overflow`]),
    /// [`ShardMapError::OrderedIoNotSupported`] for a body with the [`EffectClass::OrderedIo`] effect, and
    /// [`ShardMapError::Program`] when the body's arity or input types do not match the local shards of the declared
    /// global input types, or when the body violates the reference contract (e.g., a reference output that does not
    /// forward a reference input).
    pub fn from_program<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>>(
        body: &Program<V, O, Vec<V>, Vec<V>>,
        global_input_types: Vec<ArrayIrType>,
        shard_map: ShardMap,
    ) -> Result<Self, ShardMapError> {
        if global_input_types.len() != shard_map.in_shardings().len() {
            return Err(ShardMapError::InputTypeCountMismatch {
                expected: shard_map.in_shardings().len(),
                actual: global_input_types.len(),
            });
        }
        let global_input_types = global_input_types
            .iter()
            .enumerate()
            .map(|(index, r#type)| {
                let global =
                    shard_map.global_input_type(index, boundary_array_type(r#type).map_err(ProgramError::from)?)?;
                Ok(if r#type.is_reference() { ReferenceType::new(global).into() } else { global.into() })
            })
            .collect::<Result<Vec<ArrayIrType>, ShardMapError>>()?;
        let local_input_types = body.input_types();
        check_count!("input", local_input_types, global_input_types.len(), ProgramError);
        for (index, (global, local)) in global_input_types.iter().zip(&local_input_types).enumerate() {
            let expected =
                shard_map.local_input_type(index, boundary_array_type(global).map_err(ProgramError::from)?)?;
            if global.is_reference() != local.is_reference()
                || !shard_map_boundary_types_match(boundary_array_type(local).map_err(ProgramError::from)?, &expected)
            {
                return Err(ProgramError::InvalidArgument {
                    message: format!(
                        "`{SHARD_MAP_OPERATION_NAME}` body input #{index} has type `{local}`, but its global input \
                         requires local {} `{expected}`",
                        if global.is_reference() { "reference referent" } else { "array type" },
                    ),
                }
                .into());
            }
        }
        let local_output_types = body.output_types();
        let referents = local_output_types
            .iter()
            .map(|r#type| boundary_array_type(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        let global_outputs = shard_map.global_output_types(&referents)?;
        let output_types = local_output_types
            .iter()
            .zip(global_outputs)
            .map(|(local, global)| {
                if local.is_reference() {
                    ArrayIrType::Reference(ReferenceType::new(global))
                } else {
                    ArrayIrType::Array(global)
                }
            })
            .collect::<Vec<_>>();
        Self::validate_body_effects(body.effects().classes())?;
        let analysis = body.entry_region_ref().reference_analysis(0).map_err(ProgramError::from)?;
        let output_forwarding = analysis
            .output_roots()
            .iter()
            .map(|root| match root {
                Some(ReferenceRoot::RegionInput { region, input_index }) if *region == body.entry_region_ref().id() => {
                    Some(*input_index)
                }
                _ => None,
            })
            .collect();
        let operation = Self::from_boundary(shard_map, global_input_types.clone(), output_types)
            .with_output_forwarding(output_forwarding)?;
        // The reference contract is validated first, so that a reference output that does not forward a reference input
        // (e.g., one allocated inside the body) is reported with its precise cause rather than as an undeclared output
        // forwarding by type inference.
        operation.validate_reference_body(body.entry_region_ref())?;
        operation
            .infer_output_types(&global_input_types, &[body.entry_region_ref().interface()])
            .map_err(ProgramError::from)?;
        Ok(operation)
    }

    /// Creates a new [`ShardMapOperation`] directly from its boundary parts, skipping the validation of
    /// [`Self::from_program`] and declaring every output as a value output. Transform rules use it to rebuild a
    /// boundary that they derive from one that was already validated, and the attachment of the body that realizes the
    /// boundary still validates the result (refer to the `# Boundary Validation` section of the type documentation). A
    /// boundary with reference outputs declares them through [`Self::with_output_forwarding`].
    pub(crate) fn from_boundary<Inputs, Outputs>(
        shard_map: ShardMap,
        input_types: Inputs,
        output_types: Outputs,
    ) -> Self
    where
        Inputs: IntoIterator<Item: Into<ArrayIrType>>,
        Outputs: IntoIterator<Item: Into<ArrayIrType>>,
    {
        let input_types = input_types.into_iter().map(Into::into).collect::<Vec<_>>();
        let output_types = output_types.into_iter().map(Into::into).collect::<Vec<_>>();
        let output_forwarding = vec![None; output_types.len()];
        Self { shard_map, input_types, output_types, output_forwarding }
    }

    /// Returns a copy of this operation whose global output types are replaced by `global_output_types`, keeping the
    /// manual SPMD metadata, global input types, and output forwarding unchanged. Forward-mode differentiation uses
    /// this to align a tangent boundary's global output types with the tangent descriptors derived from the staged
    /// primal `shard_map`'s output types.
    pub(crate) fn with_global_output_types(
        mut self,
        global_output_types: Vec<ArrayIrType>,
    ) -> Result<Self, ShardMapError> {
        if global_output_types.len() != self.output_types.len() {
            return Err(ShardMapError::OutputTypeCountMismatch {
                expected: self.output_types.len(),
                actual: global_output_types.len(),
            });
        }
        self.output_types = global_output_types;
        Ok(self)
    }

    /// Returns a copy of this operation whose output forwarding is replaced by `output_forwarding` (refer to the
    /// documentation of [`output_forwarding`](Self::output_forwarding)), which must have one entry per global output.
    pub(crate) fn with_output_forwarding(
        mut self,
        output_forwarding: Vec<Option<usize>>,
    ) -> Result<Self, ShardMapError> {
        if output_forwarding.len() != self.output_types.len() {
            return Err(ShardMapError::OutputForwardingCountMismatch {
                expected: self.output_types.len(),
                actual: output_forwarding.len(),
            });
        }
        self.output_forwarding = output_forwarding;
        Ok(self)
    }

    /// Returns the manual SPMD metadata governing the attached body region (i.e., the mesh, the boundary shardings, and
    /// the active manual axes).
    #[inline]
    pub fn shard_map(&self) -> &ShardMap {
        &self.shard_map
    }

    /// Returns the global input types declared at the shard-map boundary, one per input. A reference position carries
    /// the global reference type, whose referent is placed by the input sharding.
    #[inline]
    pub fn global_input_types(&self) -> &[ArrayIrType] {
        &self.input_types
    }

    /// Returns the global output types declared at the shard-map boundary, one per output. They retain the placement
    /// and variation that the body and the output shardings establish, and a reference position carries the global
    /// reference type of the input that the output forwards.
    #[inline]
    pub fn global_output_types(&self) -> &[ArrayIrType] {
        &self.output_types
    }

    /// Returns, for each global output, the input whose reference the output forwards by identity, or [`None`] for a
    /// value output. Every reference output must name its forwarded input here: the body's reference outputs can only
    /// forward the body's reference inputs, and a reference allocated inside the body cannot escape. Transform paths
    /// belong to accesses and produce no reference outputs. The forwarded input's sharding must equal the output's
    /// sharding.
    #[inline]
    pub(crate) fn output_forwarding(&self) -> &[Option<usize>] {
        &self.output_forwarding
    }

    /// Validates the effects of a shard-map body: a body with the [`EffectClass::OrderedIo`] effect is rejected,
    /// because `OrderedIo` promises one order across all devices, which independent per-device execution of the body
    /// cannot provide. [`EffectClass::DeviceOrderedIo`] orders effects on each device separately and is supported, like
    /// every other effect class.
    fn validate_body_effects(effects: EffectClasses) -> Result<(), ShardMapError> {
        if effects.contains(EffectClass::OrderedIo) {
            return Err(ShardMapError::OrderedIoNotSupported);
        }
        Ok(())
    }

    /// Validates the attached local `body` against the reference contract of the ``# References Under `shard_map` ``
    /// section of the [module documentation](mod@crate::operations::sharding::shard_map), through the body's retained
    /// reference analysis. A body without references passes trivially.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::Reference`] holding the reference analysis error `InvalidReferenceCapture` (refer to
    /// [`ReferenceAnalysisError`](crate::programs::ReferenceAnalysisError)) when the body uses a captured reference (a
    /// reference must be an explicit input with an input sharding), and the other reference analysis errors of the
    /// body, and [`ProgramError::MalformedProgram`] when a reference output is rooted in an allocation made inside the
    /// body, when a reference output forwards an input that the operation does not declare, or when the forwarded
    /// input's sharding differs from the output's sharding. Mutations of reference inputs that are replicated along an
    /// active manual axis need no check here, because type inference already requires them to keep every copy
    /// identical (refer to the module documentation).
    fn validate_reference_body<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>>(
        &self,
        body: RegionRef<'_, V, O>,
    ) -> Result<(), ProgramError> {
        let name = self.name();
        let analysis = body.reference_analysis(0)?;
        for (index, root) in analysis.output_roots().iter().enumerate() {
            let forwarded = match root {
                None => continue,
                Some(ReferenceRoot::RegionInput { region, input_index }) if *region == body.id() => *input_index,
                Some(_) => {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{name}` output #{index} is a reference allocated inside the shard-map body, which cannot \
                         escape; a reference output must forward a reference input",
                    )));
                }
            };
            if self.output_forwarding().get(index).copied().flatten() != Some(forwarded) {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{name}` output #{index} forwards reference input #{forwarded} but the operation does not \
                     declare that forwarding",
                )));
            }
            if self.shard_map.out_shardings()[index] != self.shard_map.in_shardings()[forwarded] {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{name}` output #{index} forwards reference input #{forwarded} but its output sharding `{}` \
                     differs from the input sharding `{}`",
                    self.shard_map.out_shardings()[index],
                    self.shard_map.in_shardings()[forwarded],
                )));
            }
        }
        Ok(())
    }
}

impl Display for ShardMapOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ShardMapOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        SHARD_MAP_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("body")] }
    }

    // Every input must be placed over the mesh of this map up to its axis types (if at all) and agree with the declared
    // global boundary type at its position up to carried placement metadata (but with exactly the same manual
    // variation), an array input directly and a reference input through its referent. Every body input must be the
    // local shard that its input sharding derives from the declared global type (for a reference, of its referent), and
    // every ordinary body output must derive the declared global output type through its output sharding, which also
    // checks the variation required of outputs tiled along manual axes. A reference output must forward a declared
    // reference input by identity under an equal output sharding, and its declared type must describe that input, while
    // an array output must not declare a forwarding.
    // Whether the body forwards the reference outputs it declares needs reference analysis rather than interface types,
    // so `validate_reference_body` checks that property instead (refer to the `# Boundary Validation` section of the
    // type documentation).
    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        let name = self.name();
        let body_interface =
            shard_map_body_interface(region_interfaces, self.input_types.len(), self.output_types.len())?;
        Self::validate_body_effects(body_interface.effects()).map_err(TypeError::custom)?;
        check_count!("input sharding", self.shard_map.in_shardings(), self.input_types.len(), TypeError);
        check_count!("output sharding", self.shard_map.out_shardings(), self.output_types.len(), TypeError);
        check_count!("input", input_types, self.input_types.len(), TypeError);
        for (index, (actual, declared)) in input_types.iter().zip(&self.input_types).enumerate() {
            // An input placed over another mesh has no placement at this boundary, whose shardings and transposition
            // are defined over the mesh of this map only.
            let actual_sharding = boundary_array_type(actual)?.sharding();
            self.shard_map.validate_input_mesh(actual_sharding, index).map_err(TypeError::custom)?;
            // Only values of a manual region that already made an axis manual vary along it, so an input that varies
            // along one of this map's manual axes means that this map makes that axis manual a second time.
            if let Some(axis_name) = self
                .shard_map
                .manual_axes()
                .iter()
                .find(|axis| actual_sharding.is_some_and(|sharding| sharding.varying_manual_axes().contains(*axis)))
            {
                let error =
                    ShardMapError::InputVariesAlongManualAxis { input_index: index, axis_name: axis_name.clone() };
                return Err(TypeError::custom(error));
            }
            let body_input_type = &body_interface.input_types()[index];
            match declared {
                ArrayIrType::Reference(declared) => {
                    let actual = <&ReferenceType<ArrayType>>::try_from(actual)?;
                    if !shard_map_boundary_types_match(actual.referent(), declared.referent()) {
                        return Err(TypeError::invalid(format!(
                            "`{name}` input #{index} has type `{actual}` but its declared global input type is \
                             `{declared}`",
                        )));
                    }
                    let local_referent =
                        self.shard_map.local_input_type(index, declared.referent()).map_err(TypeError::custom)?;
                    let body_input_type = <&ReferenceType<ArrayType>>::try_from(body_input_type)?;
                    if !shard_map_boundary_types_match(body_input_type.referent(), &local_referent) {
                        return Err(TypeError::invalid(format!(
                            "`{name}` body input #{index} has type `{body_input_type}` but the local shard of \
                             reference input #{index} is `{}`",
                            ReferenceType::new(local_referent),
                        )));
                    }
                }
                declared => {
                    let actual = <&ArrayType>::try_from(actual)?;
                    let declared = <&ArrayType>::try_from(declared)?;
                    if !shard_map_boundary_types_match(actual, declared) {
                        return Err(TypeError::invalid(format!(
                            "`{name}` input #{index} has type `{actual}` but its declared global input type is \
                             `{declared}`",
                        )));
                    }
                    let local_type = self.shard_map.local_input_type(index, declared).map_err(TypeError::custom)?;
                    let body_input_type = <&ArrayType>::try_from(body_input_type)?;
                    if !shard_map_boundary_types_match(body_input_type, &local_type) {
                        return Err(TypeError::invalid(format!(
                            "`{name}` body input #{index} has type `{body_input_type}` but the local shard of input \
                             #{index} is `{local_type}`",
                        )));
                    }
                }
            }
        }
        let mut output_types = Vec::with_capacity(self.output_types.len());
        for (index, declared) in self.output_types.iter().enumerate() {
            let body_output_type = &body_interface.output_types()[index];
            match declared {
                ArrayIrType::Reference(declared) => {
                    <&ReferenceType<ArrayType>>::try_from(body_output_type)?;
                    let Some(forwarded) = self.output_forwarding[index] else {
                        return Err(TypeError::invalid(format!(
                            "`{name}` output #{index} is a reference whose forwarded input the operation does not \
                             declare",
                        )));
                    };
                    if forwarded >= input_types.len() || body_output_type != &body_interface.input_types()[forwarded] {
                        return Err(TypeError::invalid(format!(
                            "`{name}` output #{index} does not forward reference input #{forwarded} by identity",
                        )));
                    }
                    if self.shard_map.out_shardings()[index] != self.shard_map.in_shardings()[forwarded] {
                        return Err(TypeError::invalid(format!(
                            "`{name}` output #{index} forwards reference input #{forwarded} but its output sharding \
                             `{}` differs from the input sharding `{}`",
                            self.shard_map.out_shardings()[index],
                            self.shard_map.in_shardings()[forwarded],
                        )));
                    }
                    // The output is the forwarded input itself, so its declared type must describe that input.
                    let forwarded_type = <&ReferenceType<ArrayType>>::try_from(&input_types[forwarded])?;
                    if !shard_map_boundary_types_match(forwarded_type.referent(), declared.referent()) {
                        return Err(TypeError::invalid(format!(
                            "`{name}` output #{index} has declared type `{declared}`, but it forwards reference \
                             input #{forwarded} of type `{forwarded_type}`",
                        )));
                    }
                    output_types.push(input_types[forwarded].clone());
                }
                declared => {
                    // Only a reference output can forward an input by identity.
                    if let Some(forwarded) = self.output_forwarding[index] {
                        return Err(TypeError::invalid(format!(
                            "`{name}` output #{index} is an array but the operation declares that it forwards input \
                             #{forwarded}; only reference outputs forward inputs",
                        )));
                    }
                    let body_output_type = <&ArrayType>::try_from(body_output_type)?;
                    let declared = <&ArrayType>::try_from(declared)?;
                    let global_type =
                        self.shard_map.global_output_type(index, body_output_type).map_err(TypeError::custom)?;
                    if !shard_map_boundary_types_match(&global_type, declared) {
                        return Err(TypeError::invalid(format!(
                            "`{name}` body output #{index} has type `{body_output_type}`, whose global type \
                             `{global_type}` does not match the declared global output type `{declared}`",
                        )));
                    }
                    output_types.push(ArrayIrType::Array(declared.clone()));
                }
            }
        }
        Ok(output_types)
    }

    // The body's inputs mirror the instruction inputs one for one and its outputs are the operation's outputs, exactly
    // like a jitted call.
    #[inline]
    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        if region_index == 0 {
            InputRegionProvenance::Input { index: input_index }
        } else {
            InputRegionProvenance::None
        }
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![OutputRegionProvenance { region_index: 0, output_index }]
    }

    #[inline]
    fn region_data_flow(&self) -> RegionDataFlow<'_> {
        RegionDataFlow::Provenance
    }

    // Instruction inputs map onto the body inputs one for one, so the inputs that the body does not use (once its dead
    // outputs are dropped) are dropped together with their input shardings and declared global input types, and dead
    // outputs are dropped together with their output shardings, declared global output types, and forwarding entries,
    // which keeps the boundary metadata aligned with the pruned body. The region liveness keeps every input that the
    // body still needs for work that is retained when unused (e.g., an input read by an observable effect). A boundary
    // with a reference position keeps its whole boundary, because the reference contract (including the input positions
    // that the output forwarding names) was validated against it. A body whose ordinary outputs are all dead but which
    // has effects that remain observable is retained with no outputs, and its effects keep their per-device order.
    fn prune_boundary(
        &self,
        input_count: usize,
        used_outputs: &[bool],
        regions: &mut dyn RegionLiveness,
    ) -> Result<Option<OperationBoundaryPruning<Self>>, ProgramError> {
        if self.input_types.iter().chain(&self.output_types).any(Type::is_reference) {
            return Ok(None);
        }
        check_count!("output", used_outputs, self.output_types.len(), ProgramError);
        let kept_inputs = regions.used_region_inputs(0, used_outputs)?;
        check_count!("input", kept_inputs, input_count, ProgramError);
        check_count!("input", kept_inputs, self.input_types.len(), ProgramError);
        let input_indices = (0..self.input_types.len()).filter(|&index| kept_inputs[index]).collect::<Vec<_>>();
        let output_indices = (0..self.output_types.len()).filter(|&index| used_outputs[index]).collect::<Vec<_>>();
        let operation = Self {
            shard_map: ShardMap::from_shardings(
                self.shard_map.mesh().clone(),
                input_indices.iter().map(|&index| self.shard_map.in_shardings()[index].clone()).collect(),
                output_indices.iter().map(|&index| self.shard_map.out_shardings()[index].clone()).collect(),
                self.shard_map.manual_axes().to_vec(),
            ),
            input_types: input_indices.iter().map(|&index| self.input_types[index].clone()).collect(),
            output_types: output_indices.iter().map(|&index| self.output_types[index].clone()).collect(),
            output_forwarding: output_indices.iter().map(|&index| self.output_forwarding[index]).collect(),
        };
        Ok(Some(OperationBoundaryPruning { operation, kept_inputs, kept_outputs: used_outputs.to_vec() }))
    }

    // The body is traced through a fresh-root context that discards captures, so, like a jitted call and unlike nested
    // control flow (which inherits the namespace of its parent), it establishes a fresh capture namespace that contains
    // no captures.
    #[inline]
    fn region_capture_input_count(&self, region_index: usize) -> Option<usize> {
        (region_index == 0).then_some(0)
    }

    #[inline]
    fn reference_output_identity_input(&self, output_index: usize) -> Option<usize> {
        self.output_forwarding().get(output_index).copied().flatten()
    }

    // Every field a consumer can observe is rendered, because this rendering is the metadata fingerprint that the debug
    // transform-cache diagnostic compares programs by: a field this rendering drops is a field whose corruption that
    // diagnostic cannot see. All of `mesh`, `in_shardings`, `out_shardings`, and `manual_axes` steer differentiation,
    // transposition, and backend lowering while being invisible to the instruction's rendered atom types, and so do the
    // global boundary types: an input type only has to match the declared global input types up to its dimension
    // shardings. The declared output types retain the placement and variation established by the body and its output
    // shardings. The output forwarding is rendered exactly when some output forwards a reference input, so an all-value
    // boundary renders without it and a boundary with a forwarded reference output never renders like one without. No
    // state is elided. Every field is sequence- or scalar-valued, so the rendering is deterministic without any
    // ordering normalization. Every mesh axis name is rendered as a single-quoted literal with Rust's character escapes
    // (by the `Display` implementation of `MeshAxis` for the mesh axes and by `render_mesh_axis_name` for the manual
    // axes), so that no name can imitate the quoting or the `, ` separator that `OperationFormatter::list` joins items
    // with: the two-name list `["a", "b"]` and the one-name list `["a', 'b"]` render distinctly, and a collision could
    // not hide a genuinely different manual SPMD boundary from that diagnostic. `Display` renders the same text at
    // indentation zero.
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let manual_axes = self.shard_map.manual_axes().iter().map(|axis| render_mesh_axis_name(axis.as_str()));
        OperationFormatter::new(formatter, indentation, SHARD_MAP_OPERATION_NAME)?.bracketed(|operation| {
            // The mesh is redundant with any rendered sharding, which embeds it, but a boundary may have no
            // shardings at all, so it is rendered unconditionally.
            operation.list("mesh", self.shard_map.mesh().axes())?;
            operation.list("in_shardings", self.shard_map.in_shardings())?;
            operation.list("out_shardings", self.shard_map.out_shardings())?;
            operation.list("manual_axes", manual_axes)?;
            operation.list("global_input_types", &self.input_types)?;
            operation.list("global_output_types", &self.output_types)?;
            if self.output_forwarding.iter().any(Option::is_some) {
                let forwarding = self.output_forwarding.iter().map(|forwarded| match forwarded {
                    Some(input_index) => input_index.to_string(),
                    None => "_".to_string(),
                });
                operation.list("output_forwarding", forwarding)?;
            }
            Ok(())
        })
    }
}

// A shard map with reference inputs discharges its *local* body through the driver's isolated region rebuild. Each
// reference input enters the rebuilt body as a boundary view (refer to `ReferenceDischargeRegionInput::View`): the body
// receives `ref<T_local>` while the caller allocation holds `ref<T_global>`, so the view is fresh local state typed by
// the body input and initialized from the local shard, and every mutated view publishes its final local state through
// `ReferenceDischargeRegionOutput::View`. The rebuilt boundary keeps every input position (a reference position becomes
// the global referent, an array sharded by the same input sharding), drops the forwarded reference outputs (the caller
// already holds those handles), and publishes each mutated input's final state as a trailing output whose output
// sharding is that input's input sharding, so the stateful ABI commits each device's shard into the global referent.
// Allocations made inside the body are discharged under the caller's discharge targets, like the allocations of any
// other rebuilt region, so an unselected one survives in the rebuilt body.
//
// Two reference inputs naming one allocation would become two independent views, silently losing their aliasing, so
// the rule rejects them, enforcing the contract that distinct reference inputs are distinct allocations. It also
// refuses a preserved reference, because a manual region threads state, not destination references, and it summarizes
// the body first so that the shared access-policy and consumption checks run exactly as for other structured rules. A
// body without references replays verbatim.
impl<C, P> ReferenceDischargeableOperation<C, P> for ShardMapOperation
where
    C: Context<Type = ArrayIrType, Operation: From<ShardMapOperation>>,
    P: ReferenceDischargePolicy<C, Referent = ArrayType>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        let name = self.name();
        check_count!("input", inputs, self.input_types.len(), ProgramError);
        let body = driver.region(0)?;
        check_count!("input", body.input_ids(), inputs.len(), ProgramError);
        check_count!("output", body.output_ids(), self.output_types.len(), ProgramError);
        let allocations =
            inputs.iter().map(|input| context.boundary_allocation(input)).collect::<Result<Vec<_>, _>>()?;
        if allocations.iter().all(Option::is_none) && !body.contains_references_in_closure() {
            return discharge_reference_free_operation(self, context, driver, inputs);
        }
        for (second_input_index, allocation) in allocations.iter().enumerate() {
            if let Some(allocation) = allocation
                && let Some(first_input_index) =
                    allocations[..second_input_index].iter().position(|candidate| candidate == &Some(*allocation))
            {
                let error = ShardMapError::RepeatedReferenceInputAllocation { first_input_index, second_input_index };
                return Err(error.into());
            }
        }
        self.validate_reference_body(body)?;
        let summary = context.region_summary(self, 0, body, allocations.as_slice())?;
        for (index, allocation) in allocations.iter().enumerate() {
            if let Some(allocation) = allocation
                && !context.is_allocation_discharged(*allocation)?
            {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{name}` does not thread a preserved reference through its body, but input #{index} \
                         denotes preserved {allocation}; discharge it or pass a value",
                    ),
                });
            }
        }

        // The forwarded reference outputs are dropped from the body before it is rebuilt: a public output that is a
        // discharged reference has no state to publish, and the caller keeps the handle it passed. The body region is
        // normalized in place, keeping its atoms and instructions, because discharge targets name source region, atom,
        // and instruction identities that a copied program would renumber.
        let value_output_indices = (0..self.output_types.len())
            .filter(|&index| self.output_forwarding[index].is_none())
            .collect::<Vec<_>>();
        let normalized_arena = (value_output_indices.len() != self.output_types.len())
            .then(|| {
                let region = body.region();
                let output_ids = value_output_indices.iter().map(|&index| region.output_ids()[index]).collect();
                let normalized = Region::new(
                    region.atoms().to_vec(),
                    region.input_ids().to_vec(),
                    output_ids,
                    region.instructions().to_vec(),
                );
                let mut regions = body.arena().iter().take(body.id().index()).cloned().collect::<Vec<_>>();
                regions.push(normalized);
                RegionArena::from_regions(regions)
            })
            .transpose()?;
        let normalized_body = match &normalized_arena {
            Some(arena) => RegionRef::new(arena, body.id())?,
            None => body,
        };

        // Every reference input enters as a boundary view, and the final local state of every input that the body
        // mutates follows the value outputs, in input order.
        let mutated_inputs = allocations
            .iter()
            .enumerate()
            .filter_map(|(index, allocation)| {
                allocation.filter(|allocation| summary.is_mutated(*allocation)).map(|_| index)
            })
            .collect::<Vec<_>>();
        let boundary = ReferenceDischargeRegionBoundary::new(
            self,
            0,
            allocations.iter().map(|allocation| match allocation {
                Some(allocation) => ReferenceDischargeRegionInput::View(*allocation),
                None => ReferenceDischargeRegionInput::Value,
            }),
            ReferenceDischargeRegionBoundaryInsertion::new(Vec::new(), inputs.len()),
            [ReferenceDischargeRegionBoundaryInsertion::new(
                mutated_inputs.iter().map(|&index| ReferenceDischargeRegionOutput::View(index)).collect(),
                value_output_indices.len(),
            )],
        );
        let program = driver.rebuild_region(context, normalized_body, &boundary)?.into_program();

        // Rebuild the boundary: every input keeps its position and input sharding, the value outputs keep theirs, and
        // one final-state output per mutated reference input follows them under that input's input sharding. The
        // final state is typed by the input it replaces (the allocation's current state) rather than by the declared
        // global referent, because the declared boundary type may omit sharding metadata that the allocation carries
        // and the discharged state must match the allocation's referent type exactly.
        let boundary_inputs = inputs
            .iter()
            .zip(&allocations)
            .map(|(input, allocation)| match allocation {
                Some(allocation) => context
                    .boundary_value(&ReferenceDischargeValue::Reference(context.allocation_reference(*allocation)?)),
                None => context.boundary_value(input),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input_types = self
            .input_types
            .iter()
            .map(|r#type| boundary_array_type(r#type).cloned().map(ArrayIrType::Array))
            .collect::<Result<Vec<_>, _>>()?;
        let mut output_types =
            value_output_indices.iter().map(|&index| self.output_types[index].clone()).collect::<Vec<_>>();
        let mut out_shardings = value_output_indices
            .iter()
            .map(|&index| self.shard_map.out_shardings()[index].clone())
            .collect::<Vec<_>>();
        for &index in &mutated_inputs {
            output_types.push(boundary_inputs[index].r#type().into_owned());
            out_shardings.push(self.shard_map.in_shardings()[index].clone());
        }
        let shard_map = ShardMap::from_shardings(
            self.shard_map.mesh().clone(),
            self.shard_map.in_shardings().to_vec(),
            out_shardings,
            self.shard_map.manual_axes().to_vec(),
        );
        let operation = ShardMapOperation::from_boundary(shard_map, input_types, output_types);
        let mut outputs = context.parent().bind(operation, vec![program], &boundary_inputs)?;
        check_count!("output", outputs, value_output_indices.len() + mutated_inputs.len(), ProgramError);
        let final_states = outputs.split_off(value_output_indices.len());
        for (index, final_state) in mutated_inputs.iter().zip(final_states) {
            let allocation = allocations[*index].unwrap();
            context.set_discharged_state(allocation, final_state, true)?;
        }

        // A forwarded reference output is reported as the handle the caller already holds at the forwarded position.
        let mut outputs = outputs.into_iter();
        Ok(self
            .output_forwarding
            .iter()
            .map(|forwarded| match forwarded {
                Some(input_index) => inputs[*input_index].clone(),
                None => ReferenceDischargeValue::Value(outputs.next().unwrap()),
            })
            .collect())
    }
}

// Interpretation of a `shard_map` over the reference backend's `Array` values emulates its devices in lockstep (refer
// to the documentation of the `emulation` module): the body is replayed once for all devices, ordinary operations are
// bound once per device through the driver's `bind` request with exactly that device's local values, and collectives
// over the active manual axes are computed across the devices. Interpreting the local body once over global values
// would ignore the per-device partitioning, so the body is never replayed through `interpret_region`. The rule needs
// only payload projection from the operation family, to recognize collectives and region operations inside the body,
// and it serves every domain over `ArrayIrValue<Array>` (e.g., the eager context of `ArrayIrOperation<Array>` and of
// kernel operation families). Backends with a device runtime execute the complete boundary through their own
// `Context::bind` instead (e.g., by compiling the whole manual computation for their devices).
impl<C> InterpretableOperation<C> for ShardMapOperation
where
    C: Domain<
            Type = ArrayIrType,
            Value = ArrayIrValue<Array>,
            Constant = ArrayIrValue<Array>,
            Operation: OperationPayloadProjection,
        >,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        emulation::interpret_shard_map(self, context, driver, inputs)
    }
}

// Online partial-evaluation rule for a staged `shard_map`, the map-boundary sibling of the jitted-call rule: it splits
// the local body against the caller's known-ness while preserving the `shard_map` boundary, its mesh, and its
// shardings on both sides.
//
// The split fires whenever the known-ness of the inputs is mixed (i.e., some inputs are known and some are not).
// All-known and all-unknown calls defer to the default fold-or-residualize behavior, which preserves the original
// boundary exactly. Whether the known inputs are concrete values of an eager known-side context or tracers into a live
// outer trace does not matter, as in JAX's `_shard_map_partial_eval`, which always binds its known `shard_map`: the
// known half is bound into the known-side context through the default fold-or-residualize policy, which executes it
// under an eager context (e.g., through the reference backend's emulation or a device runtime) and stages it into the
// outer program under a staging context. An eager known-side context therefore needs to execute the known `shard_map`,
// which is the same requirement that binding an all-known call through it imposes. The jitted-call rule follows the
// same policy, so that every region-boundary operation feeds the shared `PartitionedProgram` protocol under the same
// known-ness contract.
//
// When the split fires, an output that merely forwards a known input (i.e., whose body output is that body input) under
// an output sharding equal to the input sharding is that global input itself, so it is returned as the known input
// whenever the input already has the declared output type, without any known-side work, and the remaining outputs are
// split as follows. This shortcut is a Ryft extension: JAX's `_shard_map_partial_eval` returns every known output from
// its known `shard_map`. The *local* body program is split through the shared `PartitionedProgram` machinery, and the
// original boundary is kept whole only when the known half has neither instructions nor known outputs; known outputs
// that need no known instruction (e.g., forwarded inputs placed by another sharding) still get a known-side
// `shard_map`, so they stay known, as JAX always binds its known half. Residual edges that would merely repeat a known
// value are forwarded (refer to `PartitionedProgram::forward_residuals`): an edge that the known half forwards from one
// of its inputs is fed to the residual side directly from the corresponding known boundary input (JAX's `in_fwd`),
// under that input's global type and input sharding, and an edge that is a known output is fed from that known
// boundary output (JAX's `out_fwd`), under that output's global type and with its output sharding as the input
// sharding. Both deliver exactly the local value that the edge carries, so the known half then neither returns the
// value a second time nor, when nothing else uses a forwarded input, receives that input. The known side is rewrapped
// as a `shard_map` whose global outputs are the fully known boundary outputs followed by the remaining known-to-unknown
// residual edges (each edge's global type and sharding derived through `residual_boundary`), bound into the enclosing
// known-side context over the known boundary inputs that it uses. The residual side is rewrapped as a `shard_map` over
// the surviving unknown boundary inputs, the forwarded known boundary inputs and outputs, and those residual edges, and
// emitted into the residual program.
impl<C> PartiallyEvaluatableOperation<C> for ShardMapOperation
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<ShardMapOperation>
                           + OperationProjection<
                ArrayType,
                Projected: From<ReshapeOperation> + From<BroadcastOperation> + From<ParallelVaryOperation>,
            > + From<DimensionToScalarOperation>
                           + From<DimensionFromScalarOperation>
                           + OperationPayloadProjection,
        >,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Split only a boundary with mixed known-ness; all-known and all-unknown calls keep the default
        // fold-or-residualize behavior and therefore the original boundary.
        if inputs.iter().all(PartialEvaluationValue::is_known) || inputs.iter().all(PartialEvaluationValue::is_unknown)
        {
            return context.fold_or_residualize(
                self.clone(),
                driver.regions().map(|region| region.to_program()).collect(),
                inputs,
            );
        }
        // Split the local body through the shared online boundary machinery. The body's inputs are index-aligned
        // with the boundary inputs and carry the *local* types, so both split sides stay local programs.
        let body_program = driver.region(0)?;
        // A reference input, a reference output, or a reference access anywhere in the body keeps the shard map
        // whole, for the same reason a `jit_call` stays whole: splitting would spread the accesses to one root across
        // two manual regions whose known side runs first, and the default rule preserves effect order by placing the
        // whole application on one side instead. The contract is validated first so a malformed body is reported as
        // such rather than residualized silently.
        if body_has_references(body_program) {
            self.validate_reference_body(body_program)?;
            return context.fold_or_residualize(self.clone(), vec![body_program.to_program()], inputs);
        }
        // Outputs that merely forward a known input are that input itself (a Ryft extension, as explained above): when
        // the output sharding equals the input sharding and the input already has the declared output type, the global
        // output is the global input, so it stays known without any known-side `shard_map`. The remaining outputs are
        // split below.
        let forwarded_inputs = (0..self.output_types.len())
            .map(|output_index| {
                let output = body_program.output_ids()[output_index];
                let input_index = body_program.input_ids().iter().position(|input| *input == output)?;
                (inputs[input_index].is_known()
                    && self.shard_map.out_shardings()[output_index] == self.shard_map.in_shardings()[input_index]
                    && *inputs[input_index].r#type() == self.output_types[output_index])
                    .then_some(input_index)
            })
            .collect::<Vec<_>>();
        if forwarded_inputs.iter().all(Option::is_none) {
            return partially_evaluate_split_shard_map(self, context, driver, body_program, inputs);
        }
        let kept_outputs =
            (0..self.output_types.len()).filter(|&index| forwarded_inputs[index].is_none()).collect::<Vec<_>>();
        let operation = Self {
            shard_map: ShardMap::from_shardings(
                self.shard_map.mesh().clone(),
                self.shard_map.in_shardings().to_vec(),
                kept_outputs.iter().map(|&index| self.shard_map.out_shardings()[index].clone()).collect(),
                self.shard_map.manual_axes().to_vec(),
            ),
            input_types: self.input_types.clone(),
            output_types: kept_outputs.iter().map(|&index| self.output_types[index].clone()).collect(),
            output_forwarding: kept_outputs.iter().map(|&index| self.output_forwarding[index]).collect(),
        };
        let body = body_program.to_program().with_outputs(&kept_outputs)?;
        // A body whose outputs are all forwarded leaves no work behind unless it is retained when unused, which is the
        // dead-code criterion that simplification and boundary pruning apply to an all-dead `shard_map` (and that the
        // other rules of this operation use). Its effect classes alone are not the right criterion: deferred work
        // carries no effect class but must survive, while an unused local allocation carries one but need not.
        let outputs = if kept_outputs.is_empty() && !body.effects().is_retained_when_unused() {
            Vec::new()
        } else {
            partially_evaluate_split_shard_map(&operation, context, driver, body.entry_region_ref(), inputs)?
        };
        let mut outputs = outputs.into_iter();
        Ok(forwarded_inputs
            .into_iter()
            .map(|forwarded| match forwarded {
                Some(input_index) => inputs[input_index].clone(),
                None => outputs.next().unwrap(),
            })
            .collect())
    }
}

// Batching rule for `ShardMapOperation`, following JAX's `_shard_map_batch`. The logical batch axis of a mapped input
// or output is the axis that the batching context maps over, and it physically becomes one more dimension of that
// value's global boundary type (of its referent, for a reference), at the value's batch axis and with the static batch
// extent. For a replicated batch axis, that dimension is unpartitioned (JAX's `batch_spec(spec, axis, None)`), and
// otherwise it is partitioned only along free axes (refer to the placement paragraph below), so the active manual axes
// of the batched map keep partitioning exactly the dimensions that they partitioned before batching: the local type of
// a mapped value is its unbatched local type with the whole batch dimension inserted at the same position, and every
// device runs the structurally batched local body over all batch items of its own shards. Collectives over manual axes
// inside the body combine those shards elementwise, so they pass the batch dimension through untouched.
//
// Unbatched inputs keep their boundary types and shardings and enter the batched body unbatched, where its structural
// batching aligns them with the mapped values that they meet, so mapped and unbatched inputs mix freely. Once the body
// is batched, the rule restores the boundary by inserting the batch dimension into the input sharding and declared
// global type of every mapped input, and into the output sharding and declared global type of every output that the
// batched body maps, at the position that the batched body reports. The body is batched at the same batching level
// (only its placement may differ, as explained below), so a collective over the name of this level inside the body
// (which the body inherits from the enclosing named axes, refer to `shard_map_in_context`) is consumed by this level,
// exactly as JAX traces the body under its `BatchTrace`. A call whose inputs are all unbatched computes the same values
// for every batch item unless its body refers to this level by name, so only an anonymous level keeps such a call's
// boundary and binds it unchanged. A named level batches the body structurally even then: an output that the consumed
// collectives map (e.g., the `axis_index` of this level) takes the mapped path, while a body whose outputs all stay
// unbatched keeps the boundary of the call.
//
// References batch like arrays, through their referents: a mapped reference input `ref<T>` crosses as `ref<T'>`, where
// `T'` is `T` with the batch dimension inserted, the body reads and writes per-item values at that axis, and a
// forwarded reference output carries the batch axis of the input that it forwards (which the rule validates). An
// unbatched reference input keeps its boundary and is shared by every batch item, so the reference rules of the body
// reject any write of a batched value into it, as JAX's `_swap_vmap` does. The ownership contract of the boundary is
// unchanged, because the batch dimension is never partitioned along an active manual axis.
//
// A batching level may place its batch axis on mesh axes (`BatchingContext::axis_sharding`, which batching infers from
// the shardings of the mapped inputs), and the rule keeps that placement at the boundary, where it partitions the batch
// dimension of every mapped input and output. An `Auto` placement axis is dropped, as the boundary drops `Auto` axes
// from every sharding, and an unconstrained placement is rejected, because the boundary needs a pinned placement. A
// placement axis that an input or output sharding names is rejected (`ShardMapError::BatchAxisPlacedOnSpecifiedAxis`),
// as JAX rejects a `spmd_axis_name` that its specifications mention. Every other placement axis is free in the batched
// map, so the batch dimension keeps its global extent inside the body (as JAX's explicit-axis branch keeps the
// placement in the outer sharding only), and collectives over the batch axis inside the body still see every batch
// item. An `Explicit` axis or a manual axis that is not active in this map is free already. An active manual axis is
// the analogue of JAX's `spmd_axis_name`, but JAX's design (a local batch dimension that varies along that axis inside
// the body) needs operation-level batching rules that align manual variation, which this rule cannot provide. Instead,
// the rule removes such an axis from the active manual axes of the batched map, so that it becomes free and carries the
// batch dimension. That leaves the semantics of the body unchanged when every device along the axis computes the same
// values, which holds when no input or output sharding names the axis (so every input is replicated along it and every
// output must be invariant along it), when no value of the body varies along it and no collective or `axis_index` of
// the body names it (which excludes communication along it even between invariant values, such as a meshless
// permutation, whose output is typed by its input), and when the body has no `DeviceOrderedIo` effect, whose per-device
// executions would change. Recognizing those collectives is why the operation family must support payload projection.
// Each violation is rejected with its own `ShardMapError`, as is a placement on every active manual axis, because the
// batched map must keep one. No manual placement axis is manual inside the batched body, but a batched value placed on
// a manual axis would vary along it, so the body is batched with a context whose placement keeps only the `Explicit`
// placement axes (boundary validation compares local types up to their placement, so the local types still match). The
// batched boundary is validated through `ShardMap::new_within` against the enclosing manual axes that the inputs
// witness by varying along them, which rejects a placement on an axis that an enclosing manual region made manual.
//
// Shard-map boundaries are static (refer to the `# Static Boundaries` section of the documentation of
// `ShardMapOperation`), so the batch extent must be static whenever a boundary position is mapped or the body uses it,
// and is rejected with `ShardMapError::DynamicBatchExtentNotSupported` otherwise. That rejection is deliberate rather
// than a gap: padding the batch to a static bound would run the body on padding items, which effectful bodies and
// bodies that write through references would observe, and JAX also supports only static batch sizes here (its
// `_shard_map_batch` computes them with integer arithmetic). The structurally batched body receives the extent as a
// leading dimension input (refer to `ArrayIrBatchingPolicy`), which the rule replaces by a constant of the same static
// type inside the body (or drops when the body does not use it), so that no dimension crosses the boundary. Ragged
// inputs are rejected because their per-item extents would be lost at the boundary. Supporting them needs ragged-aware
// structural program batching that threads the per-item lengths through the boundary as inputs, which `scan`,
// `condition`, and kernels need as well and which should therefore be shared among them rather than built for this rule
// alone.
impl<C> BatchableOperation<C, ArrayIrBatchingPolicy> for ShardMapOperation
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<ShardMapOperation> + From<ConstantOperation<DimensionValue>> + OperationPayloadProjection,
        >,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        check_count!("input", inputs, self.input_types.len(), ProgramError);
        let input_values = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();

        // A call whose inputs are all unbatched computes the same values for every batch item when no collective in its
        // body can refer to this level, which holds for an anonymous level, so it keeps its boundary.
        let input_positions = inputs.iter().map(ArrayIrBatch::batch_axis_position).collect::<Vec<_>>();
        if context.axis_name().is_none() && input_positions.iter().all(Option::is_none) {
            let outputs = context.parent().bind(self.clone(), vec![driver.region(0)?.to_program()], &input_values)?;
            return Ok(outputs.into_iter().map(ArrayIrBatch::replicated).collect::<Vec<_>>().into());
        }
        // Ragged batching needs ragged-aware structural program batching shared with `scan`, `condition`, and kernels,
        // and dynamic extents are rejected deliberately (refer to the rationale above).
        ArrayIrBatch::reject_ragged_inputs(self, inputs)?;
        let extent_type = <&DimensionType>::try_from(context.axis_extent().r#type().as_ref())?.clone();
        let static_extent = || {
            extent_type.extent().ok_or_else(|| {
                BatchingError::from(ProgramError::from(ShardMapError::DynamicBatchExtentNotSupported {
                    extent_type: extent_type.clone(),
                }))
            })
        };
        if input_positions.iter().any(Option::is_some) {
            static_extent()?;
        }

        // Batch the local body at the mapped input axes, and replace its leading extent input and output by a constant
        // of the extent's own static type, so that the batched body keeps the boundary of the source body. A body that
        // does not use the extent (e.g., one whose values are all unbatched) drops it instead, so that it needs no
        // static extent. Inside the body, the batch dimension keeps only the `Explicit` axes of its placement, because
        // a placement on a manual axis would make the batched body values vary along it (refer to the rationale above).
        let body = driver.region(0)?;
        let input_axes = inputs.iter().map(ArrayIrBatch::batch_axis).collect::<Vec<_>>();
        let body_context = match context.axis_sharding() {
            ShardingDimension::Sharded(axis_names) => {
                let mesh = self.shard_map.mesh();
                let explicit_axes = axis_names
                    .iter()
                    .filter(|axis_name| mesh.axis_type(axis_name) == Some(MeshAxisType::Explicit))
                    .cloned()
                    .collect::<Vec<_>>();
                let body_placement = if explicit_axes.is_empty() {
                    ShardingDimension::Replicated
                } else {
                    ShardingDimension::Sharded(explicit_axes)
                };
                context.clone().with_axis_sharding(body_placement)
            }
            ShardingDimension::Replicated | ShardingDimension::Unconstrained => context.clone(),
        };
        let (batched_body, output_axes) = driver
            .batch_program(&body_context, body, input_axes.as_slice(), ProgramBatchingOutputAxesPolicy::Natural)?
            .into_parts();
        check_count!("output", batched_body.output_ids(), output_axes.len() + 1, ProgramError);
        let (live_body, live_inputs) =
            batched_body.filtered(batched_body.input_ids(), &batched_body.output_ids()[1..], &[])?;
        let batched_body = {
            let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
            let body_extent = if live_inputs.contains(&0) {
                let extent = DimensionValue::new(extent_type.clone(), static_extent()?).map_err(ProgramError::from)?;
                Some(builder.add_instruction(ConstantOperation::new(extent), Vec::new(), Vec::new(), None)?[0])
            } else {
                None
            };
            let body_inputs = batched_body.input_types()[1..]
                .iter()
                .cloned()
                .map(|r#type| builder.add_input(r#type))
                .collect::<Vec<_>>();
            let live_body_inputs = live_inputs
                .iter()
                .map(|&index| if index == 0 { body_extent.unwrap() } else { body_inputs[index - 1] })
                .collect::<Vec<_>>();
            let body_outputs = builder.splice_program(&live_body, live_body_inputs.as_slice())?;
            builder
                .build::<Vec<C::Constant>, Vec<C::Constant>>(
                    body_outputs,
                    vec![Placeholder; body_inputs.len()],
                    vec![Placeholder; output_axes.len()],
                )?
                .simplified()?
        };

        // The batched body reports the logical batch axis of each output, which `normalize_batch_axis` resolves to the
        // physical position of the batch dimension in the batched body's output type (in its referent, for a reference
        // output). A forwarded reference output is the reference that it forwards, so it must keep that input's batch
        // axis, which the structural batching of references guarantees and which is validated here defensively.
        let output_positions = output_axes
            .iter()
            .zip(batched_body.output_types())
            .map(|(axis, r#type)| Ok(boundary_array_type(&r#type)?.normalize_batch_axis(*axis)?.1))
            .collect::<Result<Vec<_>, BatchingError>>()?;
        for (output_index, input_index) in self.output_forwarding.iter().enumerate() {
            if let Some(input_index) = input_index
                && output_positions[output_index] != input_positions[*input_index]
            {
                return Err(ProgramError::MalformedProgram(format!(
                    "batching `{SHARD_MAP_OPERATION_NAME}` output #{output_index} moves the batch axis of the \
                     reference input #{input_index} that it forwards",
                ))
                .into());
            }
        }

        // Restore the boundary: insert the batch dimension, with the static extent and the boundary placement of the
        // batch axis, into the sharding and declared global type (or global referent) of every mapped input and output
        // at its physical batch axis position, while unbatched inputs and outputs keep their boundary unchanged.
        let mut shard_map = self.shard_map.clone();
        let mut input_types = self.input_types.clone();
        let mut output_types = self.output_types.clone();
        if input_positions.iter().chain(&output_positions).any(Option::is_some) {
            let placement_axes: &[String] = match context.axis_sharding() {
                ShardingDimension::Sharded(axis_names) => axis_names,
                ShardingDimension::Replicated => &[],
                ShardingDimension::Unconstrained => {
                    return Err(BatchingError::UnsupportedOperation {
                        message: format!(
                            "batching a `{SHARD_MAP_OPERATION_NAME}` over mapped inputs requires a batch axis with a \
                             pinned placement, but the batch axis is placed as `{}`",
                            context.axis_sharding(),
                        ),
                    });
                }
            };
            let to_batching_error = |error: ShardMapError| BatchingError::from(ProgramError::from(error));
            let (placement, manual_axes) =
                self.shard_map.batched_placement(placement_axes, driver.region(0)?).map_err(to_batching_error)?;
            let batch_dimension = Dimension::Static(static_extent()?);
            let batch_boundary = |sharding: &mut Sharding, r#type: &mut ArrayIrType, position: usize| {
                // The batch dimension is inserted without adding manual variation: inside the batched map, its
                // placement names only free axes, along which values never vary.
                *sharding = sharding.with_inserted_dimension(position, placement.clone()).map_err(TypeError::from)?;
                let array_type = boundary_array_type(r#type)?;
                let referent_sharding = array_type
                    .sharding()
                    .map(|sharding| sharding.with_inserted_dimension(position, placement.clone()))
                    .transpose()
                    .map_err(TypeError::from)?;
                let referent = array_type
                    .with_inserted_dimension(position, batch_dimension.clone())?
                    .with_sharding(referent_sharding)
                    .map_err(TypeError::from)?;
                *r#type = if r#type.is_reference() { ReferenceType::new(referent).into() } else { referent.into() };
                Ok::<_, BatchingError>(())
            };
            let mut in_shardings = self.shard_map.in_shardings().to_vec();
            let mut out_shardings = self.shard_map.out_shardings().to_vec();
            for (index, position) in input_positions.iter().enumerate() {
                if let Some(position) = position {
                    batch_boundary(&mut in_shardings[index], &mut input_types[index], *position)?;
                }
            }
            for (index, position) in output_positions.iter().enumerate() {
                if let Some(position) = position {
                    batch_boundary(&mut out_shardings[index], &mut output_types[index], *position)?;
                }
            }

            // Only values of a manual region that made an axis manual vary along it, so the manual mesh axes that are
            // not active in this map but along which an input varies are manual in enclosing manual regions. The
            // batched boundary is validated against them, which rejects a batch axis placed on such an axis.
            let mesh = self.shard_map.mesh();
            let mut enclosing_manual_axes = BTreeSet::new();
            for input in inputs {
                let input_type = input.value().r#type();
                if let Some(sharding) = boundary_array_type(&input_type)?.sharding() {
                    enclosing_manual_axes.extend(sharding.varying_manual_axes().iter().cloned());
                }
            }
            let enclosing_manual_axes = enclosing_manual_axes
                .iter()
                .map(String::as_str)
                .filter(|axis_name| {
                    mesh.axis_type(axis_name) == Some(MeshAxisType::Manual)
                        && !self.shard_map.manual_axes().iter().any(|manual_axis| manual_axis == axis_name)
                })
                .collect::<HashSet<_>>();
            shard_map =
                ShardMap::new_within(mesh.clone(), in_shardings, out_shardings, manual_axes, &enclosing_manual_axes)
                    .map_err(to_batching_error)?;
        }
        let operation = ShardMapOperation::from_boundary(shard_map, input_types, output_types)
            .with_output_forwarding(self.output_forwarding.clone())
            .map_err(ProgramError::from)?;
        let outputs = context.parent().bind(operation, vec![batched_body], &input_values)?;
        check_count!("output", outputs, output_axes.len(), ProgramError);
        Ok(outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayIrBatch::new(output, axis))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

// Capture-free forward-mode (JVP) rule for `ShardMapOperation`, binding either one fused `shard_map` or a primal
// `shard_map` and a tangent `shard_map` as ordinary operations of the active context's operation family: a staging
// context stages them over its shared builder, while an eager context executes them immediately.
//
// Both forms realize the identity `jvp(shard_map(f)) = shard_map(jvp f)`: rather than capturing the global primals as
// residual factors and staging a linear `shard_map`, the rule keeps the manual region intact, and the two-map form
// threads every residual as a plain primal input edge between its two `shard_map`s, so no symbolic capture is ever
// introduced. The enclosing partial-evaluation split then discovers the residual edges structurally, exactly as for the
// jitted-call rule.
//
// For the two-map form, the rule linearizes the body capture-free through `shard_map_bodies` under the inputs' activity
// mask (an input with a live tangent is active, while an input with a structurally zero tangent and a plumbing
// reference input that receives no tangent reference are not, as under JAX's `which_nz`), giving a primal body
// `inputs -> [outputs..., residual_edges...]` and a tangent body
// `[active(input_tangents)..., residuals...] -> [live(output_tangents)...]`, where a residual that is a primal body
// input or output is not a residual edge but is taken from that boundary input or output (JAX's `in_fwd` and
// `out_fwd`), and repeated residuals are deduplicated. It then stages the primal `shard_map` over the input primals
// (recovering the primal outputs followed by the residual edge values), stages the tangent `shard_map` over the active
// input tangents followed by the residual values (recovering the live output tangents, including forwarded active
// tangent references), and pairs each primal output with its tangent or a structural zero for an output without a live
// tangent. When no output tangent is live and the tangent body has no observable effects, the tangent `shard_map` is
// not staged at all, and the source `shard_map` is bound as the primal one. Both sub-programs are built over the
// context's own constant and operation families, and every nested request goes through `driver`, so the rule carries no
// differentiation obligation on its operation family.
//
// The two-map form above is what a partitioned differentiation context (i.e., a linearization, whose primal and tangent
// work go to separate contexts) needs, as in JAX's `_shard_map_linearize`. A fused context (i.e., one whose primal and
// tangent contexts are the same context, as for `jvp` and for the fused programs that the rules of higher-order
// operations derive) instead binds one `shard_map`, as JAX's `_shard_map_jvp` does, whose body is the fused JVP program
// of the body over `[inputs..., active(input_tangents)...]` (refer to `fused_shard_map_jvp`). Its outputs are the
// outputs followed by the live output tangents, so an eager context runs one manual computation per `shard_map` and
// materializes no residual edges, and a fused program that is later partitioned (e.g., by the `while`, `scan`,
// `condition`, rematerialization, or custom-function rules) is split by the partial-evaluation rule of this operation
// into a known map (computing the outputs and the residuals) and a residual map (computing the output tangents). That
// split must keep every output known, so a body that the partial-evaluation rule keeps whole keeps the two-map form
// even in a fused context: a body with references (refer to `body_has_references`) and a body with a dynamically shaped
// value anywhere in its region closure, which could be a residual that cannot cross the static boundary (refer to
// `body_has_dynamically_shaped_values`). So does a body none of whose output tangents is live, for which the two-map
// form binds no tangent map at all.
impl<C> DifferentiableOperation<C> for ShardMapOperation
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<ShardMapOperation>
                           + OperationProjection<
                ArrayType,
                Projected: From<ReshapeOperation> + From<BroadcastOperation> + From<ParallelVaryOperation>,
            > + From<DimensionToScalarOperation>
                           + From<DimensionFromScalarOperation>
                           + From<ReferenceNewOperation<ArrayType, ArrayIrType>>
                           + From<ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>>,
        > + Zero<C::Value>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let output_count = self.output_types.len();
        check_count!("input", inputs, self.input_types.len(), ProgramError);

        let body_program = driver.region(0)?;
        self.validate_reference_body(body_program)?;
        // A structurally zero tangent is inactive, like a missing one (JAX's `which_nz`), so it gets no tangent
        // boundary slot, and every output that depends only on such inputs receives a structurally zero tangent.
        let activity = inputs
            .iter()
            .map(|input| input.is_tangent_active() && !input.tangent().is_zero())
            .collect::<Vec<_>>();
        if std::ptr::eq(context.primal(), context.tangent())
            && !body_has_references(body_program)
            && !body_has_dynamically_shaped_values(body_program)
            && let Some(outputs) = fused_shard_map_jvp(self, context, driver, body_program, inputs, &activity)?
        {
            return Ok(outputs);
        }
        let ShardMapBodies { primal_operation, primal_body, tangent, output_activity } =
            shard_map_bodies(self, driver, body_program, activity.as_slice())?;

        // Bind the primal `shard_map`, recovering the primal outputs followed by the residual edge values. Without a
        // tangent `shard_map`, every output tangent is a structural zero.
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_output_count = primal_operation.global_output_types().len();
        let mut primal_outputs = context.primal().bind(primal_operation, vec![primal_body], &primal_inputs)?;
        check_count!("output", primal_outputs, primal_output_count, ProgramError);
        let Some(ShardMapTangent { operation: tangent_operation, body: tangent_body, residuals }) = tangent else {
            return primal_outputs.into_iter().map(DifferentiationDual::new_with_zero_tangent).collect();
        };
        let residual_edges = primal_outputs.split_off(output_count);

        // The primal `shard_map` may re-embed its outputs into the caller's ambient sharding envelope. The tangent
        // `shard_map` carries residual inputs and therefore cannot infer that envelope through the single-input
        // adaptation path, so derive its output descriptors from the adapted primal outputs while retaining the
        // tangent element representation.
        let tangent_output_types = primal_outputs
            .iter()
            .zip(&output_activity)
            .filter(|(_, active)| **active)
            .map(|(output, _)| output.r#type().tangent())
            .collect::<Result<Vec<_>, _>>()?;
        let tangent_operation =
            tangent_operation.with_global_output_types(tangent_output_types).map_err(ProgramError::from)?;

        // Bind the tangent `shard_map` over the active input tangents followed by the residual values, which are primal
        // inputs, primal outputs, or residual edge values, recovering one output tangent per live output tangent.
        // Structurally zero tangents are inactive, so every active input tangent is a live value and materializing it
        // only unwraps it, while an inactive input (a structural zero, a plumbing reference, or a zero-space input) has
        // no tangent boundary slot.
        let mut tangent_inputs = inputs
            .iter()
            .zip(&activity)
            .filter(|(_, active)| **active)
            .map(|(input, _)| input.tangent().clone().materialize(context.tangent()))
            .collect::<Result<Vec<_>, _>>()?;
        for residual in residuals {
            let value = match residual {
                ShardMapResidual::Input(index) => primal_inputs[index].clone(),
                ShardMapResidual::Output(index) => primal_outputs[index].clone(),
                ShardMapResidual::Edge(edge) | ShardMapResidual::ReferenceSnapshot(edge) => {
                    residual_edges[edge].clone()
                }
            };
            tangent_inputs.push(context.primal_to_tangent(value)?);
        }
        let tangent_outputs = context.tangent().bind(tangent_operation, vec![tangent_body], &tangent_inputs)?;
        check_count!("output", tangent_outputs, output_activity.iter().filter(|active| **active).count(), ProgramError);
        let mut tangent_outputs = tangent_outputs.into_iter();
        primal_outputs
            .into_iter()
            .zip(output_activity)
            .map(|(primal, active)| {
                if active {
                    DifferentiationDual::new(primal, tangent_outputs.next().unwrap())
                } else {
                    DifferentiationDual::new_with_zero_tangent(primal)
                }
            })
            .collect()
    }
}

// Transpose rule for a primal tangent `ShardMapOperation`, forwarding to `transpose_primal_shard_map` with the
// cotangent references of its linear reference-typed inputs resolved (and allocated on first use) through the
// enclosing `TranspositionContext`. The body transposition is requested through the instruction-scoped driver, so
// instantiating this implementation for a closed operation family introduces no recursive `TransposableOperation`
// obligation on that family.
//
// The forward boundary implicitly places each caller input by its input sharding and moves it into device memory, so
// the assembled input cotangents are reconciled with the caller's cotangent types (refer to
// `reconcile_input_cotangent`): a `reshard` restores the caller's placement over `Explicit` axes unless the caller
// also places a dimension along a manual axis, a placement-only `broadcast` restores the remaining (e.g., manual-axis)
// placement, and a `transfer_to_memory` restores the caller's memory kind. This is why the array projection of `O`
// must provide `ReshardOperation`, `BroadcastOperation`, and `TransferToMemoryOperation`, while `ResidualZeroProvider`
// serves the cotangent destinations, and the reference operations serve the per-device accumulators of the replicated
// destinations that are accumulated locally (i.e., those of inputs that the body does not mutate; refer to
// `local_accumulator_destinations`).
impl<V, O> TransposableOperation<V, O> for ShardMapOperation
where
    V: Value<Type = ArrayIrType>,
    O: Operation<Type = ArrayIrType>
        + From<AddOperation<ArrayIrType>>
        + From<ShardMapOperation>
        + ResidualZeroProvider<ArrayIrType, Operation = O>
        + OperationProjection<
            ArrayType,
            Projected: From<BroadcastOperation> + From<ReshardOperation> + From<TransferToMemoryOperation>,
        > + From<ReferenceNewOperation<ArrayType, ArrayIrType>>
        + From<ReferenceFreezeOperation<ArrayType, ArrayIrType>>
        + From<ReferenceAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("output", outputs, self.output_types.len(), ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
        let cotangents = context.cotangent_destinations(driver, inputs, accumulators)?;
        let contributions = transpose_primal_shard_map(self, context, driver, inputs, outputs, &cotangents)?;
        check_count!("input", contributions, accumulators.len(), ProgramError);
        for (accumulator, contribution) in accumulators.iter().zip(contributions) {
            accumulator.accumulate(context, contribution)?;
        }
        Ok(())
    }
}

/// Local value that a `shard_map` body closure traced for the [`Domain`] `C` receives and returns: a [`DomainTracer`]
/// of `C` projected onto its array member. Closure bodies (refer to [`shard_map`], [`shard_map_with_options`],
/// [`shard_map_in_context`], [`trace_shard_map`], [`trace_shard_map_with_options`], and
/// [`trace_shard_map_with_named_axes`]) are array-only, so every local input and output of such a body is one of these
/// values. Bodies with reference boundaries are traced explicitly and checked through
/// [`ShardMapOperation::from_program`] instead. Shard-map boundaries are array or reference positions only, so
/// first-class dimensions may appear only inside a body.
pub type ShardMapTracer<C> = ProjectedValue<ArrayType, DomainTracer<C>>;

/// Local context that a `shard_map` body closure traced for the [`Domain`] `C` receives from [`shard_map_in_context`]
/// and [`trace_shard_map_with_named_axes`]: the array view (refer to [`ProjectedContext`]) of the fresh
/// [`DomainTracingContext`] of `C` that the body is traced in. It is the [`Domain`](Value::Domain) of the body's
/// [`ShardMapTracer`]s, so values that it creates through its capabilities are [`ShardMapTracer`]s as well (e.g.,
/// `context.axis_index("x")` returns the coordinate of the executing device along the manual axis `x`, and
/// [`Context::lift`] stages an array constant), and [`ProjectedContext::parent`] returns the underlying composite
/// tracing context.
pub type ShardMapContext<C> = ProjectedContext<DomainTracingContext<C>, ArrayType>;

/// Invokes a manual SPMD computation over `mesh` on `inputs`, running `function` once per device on the local shards of
/// the inputs and assembling the global outputs from its local outputs. This is the analogue of
/// [`jax.shard_map`](https://docs.jax.dev/en/latest/_autosummary/jax.shard_map.html), with every mesh axis of type
/// [`Manual`](MeshAxisType::Manual) active that no enclosing manual region already made manual. It binds the
/// computation in the [`Context`] of its inputs, like [`shard_map_with_options`], whose documentation explains the
/// handling of invocations without inputs. Refer to [`shard_map_in_context`] for the complete semantics, including how
/// the body is traced and bound.
///
/// # Parameters
///
///   - `function`: Body closure traced over the local shards of `inputs`.
///   - `inputs`: Structured global input values, whose leaves are array projections of composite values.
///   - `mesh`: Logical mesh that the manual computation is defined over.
///   - `in_shardings`: Shardings of the global inputs (i.e., JAX's `in_specs`), with the same structure as `inputs`.
///   - `out_shardings`: Shardings of the global outputs (i.e., JAX's `out_specs`), with the same structure as the
///     outputs of `function`.
///
/// # Errors
///
/// Returns the errors of [`shard_map_with_options`].
pub fn shard_map<C, V, F, Input, Output>(
    function: F,
    inputs: Input,
    mesh: LogicalMesh,
    in_shardings: Input::To<Sharding>,
    out_shardings: Output::To<Sharding>,
) -> Result<Output::To<ProjectedValue<ArrayType, V>>, ShardMapError>
where
    C: Context<Type = ArrayIrType, Value = V, Operation: From<ShardMapOperation>> + NamedAxes,
    V: Value<Type = ArrayIrType, Domain = C> + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
    ShardMapTracer<C>: ParallelVary,
    Input: Parameterized<ProjectedValue<ArrayType, V>>,
    Input::Family: ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTracer<C>>,
    Output: Parameterized<ShardMapTracer<C>>,
    Output::Family: ParameterizedFamily<Sharding> + ParameterizedFamily<ProjectedValue<ArrayType, V>>,
    F: FnOnce(Input::To<ShardMapTracer<C>>) -> Output,
{
    shard_map_with_options(function, inputs, mesh, in_shardings, out_shardings, Vec::new())
}

/// Invokes a manual SPMD computation over `mesh` on `inputs` with an explicit selection of active manual axes, binding
/// it in the [`Context`] `C` of its inputs. This is [`shard_map_in_context`] in the [`Domain`](Value::Domain) of the
/// first input, so that `C` is never named at a call site, with a body closure that receives only its local values.
/// Refer to [`shard_map_in_context`] for the complete semantics, including how the body is traced and bound.
///
/// When `inputs` has no leaves, no input supplies a context to bind the operation in, so an invocation with outputs is
/// rejected with [`ShardMapError::MissingTracedInvocationDomain`]. [`shard_map_in_context`] binds such an invocation in
/// an explicitly provided context instead, and its body closure also receives a context through which to create values
/// (e.g., JAX's `shard_map(lambda: axis_index('x'), in_specs=(), out_specs=P('x'))`). An invocation without inputs and
/// outputs still traces its body (in a fresh [`DomainTracingContext`] of `C`, which needs no context instance, and
/// without inherited named axes), so that its boundary and body are validated, and returns an empty result. Such a body
/// receives neither a value nor a context, so it can neither compute values nor stage effects, and dropping it discards
/// nothing observable.
///
/// # Parameters
///
///   - `function`: Body closure traced over the local shards of `inputs`.
///   - `inputs`: Structured global input values, whose leaves are array projections of composite values.
///   - `mesh`: Logical mesh that the manual computation is defined over.
///   - `in_shardings`: Shardings of the global inputs (i.e., JAX's `in_specs`), with the same structure as `inputs`.
///   - `out_shardings`: Shardings of the global outputs (i.e., JAX's `out_specs`), with the same structure as the
///     outputs of `function`.
///   - `manual_axes`: Active manual mesh axes. An empty list selects every mesh axis of type
///     [`Manual`](MeshAxisType::Manual) that no enclosing manual region already made manual.
///
/// # Errors
///
/// Returns [`ShardMapError::InputTypeCountMismatch`] when `in_shardings` and `inputs` have different numbers of leaves,
/// [`ShardMapError::MissingTracedInvocationDomain`] when `inputs` has no leaves but `out_shardings` does, and the
/// errors of [`shard_map_in_context`] otherwise.
pub fn shard_map_with_options<C, V, F, Input, Output>(
    function: F,
    inputs: Input,
    mesh: LogicalMesh,
    in_shardings: Input::To<Sharding>,
    out_shardings: Output::To<Sharding>,
    manual_axes: Vec<String>,
) -> Result<Output::To<ProjectedValue<ArrayType, V>>, ShardMapError>
where
    C: Context<Type = ArrayIrType, Value = V, Operation: From<ShardMapOperation>> + NamedAxes,
    V: Value<Type = ArrayIrType, Domain = C> + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
    ShardMapTracer<C>: ParallelVary,
    Input: Parameterized<ProjectedValue<ArrayType, V>>,
    Input::Family: ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTracer<C>>,
    Output: Parameterized<ShardMapTracer<C>>,
    Output::Family: ParameterizedFamily<Sharding> + ParameterizedFamily<ProjectedValue<ArrayType, V>>,
    F: FnOnce(Input::To<ShardMapTracer<C>>) -> Output,
{
    let context = inputs.parameters().next().map(|input| input.value().domain());
    if let Some(context) = context {
        let function = |_: &ShardMapContext<C>, local_inputs| function(local_inputs);
        return shard_map_in_context(&context, function, inputs, mesh, in_shardings, out_shardings, manual_axes);
    }
    let input_structure = inputs.parameter_structure();
    let output_structure = out_shardings.parameter_structure();
    let output_paths = out_shardings.parameter_paths().collect::<Vec<_>>();
    let in_shardings = in_shardings.into_parameters().collect::<Vec<_>>();
    if !in_shardings.is_empty() {
        return Err(ShardMapError::InputTypeCountMismatch { expected: in_shardings.len(), actual: 0 });
    }
    // Without inputs, no input supplies a context to bind the operation in, so outputs cannot be returned. A body
    // without outputs is still traced, in a fresh trace of `C` that needs no context instance, so that it is validated.
    // It receives neither a value nor a context, so it cannot stage effects that dropping it would discard.
    if output_structure.parameter_count() != 0 {
        return Err(ShardMapError::MissingTracedInvocationDomain);
    }
    let shard_map = ShardMap::new(mesh, in_shardings, out_shardings.into_parameters().collect(), manual_axes)?;
    trace_shard_map_body::<C, _>(
        shard_map,
        |_, local_inputs| {
            let local_inputs = Input::To::<ShardMapTracer<C>>::from_parameters(input_structure, local_inputs)?;
            let local_outputs = function(local_inputs);
            validate_output_structure(&output_paths, &local_outputs)?;
            Ok(local_outputs.into_parameters().collect())
        },
        Vec::new(),
        Vec::new(),
    )?;
    Ok(Output::To::<ProjectedValue<ArrayType, V>>::from_parameters(output_structure, Vec::new())?)
}

/// Invokes a manual SPMD computation over `mesh` on `inputs` in the explicitly provided `context`, with an explicit
/// selection of active manual axes, running `function` once per device coordinate along those axes on the local shards
/// of the inputs and assembling the global outputs from its local outputs. This is the analogue of
/// [`jax.shard_map`](https://docs.jax.dev/en/latest/_autosummary/jax.shard_map.html), whose `axis_names` parameter
/// `manual_axes` mirrors, and the function that [`shard_map`] and [`shard_map_with_options`] delegate to with the
/// context of their first input.
///
/// The leaves of `inputs` are array projections of composite values (i.e., values of type [`ArrayIrType`]) of the
/// [`Context`] `C`, in which the operation is bound. Every input must belong to `context`; binding validates this where
/// the context tracks value ownership (e.g., a staging context rejects tracers of another trace). `function` is traced
/// once, in a fresh [`DomainTracingContext`] of `C`, and receives the [`ShardMapContext`] of that trace together with
/// [`ShardMapTracer`]s whose types are the local shards of the input types (refer to [`ShardMap::local_input_type`]).
/// The resulting checked [`ShardMapOperation`] is bound in `context` on the input values with the traced body attached
/// as its `body` region. The outputs are array projections of the values that this binding returns, with the structure
/// of `out_shardings`, which the outputs of `function` must have as well. The boundary types are derived as follows:
///
///   - **Inputs.** The type of each input is normalized into its global boundary type: its data type, shape, and layout
///     under its input sharding, together with the manual variation of enclosing manual regions that it carries. Its
///     memory kind is not part of the boundary type, so the local body inputs carry no memory kind, unlike the local
///     body types that a caller passes to [`ShardMapOperation::from_program`], which keep theirs. Every input type must
///     have a static shape that the manual partition counts of its input sharding divide.
///   - **Outputs.** A body output that does not yet vary along an active manual axis that its output sharding tiles is
///     marked as varying along it with `parallel_vary`, so that invariant results (e.g., values computed only from
///     replicated inputs) can be returned as tiled outputs. [`ShardMapOperation::from_program`], in contrast, rejects
///     such bodies. A body output that varies along an active manual axis that its output sharding does not mention is
///     rejected, as under JAX's default `check_vma=True`. The global output types are then derived through the output
///     shardings.
///
/// The body context lets `function` create values that derive from no input, through the capabilities of that context
/// (e.g., `context.axis_index("x")` for the coordinate of the executing device along the active manual axis `x`, or
/// [`Context::lift`] for an array constant), and stage effects on them. This is what makes bodies without inputs
/// useful: JAX's `shard_map(lambda: axis_index('x'), in_specs=(), out_specs=P('x'))` is
/// `shard_map_in_context(&context, |context, ()| .., (), mesh, (), sharding, ..)`, whose `shard_map` instruction is
/// bound in `context` without inputs and returns the global vector of device coordinates. A body without inputs
/// computes per-device values from the device coordinates and from constants only, and its manual variation records
/// them like any other body value, so an output that does not vary along a tiled axis is marked as varying along it as
/// explained above. The body context is the only handle through which a body without inputs creates values: values of
/// `context` (i.e., of the enclosing trace) belong to another trace and cannot be returned from the body.
///
/// The body resolves named axes against exactly the following bindings: the enclosing [`NamedAxis::Mesh`] bindings of
/// `context` whose names are axes of `mesh` (so that a nested `shard_map` over the axes of an enclosing manual region
/// can use them), the enclosing [`NamedAxis::Batched`] bindings of `context` whose names are not axes of `mesh` (i.e.,
/// the axes of enclosing `batch` levels, as JAX traces the body of a batched `shard_map` under its `BatchTrace`), and
/// one [`NamedAxis::Mesh`] binding per active manual axis of this computation, which shadows any enclosing binding of
/// the same name. Mesh bindings whose names are not axes of `mesh` and batched bindings whose names are axes of `mesh`
/// are not visible, and a mesh binding whose name is an axis of `mesh` must belong to the same device mesh as `mesh`,
/// since names alone cannot tell an enclosing manual axis from an axis of another mesh that shares its name.
/// Collectives inside the body can therefore communicate along the active manual axes and along the inherited axes: a
/// collective over the name of an enclosing `batch` level reduces over its batch items, because the batching rule of
/// [`ShardMapOperation`] batches the body with that level, which consumes the collective (e.g., the body
/// `|x| x.parallel_reduce(ReductionKind::Sum, "items")` under `batch` over `items` sums its local shards over the batch
/// items). A collective over any other name fails while the body is traced.
///
/// The inherited mesh bindings name the axes that enclosing manual regions already made manual (as `mesh.manual_axes`
/// does in JAX), and this computation never makes such an axis manual again, which would drop the variation that the
/// values of the enclosing regions carry along it: an empty `manual_axes` selects only the manual mesh axes that are
/// not already manual, an explicitly requested axis that is already manual is rejected, and so is a sharding that names
/// one. JAX silently drops already manual axes from an explicit request instead; rejecting the request reports the
/// mistake where it is made. The mesh must still type every inherited axis [`Manual`](MeshAxisType::Manual) (as JAX
/// requires the mesh of a nested `shard_map` to match its context mesh), so a nested map over the axes of an enclosing
/// manual region uses the same mesh and selects its own manual axes among the remaining ones (e.g., an outer map over
/// `x` of a mesh whose axes `x` and `y` are both manual, with `manual_axes = ["x"]`, around an inner map over `y`).
///
/// The body is traced in a fresh trace that owns no capture table, so a body that registers a capture (refer to
/// [`CapturingContext::capture`](crate::CapturingContext::capture)) is rejected with
/// [`ProgramError::DiscardedCaptures`], and its leaves are arrays only: reference boundaries require tracing the body
/// explicitly and constructing the operation through [`ShardMapOperation::from_program`], and first-class dimensions
/// may appear only inside the body, since shard-map boundaries are array or reference positions only.
///
/// # Parameters
///
///   - `context`: Context in which the operation is bound, which every leaf of `inputs` must belong to, and whose
///     named axes the body inherits as explained above.
///   - `function`: Body closure traced over the body context and the local shards of `inputs`.
///   - `inputs`: Structured global input values, whose leaves are array projections of composite values.
///   - `mesh`: Logical mesh that the manual computation is defined over.
///   - `in_shardings`: Shardings of the global inputs (i.e., JAX's `in_specs`), with the same structure as `inputs`.
///   - `out_shardings`: Shardings of the global outputs (i.e., JAX's `out_specs`), with the same structure as the
///     outputs of `function`.
///   - `manual_axes`: Active manual mesh axes. An empty list selects every mesh axis of type
///     [`Manual`](MeshAxisType::Manual) that no enclosing manual region already made manual.
///
/// # Errors
///
/// Returns [`ShardMapError::InputTypeCountMismatch`] when `in_shardings` and `inputs` have different numbers of leaves,
/// the errors of [`ShardMap::new`] for invalid mesh, sharding, and manual-axis combinations,
/// [`ShardMapError::AllManualAxesAlreadyManual`] when every manual mesh axis is already manual in an enclosing manual
/// region, [`ShardMapError::AxisAlreadyManual`] or [`ShardMapError::SpecificationNamesEnclosingManualAxis`] when
/// `manual_axes` or a sharding names such an axis, [`ShardMapError::EnclosingManualAxisNotManual`] when `mesh` does not
/// type such an axis [`Manual`](MeshAxisType::Manual), [`ShardMapError::EnclosingManualAxisMeshMismatch`] when such an
/// axis belongs to another device mesh, [`ShardMapError::Parameter`] when the outputs of `function` do not have the
/// structure of `out_shardings`, the boundary derivation errors (e.g., [`ShardMapError::InputMeshMismatch`],
/// [`ShardMapError::DynamicShapeNotSupported`], or [`ShardMapError::ManualAxisIntroducesPadding`]), the errors of
/// [`ShardMapOperation::from_program`] for the traced body (e.g., [`ShardMapError::OrderedIoNotSupported`]), and
/// [`ShardMapError::Program`] when tracing the body or binding the operation fails (e.g., when an input does not belong
/// to `context`).
pub fn shard_map_in_context<C, V, F, Input, Output>(
    context: &C,
    function: F,
    inputs: Input,
    mesh: LogicalMesh,
    in_shardings: Input::To<Sharding>,
    out_shardings: Output::To<Sharding>,
    manual_axes: Vec<String>,
) -> Result<Output::To<ProjectedValue<ArrayType, V>>, ShardMapError>
where
    C: Context<Type = ArrayIrType, Value = V, Operation: From<ShardMapOperation>> + NamedAxes,
    V: Value<Type = ArrayIrType, Domain = C> + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
    ShardMapTracer<C>: ParallelVary,
    Input: Parameterized<ProjectedValue<ArrayType, V>>,
    Input::Family: ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTracer<C>>,
    Output: Parameterized<ShardMapTracer<C>>,
    Output::Family: ParameterizedFamily<Sharding> + ParameterizedFamily<ProjectedValue<ArrayType, V>>,
    F: FnOnce(&ShardMapContext<C>, Input::To<ShardMapTracer<C>>) -> Output,
{
    let input_structure = inputs.parameter_structure();
    let output_structure = out_shardings.parameter_structure();
    let output_paths = out_shardings.parameter_paths().collect::<Vec<_>>();
    let in_shardings = in_shardings.into_parameters().collect::<Vec<_>>();
    let global_input_types = inputs.parameters().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    if global_input_types.len() != in_shardings.len() {
        return Err(ShardMapError::InputTypeCountMismatch {
            expected: in_shardings.len(),
            actual: global_input_types.len(),
        });
    }
    let inputs = inputs.into_parameters().map(ProjectedValue::into_value).collect::<Vec<_>>();
    let (shard_map, outer_named_axes) = shard_map_within_named_axes(
        mesh,
        in_shardings,
        out_shardings.into_parameters().collect(),
        manual_axes,
        &context.named_axes(),
    )?;
    let traced = trace_shard_map_body::<C, _>(
        shard_map,
        |body_context, local_inputs| {
            let local_inputs = Input::To::<ShardMapTracer<C>>::from_parameters(input_structure, local_inputs)?;
            let local_outputs = function(body_context, local_inputs);
            validate_output_structure(&output_paths, &local_outputs)?;
            Ok(local_outputs.into_parameters().collect())
        },
        global_input_types,
        outer_named_axes,
    )?;
    let outputs = context
        .bind(traced.operation, vec![traced.body], inputs.as_slice())?
        .into_iter()
        .map(|output| ValueProjection::<ArrayType>::into_projected(output).map_err(ProgramError::from))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Output::To::<ProjectedValue<ArrayType, V>>::from_parameters(output_structure, outputs)?)
}

/// Traces `function` as the local body of a manual SPMD computation over `mesh` for global inputs of the provided
/// types, without binding it, with every mesh axis of type [`Manual`](MeshAxisType::Manual) active. This is the
/// descriptor-only counterpart of [`shard_map`]. Refer to [`trace_shard_map_with_options`] for the complete semantics,
/// including how the caller names the [`Domain`] that the body is traced in, and for a manual-axis subset.
///
/// # Parameters
///
///   - `function`: Body closure traced over the local shards of global inputs of type `global_input_types`.
///   - `global_input_types`: Structured global input types.
///   - `mesh`: Logical mesh that the manual computation is defined over.
///   - `in_shardings`: Shardings of the global inputs (i.e., JAX's `in_specs`), with the same structure as
///     `global_input_types`.
///   - `out_shardings`: Shardings of the global outputs (i.e., JAX's `out_specs`), with the same structure as the
///     outputs of `function`.
///
/// # Errors
///
/// Returns the errors of [`trace_shard_map_with_options`].
pub fn trace_shard_map<C, F, Input, Output>(
    function: F,
    global_input_types: Input,
    mesh: LogicalMesh,
    in_shardings: Input::To<Sharding>,
    out_shardings: Output::To<Sharding>,
) -> Result<TracedShardMap<C, Input, Output>, ShardMapError>
where
    C: Domain<Type = ArrayIrType>,
    ShardMapTracer<C>: ParallelVary,
    Input: Parameterized<ArrayType>,
    Input::Family: ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTracer<C>>,
    Output: Parameterized<ArrayType>,
    Output::Family: ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTracer<C>>,
    F: FnOnce(Input::To<ShardMapTracer<C>>) -> Output::To<ShardMapTracer<C>>,
{
    trace_shard_map_with_options(function, global_input_types, mesh, in_shardings, out_shardings, Vec::new())
}

/// Traces `function` as the local body of a manual SPMD computation over `mesh` for global inputs of the provided
/// types with an explicit selection of active manual axes, without binding it, and returns the [`TracedShardMap`] that
/// holds the checked [`ShardMapOperation`], its global and local boundary types, and the traced body. This is the
/// descriptor-only counterpart of [`shard_map_with_options`], whose boundary derivation and output variation it shares
/// (refer to the documentation of [`shard_map_in_context`] for details), for callers that inspect or lower a shard-map
/// body on its own.
///
/// No input value supplies a [`Domain`] here, so the caller names the [`Domain`] `C` whose staged constant and
/// operation families the body is traced in (e.g., as the first generic argument or through the annotated result
/// type). The body is traced in a fresh [`DomainTracingContext`] of `C` that binds only the active manual axes of this
/// computation as [`NamedAxis::Mesh`] bindings, since no enclosing context exists to inherit named axes from. A body
/// that is meant to be bound inside enclosing manual regions or `batch` levels is traced with
/// [`trace_shard_map_with_named_axes`] instead, which inherits their axes (excluding the axes of enclosing manual
/// regions from the manual-axis selection), so that collectives inside the body can refer to them (refer to
/// [`shard_map_in_context`]). That function also passes the body context to its closure, so a body that creates values
/// through that context (e.g., a body without inputs) is traced with it as well, with empty `named_axes`.
///
/// # Parameters
///
///   - `function`: Body closure traced over the local shards of global inputs of type `global_input_types`.
///   - `global_input_types`: Structured global input types.
///   - `mesh`: Logical mesh that the manual computation is defined over.
///   - `in_shardings`: Shardings of the global inputs (i.e., JAX's `in_specs`), with the same structure as
///     `global_input_types`.
///   - `out_shardings`: Shardings of the global outputs (i.e., JAX's `out_specs`), with the same structure as the
///     outputs of `function`.
///   - `manual_axes`: Active manual mesh axes. An empty list selects every mesh axis of type
///     [`Manual`](MeshAxisType::Manual).
///
/// # Errors
///
/// Returns [`ShardMapError::InputTypeCountMismatch`] when `in_shardings` and `global_input_types` have different
/// numbers of leaves, the errors of [`ShardMap::new`] for invalid mesh, sharding, and manual-axis combinations,
/// [`ShardMapError::Parameter`] when the outputs of `function` do not have the structure of `out_shardings`, the
/// boundary derivation errors (e.g., [`ShardMapError::InputMeshMismatch`] or
/// [`ShardMapError::DynamicShapeNotSupported`]), the errors of [`ShardMapOperation::from_program`] for the traced body
/// (e.g., [`ShardMapError::OrderedIoNotSupported`]), and [`ShardMapError::Program`] when tracing the body fails.
pub fn trace_shard_map_with_options<C, F, Input, Output>(
    function: F,
    global_input_types: Input,
    mesh: LogicalMesh,
    in_shardings: Input::To<Sharding>,
    out_shardings: Output::To<Sharding>,
    manual_axes: Vec<String>,
) -> Result<TracedShardMap<C, Input, Output>, ShardMapError>
where
    C: Domain<Type = ArrayIrType>,
    ShardMapTracer<C>: ParallelVary,
    Input: Parameterized<ArrayType>,
    Input::Family: ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTracer<C>>,
    Output: Parameterized<ArrayType>,
    Output::Family: ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTracer<C>>,
    F: FnOnce(Input::To<ShardMapTracer<C>>) -> Output::To<ShardMapTracer<C>>,
{
    trace_shard_map_with_named_axes(
        |_: &ShardMapContext<C>, local_inputs| function(local_inputs),
        global_input_types,
        mesh,
        in_shardings,
        out_shardings,
        manual_axes,
        Vec::new(),
    )
}

/// Traces `function` as the local body of a manual SPMD computation over `mesh` like [`trace_shard_map_with_options`],
/// for a body that is meant to be bound in the named-axis scope `named_axes` of an enclosing context (typically
/// `enclosing_context.named_axes()`, refer to [`NamedAxes::named_axes`]). This is the descriptor-only counterpart of
/// [`shard_map_in_context`], for callers that trace a nested body on its own (e.g., to attach it to a
/// [`ShardMapOperation`] that they bind themselves).
///
/// Like the body closure of [`shard_map_in_context`], `function` receives the [`ShardMapContext`] of the trace that the
/// body is traced in together with its local inputs, so that it can create values that derive from no input through
/// the capabilities of that context (e.g., `context.axis_index("x")`). This makes descriptor-only bodies without inputs
/// expressible (e.g., JAX's `shard_map(lambda: axis_index('x'), in_specs=(), out_specs=P('x'))` traced for the global
/// input types `()`). [`trace_shard_map_with_options`] delegates to this function with no named axes and a closure
/// that ignores the context.
///
/// The body inherits exactly the bindings that [`shard_map_in_context`] inherits from its context: the
/// [`NamedAxis::Mesh`] bindings of `named_axes` whose names are axes of `mesh` (which must belong to the same device
/// mesh) and the [`NamedAxis::Batched`] bindings of `named_axes` whose names are not, together with the active manual
/// axes of this computation. As there, the inherited mesh bindings name the axes that enclosing manual regions already
/// made manual, so collectives inside the body can communicate along them, an empty `manual_axes` selects only the
/// manual mesh axes that are not already manual, and a manual-axis selection or sharding that names an already manual
/// axis is rejected, while the inherited batched bindings name the axes of enclosing `batch` levels, whose batching of
/// the bound shard map consumes the collectives over them. When several bindings of `named_axes` share a name, the
/// first one is used, which is the one that [`NamedAxes::named_axis`] resolves when `named_axes` comes from
/// [`NamedAxes::named_axes`]. The global input types describe the values that the enclosing region passes, including
/// the manual variation that they carry along the inherited axes. A traced shard map whose boundary varies along
/// inherited axes therefore describes a computation nested in those enclosing manual regions and is only meaningful
/// inside them.
///
/// # Parameters
///
///   - `function`: Body closure traced over the body context and the local shards of global inputs of type
///     `global_input_types`.
///   - `global_input_types`: Structured global input types.
///   - `mesh`: Logical mesh that the manual computation is defined over.
///   - `in_shardings`: Shardings of the global inputs (i.e., JAX's `in_specs`), with the same structure as
///     `global_input_types`.
///   - `out_shardings`: Shardings of the global outputs (i.e., JAX's `out_specs`), with the same structure as the
///     outputs of `function`.
///   - `manual_axes`: Active manual mesh axes. An empty list selects every mesh axis of type
///     [`Manual`](MeshAxisType::Manual) that no enclosing manual region already made manual.
///   - `named_axes`: Named axes in scope where the traced shard map is meant to be bound, innermost first.
///
/// # Errors
///
/// Returns the errors of [`trace_shard_map_with_options`], as well as
/// [`ShardMapError::AllManualAxesAlreadyManual`] when every manual mesh axis is already manual in an enclosing manual
/// region, [`ShardMapError::AxisAlreadyManual`] or [`ShardMapError::SpecificationNamesEnclosingManualAxis`] when
/// `manual_axes` or a sharding names such an axis, [`ShardMapError::EnclosingManualAxisNotManual`] when `mesh` does not
/// type such an axis [`Manual`](MeshAxisType::Manual), and [`ShardMapError::EnclosingManualAxisMeshMismatch`] when such
/// an axis belongs to another device mesh.
pub fn trace_shard_map_with_named_axes<C, F, Input, Output>(
    function: F,
    global_input_types: Input,
    mesh: LogicalMesh,
    in_shardings: Input::To<Sharding>,
    out_shardings: Output::To<Sharding>,
    manual_axes: Vec<String>,
    named_axes: Vec<(String, NamedAxis)>,
) -> Result<TracedShardMap<C, Input, Output>, ShardMapError>
where
    C: Domain<Type = ArrayIrType>,
    ShardMapTracer<C>: ParallelVary,
    Input: Parameterized<ArrayType>,
    Input::Family: ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTracer<C>>,
    Output: Parameterized<ArrayType>,
    Output::Family: ParameterizedFamily<Sharding> + ParameterizedFamily<ShardMapTracer<C>>,
    F: FnOnce(&ShardMapContext<C>, Input::To<ShardMapTracer<C>>) -> Output::To<ShardMapTracer<C>>,
{
    let output_paths = out_shardings.parameter_paths().collect::<Vec<_>>();
    let (shard_map, outer_named_axes) = shard_map_within_named_axes(
        mesh,
        in_shardings.into_parameters().collect(),
        out_shardings.into_parameters().collect(),
        manual_axes,
        &named_axes,
    )?;
    let input_structure = global_input_types.parameter_structure();
    let output_structure = RefCell::new(None);
    let traced = trace_shard_map_body::<C, _>(
        shard_map,
        |body_context, local_inputs| {
            let local_inputs = Input::To::<ShardMapTracer<C>>::from_parameters(input_structure.clone(), local_inputs)?;
            let local_outputs = function(body_context, local_inputs);
            validate_output_structure(&output_paths, &local_outputs)?;
            output_structure.replace(Some(local_outputs.parameter_structure()));
            Ok(local_outputs.into_parameters().collect())
        },
        global_input_types.into_parameters().collect(),
        outer_named_axes,
    )?;
    let output_structure = output_structure.into_inner().ok_or_else(|| {
        ProgramError::MalformedProgram(format!(
            "`{SHARD_MAP_OPERATION_NAME}` tracing completed without recording its output structure",
        ))
    })?;
    Ok(TracedShardMap {
        operation: traced.operation,
        global_input_types: Input::from_parameters(input_structure.clone(), traced.global_input_types)?,
        local_input_types: Input::from_parameters(input_structure, traced.local_input_types)?,
        global_output_types: Output::from_parameters(output_structure.clone(), traced.global_output_types)?,
        local_output_types: Output::from_parameters(output_structure, traced.local_output_types)?,
        body: traced.body,
    })
}

/// Traced `shard_map` body together with its checked boundary, as returned by [`trace_shard_map`],
/// [`trace_shard_map_with_options`], and [`trace_shard_map_with_named_axes`]. It owns the [`ShardMapOperation`] that
/// carries the boundary metadata, the structured global and local boundary types, and the traced local body over the
/// staged constant and operation families of the [`Domain`] `C`, which is the region that the operation attaches when
/// it is bound. Backends lower it through their own lowering functions.
pub struct TracedShardMap<C: Domain, Input, Output> {
    /// Checked shard-map boundary of the traced body, which owns its manual SPMD metadata.
    operation: ShardMapOperation,

    /// Global input types of the boundary, normalized from the caller's input types.
    global_input_types: Input,

    /// Local input types of the traced body.
    local_input_types: Input,

    /// Global output types of the boundary, derived from the local output types through the output shardings.
    global_output_types: Output,

    /// Local output types of the traced body.
    local_output_types: Output,

    /// Simplified traced local body, flattened to the attachment boundary of [`Self::operation`].
    body: FlatProgram<C>,
}

impl<C: Domain, Input, Output> TracedShardMap<C, Input, Output> {
    /// Returns the checked [`ShardMapOperation`] of the traced body, whose [`ShardMapOperation::shard_map`] holds the
    /// mesh, the boundary shardings, and the active manual axes.
    #[inline]
    pub fn operation(&self) -> &ShardMapOperation {
        &self.operation
    }

    /// Returns the global input types of the boundary: the caller's input types under their input shardings, without
    /// memory kinds (refer to the documentation of [`shard_map_in_context`] for details).
    #[inline]
    pub fn global_input_types(&self) -> &Input {
        &self.global_input_types
    }

    /// Returns the local input types of the traced body (i.e., the shards that each device receives).
    #[inline]
    pub fn local_input_types(&self) -> &Input {
        &self.local_input_types
    }

    /// Returns the global output types of the boundary, derived from the local output types through the output
    /// shardings.
    #[inline]
    pub fn global_output_types(&self) -> &Output {
        &self.global_output_types
    }

    /// Returns the local output types of the traced body, including the manual variation that tiled outputs acquire.
    #[inline]
    pub fn local_output_types(&self) -> &Output {
        &self.local_output_types
    }

    /// Returns the simplified traced local body, flattened to the attachment boundary of [`Self::operation`].
    #[inline]
    pub fn body(&self) -> &FlatProgram<C> {
        &self.body
    }
}

/// Returns `true` when two shard-map boundary types agree apart from carried placement metadata: their element types,
/// shapes, and layouts are equal, they vary along exactly the same mesh axes (a type without a sharding varies along
/// none), and, whenever `expected` carries a sharding, they have the same unreduced and reduced axes (a type without a
/// sharding has none). Dimension shardings, memory kinds, and the reduction state of `actual` when `expected` carries
/// no sharding are tolerated.
///
/// Variation is compared over every mesh axis, including axes that are not manual axes of the mesh of `expected`,
/// because the local body types derive their variation from the declared types: an input that varies along an axis of
/// an enclosing manual region that its declared type omits would otherwise reach the body typed as invariant along it,
/// and the body would type its per-device results as replicated (JAX's `_shard_map_typecheck` likewise requires the
/// body inputs to be the shards of the actual input types).
fn shard_map_boundary_types_match(actual: &ArrayType, expected: &ArrayType) -> bool {
    let no_axes = BTreeSet::new();
    let actual_varying_axes = actual.sharding().map_or(&no_axes, Sharding::varying_manual_axes);
    let expected_varying_axes = expected.sharding().map_or(&no_axes, Sharding::varying_manual_axes);
    actual.data_type() == expected.data_type()
        && actual.shape() == expected.shape()
        && actual.layout() == expected.layout()
        && actual_varying_axes == expected_varying_axes
        && match (actual.sharding(), expected.sharding()) {
            (_, None) => true,
            (Some(actual), Some(expected)) => {
                actual.unreduced_axes() == expected.unreduced_axes() && actual.reduced_axes() == expected.reduced_axes()
            }
            (None, Some(expected)) => expected.unreduced_axes().is_empty() && expected.reduced_axes().is_empty(),
        }
}

/// Returns `true` when two meshes describe the same device mesh, i.e., when they have the same axes, with the same
/// names and sizes in the same order, irrespective of their axis types. The mesh of a `shard_map` designates the axes
/// that it makes manual through its axis types, so the mesh of a caller's value may differ from it in its axis types
/// only (e.g., in an axis that is `Explicit` for the caller and `Manual` for the `shard_map`).
fn same_device_mesh(left: &LogicalMesh, right: &LogicalMesh) -> bool {
    left.axes().len() == right.axes().len()
        && left
            .axes()
            .iter()
            .zip(right.axes())
            .all(|(left, right)| left.name() == right.name() && left.size() == right.size())
}

/// Returns the global boundary type that a shard-map input or output carries as an array: the type itself for an
/// array position and the global referent for a reference position. A dimension has no shard-map boundary type.
fn boundary_array_type(r#type: &ArrayIrType) -> Result<&ArrayType, TypeError> {
    match r#type {
        ArrayIrType::Reference(r#type) => Ok(r#type.referent()),
        r#type => <&ArrayType>::try_from(r#type),
    }
}

/// Validates the single attached shard-map body boundary and returns its interface.
fn shard_map_body_interface<T: Type>(
    region_interfaces: &[RegionInterface<T>],
    input_count: usize,
    output_count: usize,
) -> Result<&RegionInterface<T>, TypeError> {
    check_count!("region", region_interfaces, 1, TypeError);
    let interface = &region_interfaces[0];
    check_count!("body input", interface.input_types(), input_count, TypeError);
    check_count!("body output", interface.output_types(), output_count, TypeError);
    Ok(interface)
}

/// Returns the global type of the edge that carries residual `residual_index` of local type `local_type` across a
/// shard-map boundary, together with the sharding that places it there, such that the local shard of the edge is the
/// residual's [`promoted_residual_type`]. A residual that is replicated along every active manual axis crosses as
/// itself, under its local sharding. A residual that varies along some active manual axes has one distinct value per
/// device along those axes, so its leading dimension is tiled along them (in mesh order and ahead of the free axes that
/// already place that dimension), and the global edge concatenates the local values along that dimension. This is
/// JAX's residual specification `P(order_wrt_mesh(mesh, vma))` in `_shard_map_partial_eval` and
/// `_shard_map_linearize`, and, because the local shard is the residual itself, neither body converts a non-scalar
/// residual at the boundary (a varying scalar is promoted to a single-element vector first, as by JAX's
/// `_promote_scalar_residuals`). The edge keeps the variation of the residual along the manual axes of enclosing
/// manual regions, as well as its unreduced and reduced axes, layout, and memory kind.
///
/// # Errors
///
/// Returns [`ShardMapError::DynamicResidualNotSupported`] when `local_type` has a dynamic dimension, which would carry
/// a dimension identity defined inside the body across the static shard-map boundary (refer to the `# Static
/// Boundaries` section of the documentation of [`ShardMapOperation`]), [`ShardMapError::Overflow`] when the global
/// extent of the tiled dimension overflows `usize`, and [`ShardMapError::Sharding`] or [`ShardMapError::Program`] when
/// the global type or its sharding cannot be constructed.
fn residual_boundary(
    residual_index: usize,
    local_type: &ArrayType,
    shard_map: &ShardMap,
) -> Result<(ArrayType, Sharding), ShardMapError> {
    if let Some(dimension) = local_type
        .shape()
        .dimensions()
        .iter()
        .position(|dimension| !matches!(dimension, Dimension::Static(_)))
    {
        return Err(ShardMapError::DynamicResidualNotSupported { residual_index, dimension });
    }
    let axes = residual_manual_axes(local_type, shard_map);
    if axes.is_empty() {
        let local_sharding = local_type
            .sharding()
            .cloned()
            .unwrap_or_else(|| Sharding::replicated(shard_map.mesh().clone(), local_type.rank()));
        return Ok((local_type.clone(), local_sharding));
    }
    let boundary_type = promoted_residual_type(local_type, shard_map).map_err(ProgramError::from)?;
    let local_sharding = boundary_type.sharding().unwrap();
    let mut dimensions = local_sharding.dimensions().to_vec();
    dimensions[0] = match &dimensions[0] {
        ShardingDimension::Sharded(free_axes) => {
            ShardingDimension::Sharded(axes.iter().chain(free_axes).cloned().collect())
        }
        ShardingDimension::Replicated | ShardingDimension::Unconstrained => ShardingDimension::Sharded(axes.clone()),
    };
    let sharding = local_sharding.with_dimensions(dimensions)?.with_varying_manual_axes(
        local_sharding
            .varying_manual_axes()
            .iter()
            .filter(|axis| !shard_map.manual_axes().contains(axis))
            .cloned(),
    )?;
    let mut shape = boundary_type.static_shape().unwrap().dimensions().to_vec();
    shape[0] = axes.iter().try_fold(shape[0], |extent, axis| {
        extent
            .checked_mul(shard_map.mesh().axis_size(axis).unwrap())
            .ok_or_else(|| ShardMapError::Overflow {
                context: format!("computing the global extent of residual #{residual_index}"),
            })
    })?;
    let shape = Shape::new(shape.into_iter().map(Dimension::Static).collect());
    let global_type = boundary_type.with_shape(shape).with_sharding(sharding.clone())?;
    Ok((global_type, sharding))
}

/// Returns the active manual axes along which this residual's local value varies.
fn residual_manual_axes(local_type: &ArrayType, shard_map: &ShardMap) -> Vec<String> {
    shard_map
        .manual_axes()
        .iter()
        .filter(|axis| local_type.sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(*axis)))
        .cloned()
        .collect()
}

/// Returns the local type with which a residual of local type `local_type` crosses a shard-map boundary (refer to
/// [`residual_boundary`]): the residual's own type, except that a scalar that varies along some active manual axis
/// gains a leading dimension of size one, without changing its element or variation metadata, so that its global edge
/// has a dimension to tile along those axes.
fn promoted_residual_type(local_type: &ArrayType, shard_map: &ShardMap) -> Result<ArrayType, TypeError> {
    if local_type.rank() != 0 || residual_manual_axes(local_type, shard_map).is_empty() {
        return Ok(local_type.clone());
    }
    local_type.with_inserted_dimension(0, Dimension::Static(1))
}

/// Returns the local array type that carries a residual edge of local type `local_type` across a shard-map boundary,
/// whose positions must be arrays or references. An array residual crosses as itself. A first-class dimension residual
/// crosses as its integer scalar value, typed as varying over every active manual axis because dimension types record
/// no manual variation: each device then keeps the value that it computed instead of relying on all devices agreeing.
/// The edge records no variation along the axes of enclosing manual regions, which the operation cannot name in general
/// (e.g., a body can derive a dimension from the index of an enclosing axis that no boundary type mentions). That
/// omission loses no value: every device of an enclosing manual region runs its own instances of both maps, so the
/// edge never crosses devices along such an axis, and the value only redefines a dimension, whose type carries no
/// variation, so no collective or transposition rule consumes the edge's variation along such an axis.
/// [`reshape_program_boundary`] converts between the dimension and its edge on either side of the boundary.
fn residual_edge_type(local_type: &ArrayIrType, shard_map: &ShardMap) -> Result<ArrayType, ProgramError> {
    match local_type {
        ArrayIrType::Dimension(_) => {
            let sharding = Sharding::replicated(shard_map.mesh().clone(), 0)
                .with_varying_manual_axes(shard_map.manual_axes().iter().cloned())
                .map_err(TypeError::from)?;
            Ok(ArrayType::scalar(DIMENSION_DATA_TYPE).with_sharding(sharding).map_err(TypeError::from)?)
        }
        _ => Ok(<&ArrayType>::try_from(local_type)?.clone()),
    }
}

/// Rewraps a body's boundary with element-preserving conversions. Arrays only change their shape (e.g., between a
/// varying scalar residual and its [`promoted_residual_type`]), and reference positions must remain unchanged. A
/// first-class dimension leaves a body as its integer scalar value, placed on the mesh of the target edge and marked
/// varying over that edge's manual axes before it is promoted, and re-enters a body by recovering that scalar and
/// redefining the same dimension variable from it (refer to [`residual_edge_type`]).
fn reshape_program_boundary<V: Value<Type = ArrayIrType>, O>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    input_types: Vec<ArrayIrType>,
    output_types: Vec<ArrayIrType>,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    O: Operation<Type = ArrayIrType>
        + OperationProjection<
            ArrayType,
            Projected: From<ReshapeOperation> + From<BroadcastOperation> + From<ParallelVaryOperation>,
        > + From<DimensionToScalarOperation>
        + From<DimensionFromScalarOperation>,
{
    let mut builder = ProgramBuilder::new();
    let inputs = input_types.iter().cloned().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
    let bind = |builder: &mut ProgramBuilder<V, O>, operation: O, value| {
        Ok::<_, ProgramError>(builder.add_instruction(operation, Vec::new(), vec![value], None)?[0])
    };

    // Array conversions are homogeneous array operations, which every family lifts through its array member.
    let array = |operation: <O as OperationProjection<ArrayType>>::Projected| O::from(operation);
    let reshape_array = |builder: &mut ProgramBuilder<V, O>, value, target: &ArrayType| {
        let operation = ReshapeOperation::new(target.shape().clone()).with_output_sharding(target.sharding().cloned());
        bind(builder, array(operation.into()), value)
    };
    let reshape = |builder: &mut ProgramBuilder<V, O>, value, source: &ArrayIrType, target: &ArrayIrType| {
        if source == target {
            return Ok(value);
        }
        match (source, target) {
            (ArrayIrType::Dimension(_), ArrayIrType::Array(target)) => {
                let mut value = bind(builder, DimensionToScalarOperation.into(), value)?;
                if let Some(sharding) = target.sharding() {
                    let mesh_type = ArrayType::scalar(target.data_type())
                        .with_sharding(Sharding::replicated(sharding.mesh().clone(), 0))
                        .map_err(TypeError::from)?;
                    let operation = BroadcastOperation::new(mesh_type, Vec::new());
                    value = bind(builder, array(operation.into()), value)?;
                    for axis_name in sharding.varying_manual_axes() {
                        let operation = ParallelVaryOperation::new(axis_name.clone());
                        value = bind(builder, array(operation.into()), value)?;
                    }
                }
                if target.rank() == 0 { Ok(value) } else { reshape_array(builder, value, target) }
            }
            (ArrayIrType::Array(source), ArrayIrType::Dimension(target)) => {
                let mut value = value;
                if source.rank() != 0 {
                    let scalar_sharding = source
                        .sharding()
                        .map(|sharding| sharding.with_dimensions(Vec::new()))
                        .transpose()
                        .map_err(TypeError::from)?;
                    let scalar_type = ArrayType::scalar(source.data_type())
                        .with_memory(source.memory())
                        .with_sharding(scalar_sharding)
                        .map_err(TypeError::from)?;
                    value = reshape_array(builder, value, &scalar_type)?;
                }
                let operation = DimensionFromScalarOperation::new(target.variable().clone());
                bind(builder, operation.into(), value)
            }
            _ => reshape_array(builder, value, <&ArrayType>::try_from(target)?),
        }
    };
    let converted_inputs = inputs
        .iter()
        .copied()
        .zip(input_types.iter().zip(program.input_types()))
        .map(|(input, (source, target))| reshape(&mut builder, input, source, &target))
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = builder.splice_program(program, &converted_inputs)?;
    let outputs = outputs
        .into_iter()
        .zip(program.output_types().iter().zip(&output_types))
        .map(|(output, (source, target))| reshape(&mut builder, output, source, target))
        .collect::<Result<Vec<_>, _>>()?;
    builder.build(outputs, vec![Placeholder; inputs.len()], vec![Placeholder; output_types.len()])
}

/// Splits the `body` of a `shard_map` with mixed known and unknown inputs into a known-side `shard_map`, bound in
/// the known-side context, and a residual `shard_map`, emitted into the residual program (refer to the documentation
/// of the [`PartiallyEvaluatableOperation`] implementation of [`ShardMapOperation`] for the protocol), and returns the
/// reassembled outputs of `operation`. A split that would hoist no work or that would carry a dynamically shaped
/// residual across the boundary keeps `operation` whole instead.
fn partially_evaluate_split_shard_map<C, D: PartialEvaluationDriver<C>>(
    operation: &ShardMapOperation,
    context: &PartialEvaluationContext<C>,
    driver: &D,
    body: RegionRef<'_, C::Constant, C::Operation>,
    inputs: &[PartialEvaluationValue<C::Value>],
) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<ShardMapOperation>
                           + OperationProjection<
                ArrayType,
                Projected: From<ReshapeOperation> + From<BroadcastOperation> + From<ParallelVaryOperation>,
            > + From<DimensionToScalarOperation>
                           + From<DimensionFromScalarOperation>
                           + OperationPayloadProjection,
        >,
{
    let input_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();
    let partition = driver.partition_program(context, body, input_known.as_slice())?;
    // A trivial partition hoists no work, so keep the original boundary and let the default materialize the knowns
    // directly as residual feeders. A partition is trivial when it determines no known output and its known program
    // only converts known inputs between representations (i.e., reshapes them or redefines first-class dimensions from
    // their integer scalar values), because its known side could then only convert known inputs into residual edges,
    // which the residual side would read through the same free conversions. In particular, the tangent map of a
    // partitioned linearization (refer to `shard_map_bodies`) is bound in a partial-evaluation context in which its
    // residual edges are known, and its body starts by unpacking the varying scalar and first-class dimension residuals
    // from their edges (refer to `reshape_program_boundary`). Hoisting those unpacking conversions would only split off
    // a known map that unpacks and repacks the same edges. A known output keeps the split even without known
    // instructions, so that it stays known (as in JAX, which always binds the known `shard_map`).
    if partition.outputs().iter().all(|output| output.is_unknown())
        && partition.known_program().instructions().iter().all(|instruction| {
            let operation = instruction.operation();
            operation.projected_payload::<ReshapeOperation>().is_some()
                || operation.projected_payload::<DimensionFromScalarOperation>().is_some()
        })
    {
        return context.fold_or_residualize(operation.clone(), vec![body.to_program()], inputs);
    }
    // Feed the residual side directly from the known boundary inputs and known boundary outputs that residual edges
    // would merely repeat (JAX's `in_fwd` and `out_fwd`), so that the known `shard_map` returns each value once.
    let partition = partition.forward_residuals()?;
    let known_output_indices = partition
        .outputs()
        .iter()
        .enumerate()
        .filter_map(|(index, output)| output.is_known().then_some(index))
        .collect::<Vec<_>>();

    // Derive each residual edge's global boundary type and sharding from its local type. A dynamically shaped
    // residual cannot cross the boundary (refer to the `# Static Boundaries` section of the documentation of
    // `ShardMapOperation`), so a split that needs one keeps the original boundary whole, like a trivial partition.
    let mesh = operation.shard_map.mesh();
    let residual_edge_types = partition.known_program().output_types()[known_output_indices.len()..]
        .iter()
        .map(|edge_type| residual_edge_type(edge_type, &operation.shard_map))
        .collect::<Result<Vec<_>, _>>()?;
    if residual_edge_types.iter().any(|edge_type| edge_type.static_shape().is_none()) {
        return context.fold_or_residualize(operation.clone(), vec![body.to_program()], inputs);
    }
    let residual_edge_boundaries = residual_edge_types
        .iter()
        .enumerate()
        .map(|(index, edge_type)| residual_boundary(index, edge_type, &operation.shard_map))
        .collect::<Result<Vec<_>, _>>()?;

    // Gather the known-side boundary metadata: shardings and global types per original index, with the residual
    // edges appended.
    let known_global_input_types = partition
        .known_input_indices()
        .iter()
        .map(|&index| operation.input_types[index].clone())
        .collect::<Vec<_>>();
    let known_in_shardings = partition
        .known_input_indices()
        .iter()
        .map(|&index| operation.shard_map.in_shardings()[index].clone())
        .collect::<Vec<_>>();
    let mut known_global_output_types =
        known_output_indices.iter().map(|&index| operation.output_types[index].clone()).collect::<Vec<_>>();
    let mut known_out_shardings = known_output_indices
        .iter()
        .map(|&index| operation.shard_map.out_shardings()[index].clone())
        .collect::<Vec<_>>();
    for (global_type, sharding) in residual_edge_boundaries.iter() {
        known_global_output_types.push(global_type.clone().into());
        known_out_shardings.push(sharding.clone());
    }

    // Gather the staged-side boundary metadata: the unknown and forwarded known boundary inputs, the forwarded known
    // boundary outputs, and the residual edges, in residual input order, and the residual-owned outputs.
    let mut staged_global_input_types = Vec::with_capacity(partition.residual_inputs().len());
    let mut staged_in_shardings = Vec::with_capacity(partition.residual_inputs().len());
    for source in partition.residual_inputs().iter() {
        match *source {
            // An unknown or forwarded known boundary input crosses under its declared global type and input sharding.
            ResidualInputSource::UnknownInput(index) | ResidualInputSource::KnownInput(index) => {
                staged_global_input_types.push(operation.input_types[index].clone());
                staged_in_shardings.push(operation.shard_map.in_shardings()[index].clone());
            }
            // A forwarded known boundary output crosses under its declared global type, with its output sharding as
            // the input sharding, which places exactly the local value that the known body returns for it.
            ResidualInputSource::KnownOutput(index) => {
                let output_index = known_output_indices[index];
                staged_global_input_types.push(operation.output_types[output_index].clone());
                staged_in_shardings.push(operation.shard_map.out_shardings()[output_index].clone());
            }
            ResidualInputSource::ResidualEdge(edge) => {
                let (global_type, sharding) = &residual_edge_boundaries[edge];
                staged_global_input_types.push(global_type.clone().into());
                staged_in_shardings.push(sharding.clone());
            }
        }
    }
    let mut staged_global_output_types = Vec::new();
    let mut staged_out_shardings = Vec::new();
    for (index, output) in partition.outputs().iter().enumerate() {
        if output.is_unknown() {
            staged_global_output_types.push(operation.output_types[index].clone());
            staged_out_shardings.push(operation.shard_map.out_shardings()[index].clone());
        }
    }

    // Both bodies convert each residual edge between its local type and its local boundary type, which differ only for
    // varying scalars and for first-class dimensions (refer to `promoted_residual_type` and `residual_edge_type`).
    let edge_local_type = |r#type: &ArrayIrType| -> Result<ArrayIrType, ProgramError> {
        let edge_type = residual_edge_type(r#type, &operation.shard_map)?;
        Ok(promoted_residual_type(&edge_type, &operation.shard_map)?.into())
    };
    let known_program = partition.known_program().clone();
    let mut known_output_types = known_program.output_types();
    for r#type in &mut known_output_types[known_output_indices.len()..] {
        *r#type = edge_local_type(r#type)?;
    }
    let known_input_types = known_program.input_types();
    let known_body = reshape_program_boundary(&known_program, known_input_types, known_output_types)?;
    let residual_program = partition.residual_program().clone();
    let mut residual_input_types = residual_program.input_types();
    for (source, r#type) in partition.residual_inputs().iter().zip(&mut residual_input_types) {
        if matches!(source, ResidualInputSource::ResidualEdge(_)) {
            *r#type = edge_local_type(r#type)?;
        }
    }
    let residual_output_types = residual_program.output_types();
    let residual_body = reshape_program_boundary(&residual_program, residual_input_types, residual_output_types)?;

    // Bind the known-side `shard_map` into the enclosing known-side context, emit the residual `shard_map` over its
    // residual sources, and reassemble the original outputs.
    context.inline_partitioned_program(
        partition,
        inputs,
        |_known_program| {
            let known_shard_map = ShardMap::from_shardings(
                mesh.clone(),
                known_in_shardings,
                known_out_shardings,
                operation.shard_map.manual_axes().to_vec(),
            );
            let known_operation =
                ShardMapOperation::from_boundary(known_shard_map, known_global_input_types, known_global_output_types);
            (known_operation, vec![known_body])
        },
        |_residual_program| {
            let staged_shard_map = ShardMap::from_shardings(
                mesh.clone(),
                staged_in_shardings,
                staged_out_shardings,
                operation.shard_map.manual_axes().to_vec(),
            );
            let staged_operation = ShardMapOperation::from_boundary(
                staged_shard_map,
                staged_global_input_types,
                staged_global_output_types,
            );
            (staged_operation, vec![residual_body])
        },
    )
}

/// Source of one residual input of the tangent `shard_map` that [`shard_map_bodies`] derives. A residual that is a
/// primal body input or a primal body output needs no residual edge, and a residual that repeats another one shares
/// its tangent input, as under the residual forwarding (`in_fwd` and `out_fwd`) and deduplication of JAX's
/// `_shard_map_linearize`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ShardMapResidual {
    /// Residual that is primal body input `index` itself. The tangent `shard_map` receives it directly from the
    /// corresponding primal boundary input, under that input's global type and input sharding. This is the only way
    /// in which a reference residual crosses by identity, which includes a reference residual that aliases a reference
    /// input.
    Input(usize),

    /// Residual that is primal body output `index` itself. The tangent `shard_map` receives it from the corresponding
    /// primal boundary output, under that output's global type and output sharding.
    Output(usize),

    /// Residual computed inside the body, which the primal `shard_map` returns as its residual edge `index` (i.e., as
    /// its output that follows the original outputs and the preceding residual edges), tiled along the manual axes
    /// along which it varies (refer to [`residual_boundary`]).
    Edge(usize),

    /// Residual that is a reference allocated inside the body, which crosses as a snapshot of its final state, since a
    /// reference cannot be carried by a residual edge. The primal body reads the complete referent after its last
    /// instruction, and the primal `shard_map` returns that value as its residual edge `index` (refer to
    /// [`Self::Edge`]). The tangent body allocates a fresh reference from that edge before its first instruction and
    /// uses it in place of the primal reference. Aliases of one allocation share one edge.
    ///
    /// The tangent body therefore observes the final primal state, which is also the state that a tangent program
    /// observes through a reference residual outside `shard_map`, since it runs after the whole primal program (and
    /// an allocation made inside the body cannot escape it, so nothing else accesses it in between). Because the
    /// tangent body owns its reference, every invocation of the tangent `shard_map` (e.g., every application of a
    /// pushforward or pullback) starts from that state, which is how repeated-residual partitioning treats state that
    /// tangent work mutates (it allocates such state afresh for every residual call). This differs from the behavior
    /// outside `shard_map` only when an opaque tangent rule mutates the reference, for example, a custom backward rule
    /// that writes into a reference residual: outside `shard_map`, that write updates the primal allocation, so a
    /// later application observes it, while under `shard_map`, each application observes the final primal state.
    ReferenceSnapshot(usize),
}

/// Primal and tangent `shard_map` boundaries and bodies into which [`shard_map_bodies`] splits one shard-map body.
struct ShardMapBodies<C: Domain> {
    /// Primal boundary, which keeps every input and output of the source boundary and gains the residual edges as
    /// trailing outputs.
    primal_operation: ShardMapOperation,

    /// Primal body, which maps the local inputs to the local outputs followed by the local residual edges.
    primal_body: FlatProgram<C>,

    /// Tangent `shard_map`, or [`None`] when no output tangent is live and the tangent body has no observable effects,
    /// in which case there is nothing to bind and the primal boundary and body are the source ones.
    tangent: Option<ShardMapTangent<C>>,

    /// One entry per output recording whether the output has a live tangent, and therefore a tangent boundary slot.
    output_activity: Vec<bool>,
}

/// Tangent `shard_map` boundary and body that [`shard_map_bodies`] derives, together with the sources of its residual
/// inputs.
struct ShardMapTangent<C: Domain> {
    /// Tangent boundary, which consumes the active input tangents followed by the residuals and produces the live
    /// output tangents.
    operation: ShardMapOperation,

    /// Tangent body, which maps the active local input tangents followed by the local residuals to the live local
    /// output tangents.
    body: FlatProgram<C>,

    /// Source of each residual input of [`Self::operation`], in input order.
    residuals: Vec<ShardMapResidual>,
}

/// Fuse-linearizes a shard-map body capture-free under the inputs' activity mask into a primal body and a tangent
/// body that thread residuals as plain input edges across the shard-map boundary. Returns both bodies with their
/// boundary operations, the sources of the tangent body's residuals, and the output-activity mask used to reconstruct
/// the caller's output duals (refer to [`ShardMapBodies`]).
///
/// The borrowed `body` region is linearized once through `driver`, yielding a primal sub-program
/// `local_inputs -> [local_outputs..., local_residuals...]` and a tangent sub-program
/// `[active(local_input_tangents)..., local_residuals...] -> [local_output_tangents...]` together with the residual
/// count. The tangent sub-program keeps only the live output tangents, and the returned output-activity mask marks the
/// outputs that have one: an output whose tangent depends on no active input tangent (e.g., one computed only from
/// inputs with structurally zero tangents) has no tangent boundary slot and receives a structurally zero tangent, as
/// under JAX's `which_nz_out`. When no output tangent is live and the tangent sub-program has no observable effects
/// (e.g., when every active tangent only reaches integer outputs), no tangent `shard_map` is returned and the primal
/// one is the source boundary with its body.
///
/// Otherwise, each residual is classified by [`ShardMapResidual`]: a residual that is a primal body input or output
/// is fed to the tangent boundary from the corresponding primal boundary input or output, a residual that repeats
/// another one shares its residual input, and only the remaining residuals become residual edges of the primal
/// boundary, each tiled along the manual axes along which it varies by [`residual_boundary`] (so that its local shard
/// is the residual itself). A reference residual that denotes a reference input of the body (directly or through an
/// alias) is that input, which it forwards by identity. A reference residual allocated inside the body crosses as a
/// snapshot of its final state ([`ShardMapResidual::ReferenceSnapshot`]): the primal body reads the complete referent
/// after its last instruction into a residual edge, and the tangent body allocates a fresh reference from that edge
/// before its first instruction. Any other reference residual is rejected with
/// [`ShardMapError::ReferenceResidualNotSupported`]. The primal boundary keeps every input and gains the residual edges
/// as trailing outputs, and the tangent boundary consumes the active inputs' tangents (for an active reference input,
/// a tangent reference `ref<tangent(T)>` under the primal's input sharding) followed by the residuals. A reference
/// output that forwards an active reference input forwards that input's tangent reference in the tangent boundary, and
/// one that forwards an inactive reference input keeps its primal handle and has no tangent boundary slot. This is the
/// shard-map counterpart of the jitted-call rule, realizing `jvp(shard_map(f)) = shard_map(jvp f)` without introducing
/// symbolic captures.
///
/// # Parameters
///
///   - `operation`: Boundary metadata of the primal shard map being linearized.
///   - `driver`: Call-scoped access to the active differentiation machinery.
///   - `body`: Borrowed `body` region of that shard map.
///   - `activity`: One entry per input recording whether the tangent body receives a tangent at that position.
fn shard_map_bodies<C, D: DifferentiationDriver<C>>(
    operation: &ShardMapOperation,
    driver: &D,
    body: RegionRef<'_, C::Constant, C::Operation>,
    activity: &[bool],
) -> Result<ShardMapBodies<C>, DifferentiationError>
where
    C: Context<
            Type = ArrayIrType,
            Operation: OperationProjection<
                ArrayType,
                Projected: From<ReshapeOperation> + From<BroadcastOperation> + From<ParallelVaryOperation>,
            > + From<DimensionToScalarOperation>
                           + From<DimensionFromScalarOperation>
                           + From<ReferenceNewOperation<ArrayType, ArrayIrType>>
                           + From<ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>>,
        >,
{
    let output_count = operation.output_types.len();
    check_count!("input", activity, operation.input_types.len(), ProgramError);
    let input_indices = activity
        .iter()
        .enumerate()
        .filter_map(|(index, &active)| active.then_some(index))
        .collect::<Vec<_>>();
    let linearization = driver.linearize_program(body, &input_indices)?;
    let (primal_program, tangent_program, residual_count) = linearization.into_parts();
    // Linearization validates this partition: the tangent inputs precede the trailing primal residuals. Dependence
    // must start from that compact tangent-input prefix, not from the original primal input indices or the residuals.
    let tangent_input_count = tangent_program.input_count() - residual_count;

    // Map compact tangent outputs back to primal output order, excluding the primal program's trailing residuals.
    // Zero differential spaces and inactive reference outputs have no tangent slot and consume no dependence entry.
    // Every remaining output that depends on no tangent input is a structural zero, so it does not cross the boundary.
    let tangent_slots = primal_program.entry_region_ref().tangent_output_mask(&input_indices)?;
    let mut dependence = tangent_program.output_dependence(&(0..tangent_input_count).collect::<Vec<_>>())?.into_iter();
    let output_activity = tangent_slots[..output_count]
        .iter()
        .map(|&slot| slot && dependence.next().unwrap())
        .collect::<Vec<_>>();
    let live_tangent_slots = (0..output_count)
        .filter(|&index| tangent_slots[index])
        .enumerate()
        .filter_map(|(slot, index)| output_activity[index].then_some(slot))
        .collect::<Vec<_>>();
    // Output projection keeps the complete tangent/residual input boundary and instructions with observable effects.
    // When all outputs are live, consume the original program handle without rebuilding its boundary.
    let tangent_program = if live_tangent_slots.len() == tangent_program.output_count() {
        Arc::unwrap_or_clone(tangent_program)
    } else {
        tangent_program.with_outputs(&live_tangent_slots)?
    };

    // A tangent body without live outputs is linear and computes nothing observable unless it has effects (e.g., a
    // write into a tangent reference), so no tangent `shard_map` is bound, and the primal `shard_map` is the source
    // one, whose residuals nothing would consume.
    if !output_activity.contains(&true) && !tangent_program.effects().is_retained_when_unused() {
        return Ok(ShardMapBodies {
            primal_operation: operation.clone(),
            primal_body: body.to_program(),
            tangent: None,
            output_activity,
        });
    }

    // Classify the residuals (JAX's `in_fwd`, `out_fwd`, and residual deduplication). A residual that is a primal body
    // input or output already crosses the boundary as that input or output, so it is fed to the tangent boundary from
    // there instead of being carried by a residual edge, and a residual that repeats an earlier one shares its tangent
    // input. A reference residual preserves the identity of the primal reference rather than a snapshot of its state
    // (refer to the documentation of `Linearization::tangent`), so one that denotes a reference input (directly or
    // through an alias) crosses as that input, under its input sharding. A reference cannot be carried by a residual
    // edge, so one allocated inside the body crosses as a snapshot of its final state instead (refer to
    // `ShardMapResidual::ReferenceSnapshot`), with one edge per allocation. The reference analysis resolves constants,
    // so that any other reference residual (i.e., one that refers to state outside the body, which no valid body
    // does) is rejected precisely. That rejection is defensive: builders reject reference constants and the analysis
    // rejects unbound captures, so no checked linearization reaches it (refer to the documentation of
    // `ShardMapError::ReferenceResidualNotSupported`).
    let primal_input_ids = primal_program.input_ids();
    let primal_output_ids = primal_program.output_ids();
    let primal_output_types = primal_program.output_types();
    let mut residuals = Vec::new();
    let mut residual_slots = Vec::with_capacity(residual_count);
    let mut edge_positions = Vec::new();
    let mut edge_roots = Vec::new();
    let mut reference_analysis = None;
    for position in output_count..output_count + residual_count {
        let atom = primal_output_ids[position];
        let is_reference = primal_output_types[position].is_reference();
        let input_index = primal_input_ids.iter().position(|input| *input == atom);
        let output_index = primal_output_ids[..output_count]
            .iter()
            .position(|output| *output == atom)
            .filter(|&index| !is_reference && !operation.output_types[index].is_reference());
        let residual = match (input_index, output_index) {
            (Some(index), _) => ShardMapResidual::Input(index),
            (None, Some(index)) => ShardMapResidual::Output(index),
            (None, None) if is_reference => {
                if reference_analysis.is_none() {
                    let analysis = primal_program
                        .entry_region_ref()
                        .reference_analysis_with_configuration(Some(0), true, &[])
                        .map_err(ProgramError::from)?;
                    reference_analysis = Some(analysis);
                }
                match reference_analysis.as_ref().unwrap().output_roots()[position] {
                    Some(ReferenceRoot::RegionInput { input_index, .. }) => ShardMapResidual::Input(input_index),
                    Some(root @ ReferenceRoot::Allocation { .. }) => {
                        match edge_roots.iter().position(|edge_root| *edge_root == Some(root)) {
                            Some(edge) => ShardMapResidual::ReferenceSnapshot(edge),
                            None => {
                                edge_positions.push(position);
                                edge_roots.push(Some(root));
                                ShardMapResidual::ReferenceSnapshot(edge_positions.len() - 1)
                            }
                        }
                    }
                    _ => {
                        let residual_index = position - output_count;
                        let error = ShardMapError::ReferenceResidualNotSupported { residual_index };
                        return Err(ProgramError::from(error).into());
                    }
                }
            }
            (None, None) => match edge_positions.iter().position(|&edge| primal_output_ids[edge] == atom) {
                Some(edge) => ShardMapResidual::Edge(edge),
                None => {
                    edge_positions.push(position);
                    edge_roots.push(None);
                    ShardMapResidual::Edge(edge_positions.len() - 1)
                }
            },
        };
        let slot = residuals.iter().position(|candidate| *candidate == residual).unwrap_or_else(|| {
            residuals.push(residual);
            residuals.len() - 1
        });
        residual_slots.push(slot);
    }

    // The local types of the residual edges are authoritative and back their boundary types on both bodies. The edge of
    // a reference snapshot carries the referent of that reference.
    let shard_map = operation.shard_map();
    let mesh = shard_map.mesh();
    let mut edge_global_types = Vec::with_capacity(edge_positions.len());
    let mut edge_shardings = Vec::with_capacity(edge_positions.len());
    let mut edge_local_types = Vec::with_capacity(edge_positions.len());
    for &position in &edge_positions {
        let edge_type = match &primal_output_types[position] {
            ArrayIrType::Reference(reference) => reference.referent().clone(),
            r#type => residual_edge_type(r#type, shard_map)?,
        };
        let (global_type, sharding) =
            residual_boundary(position - output_count, &edge_type, shard_map).map_err(ProgramError::from)?;
        edge_global_types.push(ArrayIrType::Array(global_type));
        edge_shardings.push(sharding);
        edge_local_types.push(ArrayIrType::from(promoted_residual_type(&edge_type, shard_map)?));
    }

    // The primal body returns the original outputs followed by the residual edges only, each converted to its local
    // boundary type (which only promotes varying scalars and recovers the integer scalars of dimensions). The edge of a
    // reference snapshot is the complete referent, read after the last instruction of the body. A read leaves the
    // allocation usable, so the snapshot stays valid even when the read is later scheduled ahead of uses of the
    // reference that do not access it (e.g., passing it to an opaque call), while the declared read effect keeps it
    // ordered after every access.
    let primal_program = primal_program.with_outputs(&(0..output_count).chain(edge_positions).collect::<Vec<_>>())?;
    let primal_program = if edge_roots.iter().any(Option::is_some) {
        let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
        let inputs =
            primal_program.input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let mut outputs = builder.splice_program(&primal_program, inputs.as_slice())?;
        for (output, root) in outputs[output_count..].iter_mut().zip(&edge_roots) {
            if root.is_some() {
                let operation = ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
                let operation = C::Operation::from(operation);
                *output = builder.add_instruction(operation, Vec::new(), vec![*output], None)?[0];
            }
        }
        let output_count = outputs.len();
        builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
            outputs,
            vec![Placeholder; inputs.len()],
            vec![Placeholder; output_count],
        )?
    } else {
        primal_program
    };
    let mut primal_output_types = primal_program.output_types();
    primal_output_types[output_count..].clone_from_slice(&edge_local_types);
    let primal_input_types = primal_program.input_types();
    let primal_program = reshape_program_boundary(&primal_program, primal_input_types, primal_output_types)?;

    // The tangent body receives one residual input per distinct residual, in order of first use, so it is rebuilt over
    // those inputs whenever forwarding, deduplication, or a reference snapshot changed its residual inputs, and the
    // residual edges are converted back to their local types on entry. A reference snapshot enters as its referent,
    // from which the tangent body allocates the reference that it uses in place of the primal one before its first
    // instruction.
    let tangent_program = if residuals.iter().copied().eq((0..residual_count).map(ShardMapResidual::Edge)) {
        tangent_program
    } else {
        let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
        let tangent_input_types = tangent_program.input_types();
        let mut tangent_inputs = tangent_input_types[..tangent_input_count]
            .iter()
            .cloned()
            .map(|r#type| builder.add_input(r#type))
            .collect::<Vec<_>>();
        let mut residual_inputs = vec![None; residuals.len()];
        for (index, &slot) in residual_slots.iter().enumerate() {
            residual_inputs[slot].get_or_insert_with(|| {
                match (&residuals[slot], &tangent_input_types[tangent_input_count + index]) {
                    (ShardMapResidual::ReferenceSnapshot(_), ArrayIrType::Reference(reference)) => {
                        builder.add_input(ArrayIrType::Array(reference.referent().clone()))
                    }
                    (_, r#type) => builder.add_input(r#type.clone()),
                }
            });
        }
        let residual_inputs = residual_inputs
            .into_iter()
            .zip(&residuals)
            .map(|(input, residual)| match residual {
                ShardMapResidual::ReferenceSnapshot(_) => {
                    let operation = C::Operation::from(ReferenceNewOperation::<ArrayType, ArrayIrType>::new());
                    Ok(builder.add_instruction(operation, Vec::new(), vec![input.unwrap()], None)?[0])
                }
                _ => Ok(input.unwrap()),
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        tangent_inputs.extend(residual_slots.iter().map(|&slot| residual_inputs[slot]));
        let outputs = builder.splice_program(&tangent_program, tangent_inputs.as_slice())?;
        let output_count = outputs.len();
        builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
            outputs,
            vec![Placeholder; tangent_input_count + residuals.len()],
            vec![Placeholder; output_count],
        )?
    };
    let mut tangent_input_types = tangent_program.input_types();
    for (r#type, residual) in tangent_input_types[tangent_input_count..].iter_mut().zip(&residuals) {
        if let ShardMapResidual::Edge(edge) | ShardMapResidual::ReferenceSnapshot(edge) = residual {
            *r#type = edge_local_types[*edge].clone();
        }
    }
    let tangent_output_types = tangent_program.output_types();
    let tangent_program = reshape_program_boundary(&tangent_program, tangent_input_types, tangent_output_types)?;

    let primal_operation = ShardMapOperation::from_boundary(
        ShardMap::from_shardings(
            mesh.clone(),
            shard_map.in_shardings().to_vec(),
            shard_map.out_shardings().iter().cloned().chain(edge_shardings.iter().cloned()).collect(),
            shard_map.manual_axes().to_vec(),
        ),
        operation.input_types.clone(),
        operation.output_types.iter().cloned().chain(edge_global_types.iter().cloned()).collect::<Vec<_>>(),
    )
    .with_output_forwarding(
        operation.output_forwarding.iter().copied().chain(vec![None; edge_global_types.len()]).collect(),
    )
    .map_err(ProgramError::from)?;

    // The tangent boundary compacts the inputs by the activity mask, so a forwarded reference output must be remapped
    // onto the tangent position of the input it forwards. Each residual crosses under the global type and sharding of
    // the primal boundary input, primal boundary output, or residual edge that it comes from.
    let mut tangent_positions = vec![None; activity.len()];
    let mut tangent_in_shardings = Vec::with_capacity(activity.len() + residuals.len());
    let mut tangent_input_types = Vec::with_capacity(activity.len() + residuals.len());
    for (index, _) in activity.iter().enumerate().filter(|(_, active)| **active) {
        tangent_positions[index] = Some(tangent_input_types.len());
        tangent_in_shardings.push(shard_map.in_shardings()[index].clone());
        tangent_input_types.push(operation.input_types[index].tangent()?);
    }
    for residual in &residuals {
        let (sharding, r#type) = match *residual {
            ShardMapResidual::Input(index) => (&shard_map.in_shardings()[index], &operation.input_types[index]),
            ShardMapResidual::Output(index) => (&shard_map.out_shardings()[index], &operation.output_types[index]),
            ShardMapResidual::Edge(edge) | ShardMapResidual::ReferenceSnapshot(edge) => {
                (&edge_shardings[edge], &edge_global_types[edge])
            }
        };
        tangent_in_shardings.push(sharding.clone());
        tangent_input_types.push(r#type.clone());
    }
    let tangent_output_types = operation
        .output_types
        .iter()
        .zip(&output_activity)
        .filter(|(_, active)| **active)
        .map(|(r#type, _)| r#type.tangent())
        .collect::<Result<Vec<_>, _>>()?;
    let tangent_forwarding = operation
        .output_forwarding
        .iter()
        .zip(&output_activity)
        .filter(|(_, active)| **active)
        .map(|(forwarded, _)| forwarded.map(|index| tangent_positions[index].unwrap()))
        .collect();
    let tangent_out_shardings = shard_map
        .out_shardings()
        .iter()
        .zip(&output_activity)
        .filter(|(_, active)| **active)
        .map(|(sharding, _)| sharding.clone())
        .collect();
    let tangent_operation = ShardMapOperation::from_boundary(
        ShardMap::from_shardings(
            mesh.clone(),
            tangent_in_shardings,
            tangent_out_shardings,
            shard_map.manual_axes().to_vec(),
        ),
        tangent_input_types,
        tangent_output_types,
    )
    .with_output_forwarding(tangent_forwarding)
    .map_err(ProgramError::from)?;

    Ok(ShardMapBodies {
        primal_operation,
        primal_body: primal_program,
        tangent: Some(ShardMapTangent { operation: tangent_operation, body: tangent_program, residuals }),
        output_activity,
    })
}

/// Binds the fused forward-mode `shard_map` of `operation` in the shared primal and tangent context of `context` (refer
/// to the forward-mode rule of [`ShardMapOperation`]) and returns the output duals, or [`None`] when no output tangent
/// is live, in which case nothing is bound. The fused body is the fused JVP program of `body` with respect to the
/// inputs that `activity` marks as active, which maps `[inputs..., active(input_tangents)...]` to
/// `[outputs..., output_tangents...]`, projected onto the outputs followed by the live output tangents with its dead
/// work removed (refer to [`Program::with_outputs`] and [`Program::simplified`]): an output tangent is live when
/// it depends on some active input tangent (JAX's `which_nz_out`), and every other output receives a structurally zero
/// tangent. Each input tangent crosses the boundary under the input sharding of its primal, and each output tangent
/// under the output sharding of its primal, as in JAX's `_shard_map_jvp`.
fn fused_shard_map_jvp<C, D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
    operation: &ShardMapOperation,
    context: &DifferentiationContext<C, P>,
    driver: &D,
    body: RegionRef<'_, C::Constant, C::Operation>,
    inputs: &[DifferentiationDual<C::Value>],
    activity: &[bool],
) -> Result<Option<Vec<DifferentiationDual<C::Value>>>, DifferentiationError>
where
    C: Context<Type = ArrayIrType, Operation: From<ShardMapOperation>> + Zero<C::Value>,
{
    let input_count = inputs.len();
    let output_count = operation.output_types.len();
    let input_indices = activity
        .iter()
        .enumerate()
        .filter_map(|(index, &active)| active.then_some(index))
        .collect::<Vec<_>>();
    let tangent_slots = body.tangent_output_mask(&input_indices)?;
    let fused_body = driver.jvp_program(body, &input_indices)?;
    check_count!("input", fused_body.input_types(), input_count + input_indices.len(), ProgramError);
    let tangent_inputs = (input_count..fused_body.input_count()).collect::<Vec<_>>();
    let mut dependence = fused_body.output_dependence(&tangent_inputs)?.into_iter().skip(output_count);
    let output_activity = tangent_slots.iter().map(|&slot| slot && dependence.next().unwrap()).collect::<Vec<_>>();
    if !output_activity.contains(&true) {
        return Ok(None);
    }
    let live_tangent_outputs =
        (0..output_count).filter(|&index| tangent_slots[index]).map(|index| output_activity[index]);
    let live_outputs = std::iter::repeat_n(true, output_count).chain(live_tangent_outputs).collect::<Vec<_>>();

    // The boundary keeps every primal output followed by the live tangent slots, retaining all inputs to match the
    // input shardings below. Simplification also removes dead work inside nested regions (e.g., unused tangent
    // allocations passed only to custom functions), while preserving observable effects and deferred work.
    let output_indices = live_outputs
        .iter()
        .enumerate()
        .filter_map(|(index, &live)| live.then_some(index))
        .collect::<Vec<_>>();
    let fused_body = fused_body.with_outputs(&output_indices)?;
    let simplified_body = fused_body.simplified()?;

    // A no-op cleanup keeps the staged order and existing transform caches instead of adopting simplification's
    // output-driven instruction order. Removing instructions remains necessary even when every output is live.
    let instruction_count = fused_body.regions().iter().map(|region| region.instructions().len()).sum::<usize>();
    let fused_body = if simplified_body.regions().iter().map(|region| region.instructions().len()).sum::<usize>()
        < instruction_count
    {
        simplified_body
    } else {
        fused_body
    };

    let shard_map = operation.shard_map();
    let mut in_shardings = shard_map.in_shardings().to_vec();
    let mut input_types = operation.input_types.clone();
    let mut fused_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
    for &index in &input_indices {
        in_shardings.push(shard_map.in_shardings()[index].clone());
        input_types.push(operation.input_types[index].tangent()?);
        fused_inputs.push(inputs[index].tangent().clone().materialize(context.tangent())?);
    }
    let mut out_shardings = shard_map.out_shardings().to_vec();
    let mut output_types = operation.output_types.clone();
    for index in (0..output_count).filter(|&index| output_activity[index]) {
        out_shardings.push(shard_map.out_shardings()[index].clone());
        output_types.push(operation.output_types[index].tangent()?);
    }
    let fused_operation = ShardMapOperation::from_boundary(
        ShardMap::from_shardings(
            shard_map.mesh().clone(),
            in_shardings,
            out_shardings,
            shard_map.manual_axes().to_vec(),
        ),
        input_types,
        output_types,
    );
    let mut outputs = context.primal().bind(fused_operation, vec![fused_body], &fused_inputs)?;
    let live_output_count = live_outputs.iter().filter(|live| **live).count();
    check_count!("output", outputs, live_output_count, ProgramError);
    let mut tangent_outputs = outputs.split_off(output_count).into_iter();
    outputs
        .into_iter()
        .zip(output_activity)
        .map(|(primal, active)| {
            if active {
                DifferentiationDual::new(primal, tangent_outputs.next().unwrap())
            } else {
                DifferentiationDual::new_with_zero_tangent(primal)
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// Returns `true` when a shard-map body has a reference input or output or accesses a reference anywhere in its region
/// closure. The partial-evaluation rule of [`ShardMapOperation`] keeps the boundary of such a body whole, so its
/// forward-mode rule never fuses it.
fn body_has_references<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>>(
    body: RegionRef<'_, V, O>,
) -> bool {
    body.input_types().iter().any(Type::is_reference)
        || body.output_types().iter().any(Type::is_reference)
        || body.contains_reference_accesses_in_closure()
}

/// Returns `true` when a value of a shard-map body, including the values of its nested regions, has a dynamically
/// shaped array type. Such a value cannot cross the static boundary between the bodies of a split (refer to
/// [`residual_boundary`]), so the partial-evaluation rule of [`ShardMapOperation`] cannot split the body across it, and
/// its forward-mode rule never fuses such a body. The values of nested regions count as well, because splitting a
/// region operation (e.g., a `condition` whose branch computes `sin` of a dynamically shaped value) moves the residuals
/// of its regions to the body itself, and when those residuals are dynamically shaped, the region operation (or the
/// whole body) stays on the unknown side of the split, which would leave the primal outputs that it computes unknown.
fn body_has_dynamically_shaped_values<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>>(
    body: RegionRef<'_, V, O>,
) -> bool {
    body.region_ids_in_closure().into_iter().any(|region_id| {
        body.arena()[region_id.index()]
            .atoms()
            .iter()
            .any(|atom| matches!(atom.r#type().as_ref(), ArrayIrType::Array(r#type) if r#type.static_shape().is_none()))
    })
}

/// Transposes a tangent [`ShardMapOperation`] while retaining its manual region boundary.
///
/// The [`TranspositionDriver`] transposes the attached body under the inputs' linearity and cotangent-destination
/// masks. Known inputs supply residual values and may appear anywhere in the input boundary. The reverse boundary
/// dualizes the shardings and boundary types of every cotangent that crosses it, including the cotangent references
/// that it forwards for the `Reference`-kind destinations that are not accumulated locally. Reference outputs forward
/// input roots and therefore have no separate output-cotangent slot. Structurally zero output cotangents have no slot
/// either: like JAX's `_shard_map_transpose`, which binds the transposed map over the nonzero output cotangents only
/// and propagates the zero ones symbolically through the body's backward pass, the body is transposed projected onto
/// its other outputs (and its reference outputs), so zero cotangents are neither materialized as global zeros nor as
/// local ones, and an input that only those outputs depend on receives a structural zero.
///
/// The reverse contributions to a replicated destination of an input that the body does not mutate (i.e., of a value
/// input or of a read-only reference input) accumulate in fresh per-device references, which update the caller's
/// cotangent destination once outside the map. The cotangent reference of a replicated reference input that the body
/// mutates crosses the transposed boundary by identity instead, like that of a sharded one (refer to
/// [`local_accumulator_destinations`]). Bodies own replica aggregation through the adjoints of their variation
/// operations, so the boundary neither normalizes output seeds nor sums replicated input contributions. Returned
/// cotangents follow the original input order, and reference, known, and ignored inputs receive structural zeros, as do
/// the inputs whose cotangents the transposed body produces as structural zeros, which are not outputs of the
/// transposed map (as in JAX's `_shard_map_transpose`).
///
/// The transposed map assembles every input cotangent under the cotangent dual of its input sharding and in device
/// memory, while the forward boundary accepted each caller input in its own placement and memory kind. Every returned
/// cotangent, and every frozen accumulator before it is added into a replicated reference destination, is therefore
/// reconciled with the caller's cotangent type by [`reconcile_input_cotangent`]: `reshard` restores the caller's
/// placement over `Explicit` axes unless the caller also places a dimension along a manual axis, a placement-only
/// `broadcast` restores the remaining placement, and `transfer_to_memory` restores the caller's memory kind, which is
/// why the array projection of `O` must provide [`ReshardOperation`], [`BroadcastOperation`], and
/// [`TransferToMemoryOperation`]. The cotangent reference of every other `Reference`-kind destination crosses the
/// transposed boundary by identity instead, exactly like a reference input of the forward boundary, so it is neither
/// resharded nor transferred.
///
/// # Parameters
///
///   - `operation`: Primal tangent `shard_map` staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `driver`: Instruction-scoped access to the attached body region and its recursive transposition machinery.
///   - `inputs`: Per-input [`PartialValue`] knowledge, mirroring the body's global inputs one-to-one. The
///     [`Unknown`](PartialValue::Unknown) entries are the input tangents, and the [`Known`](PartialValue::Known)
///     entries carry the residual tracers the pullback reads.
///   - `outputs`: Symbolic cotangents for the tangent `shard_map`'s outputs.
///   - `cotangents`: Cotangent destinations of the inputs (refer to the documentation of
///     [`TranspositionContext::cotangent_destinations`]). The body is transposed with their destination kinds, so a
///     live (`Reference`-kind) reference input accumulates into its destination using the sharded or replicated policy
///     above. A dead (`Ignore`-kind) reference input has no slot in the transposed body.
fn transpose_primal_shard_map<V: Value<Type = ArrayIrType>, O, D: TranspositionDriver<V, O>>(
    operation: &ShardMapOperation,
    context: &mut TracingContext<V, O>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    cotangents: &CotangentDestinations<Tracer<TracingContext<V, O>>>,
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError>
where
    O: Operation<Type = ArrayIrType>
        + From<ShardMapOperation>
        + OperationProvider<ArrayIrType, ZeroOperation<ArrayIrType>, Operation = O>
        + OperationProjection<
            ArrayType,
            Projected: From<BroadcastOperation> + From<ReshardOperation> + From<TransferToMemoryOperation>,
        > + From<ReferenceNewOperation<ArrayType, ArrayIrType>>
        + From<ReferenceFreezeOperation<ArrayType, ArrayIrType>>
        + From<ReferenceAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>>,
{
    let input_linearity = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    check_count!("input", input_linearity, operation.input_types.len(), ProgramError);
    check_count!("input", cotangents.kinds(), inputs.len(), ProgramError);
    check_count!("output", outputs, operation.output_types.len(), ProgramError);
    let body = driver.region(0)?;
    operation.validate_reference_body(body)?;
    let local_accumulators = local_accumulator_destinations(operation, body, cotangents.kinds())?;

    // A shard_map with no live output cotangents and no live reference input is a zero linear map, so every input
    // cotangent is zero. A live reference input keeps the shard map live, because its accumulated state cotangent
    // flows through the transposed body even when no ordinary output cotangent does. A body with deferred work or
    // observable rule effects also keeps it live.
    if outputs.iter().all(MaybeZero::is_zero)
        && !cotangents.has_reference_state_destinations()
        && !body.must_transpose()
    {
        return inputs
            .iter()
            .map(|input| input.r#type().cotangent().map(MaybeZero::Zero).map_err(ProgramError::from))
            .collect();
    }

    // A reference output forwards a body input root whose state cotangent lives in that input's cotangent reference,
    // so it owns no cotangent slot. Every other output cotangent crosses the boundary unless it is a structural zero.
    let value_output_cotangents = outputs
        .iter()
        .zip(&operation.output_types)
        .filter_map(|(cotangent, output_type)| (!output_type.is_reference()).then_some(cotangent))
        .collect::<Vec<_>>();
    let zero_output_cotangents =
        value_output_cotangents.iter().map(|cotangent| cotangent.is_zero()).collect::<Vec<_>>();

    // Transpose the tangent body's flat program, projected onto its outputs whose cotangents are not structural zeros,
    // under the same per-input linearity mask and destination kinds. The transposed program maps
    // `[nonzero_output_cotangents..., cotangent_references..., known_input_values...]` to
    // `[linear_input_cotangents...]`, in body-input order on each side, where a live reference input's cotangent is its
    // cotangent reference itself and a dead reference input has no cotangent slot; re-wrap it as a transposed shard-map
    // boundary whose shardings are permuted to match.
    let TransposedShardMap { operation: transposed_operation, body: transposed_body, zero_input_cotangents } =
        transpose_shard_map_body(
            operation,
            driver,
            input_linearity.as_slice(),
            cotangents.kinds(),
            zero_output_cotangents.as_slice(),
            local_accumulators.as_slice(),
        )?;

    // Stage a fresh `shard_map` over the transposed body on `[nonzero_output_cotangents..., cotangent_references...,
    // known_input_values...]`. Its outputs are the linear-input cotangents.
    let mut transposed_inputs = value_output_cotangents
        .into_iter()
        .filter_map(|cotangent| cotangent.as_value().cloned())
        .collect::<Vec<_>>();
    let mut reference_destinations = cotangents.references().iter();
    for (index, kind) in cotangents.kinds().iter().enumerate() {
        if *kind == CotangentDestinationKind::Reference {
            let reference = reference_destinations.next().unwrap();
            if !local_accumulators[index] {
                transposed_inputs.push(reference.clone());
            }
        }
    }
    transposed_inputs.extend(inputs.iter().filter_map(PartialValue::as_known).cloned());

    // The transposed body is attached as the shared handle the driver returned, so repeated binds of one transposed
    // body intern by `Arc` identity instead of copying the program into the trace again. A transposed map without
    // outputs and without observable effects (i.e., one whose input cotangents are all structural zeros) is not staged.
    let input_cotangents =
        if transposed_operation.output_types.is_empty() && !transposed_body.effects().is_retained_when_unused() {
            Vec::new()
        } else {
            context.stage_operation(
                transposed_operation,
                CalleeRegionDriver::new(&[transposed_body]),
                transposed_inputs.as_slice(),
            )?
        };
    let output_count = (0..inputs.len())
        .filter(|&index| {
            input_linearity[index]
                && ((cotangents.returns_cotangent(index) && !zero_input_cotangents[index]) || local_accumulators[index])
        })
        .count();
    check_count!("output", input_cotangents, output_count, ProgramError);

    // Reassemble one cotangent per input: the known inputs and the linear inputs whose cotangents are structural zeros
    // carry structural zeros, while the other linear input tangents receive the transposed `shard_map`'s outputs in
    // body-input order. A live reference tangent's output is its cotangent reference, whose contents were accumulated
    // in place, and a dead one has no output, so every reference input receives a structural zero.
    let mut input_cotangents = input_cotangents.into_iter();
    let mut reference_destinations = cotangents.references().iter();
    input_linearity
        .iter()
        .zip(inputs)
        .enumerate()
        .map(|(index, (&linear, input))| match cotangents.kind(index) {
            CotangentDestinationKind::Return if linear && !zero_input_cotangents[index] => {
                let contribution = input_cotangents.next().unwrap();
                let target = input.r#type().cotangent()?;
                Ok(MaybeZero::Value(reconcile_input_cotangent(context, index, contribution, &target)?))
            }
            CotangentDestinationKind::Reference => {
                let destination = reference_destinations.next().unwrap();
                if cotangents.is_reference_input(index) || local_accumulators[index] {
                    let cotangent = input_cotangents.next().unwrap();
                    // The frozen local accumulator of a replicated destination leaves the map under the input sharding
                    // in device memory, like a returned cotangent, so it is reconciled with the referent type of the
                    // caller's destination before it is added into that destination once.
                    if local_accumulators[index] {
                        let destination_type = destination.r#type();
                        let referent = <&ReferenceType<ArrayType>>::try_from(destination_type.as_ref())?.referent();
                        let target = ArrayIrType::Array(referent.clone());
                        let cotangent = reconcile_input_cotangent(context, index, cotangent, &target)?;
                        context.bind(
                            ReferenceAddUpdateOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new(),
                            Vec::new(),
                            &[destination.clone(), cotangent],
                        )?;
                    }
                }
                input.r#type().cotangent().map(MaybeZero::Zero).map_err(ProgramError::from)
            }
            CotangentDestinationKind::Return | CotangentDestinationKind::Ignore => {
                input.r#type().cotangent().map(MaybeZero::Zero).map_err(ProgramError::from)
            }
        })
        .collect()
}

/// Returns one entry per input of the tangent `operation` recording whether its cotangent destination accumulates in a
/// fresh local reference that updates the caller's destination once outside the transposed map (refer to
/// [`transpose_primal_shard_map`]). That is the case for a [`Reference`](CotangentDestinationKind::Reference)-kind
/// destination of an input that is replicated along an active manual axis, unless the input is a reference that `body`
/// mutates. A local accumulator starts from zero, and its final value is added into the caller's destination once,
/// which is correct whenever the transposed body only adds into the cotangent of the input, as it does for a value
/// input or for a reference that the body only reads. It is not correct for a reference that the body mutates, whose
/// transposed mutations may also overwrite the incoming cotangent (e.g., the transpose of a write zeroes the cotangent
/// of the overwritten state), which an accumulator that starts from zero cannot express. The cotangent reference of
/// such an input crosses the transposed boundary by identity instead, like that of a sharded reference. This is sound
/// because a body mutates a replicated reference with invariant values at invariant indices only (refer to the
/// reference contract of the module documentation), so the transposed body updates every device's copy of the
/// destination identically.
fn local_accumulator_destinations<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>>(
    operation: &ShardMapOperation,
    body: RegionRef<'_, V, O>,
    destination_kinds: &[CotangentDestinationKind],
) -> Result<Vec<bool>, ProgramError> {
    check_count!("input", destination_kinds, operation.input_types.len(), ProgramError);
    let analysis = body.reference_analysis(0)?;
    Ok(destination_kinds
        .iter()
        .enumerate()
        .map(|(index, kind)| {
            let mutated = operation.input_types[index].is_reference()
                && analysis.is_mutated(ReferenceRoot::RegionInput { region: body.id(), input_index: index });
            *kind == CotangentDestinationKind::Reference
                && !operation.shard_map.input_replicated_manual_axes(index).is_empty()
                && !mutated
        })
        .collect())
}

/// Reconciles the cotangent `contribution` that a transposed `shard_map` assembled for input `input_index` with the
/// caller's cotangent type `target`, and returns the reconciled cotangent.
///
/// The transposed boundary assembles each input cotangent under the cotangent dual of its input sharding and in device
/// memory, because the forward boundary accepted the caller's input in place of a value of its declared global input
/// type: it implicitly placed the input by its input sharding instead of the caller's own placement (boundary
/// validation tolerates dimension shardings) and moved it into device memory (boundary types carry no memory kind).
/// This function undoes those implicit conversions in reverse order:
///
///   1. A placement difference over [`Explicit`](MeshAxisType::Explicit) mesh axes is a tracked sharding transition, so
///      it is undone with `reshard` to the caller's placement over those axes. The axis types of the caller's mesh
///      decide which axes are `Explicit` (the mesh of the assembled cotangent may differ from it in its axis types
///      only), [`Auto`](MeshAxisType::Auto) axes are untracked and therefore dropped from the target, and a caller
///      without a sharding is treated as replicated over the mesh. `reshard` cannot target manual axes, so this step
///      is skipped when the caller places a dimension along a manual axis of its mesh, and the next step restores the
///      complete placement instead. Resharding to the caller's placement with its manual axes projected away and then
///      re-placing the result along those manual axes would replicate the cotangent along them only to slice it again,
///      which compiles to an extra round trip of collectives, while one placement-only `broadcast` compiles to local
///      slicing.
///   2. The remaining placement difference (i.e., over manual or `Auto` axes, over `Explicit` axes when the caller also
///      places a dimension along a manual axis, or the absence of a sharding) is reconciled with a placement-only
///      `broadcast`, which preserves the manual variation and reduction state.
///   3. A memory kind difference is undone with `transfer_to_memory` into the caller's memory kind.
///
/// # Errors
///
/// Returns [`ShardMapError::CotangentTypeMismatch`] when `contribution` differs from `target` in more than its
/// placement and memory kind (i.e., in its element type, shape, layout, reference kind, mesh axes, manual variation, or
/// reduction state), which no well-formed boundary produces (refer to the documentation of that variant), and the
/// errors of staging the reconciling operations otherwise.
fn reconcile_input_cotangent<V: Value<Type = ArrayIrType>, O>(
    context: &TracingContext<V, O>,
    input_index: usize,
    mut contribution: Tracer<TracingContext<V, O>>,
    target: &ArrayIrType,
) -> Result<Tracer<TracingContext<V, O>>, ProgramError>
where
    O: Operation<Type = ArrayIrType>
        + OperationProjection<
            ArrayType,
            Projected: From<BroadcastOperation> + From<ReshardOperation> + From<TransferToMemoryOperation>,
        >,
{
    let actual = contribution.r#type().into_owned();
    if actual == *target {
        return Ok(contribution);
    }
    let mismatch =
        || ShardMapError::CotangentTypeMismatch { input_index, expected: target.clone(), actual: actual.clone() };
    let (ArrayIrType::Array(actual_array), ArrayIrType::Array(target_array)) = (&actual, target) else {
        return Err(mismatch().into());
    };
    let compatible = actual_array.data_type() == target_array.data_type()
        && actual_array.shape() == target_array.shape()
        && actual_array.layout() == target_array.layout()
        && match (actual_array.sharding(), target_array.sharding()) {
            (Some(actual), Some(target)) => {
                same_device_mesh(actual.mesh(), target.mesh())
                    && actual.varying_manual_axes() == target.varying_manual_axes()
                    && actual.unreduced_axes() == target.unreduced_axes()
                    && actual.reduced_axes() == target.reduced_axes()
            }
            (Some(sharding), None) | (None, Some(sharding)) => {
                sharding.varying_manual_axes().is_empty()
                    && sharding.unreduced_axes().is_empty()
                    && sharding.reduced_axes().is_empty()
            }
            (None, None) => true,
        };
    if !compatible {
        return Err(mismatch().into());
    }

    // Step 1: `reshard` to the caller's `Explicit`-axis placement, in the frame of the caller's mesh, unless the caller
    // places a dimension along a manual axis, in which case the `broadcast` of step 2 restores the complete placement.
    if let Some(mesh) = target_array.sharding().or(actual_array.sharding()).map(Sharding::mesh) {
        let placed_along_manual_axes = target_array.sharding().is_some_and(|sharding| {
            sharding.dimensions().iter().any(|dimension| match dimension {
                ShardingDimension::Sharded(axes) => {
                    axes.iter().any(|axis| mesh.axis_type(axis) == Some(MeshAxisType::Manual))
                }
                _ => false,
            })
        });
        let explicit_axes = |sharding: Option<&Sharding>| {
            (0..target_array.rank())
                .map(|dimension| match sharding.map(|sharding| &sharding.dimensions()[dimension]) {
                    Some(ShardingDimension::Sharded(axes)) => axes
                        .iter()
                        .filter(|axis| mesh.axis_type(axis) == Some(MeshAxisType::Explicit))
                        .cloned()
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                })
                .collect::<Vec<_>>()
        };
        if !placed_along_manual_axes && explicit_axes(actual_array.sharding()) != explicit_axes(target_array.sharding())
        {
            let target_placement = match target_array.sharding() {
                Some(sharding) => sharding
                    .without_auto_axes()
                    .without_manual_reduction_axes()
                    .with_varying_manual_axes(Vec::<String>::new())
                    .map_err(TypeError::from)?,
                None => Sharding::replicated(mesh.clone(), target_array.rank()),
            };
            let reshard =
                <O as OperationProjection<ArrayType>>::Projected::from(ReshardOperation::new(target_placement));
            contribution = context.bind(O::from(reshard), Vec::new(), &[contribution])?.remove(0);
        }
    }

    // Step 2: placement-only `broadcast` for the remaining placement difference, in the current memory kind.
    let memory = boundary_array_type(&contribution.r#type())?.memory();
    let placed_target = target_array.clone().with_memory(memory);
    if *contribution.r#type() != ArrayIrType::Array(placed_target.clone()) {
        let axes = (0..placed_target.rank()).collect();
        let broadcast =
            <O as OperationProjection<ArrayType>>::Projected::from(BroadcastOperation::new(placed_target, axes));
        contribution = context.bind(O::from(broadcast), Vec::new(), &[contribution])?.remove(0);
    }

    // Step 3: `transfer_to_memory` into the caller's memory kind.
    if memory != target_array.memory() {
        let transfer = TransferToMemoryOperation::new(target_array.memory());
        let transfer = <O as OperationProjection<ArrayType>>::Projected::from(transfer);
        contribution = context.bind(O::from(transfer), Vec::new(), &[contribution])?.remove(0);
    }
    Ok(contribution)
}

/// Transposed `shard_map` boundary and body that [`transpose_shard_map_body`] derives for
/// [`transpose_primal_shard_map`].
struct TransposedShardMap<V: Value, O> {
    /// Transposed boundary, whose outputs are the input cotangents that are not structural zeros.
    operation: ShardMapOperation,

    /// Transposed local body, attached as the shared handle that the transposition driver returned whenever it needs no
    /// rebuilding.
    body: Arc<Program<V, O, Vec<V>, Vec<V>>>,

    /// One entry per input of the source boundary recording whether its cotangent is a structural zero, which is then
    /// not an output of [`Self::operation`].
    zero_input_cotangents: Vec<bool>,
}

/// Transposes one tangent shard-map body into the reverse boundary and body consumed by
/// [`transpose_primal_shard_map`].
///
/// The tangent body's flat program is transposed with [`TranspositionDriver::transpose_program`] using the input
/// indices selected by `input_linearity` and their corresponding `destination_kinds`, producing a shared program
/// mapping `[nonzero_output_cotangents..., cotangent_references..., known_input_values...]` to
/// `[linear_input_cotangents...]`, where a `Reference`-kind input's cotangent is its cotangent reference forwarded by
/// identity and an `Ignore`-kind input has no cotangent slot. The transposed boundary permutes and dualizes the
/// original one to match: its global inputs are the cotangent descriptors of the original value outputs whose
/// cotangents are not structural zeros under the cotangent duals of their output shardings, then the cotangent
/// references `ref<cotangent(T)>` of the `Reference`-kind inputs whose destinations are not accumulated locally (refer
/// to [`local_accumulator_destinations`]) under the cotangent duals of those inputs' input shardings, then the known
/// inputs' original global inputs; its global outputs are the cotangent descriptors of the linear inputs' original
/// global inputs under the cotangent duals of their input shardings, except for the `Return`-kind input cotangents that
/// the transposed program produces as structural zeros, which the returned [`TransposedShardMap`] records instead.
///
/// When some value output cotangents are structural zeros, the body is first projected onto its other outputs and its
/// reference outputs with [`Program::with_outputs`], which keeps the whole input boundary and every instruction with
/// observable effects or deferred work. Zero output cotangents therefore never enter the transposed body, and the
/// transposition engine propagates their absence symbolically: an input that only those outputs depend on (e.g.,
/// through `neg` or `add`) is disconnected from the projected body, so the transposed program produces its cotangent as
/// a canonical `zero`, which is then a structural zero like any other. This is the analogue of JAX's
/// `_shard_map_transpose`, which runs the body's backward pass on symbolic zero output cotangents. A projected body is
/// a fresh program, so its transposition does not share the body region's retained transform cache.
///
/// # Parameters
///
///   - `operation`: Boundary metadata of the tangent `shard_map` produced by [`shard_map_bodies`], whose global
///     inputs are `[active(input_tangents)..., residuals...]` and whose global outputs are `[output_tangents...]`.
///   - `driver`: Instruction-scoped access to the attached body region and its recursive transposition machinery.
///   - `input_linearity`: Per-input linearity flags over the tangent boundary's global inputs.
///   - `destination_kinds`: Per-input cotangent destination kinds over the tangent boundary's global inputs.
///   - `zero_output_cotangents`: Per-output flags over the tangent boundary's non-reference outputs recording whether
///     the output cotangent is a structural zero, which then does not cross the transposed boundary, and whose output
///     is projected out of the body before it is transposed.
fn transpose_shard_map_body<V: Value<Type = ArrayIrType>, O, D: TranspositionDriver<V, O>>(
    operation: &ShardMapOperation,
    driver: &D,
    input_linearity: &[bool],
    destination_kinds: &[CotangentDestinationKind],
    zero_output_cotangents: &[bool],
    local_accumulators: &[bool],
) -> Result<TransposedShardMap<V, O>, ProgramError>
where
    O: Operation<Type = ArrayIrType>
        + OperationProvider<ArrayIrType, ZeroOperation<ArrayIrType>, Operation = O>
        + From<ReferenceNewOperation<ArrayType, ArrayIrType>>
        + From<ReferenceFreezeOperation<ArrayType, ArrayIrType>>,
{
    check_count!("input", input_linearity, operation.input_types.len(), ProgramError);
    check_count!("input", destination_kinds, operation.input_types.len(), ProgramError);
    check_count!("input", local_accumulators, operation.input_types.len(), ProgramError);
    // The driver takes destinations only for selected inputs; keep the full masks below to reconstruct shardings
    // in the original boundary's input order.
    let (input_indices, selected_destination_kinds): (Vec<_>, Vec<_>) = input_linearity
        .iter()
        .zip(destination_kinds)
        .enumerate()
        .filter_map(|(index, (linear, kind))| linear.then_some((index, *kind)))
        .unzip();
    let seed_count = operation.output_types.iter().filter(|r#type| !r#type.is_reference()).count();
    check_count!("output", zero_output_cotangents, seed_count, ProgramError);

    // Structurally zero output cotangents do not cross the transposed boundary, so the body is transposed projected
    // onto the other outputs. Reference outputs own no cotangent slot and are always kept, so that the projected body
    // forwards the same reference roots. Without zero output cotangents, the body region is transposed directly through
    // its retained transform cache, so that a body shared by several programs is transposed once per mask.
    let body = driver.region(0)?;
    let projected_body;
    let body = if zero_output_cotangents.contains(&true) {
        let mut zero_output_cotangents = zero_output_cotangents.iter();
        let kept_outputs = operation
            .output_types
            .iter()
            .enumerate()
            .filter(|(_, output_type)| output_type.is_reference() || !*zero_output_cotangents.next().unwrap())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        projected_body = body.to_program().with_outputs(&kept_outputs)?;
        projected_body.entry_region_ref()
    } else {
        body
    };
    let mut transposed_program = driver.transpose_program(body, &input_indices, &selected_destination_kinds)?;
    let nonzero_seed_count = zero_output_cotangents.iter().filter(|zero| !**zero).count();
    let shard_map = operation.shard_map();

    // The transposed program returns one output per linear input with a `Return` destination or with a `Reference`
    // destination of a reference input, in input order. A `Return` cotangent that it produces as a structural zero
    // (i.e., through a zero-producing instruction or as a zero constant) is not an output of the transposed boundary,
    // so the caller receives it as a symbolic zero rather than as materialized global zeros, as JAX's
    // `_shard_map_transpose` binds the transposed map over its nonzero input cotangents only (`left_specs_nz`).
    let region = transposed_program.entry_region_ref();
    let mut transposed_outputs = region.output_ids().iter();
    let zero_input_cotangents = (0..input_linearity.len())
        .map(|index| {
            let has_output = input_linearity[index]
                && (destination_kinds[index] == CotangentDestinationKind::Return
                    || (destination_kinds[index] == CotangentDestinationKind::Reference
                        && operation.input_types[index].is_reference()));
            if !has_output {
                return false;
            }
            let output = *transposed_outputs.next().unwrap();
            destination_kinds[index] == CotangentDestinationKind::Return
                && (region.atoms()[output.index()].as_constant().is_some_and(Value::is_zero)
                    || region.instructions().iter().any(|instruction| {
                        instruction
                            .outputs()
                            .iter()
                            .position(|candidate| *candidate == output)
                            .is_some_and(|output_index| instruction.operation().is_zero(output_index))
                    }))
        })
        .collect::<Vec<_>>();

    // The replicated destinations of inputs that the body does not mutate accumulate locally: each device accumulates
    // its contribution into fresh local state, and only the enclosing map updates the caller's destination once.
    // Existing destination contents are neither duplicated nor used as a per-device seed. The body's variation adjoints
    // already aggregate the contributions: an invariant primal read passed through `parallel_vary` transposes to the
    // mesh-form sum `parallel_reduce`, so the boundary performs no reduction and no output-seed normalization of its
    // own.
    if local_accumulators.contains(&true) || zero_input_cotangents.contains(&true) {
        let mut builder = ProgramBuilder::new();
        let local_destinations = destination_kinds
            .iter()
            .zip(local_accumulators)
            .filter_map(|(kind, local)| (*kind == CotangentDestinationKind::Reference).then_some(*local))
            .collect::<Vec<_>>();
        // The inputs that still cross the boundary are declared first, so that they keep their relative order and
        // precede the locally materialized accumulators in the rebuilt body.
        let input_types = transposed_program.input_types();
        let materialized = (0..input_types.len())
            .map(|position| {
                position
                    .checked_sub(nonzero_seed_count)
                    .and_then(|index| local_destinations.get(index))
                    .copied()
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        let mut body_inputs = input_types
            .iter()
            .zip(&materialized)
            .map(|(r#type, materialized)| (!materialized).then(|| builder.add_input(r#type.clone())))
            .collect::<Vec<_>>();
        let input_count = body_inputs.iter().flatten().count();
        for (position, r#type) in input_types.into_iter().enumerate() {
            if body_inputs[position].is_some() {
                continue;
            }
            // A fresh local accumulator starts from a zero of the destination referent's local type. The zero is a
            // non-differentiable constant, so it is created with the manual variation of that type directly rather
            // than through `parallel_vary`.
            let referent = <&ReferenceType<ArrayType>>::try_from(&r#type)?.referent().clone();
            let zero = O::provide(ZeroOperation::new(referent.into()), &[])?;
            let zero = builder.add_instruction(zero, Vec::new(), Vec::new(), None)?[0];
            let reference = ReferenceNewOperation::<ArrayType, ArrayIrType>::new();
            body_inputs[position] = Some(builder.add_instruction(reference, Vec::new(), vec![zero], None)?[0]);
        }
        let body_inputs = body_inputs.into_iter().flatten().collect::<Vec<_>>();
        let mut outputs = builder.splice_program(&transposed_program, &body_inputs)?.into_iter();
        let mut rebuilt_outputs = Vec::new();
        let mut destination_index = 0;
        for (index, linear) in input_linearity.iter().enumerate() {
            let destination = if destination_kinds[index] == CotangentDestinationKind::Reference {
                let destination = body_inputs[nonzero_seed_count + destination_index];
                destination_index += 1;
                Some(destination)
            } else {
                None
            };
            if !linear
                || destination_kinds[index] == CotangentDestinationKind::Ignore
                || (destination_kinds[index] == CotangentDestinationKind::Reference
                    && !operation.input_types[index].is_reference()
                    && !local_accumulators[index])
            {
                continue;
            }
            // Ordinary buffers produce no identity output. A replicated local buffer is nevertheless frozen and
            // returned, and the enclosing rule adds it into the caller's destination once outside the map, so retrieve
            // its local argument rather than consuming a body output.
            let mut output = if destination_kinds[index] == CotangentDestinationKind::Reference
                && !operation.input_types[index].is_reference()
            {
                destination.unwrap()
            } else {
                outputs.next().unwrap()
            };
            // A structurally zero cotangent is dropped, and simplification removes the zero that produced it.
            if zero_input_cotangents[index] {
                continue;
            }
            if local_accumulators[index] {
                let freeze = ReferenceFreezeOperation::<ArrayType, ArrayIrType>::new();
                output = builder.add_instruction(freeze, Vec::new(), vec![output], None)?[0];
            }
            rebuilt_outputs.push(output);
        }
        let output_count = rebuilt_outputs.len();
        // Simplification drops the zeros that produced the dropped structurally zero cotangents.
        transposed_program = Arc::new(
            builder
                .build(rebuilt_outputs, vec![Placeholder; input_count], vec![Placeholder; output_count])?
                .simplified()?,
        );
    }

    // The transposed inputs: the cotangents of the value outputs that are not structural zeros, the cotangent
    // references of the live `Reference`-kind inputs whose destinations are not accumulated locally, and the known
    // inputs, in that order.
    let mut in_shardings = Vec::new();
    let mut global_input_types = Vec::new();
    let mut zero_output_cotangents = zero_output_cotangents.iter();
    for (output_type, sharding) in operation.output_types.iter().zip(shard_map.out_shardings()) {
        if output_type.is_reference() || *zero_output_cotangents.next().unwrap() {
            continue;
        }
        in_shardings.push(sharding.cotangent());
        global_input_types.push(output_type.cotangent()?);
    }
    let mut destination_positions = vec![None; input_linearity.len()];
    for (index, kind) in destination_kinds.iter().enumerate() {
        if *kind != CotangentDestinationKind::Reference || local_accumulators[index] {
            continue;
        }
        destination_positions[index] = Some(global_input_types.len());
        in_shardings.push(shard_map.in_shardings()[index].cotangent());
        let cotangent_type = operation.input_types[index].cotangent()?;
        global_input_types.push(if operation.input_types[index].is_reference() {
            cotangent_type
        } else {
            ReferenceType::new(boundary_array_type(&cotangent_type)?.clone()).into()
        });
    }
    for (index, _) in input_linearity.iter().enumerate().filter(|(_, linear)| !**linear) {
        in_shardings.push(shard_map.in_shardings()[index].clone());
        global_input_types.push(operation.input_types[index].clone());
    }

    // The transposed outputs: one cotangent per linear input in input order, a value for a `Return` destination
    // whose cotangent is not a structural zero and the forwarded cotangent reference for a `Reference` destination; an
    // `Ignore` destination has no output.
    let mut out_shardings = Vec::new();
    let mut global_output_types = Vec::new();
    let mut output_forwarding = Vec::new();
    for (index, _) in input_linearity.iter().enumerate().filter(|(_, linear)| **linear) {
        match destination_kinds[index] {
            CotangentDestinationKind::Return if zero_input_cotangents[index] => {}
            CotangentDestinationKind::Return => {
                out_shardings.push(shard_map.in_shardings()[index].cotangent());
                global_output_types.push(operation.input_types[index].cotangent()?);
                output_forwarding.push(None);
            }
            CotangentDestinationKind::Reference
                if operation.input_types[index].is_reference() || local_accumulators[index] =>
            {
                out_shardings.push(shard_map.in_shardings()[index].cotangent());
                let cotangent_type = operation.input_types[index].cotangent()?;
                if local_accumulators[index] {
                    global_output_types.push(boundary_array_type(&cotangent_type)?.clone().into());
                    output_forwarding.push(None);
                } else {
                    global_output_types.push(cotangent_type);
                    output_forwarding.push(destination_positions[index]);
                }
            }
            CotangentDestinationKind::Reference | CotangentDestinationKind::Ignore => {}
        }
    }
    let shard_map = ShardMap::from_shardings(
        shard_map.mesh().clone(),
        in_shardings,
        out_shardings,
        shard_map.manual_axes().to_vec(),
    );
    let operation = ShardMapOperation::from_boundary(shard_map, global_input_types, global_output_types)
        .with_output_forwarding(output_forwarding)?;
    Ok(TransposedShardMap { operation, body: transposed_program, zero_input_cotangents })
}

/// Builds the [`ShardMap`] of a closure-traced `shard_map` that is bound in the named-axis scope `named_axes`
/// (innermost first, as returned by [`NamedAxes::named_axes`]), and returns it together with the enclosing bindings
/// that its body inherits. These are, first, the [`NamedAxis::Mesh`] bindings of `named_axes` whose names are axes of
/// `mesh`, in mesh order, which name the axes that enclosing manual regions already made manual (JAX's
/// `mesh.manual_axes`), so the boundary is validated against them through [`ShardMap::new_within`], and the map can
/// make none of them manual again, followed by the [`NamedAxis::Batched`] bindings of `named_axes` whose names are not
/// axes of `mesh`, in scope order, which name the axes of enclosing `batch` levels, so that collectives inside the body
/// can refer to them and the batching rule of the level that binds such a name consumes them. Every other binding
/// (i.e., a mesh binding whose name is not an axis of `mesh`, or a batched binding named like an axis of `mesh`) is not
/// visible to the body. When several bindings share a name, the first one is used, which is the one that
/// [`NamedAxes::named_axis`] resolves. A mesh binding whose name is an axis of `mesh` but whose mesh is another device
/// mesh is rejected with [`ShardMapError::EnclosingManualAxisMeshMismatch`].
fn shard_map_within_named_axes(
    mesh: LogicalMesh,
    in_shardings: Vec<Sharding>,
    out_shardings: Vec<Sharding>,
    manual_axes: Vec<String>,
    named_axes: &[(String, NamedAxis)],
) -> Result<(ShardMap, Vec<(String, NamedAxis)>), ShardMapError> {
    let mut enclosing_named_axes = Vec::new();
    for axis in mesh.axes() {
        match named_axes.iter().find(|(name, _)| name == axis.name()) {
            Some((name, binding @ NamedAxis::Mesh { mesh: enclosing_mesh, .. })) => {
                if !same_device_mesh(enclosing_mesh, &mesh) {
                    return Err(ShardMapError::EnclosingManualAxisMeshMismatch {
                        axis_name: name.clone(),
                        enclosing_mesh: enclosing_mesh.clone(),
                        mesh: mesh.clone(),
                    });
                }
                enclosing_named_axes.push((name.clone(), binding.clone()));
            }
            Some((_, NamedAxis::Batched { .. })) | None => {}
        }
    }
    let enclosing_manual_axes = enclosing_named_axes.iter().map(|(name, _)| name.as_str()).collect::<HashSet<_>>();
    let shard_map = ShardMap::new_within(mesh, in_shardings, out_shardings, manual_axes, &enclosing_manual_axes)?;
    let mut visited_names = HashSet::new();
    for (name, binding) in named_axes {
        if visited_names.insert(name.as_str())
            && matches!(binding, NamedAxis::Batched { .. })
            && shard_map.mesh().axis_index(name).is_none()
        {
            enclosing_named_axes.push((name.clone(), binding.clone()));
        }
    }
    Ok((shard_map, enclosing_named_axes))
}

/// Traces `function` as the local body of `shard_map` for the flat caller input types `global_input_types` and returns
/// the flat [`TracedShardMap`] that the closure entry points ([`shard_map_with_options`], [`shard_map_in_context`],
/// and [`trace_shard_map_with_named_axes`]) restructure or bind. The caller types are normalized through
/// [`ShardMap::global_input_type`], `function` is traced in a fresh [`DomainTracingContext`] of `C` over the
/// [`ShardMapContext`] of that trace and the local shards of those types, with the trace seeded with
/// `outer_named_axes` shadowed by the active manual axes of `shard_map`, each output receives `parallel_vary` along
/// every active manual axis that its output sharding tiles and along which it does not vary yet, and the simplified
/// body is checked through [`ShardMapOperation::from_program`], which also derives the global output types.
///
/// # Parameters
///
///   - `shard_map`: Boundary metadata of the traced body.
///   - `function`: Flat body closure over the body context and the local shard-map values.
///   - `global_input_types`: Caller's global input types, in the order of the input shardings.
///   - `outer_named_axes`: Enclosing named-axis bindings that the body inherits.
fn trace_shard_map_body<C, F>(
    shard_map: ShardMap,
    function: F,
    global_input_types: Vec<ArrayType>,
    mut outer_named_axes: Vec<(String, NamedAxis)>,
) -> Result<TracedShardMap<C, Vec<ArrayType>, Vec<ArrayType>>, ShardMapError>
where
    C: Domain<Type = ArrayIrType>,
    ShardMapTracer<C>: ParallelVary,
    F: FnOnce(&ShardMapContext<C>, Vec<ShardMapTracer<C>>) -> Result<Vec<ShardMapTracer<C>>, ProgramError>,
{
    if global_input_types.len() != shard_map.in_shardings().len() {
        return Err(ShardMapError::InputTypeCountMismatch {
            expected: shard_map.in_shardings().len(),
            actual: global_input_types.len(),
        });
    }
    let global_input_types = global_input_types
        .iter()
        .enumerate()
        .map(|(index, input_type)| shard_map.global_input_type(index, input_type))
        .collect::<Result<Vec<_>, _>>()?;
    let local_input_types = global_input_types
        .iter()
        .enumerate()
        .map(|(index, input_type)| shard_map.local_input_type(index, input_type))
        .collect::<Result<Vec<_>, _>>()?;

    // The active manual axes of this map shadow the enclosing bindings of the same names.
    let mesh = shard_map.mesh();
    outer_named_axes.retain(|(name, _)| !shard_map.manual_axes().contains(name));
    outer_named_axes.extend(shard_map.manual_axes().iter().map(|name| {
        let axis = mesh.axis_index(name).unwrap();
        let size = mesh.axis_size(name).unwrap();
        (name.clone(), NamedAxis::Mesh { mesh: mesh.clone(), axis, size })
    }));

    let (local_output_types, body) = DomainTracingContext::<C>::trace_with_context_and_named_axes(
        |context: &DomainTracingContext<C>, local_inputs: Vec<DomainTracer<C>>| {
            let local_inputs = local_inputs
                .into_iter()
                .map(|input| ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from))
                .collect::<Result<Vec<_>, _>>()?;
            // Outputs beyond the output shardings are kept as they are, so that `from_program` reports the count
            // mismatch below.
            function(&ShardMapContext::<C>::new(context.clone()), local_inputs)?
                .into_iter()
                .enumerate()
                .map(|(index, mut output)| {
                    if let Some(sharding) = shard_map.out_shardings().get(index) {
                        for axis in ShardMap::sharded_manual_axes(sharding, &shard_map.manual_axis_names()) {
                            let output_sharding = output.r#type().sharding().cloned();
                            if !output_sharding.is_some_and(|sharding| sharding.varying_manual_axes().contains(&axis)) {
                                output = output.parallel_vary(&axis)?;
                            }
                        }
                    }
                    Ok(output.into_value())
                })
                .collect::<Result<Vec<_>, ProgramError>>()
        },
        local_input_types.iter().cloned().map(ArrayIrType::Array).collect::<Vec<_>>(),
        outer_named_axes,
    )?;
    let body = body.simplified()?;
    let operation = ShardMapOperation::from_program(
        &body,
        global_input_types.iter().cloned().map(ArrayIrType::Array).collect(),
        shard_map,
    )?;
    let local_output_types = local_output_types
        .iter()
        .map(|r#type| boundary_array_type(r#type).cloned())
        .collect::<Result<Vec<_>, _>>()
        .map_err(ProgramError::from)?;
    let global_output_types = operation
        .global_output_types()
        .iter()
        .map(|r#type| boundary_array_type(r#type).cloned())
        .collect::<Result<Vec<_>, _>>()
        .map_err(ProgramError::from)?;
    Ok(TracedShardMap {
        operation,
        global_input_types,
        local_input_types,
        global_output_types,
        local_output_types,
        body,
    })
}

/// Validates that the outputs that a `shard_map` body closure returns have the structure of its output shardings, whose
/// parameter paths are `output_paths`. Without this check, outputs with as many leaves as the output shardings but a
/// different structure would be silently restructured into the structure of the output shardings, pairing leaves with
/// the wrong output shardings.
///
/// # Errors
///
/// Returns [`ParameterError::MismatchedParameterStructures`], wrapped in [`ShardMapError::Parameter`], when the
/// parameter paths of `outputs` differ from `output_paths`.
fn validate_output_structure<P: Parameter, Output: Parameterized<P>>(
    output_paths: &[ParameterPath],
    outputs: &Output,
) -> Result<(), ShardMapError> {
    let paths = outputs.parameter_paths().collect::<Vec<_>>();
    if paths != output_paths {
        let render = |paths: &[ParameterPath]| {
            format!("[{}]", paths.iter().map(ParameterPath::to_string).collect::<Vec<_>>().join(", "))
        };
        return Err(ShardMapError::Parameter(ParameterError::MismatchedParameterStructures {
            left_structure: render(output_paths),
            right_structure: render(paths.as_slice()),
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayReference, ArrayReferenceTransformIndex,
        ArraySliceAxis, DataType, DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Memory, MeshAxis,
        RaggedAxis, TiledLayout,
    };
    use crate::axes::AxisError;
    use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingLevelExtent, batch};
    use crate::captures::{CaptureReference, CapturingContext};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiationRule, Linearization, NothingSavable, differentiate_at, rematerialize};
    use crate::operations::arithmetic::{MulOperation, NegOperation, SubOperation};
    use crate::operations::collectives::axis_index::{AxisIndex, AxisIndexOperation};
    use crate::operations::collectives::parallel_permute::ParallelPermuteOperation;
    use crate::operations::collectives::parallel_ragged_all_to_all::ParallelRaggedAllToAll;
    use crate::operations::collectives::parallel_reduce::ParallelReduce;
    use crate::operations::comparisons::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::one_like::OneLikeOperation;
    use crate::operations::constants::zero_like::ZeroLikeOperation;
    use crate::operations::control_flow::condition::ConditionOperation;
    use crate::operations::control_flow::r#while::WhileOperation;
    use crate::operations::custom_functions::functions::custom_function;
    use crate::operations::custom_functions::operations::CustomFunctionTransposeOperation;
    use crate::operations::debugging::{Print, PrintOperation};
    use crate::operations::differentiation::stop_gradient::StopGradient;
    use crate::operations::exponential::Exp;
    use crate::operations::manipulation::broadcasting::DynamicBroadcastOperation;
    use crate::operations::manipulation::conversions::ConvertElementTypeOperation;
    use crate::operations::manipulation::reshaping::Reshape;
    use crate::operations::reductions::{Reduce, ReduceOperation, ReductionKind};
    use crate::operations::references::{ReferenceNew, ReferenceRead, ReferenceWrite, ReferenceWriteOperation};
    use crate::operations::trigonometric::{Cos, CosOperation, Sin, SinOperation};
    use crate::partial::{
        PartialEvaluationInput, PartialEvaluationOutput, PartitionedProgram, ResidualPolicyReference,
    };
    use crate::programs::{
        AtomId, EmptyRegionDriver, InstructionId, ReferenceAnalysisError, ReferenceDischargeTarget, ReferenceSource,
        RegionDriver,
    };

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;

    /// Tracing context over the composite test family, whose tracers are the values that `shard_map` is invoked on.
    type TestContext = TracingContext<TestValue, TestOperation>;

    /// Array projection of a [`TestContext`] tracer. It is both the input leaf of `shard_map` invocations in a
    /// [`TestContext`] trace and the local value of bodies traced for [`TestContext`] or for the eager test domain.
    type TestTracer = ShardMapTracer<TestContext>;

    /// Eager context over the composite test family, which descriptor-only tracing names as the domain of its bodies.
    type TestEagerContext = EagerContext<TestValue, TestOperation>;

    // The transform tests below stage complete programs rather than using the `check_operation_*` macros, because those
    // macros cover regionless operations over homogeneous `Array` values, whose transformed results they compare with
    // eager interpretation, while `shard_map` attaches a body region over the composite family (whose interpretation
    // the `emulation` module tests). For the same reason, the derivatives are not compared with `check_gradient!`; the
    // XLA backend tests execute differentiated shard maps on devices instead.

    /// Test-only driver that returns a predetermined transpose for its one attached source region.
    struct TestTranspositionDriver {
        /// Source region exposed to the operation rule.
        source: TestProgram,

        /// Predetermined transposed program returned by the recursive request.
        transposed: TestProgram,
    }

    impl RegionDriver<TestValue, TestOperation> for TestTranspositionDriver {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, TestValue, TestOperation>>
        where
            TestValue: 'r,
            TestOperation: 'r,
        {
            std::iter::once(self.source.entry_region_ref())
        }
    }

    impl TranspositionDriver<TestValue, TestOperation> for TestTranspositionDriver {
        fn transpose_program(
            &self,
            _region: RegionRef<'_, TestValue, TestOperation>,
            _input_indices: &[usize],
            _destination_kinds: &[CotangentDestinationKind],
        ) -> Result<Arc<TestProgram>, DifferentiationError> {
            Ok(Arc::new(self.transposed.clone()))
        }
    }

    /// Test-only driver that returns a predetermined linearization for its one attached source region.
    struct TestDifferentiationDriver {
        /// Source region exposed to the operation rule.
        source: TestProgram,

        /// Predetermined primal program of the linearization returned by the recursive request.
        primal: TestProgram,

        /// Predetermined tangent program of the linearization returned by the recursive request.
        tangent: TestProgram,

        /// Number of residuals that the primal program returns after its outputs.
        residual_count: usize,
    }

    impl RegionDriver<TestValue, TestOperation> for TestDifferentiationDriver {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, TestValue, TestOperation>>
        where
            TestValue: 'r,
            TestOperation: 'r,
        {
            std::iter::once(self.source.entry_region_ref())
        }
    }

    impl DifferentiationDriver<TestContext> for TestDifferentiationDriver {
        fn jvp_program(
            &self,
            _region: RegionRef<'_, TestValue, TestOperation>,
            _input_indices: &[usize],
        ) -> Result<Arc<TestProgram>, DifferentiationError> {
            Err(ProgramError::MalformedProgram("the test driver only linearizes".to_string()).into())
        }

        fn linearize_program(
            &self,
            _region: RegionRef<'_, TestValue, TestOperation>,
            input_indices: &[usize],
        ) -> Result<Linearization<TestValue, TestOperation>, DifferentiationError> {
            Ok(Linearization::new_with_respect_to(
                self.primal.clone(),
                self.tangent.clone(),
                self.residual_count,
                input_indices,
            )?)
        }

        fn partition_jvp_program(
            &self,
            _region: RegionRef<'_, TestValue, TestOperation>,
            _input_known: &[bool],
            _required_known_outputs: &[usize],
        ) -> Result<PartitionedProgram<TestValue, TestOperation>, DifferentiationError> {
            Err(ProgramError::MalformedProgram("the test driver only linearizes".to_string()).into())
        }

        fn bind_jvp_operation<P: DifferentiationPolicy<TestContext>>(
            &self,
            _context: &DifferentiationContext<TestContext, P>,
            _operation: &TestOperation,
            _programs: Vec<TestProgram>,
            _inputs: &[DifferentiationDual<Tracer<TestContext>>],
        ) -> Result<Vec<DifferentiationDual<Tracer<TestContext>>>, DifferentiationError> {
            Err(ProgramError::MalformedProgram("the test driver only linearizes".to_string()).into())
        }
    }

    /// Scalar `f32[]` boundary type.
    fn f32_scalar_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    /// Global `f32[size]` boundary type.
    fn f32_vector_type(size: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(size)]))
    }

    /// Two-device manual mesh over `x`.
    fn manual_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    /// Rank-one sharding over [`manual_mesh`] that shards its only dimension along `x`.
    fn sharded_along_x() -> Sharding {
        Sharding::new(manual_mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap()
    }

    /// Four-device manual mesh over `x` and `y`.
    fn manual_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
    }

    /// Mesh with a manual `data` axis and an auto `model` axis.
    fn data_model_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("data", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("model", 4, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    /// Replicated single-input, single-output boundary over the two-device manual mesh.
    fn single_input_test_shard_map() -> ShardMap {
        let mesh = manual_mesh();
        ShardMap::from_shardings(
            mesh.clone(),
            vec![Sharding::replicated(mesh.clone(), 0)],
            vec![Sharding::replicated(mesh, 0)],
            vec!["x".to_string()],
        )
    }

    /// Builds the one-input, one-output identity body over `input_type`.
    fn identity_body(input_type: impl Into<ArrayIrType>) -> TestProgram {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(input_type.into());
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }

    /// Stages `operation` over its attached `body` as the one instruction of a flat program whose inputs carry the
    /// operation's global input types and whose outputs are the instruction's outputs.
    fn shard_map_program(operation: ShardMapOperation, body: TestProgram) -> Result<TestProgram, ProgramError> {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let inputs = operation.global_input_types().iter().cloned().map(|r#type| builder.add_input(r#type));
        let inputs = inputs.collect::<Vec<_>>();
        let body_region = builder.import_program(body);
        let outputs = builder
            .add_instruction(TestOperation::ShardMap(Box::new(operation)), vec![body_region], inputs.clone(), None)?
            .to_vec();
        let output_count = outputs.len();
        builder.build::<Vec<TestValue>, Vec<TestValue>>(
            outputs,
            vec![Placeholder; inputs.len()],
            vec![Placeholder; output_count],
        )
    }

    /// Builds an identity `shard_map` boundary and body over a replicated scalar on a two-manual-axis mesh, whose
    /// boundary differs from another such boundary only in the provided manual-axis selection.
    fn metadata_fingerprint_shard_map(manual_axes: &[&str]) -> (ShardMapOperation, TestProgram) {
        let mesh = manual_mesh_2x2();
        let shard_map = ShardMap::from_shardings(
            mesh.clone(),
            vec![Sharding::replicated(mesh.clone(), 0)],
            vec![Sharding::replicated(mesh, 0)],
            manual_axes.iter().map(|axis| axis.to_string()).collect(),
        );
        let operation = ShardMapOperation::from_boundary(shard_map, vec![f32_scalar_type()], vec![f32_scalar_type()]);
        (operation, identity_body(f32_scalar_type()))
    }

    /// Builds a program staging one replicated `shard_map` over `[a, x]` whose body computes
    /// `sum(sin(broadcast(a, extent)) * x)` for a first-class dimension `extent` derived from `a`. Splitting the body
    /// between `a` and `x` (by partial evaluation or differentiation) needs the dynamically shaped intermediate
    /// `sin(broadcast(a, extent))` as a residual.
    fn dynamic_residual_shard_map_program() -> TestProgram {
        let array_type = f32_scalar_type();
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(5)).unwrap());
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let known_input = builder.add_input(array_type.clone().into());
        let runtime_input = builder.add_input(array_type.clone().into());
        let count = builder
            .add_instruction(
                ArrayOperation::ConvertElementType(ConvertElementTypeOperation::new(DataType::I64, false)),
                Vec::new(),
                vec![known_input],
                None,
            )
            .unwrap()[0];
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![count], None)
            .unwrap()[0];
        let broadcast = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![known_input, dimension], None)
            .unwrap()[0];
        let sine = builder
            .add_instruction(ArrayOperation::Sin(SinOperation::new()), Vec::new(), vec![broadcast], None)
            .unwrap()[0];
        let product = builder
            .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![sine, runtime_input], None)
            .unwrap()[0];
        let sum = builder
            .add_instruction(
                ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Sum)),
                Vec::new(),
                vec![product],
                None,
            )
            .unwrap()[0];
        let body = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let operation = ShardMapOperation::from_program(
            &body,
            vec![array_type.clone().into(), array_type.into()],
            ShardMap::new(mesh, vec![replicated.clone(), replicated.clone()], vec![replicated], vec!["x".to_string()])
                .unwrap(),
        )
        .unwrap();
        shard_map_program(operation, body).unwrap()
    }

    /// Builds a replicated shard-map boundary over `[a, x]` whose body computes `(a + a, a * x, x + a)`, so that its
    /// first output depends only on the known input `a`.
    fn mixed_known_unknown_shard_map_body() -> (ShardMapOperation, TestProgram) {
        let array_type = f32_scalar_type();
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let shard_map = ShardMap::from_shardings(
            mesh,
            vec![replicated.clone(), replicated.clone()],
            vec![replicated.clone(), replicated.clone(), replicated],
            vec!["x".to_string()],
        );
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let known_input = builder.add_input(array_type.clone().into());
        let runtime_input = builder.add_input(array_type.clone().into());
        let doubled = builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![known_input, known_input], None)
            .unwrap()[0];
        let product = builder
            .add_instruction(
                ArrayOperation::Mul(MulOperation::new()),
                Vec::new(),
                vec![known_input, runtime_input],
                None,
            )
            .unwrap()[0];
        let sum = builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![runtime_input, known_input], None)
            .unwrap()[0];
        let body = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![doubled, product, sum],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();
        let operation = ShardMapOperation::from_boundary(
            shard_map,
            vec![array_type.clone(), array_type.clone()],
            vec![array_type.clone(), array_type.clone(), array_type],
        );
        (operation, body)
    }

    /// Reference-bearing shard-map fixture over the two-device manual mesh: the boundary is
    /// `[r: ref<reference_type>, x: f32[4]] -> f32[4]`, with the reference input sharded by `reference_sharding` and
    /// the value input and the output sharded along `x`, so every device owns a `f32[2]` shard of `x` and of the
    /// output. The body performs `add_update(r_local, x_local); read(r_local)` when `mutate` holds and
    /// `read(r_local) + x_local` otherwise, over the local shards, so the reference's local referent must be `f32[2]`
    /// (a `f32[4]` referent sharded along `x`, or a `f32[2]` referent replicated along `x`).
    fn reference_shard_map(
        reference_type: ArrayType,
        reference_sharding: Sharding,
        mutate: bool,
    ) -> (ShardMapOperation, TestProgram) {
        let sharded = sharded_along_x();
        let shard_map = ShardMap::from_shardings(
            manual_mesh(),
            vec![reference_sharding, sharded.clone()],
            vec![sharded],
            vec!["x".to_string()],
        );
        let local_reference_type = ReferenceType::new(shard_map.local_input_type(0, &reference_type).unwrap());
        let reference_varies = local_reference_type
            .referent()
            .sharding()
            .is_some_and(|sharding| sharding.varying_manual_axes().contains("x"));
        let local_value_type = shard_map.local_input_type(1, &f32_vector_type(4)).unwrap();
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(ArrayIrType::Reference(local_reference_type));
            let update = builder.add_input(ArrayIrType::Array(local_value_type));
            let output = if mutate {
                builder
                    .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                    .unwrap();
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0]
            } else {
                let state = builder
                    .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
                    .unwrap()[0];
                let state = if reference_varies {
                    state
                } else {
                    builder
                        .add_instruction(ParallelVaryOperation::new("x".to_string()), Vec::new(), vec![state], None)
                        .unwrap()[0]
                };
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![state, update], None).unwrap()[0]
            };
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let operation = ShardMapOperation::from_boundary(
            shard_map,
            vec![ArrayIrType::Reference(ReferenceType::new(reference_type)), ArrayIrType::Array(f32_vector_type(4))],
            vec![ArrayIrType::Array(f32_vector_type(4))],
        );
        (operation, body)
    }

    /// Checked reference-bearing boundary built through [`ShardMapOperation::from_program`] from the mutating body of
    /// [`reference_shard_map`] (with the reference input sharded along `x`), extended to also return its reference
    /// input, so that its second output forwards that input, together with that extended body.
    fn forwarding_reference_shard_map() -> (ShardMapOperation, TestProgram) {
        let sharded = sharded_along_x();
        let (_, body) = reference_shard_map(f32_vector_type(4), sharded.clone(), true);
        let forwarding_body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let inputs = body.input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
            let mut outputs = builder.splice_program(&body, inputs.as_slice()).unwrap();
            outputs.push(inputs[0]);
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let operation = ShardMapOperation::from_program(
            &forwarding_body,
            vec![ReferenceType::new(f32_vector_type(4)).into(), f32_vector_type(4).into()],
            ShardMap::new(manual_mesh(), vec![sharded.clone(); 2], vec![sharded; 2], Vec::new()).unwrap(),
        )
        .unwrap();
        (operation, forwarding_body)
    }

    /// Checked reference-bearing shard map over the two-device manual mesh whose body computes `read(r) * x` for a
    /// reference input `r: ref<f32[4]>` and a value input `x: f32[4]`, both sharded along `x`, together with that body.
    fn reference_product_shard_map() -> (ShardMapOperation, TestProgram) {
        let sharded = sharded_along_x();
        let shard_map = ShardMap::new(manual_mesh(), vec![sharded.clone(); 2], vec![sharded], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &f32_vector_type(4)).unwrap();
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(ReferenceType::new(local_type.clone()).into());
            let value = builder.add_input(local_type.into());
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let product = builder
                .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![state, value], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let global_input_types = vec![ReferenceType::new(f32_vector_type(4)).into(), f32_vector_type(4).into()];
        (ShardMapOperation::from_program(&body, global_input_types, shard_map).unwrap(), body)
    }

    /// Traces a two-input, two-output shard map over the two-device manual mesh for the eager test domain: the first
    /// input `f32[8]` is sharded along `x` and doubled, and the second input `f32[2]` is replicated and returned as is.
    fn traced_test_shard_map() -> TracedShardMap<TestEagerContext, Vec<ArrayType>, Vec<ArrayType>> {
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        trace_shard_map(
            |inputs: Vec<TestTracer>| vec![inputs[0].clone() + inputs[0].clone(), inputs[1].clone()],
            vec![f32_vector_type(8), f32_vector_type(2)],
            mesh,
            vec![sharded.clone(), replicated.clone()],
            vec![sharded, replicated],
        )
        .unwrap()
    }

    /// Traces `function` over [`TestTracer`]s of `input_types` in a fresh [`TestContext`] seeded with `named_axes`, and
    /// returns the traced program.
    fn trace_test_program(
        function: impl FnOnce(Vec<TestTracer>) -> Vec<TestTracer>,
        input_types: Vec<ArrayType>,
        named_axes: Vec<(String, NamedAxis)>,
    ) -> TestProgram {
        TestContext::trace_with_named_axes(
            |inputs: Vec<Tracer<TestContext>>| {
                let inputs = inputs
                    .into_iter()
                    .map(|input| ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(function(inputs).into_iter().map(ProjectedValue::into_value).collect::<Vec<_>>())
            },
            input_types.into_iter().map(ArrayIrType::Array).collect::<Vec<_>>(),
            named_axes,
        )
        .unwrap()
        .1
    }

    /// Builds the program without inputs that binds, in a fresh [`TestContext`], a `shard_map` without inputs over
    /// [`manual_mesh`] whose body reshapes the coordinate of the executing device along `x` to `u64[1]` and whose
    /// output sharding tiles it along `x`, so that its output is the global vector of device coordinates (JAX's
    /// `shard_map(lambda: axis_index('x'), in_specs=(), out_specs=P('x'))`).
    fn axis_index_shard_map_program() -> TestProgram {
        let context = TestContext::new();
        let output = shard_map_in_context(
            &context,
            |context: &ShardMapContext<TestContext>, ()| context.axis_index("x").unwrap().reshape([1]).unwrap(),
            (),
            manual_mesh(),
            (),
            sharded_along_x(),
            Vec::new(),
        )
        .unwrap();
        let builder = context.builder().borrow().clone();
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![output.value().atom_id().unwrap()],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap()
    }

    /// Returns the elements of the array `values`, panicking on non-array values.
    fn arrays_f64(values: Vec<TestValue>) -> Vec<Vec<f64>> {
        values
            .into_iter()
            .map(|value| match value {
                TestValue::Array(array) => array.to_f64s(),
                value => panic!("expected an array value but got `{value}`"),
            })
            .collect()
    }

    /// Returns the array reference transform that indexes axis 0 at a dynamic position.
    fn dynamic_index_transform() -> ArrayReferenceTransform {
        ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }
    }

    /// Adds the coordinate of the executing device along `x` of [`manual_mesh`] to `builder` and returns it.
    fn add_axis_index(builder: &mut ProgramBuilder<TestValue, TestOperation>) -> AtomId {
        let operation = ArrayOperation::AxisIndex(AxisIndexOperation::new("x".to_string()).with_mesh(manual_mesh()));
        builder.add_instruction(operation, Vec::new(), Vec::new(), None).unwrap()[0]
    }

    /// Builds the local body `r -> r[axis_index("x")]` over a reference input of the provided local referent type,
    /// which reads `r` at the coordinate of the executing device along `x`.
    fn varying_index_read_body(referent: ArrayType) -> TestProgram {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(ReferenceType::new(referent).into());
        let index = add_axis_index(&mut builder);
        let read = ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new()
            .with_transforms(vec![dynamic_index_transform()]);
        let value = builder.add_instruction(read, Vec::new(), vec![reference, index], None).unwrap()[0];
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }

    /// Builds the single-input program `x -> { r = reference_new(x); update(r); read(r) }` over `x: f32[2]`. When
    /// `mapped` is set, `update` mutates `r` in the body of a `shard_map` over [`manual_mesh`] without outputs that
    /// receives `r` replicated along `x`, and otherwise it mutates `r` directly. This lets tests compare the
    /// derivatives of a body that mutates a replicated reference with those of the same mutation outside `shard_map`.
    fn replicated_reference_update_program(
        mapped: bool,
        update: impl Fn(&mut ProgramBuilder<TestValue, TestOperation>, AtomId),
    ) -> TestProgram {
        let global_type = f32_vector_type(2);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(global_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        if mapped {
            let mesh = manual_mesh();
            let shard_map =
                ShardMap::new(mesh.clone(), vec![Sharding::replicated(mesh, 1)], Vec::new(), Vec::new()).unwrap();
            let local_type = shard_map.local_input_type(0, &global_type).unwrap();
            let body = {
                let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
                let reference = builder.add_input(ReferenceType::new(local_type).into());
                update(&mut builder, reference);
                builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap()
            };
            let operation =
                ShardMapOperation::from_program(&body, vec![ReferenceType::new(global_type).into()], shard_map)
                    .unwrap();
            let body = builder.import_program(body);
            let operation = TestOperation::ShardMap(Box::new(operation));
            builder.add_instruction(operation, vec![body], vec![reference], None).unwrap();
        } else {
            update(&mut builder, reference);
        }
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }

    /// Evaluates the forward-mode derivative of the single-input `program` at `input` along unit tangents on the
    /// reference backend and returns the elements of its outputs followed by those of its output tangents.
    fn evaluate_pushforward(program: &TestProgram, input: TestValue) -> Vec<Vec<f64>> {
        let TestValue::Array(array) = &input else {
            panic!("expected an array value");
        };
        let r#type = array.r#type().into_owned();
        let ones = vec![1.0f32; Array::element_count(&r#type).unwrap()];
        let tangent = TestValue::Array(Array::from_elements(r#type, &ones).unwrap());
        arrays_f64(program.jvp().unwrap().interpret(vec![input, tangent]).unwrap())
    }

    /// Builds the program `(x, w) -> shard_map(|x, w| sin(x) * w)` over `f32[8]` inputs and output sharded along `x`.
    fn sine_product_shard_map_program() -> TestProgram {
        let sharded = sharded_along_x();
        let global_type = f32_vector_type(8).with_sharding(sharded.clone()).unwrap();
        trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |(x, w): (TestTracer, TestTracer)| x.sin().unwrap() * w,
                        (inputs[0].clone(), inputs[1].clone()),
                        manual_mesh(),
                        (sharded.clone(), sharded.clone()),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![global_type.clone(), global_type],
            Vec::new(),
        )
    }

    /// Evaluates the linearization of `program` at `primals` along `tangents` on the reference backend and returns the
    /// elements of its primal outputs followed by the elements of its output tangents.
    fn evaluate_linearization(
        program: &TestProgram,
        primals: Vec<TestValue>,
        tangents: Vec<TestValue>,
    ) -> Vec<Vec<f64>> {
        let linearization = program.linearize().unwrap();
        let mut outputs = linearization.primal().interpret(primals).unwrap();
        let residuals = outputs.split_off(outputs.len() - linearization.residual_count());
        let mut tangent_inputs = tangents;
        tangent_inputs.extend(residuals);
        let output_tangents = linearization.tangent().interpret(tangent_inputs).unwrap();
        outputs
            .into_iter()
            .chain(output_tangents)
            .map(|value| {
                let ArrayIrValue::Array(array) = value else {
                    panic!("expected an array value");
                };
                array.to_f64s()
            })
            .collect()
    }

    /// Builds a program whose unbounded `while` loop runs over the states `[counter: f32[], x: f32[4]]` (with `x`
    /// sharded along `x`), replacing `x` by `sin(x)` and decrementing `counter` while `counter` is positive. When
    /// `through_shard_map` is `true`, the loop body computes `sin(x)` inside a `shard_map`, and otherwise directly.
    fn sine_while_program(through_shard_map: bool) -> TestProgram {
        let sharded = sharded_along_x();
        let counter_type = ArrayIrType::Array(f32_scalar_type());
        let state_type = ArrayIrType::Array(f32_vector_type(4).with_sharding(sharded.clone()).unwrap());
        let condition = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let counter = builder.add_input(counter_type.clone());
            builder.add_input(state_type.clone());
            let zero = builder
                .add_instruction(ArrayOperation::ZeroLike(ZeroLikeOperation::new()), Vec::new(), vec![counter], None)
                .unwrap()[0];
            let operation = ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::GreaterThan));
            let predicate = builder.add_instruction(operation, Vec::new(), vec![counter, zero], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let counter = builder.add_input(counter_type.clone());
            let x = builder.add_input(state_type.clone());
            let one = builder
                .add_instruction(ArrayOperation::OneLike(OneLikeOperation::new()), Vec::new(), vec![counter], None)
                .unwrap()[0];
            let operation = ArrayOperation::Sub(SubOperation::new());
            let next_counter = builder.add_instruction(operation, Vec::new(), vec![counter, one], None).unwrap()[0];
            let sine = if through_shard_map {
                let shard_map =
                    ShardMap::new(manual_mesh(), vec![sharded.clone()], vec![sharded.clone()], Vec::new()).unwrap();
                let global_type = <&ArrayType>::try_from(&state_type).unwrap().clone();
                let local_type = shard_map.local_input_type(0, &global_type).unwrap();
                let sine_body = {
                    let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
                    let input = builder.add_input(local_type.into());
                    let sine = builder
                        .add_instruction(ArrayOperation::Sin(SinOperation::new()), Vec::new(), vec![input], None)
                        .unwrap()[0];
                    builder
                        .build::<Vec<TestValue>, Vec<TestValue>>(vec![sine], vec![Placeholder], vec![Placeholder])
                        .unwrap()
                };
                let operation =
                    ShardMapOperation::from_program(&sine_body, vec![state_type.clone()], shard_map).unwrap();
                let sine_body = builder.import_program(sine_body);
                builder.add_instruction(operation, vec![sine_body], vec![x], None).unwrap()[0]
            } else {
                builder
                    .add_instruction(ArrayOperation::Sin(SinOperation::new()), Vec::new(), vec![x], None)
                    .unwrap()[0]
            };
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![next_counter, sine],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let inputs = vec![builder.add_input(counter_type), builder.add_input(state_type)];
        let condition = builder.import_program(condition);
        let body = builder.import_program(body);
        let outputs = builder
            .add_instruction(WhileOperation::new(), vec![condition, body], inputs, None)
            .unwrap()
            .to_vec();
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap()
    }

    /// Traces the single-input program `x -> function(reference_new(x), x)` over an input of type `global_type`. When
    /// `sharding` is provided, the function runs in the body of a `shard_map` over [`manual_mesh`] whose input and
    /// output shardings are `sharding`, so that the body allocates the reference, and otherwise it runs directly on the
    /// global input. This lets tests compare the derivatives of a body that allocates a reference with those of the
    /// same function outside `shard_map`.
    fn body_allocated_reference_program(
        global_type: ArrayType,
        sharding: Option<Sharding>,
        function: impl Fn(Tracer<TestContext>, Tracer<TestContext>) -> Tracer<TestContext>,
    ) -> TestProgram {
        let call = |x: TestTracer| {
            let x = x.into_value();
            let reference = x.reference_new().unwrap();
            ValueProjection::<ArrayType>::into_projected(function(reference, x)).unwrap()
        };
        trace_test_program(
            |inputs| match &sharding {
                Some(sharding) => {
                    vec![shard_map(call, inputs[0].clone(), manual_mesh(), sharding.clone(), sharding.clone()).unwrap()]
                }
                None => vec![call(inputs[0].clone())],
            },
            vec![global_type],
            Vec::new(),
        )
    }

    /// Evaluates the reverse-mode linearization of the single-input `program` at `input` on the reference backend, and
    /// returns the elements of its primal outputs followed by the elements of the input cotangent that each of
    /// `application_count` applications of its pullback to unit output seeds returns. All applications reuse the
    /// residuals of one evaluation of the primal program.
    fn evaluate_pullback(program: &TestProgram, input: TestValue, application_count: usize) -> Vec<Vec<f64>> {
        let linearization = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        let pullback = linearization.pullback().unwrap();
        let mut outputs = linearization.primal().interpret(vec![input]).unwrap();
        let residuals = outputs.split_off(outputs.len() - linearization.residual_count());
        let arrays = |values: Vec<TestValue>| {
            values
                .into_iter()
                .map(|value| {
                    let ArrayIrValue::Array(array) = value else {
                        panic!("expected an array value");
                    };
                    array
                })
                .collect::<Vec<_>>()
        };
        let outputs = arrays(outputs);
        let mut pullback_inputs = outputs
            .iter()
            .map(|output| {
                let r#type = output.r#type().into_owned();
                let ones = vec![1.0f32; Array::element_count(&r#type).unwrap()];
                TestValue::Array(Array::from_elements(r#type, &ones).unwrap())
            })
            .collect::<Vec<_>>();
        pullback_inputs.extend(residuals);
        let mut elements = outputs.iter().map(Array::to_f64s).collect::<Vec<_>>();
        for _ in 0..application_count {
            let cotangents = arrays(pullback.interpret(pullback_inputs.clone()).unwrap());
            elements.extend(cotangents.iter().map(Array::to_f64s));
        }
        elements
    }

    /// Returns `sin(x)` for the plumbing reference and input `(_, x)` of the custom functions of the tests below.
    fn custom_sine((_, x): (Tracer<TestContext>, Tracer<TestContext>)) -> Result<Tracer<TestContext>, ProgramError> {
        Ok(ValueProjection::<ArrayType>::into_projected(x)?.sin()?.into_value())
    }

    #[test]
    fn test_shard_map_error_from_program_error() {
        // A program error that holds a shard-map error converts back into that error, so a shard-map error that crosses
        // a program-error boundary (e.g., from a nested shard map) is not wrapped twice.
        let error = ShardMapError::MeshHasNoManualAxes;
        assert_eq!(ShardMapError::from(ProgramError::from(error.clone())), error);

        // Every other program error is wrapped.
        let error = ProgramError::MalformedProgram("malformed body".to_string());
        assert_eq!(ShardMapError::from(error.clone()), ShardMapError::Program(error));
        let error = ProgramError::custom(ShardingError::UnknownMeshAxisName { name: "x".to_string() });
        assert_eq!(ShardMapError::from(error.clone()), ShardMapError::Program(error));
    }

    #[test]
    fn test_shard_map_error_into_program_error() {
        // Logical errors become custom program errors that callers can recover, while wrapped program errors pass
        // through unchanged.
        let error = ProgramError::from(ShardMapError::MeshHasNoManualAxes);
        assert_eq!(error.downcast_custom::<ShardMapError>(), Some(&ShardMapError::MeshHasNoManualAxes));
        assert_eq!(error.to_string(), "`shard_map` requires at least one mesh axis with type `manual`");
        let error = ProgramError::MalformedProgram("malformed body".to_string());
        assert_eq!(ProgramError::from(ShardMapError::Program(error.clone())), error);
    }

    #[test]
    fn test_shard_map_new() {
        // An empty manual-axis selection activates every manual mesh axis, in mesh order.
        let mesh = manual_mesh_2x2();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let shard_map = ShardMap::new(mesh.clone(), vec![sharded.clone()], vec![sharded.clone()], Vec::new()).unwrap();
        assert_eq!(shard_map.mesh(), &mesh);
        assert_eq!(shard_map.manual_axes(), vec!["x".to_string(), "y".to_string()].as_slice());
        assert_eq!(shard_map.in_shardings(), &[sharded.clone()]);
        assert_eq!(shard_map.out_shardings(), &[sharded]);
        assert_eq!(shard_map.in_shardings()[0].replicated_axes(), vec!["y"]);
        assert_eq!(shard_map.out_shardings()[0].replicated_axes(), vec!["y"]);

        // Specifications are stored without their auto axes, which only backend compilers place.
        let mesh = data_model_mesh();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data", "model"])]).unwrap();
        let shard_map = ShardMap::new(mesh, vec![sharding.clone()], vec![sharding.clone()], Vec::new()).unwrap();
        assert_eq!(shard_map.manual_axes(), vec!["data".to_string()].as_slice());
        assert_eq!(shard_map.in_shardings(), &[sharding.without_auto_axes()]);
        assert_eq!(shard_map.out_shardings(), &[sharding.without_auto_axes()]);
    }

    #[test]
    fn test_shard_map_new_selects_manual_axis_subset() {
        // The local shard shrinks along the active manual axis `x` only, and the free axis `y` keeps placing it.
        let mesh = manual_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap()],
            Vec::new(),
            vec!["x".into()],
        )
        .unwrap();
        assert_eq!(shard_map.manual_axes(), vec!["x".to_string()].as_slice());
        assert_eq!(
            shard_map.local_input_type(0, &f32_vector_type(8)),
            Ok(f32_vector_type(4)
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["y"])])
                        .unwrap()
                        .with_varying_manual_axes(["x"])
                        .unwrap(),
                )
                .unwrap()),
        );

        // Only manual mesh axes are active by default, so auto axes never become manual axes of the body.
        let shard_map = ShardMap::new(
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
                MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            ])
            .unwrap(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(shard_map.manual_axes(), vec!["y".to_string()].as_slice());
    }

    #[test]
    fn test_shard_map_new_rejects_mesh_without_manual_axes() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let result = ShardMap::new(mesh, Vec::new(), Vec::new(), Vec::new());
        assert_eq!(result, Err(ShardMapError::MeshHasNoManualAxes));
        assert_eq!(result.unwrap_err().to_string(), "`shard_map` requires at least one mesh axis with type `manual`");
    }

    #[test]
    fn test_shard_map_new_rejects_unknown_manual_axes() {
        let result = ShardMap::new(manual_mesh(), Vec::new(), Vec::new(), vec!["y".to_string()]);
        assert_eq!(result, Err(ShardMapError::Sharding(ShardingError::UnknownMeshAxisName { name: "y".to_string() })));
        assert_eq!(result.unwrap_err().to_string(), "unknown mesh axis name: `y`");
    }

    #[test]
    fn test_shard_map_new_rejects_non_manual_axes() {
        let result = ShardMap::new(data_model_mesh(), Vec::new(), Vec::new(), vec!["model".to_string()]);
        assert_eq!(
            result,
            Err(ShardMapError::Sharding(ShardingError::ExpectedManualMeshAxis { name: "model".to_string() })),
        );
        assert_eq!(result.unwrap_err().to_string(), "mesh axis `model` must have type manual");
    }

    #[test]
    fn test_shard_map_new_rejects_specifications_over_other_meshes() {
        // Input and output specifications must both be defined over the mesh of the manual computation.
        let mesh = manual_mesh();
        let other_mesh = manual_mesh_2x2();
        let other_sharding = Sharding::replicated(other_mesh.clone(), 1);
        let mismatch =
            ShardMapError::Sharding(ShardingError::MeshMismatch { expected: mesh.clone(), actual: other_mesh.clone() });
        assert_eq!(
            ShardMap::new(mesh.clone(), vec![other_sharding.clone()], Vec::new(), Vec::new()),
            Err(mismatch.clone()),
        );
        assert_eq!(ShardMap::new(mesh.clone(), Vec::new(), vec![other_sharding], Vec::new()), Err(mismatch.clone()));
        assert_eq!(mismatch.to_string(), format!("mesh mismatch; expected `{mesh:?}` but got `{other_mesh:?}`"));
    }

    #[test]
    fn test_shard_map_new_rejects_free_axis_before_manual_axis() {
        let mesh = data_model_mesh();
        let result = ShardMap::new(
            mesh.clone(),
            vec![Sharding::new(mesh, vec![ShardingDimension::sharded(["model", "data"])]).unwrap()],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(
            result,
            Err(ShardMapError::ManualAxisMustPrecedeFreeAxis {
                value_kind: "input",
                value_index: 0,
                dimension: 0,
                free_axis_name: "model".to_string(),
                manual_axis_name: "data".to_string(),
            }),
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "input sharding #0 dimension #0 uses free axis `model` more major than manual axis `data`",
        );
    }

    #[test]
    fn test_shard_map_new_strips_varying_manual_axes() {
        // A boundary sharding describes a placement, so the manual variation that a sharding taken from a value's type
        // carries (here, along the axis `x` of an enclosing manual region) is not part of it.
        let mesh = manual_mesh_2x2();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let varying = sharded.clone().with_varying_manual_axes(["x"]).unwrap();
        let shard_map = ShardMap::new(mesh, vec![varying.clone()], vec![varying], vec!["y".to_string()]).unwrap();
        assert_eq!(shard_map.in_shardings(), &[sharded.clone()]);
        assert_eq!(shard_map.out_shardings(), &[sharded]);
    }

    #[test]
    fn test_shard_map_mesh() {
        let mesh = manual_mesh_2x2();
        let shard_map = ShardMap::new(mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        assert_eq!(shard_map.mesh(), &mesh);
    }

    #[test]
    fn test_shard_map_in_shardings() {
        // The input shardings keep their input order and are stored in their type-level form, without auto axes.
        let mesh = data_model_mesh();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data", "model"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let shard_map =
            ShardMap::new(mesh.clone(), vec![sharded.clone(), replicated.clone()], Vec::new(), Vec::new()).unwrap();
        let expected = Sharding::new(mesh, vec![ShardingDimension::sharded(["data"])]).unwrap();
        assert_eq!(shard_map.in_shardings(), &[expected, replicated]);
    }

    #[test]
    fn test_shard_map_out_shardings() {
        // The output shardings keep their output order and are stored in their type-level form, without auto axes.
        let mesh = data_model_mesh();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data", "model"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let shard_map =
            ShardMap::new(mesh.clone(), Vec::new(), vec![replicated.clone(), sharded.clone()], Vec::new()).unwrap();
        let expected = Sharding::new(mesh, vec![ShardingDimension::sharded(["data"])]).unwrap();
        assert_eq!(shard_map.out_shardings(), &[replicated, expected]);
    }

    #[test]
    fn test_shard_map_manual_axes() {
        // The active manual axes are stored in mesh order, whatever the order of the selection.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let shard_map =
            ShardMap::new(mesh.clone(), Vec::new(), Vec::new(), vec!["z".to_string(), "x".to_string()]).unwrap();
        assert_eq!(shard_map.manual_axes(), &["x".to_string(), "z".to_string()]);
        let shard_map = ShardMap::new(mesh, Vec::new(), Vec::new(), Vec::new()).unwrap();
        assert_eq!(shard_map.manual_axes(), &["x".to_string(), "y".to_string(), "z".to_string()]);
    }

    #[test]
    fn test_shard_map_global_input_type() {
        // The boundary type keeps the data type, shape, layout, and manual variation of the caller's type, takes the
        // placement of the input sharding, and drops the memory kind.
        let mesh = manual_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()],
            Vec::new(),
            vec!["x".into()],
        )
        .unwrap();
        let layout = Some(TiledLayout::new(vec![0], Vec::new()).into());
        let input_type = f32_vector_type(8)
            .with_layout(layout.clone())
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["y"]).unwrap())
            .unwrap();
        assert_eq!(
            shard_map.global_input_type(0, &input_type),
            Ok(f32_vector_type(8)
                .with_layout(layout)
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])])
                        .unwrap()
                        .with_varying_manual_axes(["y"])
                        .unwrap(),
                )
                .unwrap()),
        );
    }

    #[test]
    fn test_shard_map_global_input_type_rejects_mismatched_reduced_axes() {
        let mesh = manual_mesh_2x2();
        let input_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
            .unwrap()
            .with_reduced_axes(["x"])
            .unwrap();
        let shard_map =
            ShardMap::new(mesh.clone(), vec![input_sharding], Vec::new(), vec!["x".into(), "y".into()]).unwrap();
        let input_type = f32_vector_type(8)
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["y"])
                    .unwrap(),
            )
            .unwrap();
        let result = shard_map.global_input_type(0, &input_type);
        assert_eq!(
            result,
            Err(ShardMapError::ShardingStateMismatch {
                value_kind: "input",
                value_index: 0,
                state_kind: "reduced axes",
                expected: vec!["x".to_string()],
                actual: vec!["y".to_string()],
            }),
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "input type #0 has reduced axes [`y`], but `shard_map` expects [`x`]",
        );
    }

    #[test]
    fn test_shard_map_global_input_type_rejects_other_meshes() {
        // An input placed over another mesh (here, a mesh that also has the explicit axis `y`) has no placement at the
        // boundary, whose shardings and transposition are defined over the mesh of the shard map only.
        let other_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let shard_map =
            ShardMap::new(manual_mesh(), vec![sharded_along_x()], vec![sharded_along_x()], Vec::new()).unwrap();
        let input_type = f32_vector_type(8)
            .with_sharding(Sharding::new(other_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap())
            .unwrap();
        let result = shard_map.global_input_type(0, &input_type);
        assert_eq!(
            result,
            Err(ShardMapError::InputMeshMismatch { input_index: 0, expected: manual_mesh(), actual: other_mesh }),
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "input type #0 is placed over mesh `['x'=2:manual, 'y'=2:explicit]`, whose axes differ from those of the \
             `shard_map` mesh `['x'=2:manual]`",
        );
    }

    #[test]
    fn test_shard_map_local_input_type() {
        // The local shard shrinks along the active manual axes of the input sharding, which become variation facts,
        // while variation along outer manual axes is preserved.
        let mesh = manual_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()],
            Vec::new(),
            vec!["x".into()],
        )
        .unwrap();
        let input_type = f32_vector_type(8)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["y"]).unwrap())
            .unwrap();
        let global_input_type = shard_map.global_input_type(0, &input_type).unwrap();
        assert_eq!(
            shard_map.local_input_type(0, &global_input_type),
            Ok(f32_vector_type(4)
                .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["x", "y"]).unwrap())
                .unwrap()),
        );
    }

    #[test]
    fn test_shard_map_local_input_type_preserves_unreduced_and_reduced_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_unreduced_axes(["y"])
            .unwrap()
            .with_reduced_axes(["z"])
            .unwrap();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![input_sharding.clone()],
            Vec::new(),
            vec!["x".into(), "y".into(), "z".into()],
        )
        .unwrap();
        let global_input_type =
            shard_map.global_input_type(0, &f32_vector_type(8).with_sharding(input_sharding).unwrap()).unwrap();
        assert_eq!(
            shard_map.local_input_type(0, &global_input_type),
            Ok(f32_vector_type(4)
                .with_sharding(
                    Sharding::replicated(mesh, 1)
                        .with_unreduced_axes(["y"])
                        .unwrap()
                        .with_reduced_axes(["z"])
                        .unwrap()
                        .with_varying_manual_axes(["x"])
                        .unwrap(),
                )
                .unwrap()),
        );
    }

    #[test]
    fn test_shard_map_local_input_type_divides_all_manual_axes() {
        let mesh = manual_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap()],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            shard_map.local_input_type(0, &f32_vector_type(16)),
            Ok(f32_vector_type(4)
                .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["x", "y"]).unwrap())
                .unwrap()),
        );
    }

    #[test]
    fn test_shard_map_local_input_type_keeps_free_axes_global() {
        // Only the manual axis `data` divides the local shape. The auto axis `model` is stripped from the stored input
        // sharding, so it neither divides the local shape nor places the local shard.
        let mesh = data_model_mesh();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data", "model"])]).unwrap()],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            shard_map.local_input_type(0, &f32_vector_type(16)),
            Ok(f32_vector_type(8)
                .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["data"]).unwrap())
                .unwrap()),
        );
    }

    #[test]
    fn test_shard_map_local_input_type_rejects_padding_from_manual_axes() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Manual).unwrap()]).unwrap();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap()],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let result = shard_map.local_input_type(0, &f32_vector_type(10));
        assert_eq!(
            result,
            Err(ShardMapError::ManualAxisIntroducesPadding {
                value_kind: "input",
                value_index: 0,
                dimension: 0,
                dimension_size: 10,
                manual_partition_count: 3,
            }),
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "input sharding #0 dimension #0 has size 10, which is not divisible by manual partition count 3",
        );
    }

    #[test]
    fn test_shard_map_local_input_type_rejects_overflowing_manual_partition_counts() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 1 << 33, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 1 << 33, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let shard_map = ShardMap::new(mesh, vec![sharding.clone()], vec![sharding], Vec::new()).unwrap();
        let result = shard_map.local_input_type(0, &f32_vector_type(8));
        assert_eq!(
            result,
            Err(ShardMapError::Overflow {
                context: "computing the manual partition count of input #0 dimension #0".to_string(),
            }),
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "overflow while computing the manual partition count of input #0 dimension #0",
        );
    }

    #[test]
    fn test_shard_map_local_input_type_rejects_rank_mismatch() {
        let mesh = manual_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap()],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let result = shard_map.local_input_type(0, &ArrayType::new_static(DataType::F32, [8, 4]));
        assert_eq!(
            result,
            Err(ShardMapError::RankMismatch { value_kind: "input", value_index: 0, sharding_rank: 1, shape_rank: 2 }),
        );
        assert_eq!(result.unwrap_err().to_string(), "input sharding #0 has rank 1, but the provided shape has rank 2");
    }

    #[test]
    fn test_shard_map_local_input_type_rejects_dynamic_shapes() {
        let mesh = manual_mesh();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap()],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("dynamic", DimensionBounds::unbounded()).into()]),
        );
        let result = shard_map.local_input_type(0, &dynamic_type);
        assert_eq!(
            result,
            Err(ShardMapError::DynamicShapeNotSupported { value_kind: "input", value_index: 0, dimension: 0 }),
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "input type #0 dimension #0 must be static at a `shard_map` boundary",
        );
    }

    #[test]
    fn test_shard_map_local_input_type_rejects_mismatched_reduction_state() {
        // The local type takes its reduction state from the input sharding, so a global type whose unreduced axes
        // differ from those of the input sharding is rejected, like by `ShardMap::global_input_type`, instead of
        // silently reaching the body with the reduction state of the sharding.
        let mesh = manual_mesh();
        let shard_map =
            ShardMap::new(mesh.clone(), vec![Sharding::replicated(mesh.clone(), 1)], Vec::new(), Vec::new()).unwrap();
        let unreduced_type = f32_vector_type(4)
            .with_sharding(Sharding::replicated(mesh, 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let result = shard_map.local_input_type(0, &unreduced_type);
        assert_eq!(
            result,
            Err(ShardMapError::ShardingStateMismatch {
                value_kind: "input",
                value_index: 0,
                state_kind: "unreduced axes",
                expected: Vec::new(),
                actual: vec!["x".to_string()],
            }),
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "input type #0 has unreduced axes [`x`], but `shard_map` expects []",
        );
    }

    #[test]
    fn test_shard_map_input_replicated_manual_axes() {
        let mesh = manual_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap(),
                Sharding::replicated(mesh, 1),
            ],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(shard_map.input_replicated_manual_axes(0), vec!["y".to_string()]);
        assert_eq!(shard_map.input_replicated_manual_axes(1), Vec::<String>::new());
        assert_eq!(shard_map.input_replicated_manual_axes(2), vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn test_shard_map_global_output_type() {
        // Variation along the active manual axes is consumed by the output sharding, while variation along outer
        // manual axes survives on the global output.
        let mesh = manual_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()],
            vec!["x".into()],
        )
        .unwrap();
        let local_output_type = f32_vector_type(4)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x", "y"]).unwrap())
            .unwrap();
        assert_eq!(
            shard_map.global_output_type(0, &local_output_type),
            Ok(f32_vector_type(8)
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])])
                        .unwrap()
                        .with_varying_manual_axes(["y"])
                        .unwrap(),
                )
                .unwrap()),
        );
    }

    #[test]
    fn test_shard_map_global_output_type_keeps_free_axes_global() {
        let mesh = data_model_mesh();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![
                Sharding::new(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()],
                )
                .unwrap(),
            ],
            Vec::new(),
        )
        .unwrap();
        let local_output_type = ArrayType::new_static(DataType::F32, [16, 8])
            .with_sharding(Sharding::replicated(mesh.clone(), 2).with_varying_manual_axes(["data"]).unwrap())
            .unwrap();
        assert_eq!(
            shard_map.global_output_type(0, &local_output_type),
            Ok(ArrayType::new_static(DataType::F32, [32, 8])
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()])
                        .unwrap(),
                )
                .unwrap()),
        );
    }

    #[test]
    fn test_shard_map_global_output_type_rejects_omitted_unreduced_axes() {
        // An output sharding never adopts unreduced axes that the body output lacks: assembling an invariant output
        // as a pending sum along `y` would scale it by the size of `y`.
        let mesh = manual_mesh_2x2();
        let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_unreduced_axes(["y"])
            .unwrap();
        let shard_map =
            ShardMap::new(mesh.clone(), Vec::new(), vec![output_sharding.clone()], vec!["x".into(), "y".into()])
                .unwrap();
        let local_output_type = f32_vector_type(4)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let result = shard_map.global_output_type(0, &local_output_type);
        assert_eq!(
            result,
            Err(ShardMapError::ShardingStateMismatch {
                value_kind: "output",
                value_index: 0,
                state_kind: "unreduced axes",
                expected: vec!["y".to_string()],
                actual: Vec::new(),
            }),
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "output type #0 has unreduced axes [], but `shard_map` expects [`y`]",
        );

        // A body output that is unreduced along `y` itself is assembled as the pending sum that the sharding declares.
        let local_output_type = f32_vector_type(4)
            .with_sharding(
                Sharding::replicated(mesh, 1)
                    .with_unreduced_axes(["y"])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            shard_map.global_output_type(0, &local_output_type),
            Ok(f32_vector_type(8).with_sharding(output_sharding).unwrap()),
        );
    }

    #[test]
    fn test_shard_map_global_output_type_rejects_extra_local_unreduced_axes() {
        let mesh = manual_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap()],
            vec!["x".into(), "y".into()],
        )
        .unwrap();
        let local_output_type = f32_vector_type(4)
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["y"])
                    .unwrap(),
            )
            .unwrap();
        let result = shard_map.global_output_type(0, &local_output_type);
        assert_eq!(
            result,
            Err(ShardMapError::ShardingStateMismatch {
                value_kind: "output",
                value_index: 0,
                state_kind: "unreduced axes",
                expected: Vec::new(),
                actual: vec!["y".to_string()],
            }),
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "output type #0 has unreduced axes [`y`], but `shard_map` expects []",
        );
    }

    #[test]
    fn test_shard_map_global_output_type_rejects_reduced_axis_mismatch() {
        let mesh = manual_mesh_2x2();
        let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
            .unwrap()
            .with_reduced_axes(["x"])
            .unwrap();
        let shard_map = ShardMap::new(mesh, Vec::new(), vec![output_sharding], vec!["x".into(), "y".into()]).unwrap();
        let result = shard_map.global_output_type(0, &f32_vector_type(4));
        assert_eq!(
            result,
            Err(ShardMapError::ShardingStateMismatch {
                value_kind: "output",
                value_index: 0,
                state_kind: "reduced axes",
                expected: vec!["x".to_string()],
                actual: Vec::new(),
            }),
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "output type #0 has reduced axes [], but `shard_map` expects [`x`]",
        );
    }

    #[test]
    fn test_shard_map_global_output_type_rejects_omitted_varying_manual_axis() {
        let mesh = manual_mesh();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap()],
            Vec::new(),
        )
        .unwrap();
        let local_output_type = f32_vector_type(4)
            .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let result = shard_map.global_output_type(0, &local_output_type);
        assert_eq!(
            result,
            Err(ShardMapError::OutputVaryingAlongUntiledManualAxis { output_index: 0, axis_name: "x".to_string() }),
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "output type #0 still varies along manual axis `x`, but its output sharding does not mention it",
        );
    }

    #[test]
    fn test_shard_map() {
        // The rendering is the metadata fingerprint of `Operation::render`. The manual SPMD boundary metadata steers
        // differentiation, transposition, and backend lowering, yet none of it is visible in the instruction's rendered
        // atom types. Rendered inequality is exactly what the debug transform-cache diagnostic consumes: it compares a
        // retained transform artifact against a freshly derived one purely by rendering, so two boundaries that differ
        // semantically must never render alike.
        let rendered_program = |manual_axes: &[&str]| {
            let (operation, body) = metadata_fingerprint_shard_map(manual_axes);
            shard_map_program(operation, body).unwrap().to_string()
        };
        let baseline = rendered_program(&["x"]);

        // The complete boundary metadata renders as deterministic operation fields beside the attached body region.
        assert_eq!(
            baseline,
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = shard_map [
                    mesh=['x'=2:manual, 'y'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, []}],
                    out_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, []}],
                    manual_axes=['x'],
                    global_input_types=[f32[]],
                    global_output_types=[f32[]],
                ] %0 [
                    body={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in (%1)"},
        );

        // Two boundaries differing only in the active manual-axis subset must render differently even though their
        // types and bodies are identical, while equal metadata still renders equally, so the inequality isolates the
        // metadata itself.
        assert_ne!(baseline, rendered_program(&["x", "y"]));
        assert_eq!(baseline, rendered_program(&["x"]));

        // Binding the operation through a tracing context whose trace owns the input stages the same instruction onto
        // that trace's builder, with the local body attached as its `body` region.
        let (operation, body) = metadata_fingerprint_shard_map(&["x"]);
        let context = TestContext::new();
        let input = context.input(ArrayIrType::Array(f32_scalar_type()));
        let outputs = context.bind(operation, vec![body], std::slice::from_ref(&input)).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                outputs.iter().map(|output| output.atom_id().unwrap()).collect(),
                vec![Placeholder],
                vec![Placeholder; outputs.len()],
            )
            .unwrap();
        assert_eq!(program.to_string(), baseline);
    }

    #[test]
    fn test_shard_map_render_escapes_axis_names() {
        // Nothing restricts the characters of a mesh axis name: `MeshAxis::new` only rejects empty names. Rendering
        // names verbatim would therefore break the fingerprint contract pinned above, because one axis named `a', 'b`
        // renders exactly like the two axes `a` and `b`, letting the debug recheck accept a corrupted boundary.

        // Boundaries over one fixed mesh that owns every adversarial name, so only `manual_axes` varies below.
        fn rendered_manual_axes(manual_axes: Vec<&str>) -> String {
            let mesh = LogicalMesh::new(
                ["a", "b", "a', 'b", "a\\", "a\\', 'b", "a\nb", "a\\nb"]
                    .into_iter()
                    .map(|name| MeshAxis::new(name, 2, MeshAxisType::Manual).unwrap())
                    .collect(),
            )
            .unwrap();
            let array_type = f32_scalar_type();
            let shard_map = ShardMap::from_shardings(
                mesh.clone(),
                vec![Sharding::replicated(mesh.clone(), 0)],
                vec![Sharding::replicated(mesh, 0)],
                manual_axes.into_iter().map(str::to_string).collect(),
            );
            ShardMapOperation::from_boundary(shard_map, vec![array_type.clone()], vec![array_type]).to_string()
        }

        // Boundaries with no shardings and no boundary types, so only the `mesh` field carries axis names.
        fn rendered_mesh(axis_names: Vec<&str>) -> String {
            let mesh = LogicalMesh::new(
                axis_names.into_iter().map(|name| MeshAxis::new(name, 2, MeshAxisType::Manual).unwrap()).collect(),
            )
            .unwrap();
            let shard_map = ShardMap::from_shardings(mesh, Vec::new(), Vec::new(), Vec::new());
            let boundary = Vec::<ArrayIrType>::new();
            ShardMapOperation::from_boundary(shard_map, boundary.clone(), boundary).to_string()
        }

        // A quote inside a name must not be able to imitate the separator between two rendered names, in either the
        // manual-axis list or the mesh, where the name is additionally followed by its size and type.
        assert_ne!(rendered_manual_axes(vec!["a", "b"]), rendered_manual_axes(vec!["a', 'b"]));
        assert_ne!(rendered_mesh(vec!["a", "b"]), rendered_mesh(vec!["a'=2:manual, 'b"]));

        // Backslashes are escaped as well, so a name ending in one cannot turn the quote that closes it into an
        // escaped quote, and control characters are escaped so that they cannot imitate their own escape codes.
        assert_ne!(rendered_manual_axes(vec!["a\\", "b"]), rendered_manual_axes(vec!["a\\', 'b"]));
        assert_ne!(rendered_manual_axes(vec!["a\nb"]), rendered_manual_axes(vec!["a\\nb"]));

        // The escaped form itself is Rust's canonical character escaping, inside the surrounding single quotes.
        assert_eq!(
            rendered_mesh(vec!["a'b", "c\\d", "e\nf"]),
            indoc! {r#"
                shard_map [
                    mesh=['a\'b'=2:manual, 'c\\d'=2:manual, 'e\nf'=2:manual],
                    in_shardings=[],
                    out_shardings=[],
                    manual_axes=[],
                    global_input_types=[],
                    global_output_types=[],
                ]"#},
        );
    }

    #[test]
    fn test_shard_map_render_includes_output_forwarding() {
        // The rendered fingerprint names the forwarded inputs exactly when some output forwards a reference.
        let sharded = sharded_along_x();
        let (operation, _) = reference_shard_map(f32_vector_type(4), sharded, true);
        assert_eq!(
            operation.to_string(),
            indoc! {"
                shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[ref<f32[4]>, f32[4]],
                    global_output_types=[f32[4]],
                ]"},
        );
        assert_eq!(
            operation.with_output_forwarding(vec![Some(0)]).unwrap().to_string(),
            indoc! {"
                shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[ref<f32[4]>, f32[4]],
                    global_output_types=[f32[4]],
                    output_forwarding=[0],
                ]"},
        );
    }

    #[test]
    fn test_shard_map_operation_from_program() {
        // The declared global input types are normalized through the input shardings, the global output types are
        // derived from the body through the output shardings, and value outputs forward no input.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let local_type = ShardMap::new(mesh.clone(), vec![sharded.clone()], vec![sharded.clone()], Vec::new())
            .unwrap()
            .local_input_type(0, &f32_vector_type(4))
            .unwrap();
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(local_type.into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let operation = ShardMapOperation::from_program(
            &body,
            vec![f32_vector_type(4).into()],
            ShardMap::new(mesh, vec![sharded.clone()], vec![sharded.clone()], Vec::new()).unwrap(),
        )
        .unwrap();
        assert_eq!(operation.shard_map().manual_axes(), &["x".to_string()]);
        assert_eq!(operation.shard_map().in_shardings(), &[sharded.clone()]);
        let global_type = f32_vector_type(4).with_sharding(sharded).unwrap();
        assert_eq!(operation.global_input_types(), &[ArrayIrType::Array(global_type.clone())]);
        assert_eq!(operation.global_output_types(), &[ArrayIrType::Array(global_type)]);
        assert_eq!(operation.output_forwarding(), &[None]);
        assert_eq!(
            operation.to_string(),
            indoc! {"
                shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                    global_output_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ]"},
        );
    }

    #[test]
    fn test_shard_map_operation_from_program_requires_tiled_output_variation() {
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let sharded = sharded_along_x();
        let array_type = ArrayType::new_static(DataType::F32, [1]).with_sharding(replicated.clone()).unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(array_type.clone().into());
        let body = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let result = ShardMapOperation::from_program(
            &body,
            vec![array_type.into()],
            ShardMap::new(mesh, vec![replicated], vec![sharded], vec!["x".to_string()]).unwrap(),
        );
        assert!(matches!(
            &result,
            Err(error @ ShardMapError::OutputNotVaryingAlongTiledManualAxis { output_index: 0, axis_name })
                if axis_name == "x"
                    && error.to_string() == "`shard_map` body output #0 must vary along tiled manual axis `x`; \
                        insert `parallel_vary` before returning the output",
        ));
    }

    #[test]
    fn test_shard_map_operation_from_program_rejects_input_type_count_mismatch() {
        let body = identity_body(f32_scalar_type());
        let result = ShardMapOperation::from_program(&body, Vec::new(), single_input_test_shard_map());
        assert_eq!(result, Err(ShardMapError::InputTypeCountMismatch { expected: 1, actual: 0 }));
        assert_eq!(result.unwrap_err().to_string(), "got 0 global input type(s), but `shard_map` expects 1");
    }

    #[test]
    fn test_shard_map_operation_from_program_rejects_output_type_count_mismatch() {
        // The body produces one output, but the boundary declares none.
        let body = identity_body(f32_scalar_type());
        let mesh = manual_mesh();
        let shard_map =
            ShardMap::new(mesh.clone(), vec![Sharding::replicated(mesh, 0)], Vec::new(), Vec::new()).unwrap();
        let result = ShardMapOperation::from_program(&body, vec![f32_scalar_type().into()], shard_map);
        assert_eq!(result, Err(ShardMapError::OutputTypeCountMismatch { expected: 0, actual: 1 }));
        assert_eq!(result.unwrap_err().to_string(), "got 1 output type(s), but `shard_map` expects 0");
    }

    #[test]
    fn test_shard_map_operation_from_program_rejects_escaping_body_allocations() {
        // A reference output must forward a reference input, so a reference allocated inside the body is rejected with
        // that cause, which the reference contract is validated for before type inference reports the output as one
        // whose forwarding the operation does not declare.
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(f32_scalar_type().into());
            let reference = builder
                .add_instruction(ReferenceNewOperation::<ArrayType, ArrayIrType>::new(), Vec::new(), vec![input], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let shard_map = ShardMap::new(mesh, vec![replicated.clone()], vec![replicated], Vec::new()).unwrap();
        let result = ShardMapOperation::from_program(&body, vec![f32_scalar_type().into()], shard_map);
        let expected = ProgramError::MalformedProgram(
            "`shard_map` output #0 is a reference allocated inside the shard-map body, which cannot escape; a \
             reference output must forward a reference input"
                .to_string(),
        );
        assert_eq!(result, Err(ShardMapError::Program(expected)));
    }

    #[test]
    fn test_shard_map_operation_with_global_output_types() {
        // Replacing the declared global output types keeps the boundary metadata, the declared global input types, and
        // the output forwarding.
        let (operation, _) = forwarding_reference_shard_map();
        let output_types = vec![
            ArrayIrType::Array(f32_vector_type(4)),
            ArrayIrType::Reference(ReferenceType::new(f32_vector_type(4))),
        ];
        let replaced = operation.clone().with_global_output_types(output_types.clone()).unwrap();
        assert_eq!(replaced.shard_map(), operation.shard_map());
        assert_eq!(replaced.global_input_types(), operation.global_input_types());
        assert_eq!(replaced.global_output_types(), output_types.as_slice());
        assert_eq!(replaced.output_forwarding(), &[None, Some(0)]);
    }

    #[test]
    fn test_shard_map_operation_with_global_output_types_rejects_output_type_count_mismatch() {
        let result = forwarding_reference_shard_map().0.with_global_output_types(Vec::new());
        assert_eq!(result, Err(ShardMapError::OutputTypeCountMismatch { expected: 2, actual: 0 }));
        assert_eq!(result.unwrap_err().to_string(), "got 0 output type(s), but `shard_map` expects 2");
    }

    #[test]
    fn test_shard_map_operation_with_output_forwarding() {
        // A boundary built from its parts declares no forwarding, and declaring the forwarding that checked
        // construction derives from the body's reference analysis reproduces the checked operation.
        let (operation, _) = forwarding_reference_shard_map();
        let boundary = ShardMapOperation::from_boundary(
            operation.shard_map().clone(),
            operation.global_input_types().to_vec(),
            operation.global_output_types().to_vec(),
        );
        assert_eq!(boundary.output_forwarding(), &[None, None]);
        assert_eq!(boundary.with_output_forwarding(vec![None, Some(0)]), Ok(operation));
    }

    #[test]
    fn test_shard_map_operation_with_output_forwarding_rejects_output_forwarding_count_mismatch() {
        let result = forwarding_reference_shard_map().0.with_output_forwarding(vec![None]);
        assert_eq!(result, Err(ShardMapError::OutputForwardingCountMismatch { expected: 2, actual: 1 }));
        assert_eq!(result.unwrap_err().to_string(), "got 1 output forwarding(s), but `shard_map` expects 2");
    }

    #[test]
    fn test_shard_map_operation_shard_map() {
        let sharded = sharded_along_x();
        assert_eq!(
            forwarding_reference_shard_map().0.shard_map(),
            &ShardMap::new(manual_mesh(), vec![sharded.clone(); 2], vec![sharded; 2], Vec::new()).unwrap(),
        );
    }

    #[test]
    fn test_shard_map_operation_global_input_types() {
        // A reference position carries the global reference type, whose referent is placed by the input sharding
        // like an array input.
        let global_type = f32_vector_type(4).with_sharding(sharded_along_x()).unwrap();
        assert_eq!(
            forwarding_reference_shard_map().0.global_input_types(),
            &[ArrayIrType::Reference(ReferenceType::new(global_type.clone())), ArrayIrType::Array(global_type)],
        );
    }

    #[test]
    fn test_shard_map_operation_global_output_types() {
        // Value and forwarded reference outputs both carry the placement that their output sharding establishes.
        let sharded = sharded_along_x();
        let global_type = f32_vector_type(4).with_sharding(sharded).unwrap();
        assert_eq!(
            forwarding_reference_shard_map().0.global_output_types(),
            &[ArrayIrType::Array(global_type.clone()), ArrayIrType::Reference(ReferenceType::new(global_type))],
        );
    }

    #[test]
    fn test_shard_map_operation_output_forwarding() {
        // Checked construction derives the forwarding from the body's reference analysis: the value output forwards
        // nothing, and the reference output forwards the reference input that the body returns.
        assert_eq!(forwarding_reference_shard_map().0.output_forwarding(), &[None, Some(0)]);
    }

    #[test]
    fn test_shard_map_type_inference() {
        let array_type = f32_scalar_type();
        let composite_array_type = ArrayIrType::Array(array_type.clone());
        let operation =
            ShardMapOperation::from_boundary(single_input_test_shard_map(), vec![array_type.clone()], vec![array_type]);
        let array_body = RegionInterface::new(
            vec![composite_array_type.clone()],
            vec![composite_array_type.clone()],
            EffectClasses::NONE,
        );

        assert_eq!(
            operation
                .infer_output_types(std::slice::from_ref(&composite_array_type), std::slice::from_ref(&array_body)),
            Ok(vec![composite_array_type.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&composite_array_type), &[]),
            Err(TypeError::invalid("expected 1 region but got 0")),
        );

        // Every input must agree with its declared global input type.
        assert_eq!(
            operation.infer_output_types(&[ArrayIrType::Array(f32_vector_type(2))], std::slice::from_ref(&array_body)),
            Err(TypeError::invalid(
                "`shard_map` input #0 has type `f32[2]` but its declared global input type is `f32[]`"
            )),
        );

        // The boundary is array-only: neither the inputs nor the body boundary may be first-class dimensions.
        let dimension_type =
            ArrayIrType::Dimension(DimensionType::new("size", DimensionBounds::positive(Some(8)).unwrap()));
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&dimension_type), std::slice::from_ref(&array_body)),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        let dimension_input_body =
            RegionInterface::new(vec![dimension_type.clone()], vec![composite_array_type.clone()], EffectClasses::NONE);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&composite_array_type),
                std::slice::from_ref(&dimension_input_body),
            ),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        let dimension_output_body =
            RegionInterface::new(vec![composite_array_type.clone()], vec![dimension_type], EffectClasses::NONE);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&composite_array_type),
                std::slice::from_ref(&dimension_output_body),
            ),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
    }

    #[test]
    fn test_shard_map_type_inference_rejects_mismatched_body_interfaces() {
        // Attaching a body validates it against the boundary's local derivation, so a checked payload cannot be rebound
        // to a same-arity body whose interface disagrees with it.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let global_type = f32_vector_type(4);
        let local_type = ShardMap::new(mesh.clone(), vec![sharded.clone()], vec![sharded.clone()], Vec::new())
            .unwrap()
            .local_input_type(0, &global_type)
            .unwrap();
        let local_sharding = local_type.sharding().cloned();
        let body = identity_body(local_type.clone());
        let operation = ShardMapOperation::from_program(
            &body,
            vec![global_type.clone().into()],
            ShardMap::new(mesh, vec![sharded.clone()], vec![sharded.clone()], Vec::new()).unwrap(),
        )
        .unwrap();
        let global_input_types = [ArrayIrType::Array(global_type)];

        // The checked body itself is accepted.
        assert_eq!(
            operation.infer_output_types(
                &global_input_types,
                &[RegionInterface::new(body.input_types(), body.output_types(), EffectClasses::NONE)],
            ),
            Ok(operation.global_output_types().to_vec()),
        );

        // Body inputs must have the shape, element type, and layout of the local shard of their global input.
        let wrong_shape = ArrayType::new_static(DataType::F32, [3]).with_sharding(local_sharding.clone()).unwrap();
        assert_eq!(
            operation.infer_output_types(
                &global_input_types,
                &[RegionInterface::new(vec![wrong_shape.clone().into()], body.output_types(), EffectClasses::NONE)],
            ),
            Err(TypeError::invalid(format!(
                "`shard_map` body input #0 has type `{wrong_shape}` but the local shard of input #0 is `{local_type}`",
            ))),
        );
        let wrong_element_type =
            ArrayType::new_static(DataType::F64, [2]).with_sharding(local_sharding.clone()).unwrap();
        assert_eq!(
            operation.infer_output_types(
                &global_input_types,
                &[RegionInterface::new(
                    vec![wrong_element_type.clone().into()],
                    body.output_types(),
                    EffectClasses::NONE,
                )],
            ),
            Err(TypeError::invalid(format!(
                "`shard_map` body input #0 has type `{wrong_element_type}` but the local shard of input #0 is \
                 `{local_type}`",
            ))),
        );
        let wrong_layout = local_type.clone().with_layout(Some(TiledLayout::new(vec![0], Vec::new()).into()));
        assert_eq!(
            operation.infer_output_types(
                &global_input_types,
                &[RegionInterface::new(vec![wrong_layout.clone().into()], body.output_types(), EffectClasses::NONE)],
            ),
            Err(TypeError::invalid(format!(
                "`shard_map` body input #0 has type `{wrong_layout}` but the local shard of input #0 is `{local_type}`",
            ))),
        );

        // Body outputs must derive the declared global output type through their output sharding.
        let wrong_output = ArrayType::new_static(DataType::F32, [3]).with_sharding(local_sharding).unwrap();
        let wrong_global_output = ArrayType::new_static(DataType::F32, [6]).with_sharding(sharded.clone()).unwrap();
        assert_eq!(
            operation.infer_output_types(
                &global_input_types,
                &[RegionInterface::new(body.input_types(), vec![wrong_output.clone().into()], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid(format!(
                "`shard_map` body output #0 has type `{wrong_output}`, whose global type `{wrong_global_output}` does \
                 not match the declared global output type `{}`",
                operation.global_output_types()[0],
            ))),
        );

        // A body output tiled along the manual axis `x` must vary along it.
        let invariant_output = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(manual_mesh(), 1))
            .unwrap();
        let result = operation.infer_output_types(
            &global_input_types,
            &[RegionInterface::new(body.input_types(), vec![invariant_output.into()], EffectClasses::NONE)],
        );
        assert_eq!(
            result,
            Err(TypeError::custom(ShardMapError::OutputNotVaryingAlongTiledManualAxis {
                output_index: 0,
                axis_name: "x".to_string(),
            })),
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "`shard_map` body output #0 must vary along tiled manual axis `x`; insert `parallel_vary` before \
             returning the output",
        );

        // Attachment through a program builder runs the same validation.
        let wrong_body = identity_body(wrong_shape.clone());
        assert_eq!(
            shard_map_program(operation, wrong_body).map(|program| program.to_string()),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`shard_map` body input #0 has type `{wrong_shape}` but the local shard of input #0 is `{local_type}`",
            )))),
        );
    }

    #[test]
    fn test_shard_map_type_inference_rejects_ordered_io_bodies() {
        // `OrderedIo` promises one order across devices, which independent per-device execution of a body cannot
        // provide, so checked construction and every attachment reject it, while `DeviceOrderedIo` bodies remain
        // supported.
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let array_type = f32_vector_type(4);
        let ordered_body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(array_type.clone().into());
            let output = builder
                .add_instruction(PrintOperation::<ArrayIrType>::new("body"), Vec::new(), vec![input], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let device_ordered_body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(array_type.clone().into());
            let output = builder
                .add_instruction(
                    PrintOperation::<ArrayIrType>::new("body").with_effect_class(EffectClass::DeviceOrderedIo),
                    Vec::new(),
                    vec![input],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let result = ShardMapOperation::from_program(
            &ordered_body,
            vec![array_type.clone().into()],
            ShardMap::new(mesh.clone(), vec![replicated.clone()], vec![replicated.clone()], Vec::new()).unwrap(),
        );
        assert_eq!(result, Err(ShardMapError::OrderedIoNotSupported));
        assert_eq!(
            result.unwrap_err().to_string(),
            "`shard_map` bodies require `DeviceOrderedIo` because `OrderedIo` demands one order across devices",
        );
        let operation = ShardMapOperation::from_program(
            &device_ordered_body,
            vec![array_type.clone().into()],
            ShardMap::new(mesh, vec![replicated.clone()], vec![replicated], Vec::new()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            operation.infer_output_types(
                &[array_type.clone().into()],
                &[RegionInterface::new(
                    device_ordered_body.input_types(),
                    device_ordered_body.output_types(),
                    device_ordered_body.effects().classes(),
                )],
            ),
            Ok(operation.global_output_types().to_vec()),
        );

        // Rebinding the checked operation to the `OrderedIo` body is rejected at attachment.
        assert_eq!(
            operation.infer_output_types(
                &[array_type.into()],
                &[RegionInterface::new(
                    ordered_body.input_types(),
                    ordered_body.output_types(),
                    ordered_body.effects().classes(),
                )],
            ),
            Err(TypeError::custom(ShardMapError::OrderedIoNotSupported)),
        );
        assert!(matches!(
            shard_map_program(operation, ordered_body),
            Err(ProgramError::Type(error))
                if error.downcast_custom::<ShardMapError>() == Some(&ShardMapError::OrderedIoNotSupported),
        ));
    }

    #[test]
    fn test_shard_map_type_inference_rejects_inputs_varying_along_manual_axes() {
        // Type inference rejects an input that already varies along a manual axis of the `shard_map`: only a value of
        // an enclosing manual region over that axis varies along it, so the map would make that axis manual a second
        // time.
        let operation = ShardMapOperation::from_boundary(
            single_input_test_shard_map(),
            vec![f32_scalar_type()],
            vec![f32_scalar_type()],
        );
        let body = identity_body(f32_scalar_type());
        let varying_type = f32_scalar_type()
            .with_sharding(Sharding::replicated(manual_mesh(), 0).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let expected = ShardMapError::InputVariesAlongManualAxis { input_index: 0, axis_name: "x".to_string() };
        assert_eq!(
            operation.infer_output_types(&[varying_type.into()], &[body.entry_region_ref().interface()]),
            Err(TypeError::custom(expected.clone())),
        );
        assert_eq!(
            expected.to_string(),
            "input type #0 already varies along manual axis `x` of the `shard_map`, which an enclosing manual region \
             therefore already made manual",
        );

        // Descriptor-only tracing has no enclosing context to exclude already manual axes, so binding a body traced
        // over all manual axes of the 2x2 mesh with specifications over `x` inside a manual region over `x` (whose
        // local values vary along `x`) is rejected when the body is attached, which also fails the enclosing trace.
        let mesh = manual_mesh_2x2();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let result: Result<TracedShardMap<TestEagerContext, ArrayType, ArrayType>, _> = trace_shard_map_with_options(
            |x: TestTracer| {
                let inner: TracedShardMap<TestEagerContext, ArrayType, ArrayType> = trace_shard_map(
                    |y: TestTracer| y.clone() + y,
                    f32_vector_type(4),
                    mesh.clone(),
                    sharding.clone(),
                    sharding.clone(),
                )
                .unwrap();
                let result = x.value().context().bind(
                    inner.operation().clone(),
                    vec![inner.body().clone()],
                    &[x.value().clone()],
                );
                assert!(matches!(
                    result,
                    Err(ProgramError::Type(error))
                        if error.downcast_custom::<ShardMapError>() == Some(&expected),
                ));
                x
            },
            f32_vector_type(8),
            mesh.clone(),
            sharding.clone(),
            sharding.clone(),
            vec!["x".to_string()],
        );
        assert!(matches!(
            result,
            Err(ShardMapError::Program(ProgramError::Type(error)))
                if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
    }

    #[test]
    fn test_shard_map_type_inference_rejects_inputs_over_other_meshes() {
        // Attaching a body rejects an input placed over a mesh other than the mesh of the shard map, whose placement
        // the boundary could not restore in reverse mode, instead of tolerating its sharding like a dimension sharding.
        let other_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let boundary =
            ShardMap::new(manual_mesh(), vec![sharded_along_x()], vec![sharded_along_x()], Vec::new()).unwrap();
        let body = identity_body(boundary.local_input_type(0, &f32_vector_type(8)).unwrap());
        let operation = ShardMapOperation::from_program(&body, vec![f32_vector_type(8).into()], boundary).unwrap();
        let input_type = f32_vector_type(8)
            .with_sharding(Sharding::new(other_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap())
            .unwrap();
        let expected = ShardMapError::InputMeshMismatch { input_index: 0, expected: manual_mesh(), actual: other_mesh };
        assert_eq!(
            operation.infer_output_types(&[input_type.clone().into()], &[body.entry_region_ref().interface()]),
            Err(TypeError::custom(expected.clone())),
        );

        // A mesh that differs only in its axis types describes the same device mesh (the mesh of a shard map designates
        // the axes that it makes manual through its axis types), so an input placed over it is accepted.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let explicit_type = f32_vector_type(8)
            .with_sharding(Sharding::new(explicit_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            operation.infer_output_types(&[explicit_type.into()], &[body.entry_region_ref().interface()]),
            Ok(vec![f32_vector_type(8).with_sharding(sharded_along_x()).unwrap().into()]),
        );

        // The closure entry points normalize the caller's input types through the boundary first, so they reject such
        // an input before tracing the body.
        trace_test_program(
            |inputs| {
                let result = shard_map(
                    |x: TestTracer| x,
                    inputs[0].clone(),
                    manual_mesh(),
                    sharded_along_x(),
                    sharded_along_x(),
                );
                assert_eq!(result, Err(expected.clone()));
                inputs
            },
            vec![input_type],
            Vec::new(),
        );
    }

    #[test]
    fn test_shard_map_type_inference_requires_declared_input_variation() {
        // A map over `x` checked through `from_program` for an unsharded `f32[8]` input is bound inside an enclosing
        // manual region over `z`, on an input that varies along `z`. Its body was checked for the declared input, which
        // is invariant along `z`, so binding it there would type the per-device results as replicated along `z`, and
        // type inference rejects the input instead: the body inputs must be the shards of the actual inputs.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let shard_map =
            ShardMap::new(mesh.clone(), vec![sharding.clone()], vec![sharding.clone()], vec!["x".into()]).unwrap();
        let invariant_body = identity_body(shard_map.local_input_type(0, &f32_vector_type(8)).unwrap());
        let invariant_operation =
            ShardMapOperation::from_program(&invariant_body, vec![f32_vector_type(8).into()], shard_map.clone())
                .unwrap();
        let varying_type = f32_vector_type(8)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["z"]).unwrap())
            .unwrap();
        let declared_type = f32_vector_type(8).with_sharding(sharding.clone()).unwrap();
        assert_eq!(
            invariant_operation
                .infer_output_types(&[varying_type.clone().into()], &[invariant_body.entry_region_ref().interface()]),
            Err(TypeError::invalid(format!(
                "`shard_map` input #0 has type `{varying_type}` but its declared global input type is \
                 `{declared_type}`",
            ))),
        );

        // Declaring the variation along `z` makes the body inputs vary along `z` as well, so the output keeps that
        // variation, and such a map in turn rejects an input that is invariant along `z`.
        let varying_declared_type =
            f32_vector_type(8).with_sharding(sharding.clone().with_varying_manual_axes(["z"]).unwrap()).unwrap();
        let varying_body = identity_body(shard_map.local_input_type(0, &varying_declared_type).unwrap());
        let varying_operation =
            ShardMapOperation::from_program(&varying_body, vec![varying_type.clone().into()], shard_map).unwrap();
        assert_eq!(varying_operation.global_input_types(), &[varying_declared_type.clone().into()]);
        assert_eq!(
            varying_operation
                .infer_output_types(&[varying_type.clone().into()], &[varying_body.entry_region_ref().interface()]),
            Ok(vec![varying_declared_type.clone().into()]),
        );
        assert_eq!(
            varying_operation
                .infer_output_types(&[f32_vector_type(8).into()], &[varying_body.entry_region_ref().interface()]),
            Err(TypeError::invalid(format!(
                "`shard_map` input #0 has type `f32[8]` but its declared global input type is \
                 `{varying_declared_type}`",
            ))),
        );

        // A map over a mesh without `z` rejects the input as well, because no type over its mesh can vary along `z`:
        // the input is placed over the mesh of the enclosing region, which is not the mesh of the map.
        let other_mesh = mesh;
        let mesh = manual_mesh();
        let sharding = sharded_along_x();
        let shard_map =
            ShardMap::new(mesh.clone(), vec![sharding.clone()], vec![sharding.clone()], Vec::new()).unwrap();
        let body = identity_body(shard_map.local_input_type(0, &f32_vector_type(8)).unwrap());
        let operation =
            ShardMapOperation::from_program(&body, vec![f32_vector_type(8).into()], shard_map.clone()).unwrap();
        let expected = ShardMapError::InputMeshMismatch { input_index: 0, expected: mesh, actual: other_mesh };
        assert_eq!(
            operation.infer_output_types(&[varying_type.clone().into()], &[body.entry_region_ref().interface()]),
            Err(TypeError::custom(expected.clone())),
        );
        assert_eq!(ShardMapOperation::from_program(&body, vec![varying_type.into()], shard_map), Err(expected));
    }

    #[test]
    fn test_shard_map_type_inference_reference_boundaries() {
        // Type inference over a reference-bearing boundary: a reference input must reach the body as a reference to its
        // local shard, and a reference output must name the input it forwards under an equal output sharding and
        // declare the type of that input.
        let sharded = sharded_along_x();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded.clone(), true);
        let body_interface = RegionInterface::new(body.input_types(), body.output_types(), body.effects().classes());
        let reference_type = ArrayIrType::Reference(ReferenceType::new(f32_vector_type(4)));
        let value_type = ArrayIrType::Array(f32_vector_type(4));
        assert_eq!(
            operation.infer_output_types(&[reference_type.clone(), value_type.clone()], &[body_interface.clone()]),
            Ok(vec![value_type.clone()]),
        );

        // Input kinds must match the declared boundary positions.
        assert_eq!(
            operation.infer_output_types(&[value_type.clone(), value_type.clone()], &[body_interface.clone()]),
            Err(TypeError::invalid("expected reference type but got array type")),
        );
        assert_eq!(
            operation.infer_output_types(&[reference_type.clone(), reference_type.clone()], &[body_interface]),
            Err(TypeError::invalid("expected array type but got reference type")),
        );

        // The body must receive the local shard of the referent, not the global referent.
        let global_body = RegionInterface::new(
            vec![reference_type.clone(), body.input_types()[1].clone()],
            body.output_types(),
            body.effects().classes(),
        );
        assert_eq!(
            operation.infer_output_types(&[reference_type.clone(), value_type.clone()], &[global_body]),
            Err(TypeError::invalid(
                "`shard_map` body input #0 has type `ref<f32[4]>` but the local shard of reference input #0 is \
                 `ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]>`",
            )),
        );

        // A reference output must declare which input it forwards, forward it by identity, and share its sharding.
        let forwarding_body = RegionInterface::new(
            body.input_types(),
            vec![body.output_types()[0].clone(), body.input_types()[0].clone()],
            body.effects().classes(),
        );
        let forwarding_operation = ShardMapOperation::from_boundary(
            ShardMap::from_shardings(
                manual_mesh(),
                vec![sharded.clone(), sharded.clone()],
                vec![sharded.clone(), sharded.clone()],
                vec!["x".to_string()],
            ),
            operation.global_input_types().to_vec(),
            vec![value_type.clone(), reference_type.clone()],
        );
        assert_eq!(
            forwarding_operation
                .infer_output_types(&[reference_type.clone(), value_type.clone()], &[forwarding_body.clone()]),
            Err(TypeError::invalid(
                "`shard_map` output #1 is a reference whose forwarded input the operation does not declare",
            )),
        );
        let forwarding_operation = forwarding_operation.with_output_forwarding(vec![None, Some(0)]).unwrap();
        assert_eq!(
            forwarding_operation
                .infer_output_types(&[reference_type.clone(), value_type.clone()], &[forwarding_body.clone()]),
            Ok(vec![value_type.clone(), reference_type.clone()]),
        );
        let misdeclared_type = ArrayIrType::Reference(ReferenceType::new(f32_vector_type(8)));
        assert_eq!(
            forwarding_operation
                .clone()
                .with_global_output_types(vec![value_type.clone(), misdeclared_type])
                .unwrap()
                .infer_output_types(&[reference_type.clone(), value_type.clone()], &[forwarding_body.clone()]),
            Err(TypeError::invalid(
                "`shard_map` output #1 has declared type `ref<f32[8]>`, but it forwards reference input #0 of type \
                 `ref<f32[4]>`",
            )),
        );

        // Only reference outputs forward inputs, so an array output cannot declare a forwarding.
        assert_eq!(
            forwarding_operation
                .clone()
                .with_output_forwarding(vec![Some(1), Some(0)])
                .unwrap()
                .infer_output_types(&[reference_type.clone(), value_type.clone()], &[forwarding_body.clone()]),
            Err(TypeError::invalid(
                "`shard_map` output #0 is an array but the operation declares that it forwards input #1; only \
                 reference outputs forward inputs",
            )),
        );

        // The body must forward the declared input by identity.
        assert_eq!(
            forwarding_operation
                .with_output_forwarding(vec![None, Some(1)])
                .unwrap()
                .infer_output_types(&[reference_type.clone(), value_type.clone()], &[forwarding_body.clone()]),
            Err(TypeError::invalid("`shard_map` output #1 does not forward reference input #1 by identity")),
        );
        let replicated = Sharding::replicated(manual_mesh(), 1);
        let mismatched_operation = ShardMapOperation::from_boundary(
            ShardMap::from_shardings(
                manual_mesh(),
                vec![sharded.clone(), sharded.clone()],
                vec![sharded.clone(), replicated.clone()],
                vec!["x".to_string()],
            ),
            operation.global_input_types().to_vec(),
            vec![value_type.clone(), reference_type.clone()],
        )
        .with_output_forwarding(vec![None, Some(0)])
        .unwrap();
        assert_eq!(
            mismatched_operation.infer_output_types(&[reference_type, value_type], &[forwarding_body]),
            Err(TypeError::invalid(format!(
                "`shard_map` output #1 forwards reference input #0 but its output sharding `{replicated}` differs from \
                 the input sharding `{sharded}`",
            ))),
        );
    }

    #[test]
    fn test_shard_map_rejects_divergent_writes_into_replicated_references() {
        // Every device holds its own copy of a reference that is replicated along `x`, so a mutation keeps those copies
        // identical only when every device writes the same value at the same position. A write at an index that varies
        // along `x` and a write of a value that varies along `x` are rejected when they are staged.
        let mesh = manual_mesh();
        let shard_map =
            ShardMap::new(mesh.clone(), vec![Sharding::replicated(mesh, 1)], Vec::new(), Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &f32_vector_type(2)).unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(ReferenceType::new(local_type.clone()).into());
        let index = add_axis_index(&mut builder);
        let static_index = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) };
        let read = ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new()
            .with_transforms(vec![static_index.clone()]);
        let value = builder.add_instruction(read, Vec::new(), vec![reference], None).unwrap()[0];
        let write = ReferenceWriteOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new()
            .with_transforms(vec![dynamic_index_transform()]);
        assert_eq!(
            builder.add_instruction(write, Vec::new(), vec![reference, value, index], None).map(|_| ()),
            Err(TypeError::invalid(
                "reference transform index `u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]` varies \
                 over manual axis `x` but the referent `f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]` does not, so a \
                 mutation through it would update a different element on every device of a referent that is identical \
                 across them; vary the referent over that axis or use an invariant index",
            )
            .into()),
        );
        let varying_value = builder
            .add_instruction(ParallelVaryOperation::new("x".to_string()), Vec::new(), vec![value], None)
            .unwrap()[0];
        let write = ReferenceWriteOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new()
            .with_transforms(vec![static_index]);
        assert_eq!(
            builder.add_instruction(write, Vec::new(), vec![reference, varying_value], None).map(|_| ()),
            Err(TypeError::invalid(
                "`reference_write` replacement type `f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]` \
                 must exactly match reference referent type `f32[][sharding={mesh<['x'=2:manual]>, []}]`",
            )
            .into()),
        );
    }

    #[test]
    fn test_shard_map_operation_region_data_flow() {
        let operation = forwarding_reference_shard_map().0;
        assert!(matches!(operation.region_data_flow(), RegionDataFlow::Provenance));
        assert_eq!(operation.input_region_provenance(0, 1), InputRegionProvenance::Input { index: 1 });
        assert_eq!(operation.input_region_provenance(1, 1), InputRegionProvenance::None);
        assert_eq!(
            operation.output_region_provenance(1),
            vec![OutputRegionProvenance { region_index: 0, output_index: 1 }],
        );
    }

    #[test]
    fn test_shard_map_boundary_pruning() {
        // Pruning a `shard_map` drops its dead outputs together with their output shardings and declared global types,
        // the inputs that the pruned body no longer reads together with their input shardings and declared global
        // types, and the body computation that only the dead outputs use.

        /// Region liveness that reports a fixed liveness for the inputs of the attached region.
        struct FixedRegionLiveness(Vec<bool>);

        impl RegionLiveness for FixedRegionLiveness {
            fn used_region_inputs(
                &mut self,
                _region_index: usize,
                _used_outputs: &[bool],
            ) -> Result<Vec<bool>, ProgramError> {
                Ok(self.0.clone())
            }
        }

        // The boundary rule filters the input shardings and declared types with the live body inputs, and the output
        // shardings and declared types with the used outputs.
        let (operation, body) = mixed_known_unknown_shard_map_body();
        let pruning = operation
            .prune_boundary(2, &[true, false, false], &mut FixedRegionLiveness(vec![true, false]))
            .unwrap()
            .unwrap();
        assert_eq!(pruning.kept_inputs, vec![true, false]);
        assert_eq!(pruning.kept_outputs, vec![true, false, false]);
        assert_eq!(
            pruning.operation,
            ShardMapOperation::from_boundary(
                ShardMap::from_shardings(
                    manual_mesh(),
                    operation.shard_map().in_shardings()[0..1].to_vec(),
                    operation.shard_map().out_shardings()[0..1].to_vec(),
                    vec!["x".to_string()],
                ),
                operation.global_input_types()[0..1].to_vec(),
                vec![f32_scalar_type()],
            ),
        );

        // Pruning a program applies the rule: using only `a * x` keeps both inputs, which its computation reads, while
        // using only `a + a` also drops the input `x`, which only the dead outputs read.
        let program = |used_output: usize| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let known_input = builder.add_input(f32_scalar_type().into());
            let runtime_input = builder.add_input(f32_scalar_type().into());
            let body = builder.import_program(body.clone());
            let outputs = builder
                .add_instruction(operation.clone(), vec![body], vec![known_input, runtime_input], None)
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![outputs[used_output]],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        assert_eq!(
            program(1).into_pruned().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, []}, {mesh<['x'=2:manual]>, []}],
                    out_shardings=[{mesh<['x'=2:manual]>, []}],
                    manual_axes=['x'],
                    global_input_types=[f32[], f32[]],
                    global_output_types=[f32[]],
                ] %0 %1 [
                    body={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = mul %0 %1
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            program(0).into_pruned().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, []}],
                    out_shardings=[{mesh<['x'=2:manual]>, []}],
                    manual_axes=['x'],
                    global_input_types=[f32[]],
                    global_output_types=[f32[]],
                ] %0 [
                    body={
                        lambda %0:f32[] .
                        let %1:f32[] = add %0 %0
                        in (%1)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_shard_map_boundary_pruning_keeps_inputs_read_by_effects() {
        // Pruning keeps an input whose only remaining reader in the body is an observable effect, while it drops an
        // input that only a dead output reads.

        // The body over `[a, x, z]` returns `(a + a, z)` and prints `x` with a dead result.
        let array_type = f32_scalar_type();
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let inputs = (0..3).map(|_| builder.add_input(array_type.clone().into())).collect::<Vec<_>>();
            let doubled =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![inputs[0], inputs[0]], None).unwrap()[0];
            builder
                .add_instruction(
                    PrintOperation::<ArrayIrType>::new("x").with_effect_class(EffectClass::DeviceOrderedIo),
                    Vec::new(),
                    vec![inputs[1]],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![doubled, inputs[2]],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let operation = ShardMapOperation::from_boundary(
            ShardMap::from_shardings(mesh, vec![replicated.clone(); 3], vec![replicated; 2], vec!["x".to_string()]),
            vec![array_type.clone(); 3],
            vec![array_type.clone(); 2],
        );
        let program = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let inputs = (0..3).map(|_| builder.add_input(array_type.clone().into())).collect::<Vec<_>>();
            let body = builder.import_program(body);
            let outputs = builder.add_instruction(operation, vec![body], inputs, None).unwrap().to_vec();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![outputs[0]], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        assert_eq!(
            program.into_pruned().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[] .
                let %3:f32[] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, []}, {mesh<['x'=2:manual]>, []}],
                    out_shardings=[{mesh<['x'=2:manual]>, []}],
                    manual_axes=['x'],
                    global_input_types=[f32[], f32[]],
                    global_output_types=[f32[]],
                ] %0 %1 [
                    body={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = add %0 %0
                            %3:f32[] = print [label=x, effect_class=device_ordered_io] %1
                        in (%2)
                    },
                ]
                in (%3)"},
        );
    }

    #[test]
    fn test_shard_map_boundary_pruning_removes_pure_and_keeps_effectful_dead_shard_maps() {
        // A pure `shard_map` whose outputs are all dead is removed, while one whose body has observable effects is kept
        // with no outputs, and kept effectful maps retain their relative order, which orders their per-device effects.
        let array_type = f32_scalar_type();
        let printing_shard_map = |label: &str| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(array_type.clone().into());
            let output = builder
                .add_instruction(
                    PrintOperation::<ArrayIrType>::new(label).with_effect_class(EffectClass::DeviceOrderedIo),
                    Vec::new(),
                    vec![input],
                    None,
                )
                .unwrap()[0];
            let body = builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            let operation = ShardMapOperation::from_boundary(
                single_input_test_shard_map(),
                vec![array_type.clone()],
                vec![array_type.clone()],
            );
            (operation, body)
        };
        let program = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let known_input = builder.add_input(array_type.clone().into());
            let runtime_input = builder.add_input(array_type.clone().into());
            let (first_operation, first_body) = printing_shard_map("first");
            let first_body = builder.import_program(first_body);
            builder.add_instruction(first_operation, vec![first_body], vec![known_input], None).unwrap();
            let pure_operation = ShardMapOperation::from_boundary(
                single_input_test_shard_map(),
                vec![f32_scalar_type()],
                vec![f32_scalar_type()],
            );
            let pure_body = builder.import_program(identity_body(f32_scalar_type()));
            builder.add_instruction(pure_operation, vec![pure_body], vec![runtime_input], None).unwrap();
            let (second_operation, second_body) = printing_shard_map("second");
            let second_body = builder.import_program(second_body);
            builder.add_instruction(second_operation, vec![second_body], vec![runtime_input], None).unwrap();
            let (mixed_operation, mixed_body) = mixed_known_unknown_shard_map_body();
            let mixed_body = builder.import_program(mixed_body);
            let outputs = builder
                .add_instruction(mixed_operation, vec![mixed_body], vec![known_input, runtime_input], None)
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![outputs[2]], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        assert_eq!(
            program.into_pruned().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let () = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, []}],
                    out_shardings=[],
                    manual_axes=['x'],
                    global_input_types=[f32[]],
                    global_output_types=[],
                ] %0 [
                    body={
                        lambda %0:f32[] .
                        let %1:f32[] = print [label=first, effect_class=device_ordered_io] %0
                        in ()
                    },
                ]
                    () = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{mesh<['x'=2:manual]>, []}],
                        out_shardings=[],
                        manual_axes=['x'],
                        global_input_types=[f32[]],
                        global_output_types=[],
                    ] %1 [
                        body={
                            lambda %0:f32[] .
                            let %1:f32[] = print [label=second, effect_class=device_ordered_io] %0
                            in ()
                        },
                    ]
                    %2:f32[] = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{mesh<['x'=2:manual]>, []}, {mesh<['x'=2:manual]>, []}],
                        out_shardings=[{mesh<['x'=2:manual]>, []}],
                        manual_axes=['x'],
                        global_input_types=[f32[], f32[]],
                        global_output_types=[f32[]],
                    ] %0 %1 [
                        body={
                            lambda %0:f32[], %1:f32[] .
                            let %2:f32[] = add %1 %0
                            in (%2)
                        },
                    ]
                in (%2)"},
        );
    }

    #[test]
    fn test_shard_map_boundary_pruning_of_fused_jvp_programs() {
        // Pruning a forward-mode program that only uses the output tangents of a fused `shard_map` drops its primal
        // outputs (and the body computation that only they use, here `a + a`) but keeps the primal computation that
        // the tangents use. The body maps `(a, x)` to `(a + a, sin(a) * x)`.
        let replicated = Sharding::replicated(manual_mesh(), 0);
        let program = trace_test_program(
            |inputs| {
                let (sum, product) = shard_map(
                    |(a, x): (TestTracer, TestTracer)| (a.clone() + a.clone(), a.sin().unwrap() * x),
                    (inputs[0].clone(), inputs[1].clone()),
                    manual_mesh(),
                    (replicated.clone(), replicated.clone()),
                    (replicated.clone(), replicated.clone()),
                )
                .unwrap();
                vec![sum, product]
            },
            vec![f32_scalar_type(), f32_scalar_type()],
            Vec::new(),
        );
        let jvp = program.jvp().unwrap();
        let tangent_outputs = jvp.output_ids()[2..].to_vec();
        let program = jvp.filtered(jvp.input_ids(), &tangent_outputs, &[]).unwrap().0;
        let scalar = f32_scalar_type().with_sharding(replicated.clone()).unwrap();
        assert_eq!(
            program.into_pruned().unwrap().to_string(),
            formatdoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[], %3:f32[] .
                let %4:{scalar}, %5:{scalar} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}, {replicated}, {replicated}, {replicated}],
                    out_shardings=[{replicated}, {replicated}],
                    manual_axes=['x'],
                    global_input_types=[{scalar}, {scalar}, {scalar}, {scalar}],
                    global_output_types=[{scalar}, {scalar}],
                ] %0 %1 %2 %3 [
                    body={{
                        lambda %0:{scalar}, %1:{scalar}, %2:{scalar}, %3:{scalar} .
                        let %4:{scalar} = add %2 %2
                            %5:{scalar} = sin %0
                            %6:{scalar} = cos %0
                            %7:{scalar} = mul %6 %2
                            %8:{scalar} = mul %1 %7
                            %9:{scalar} = mul %5 %3
                            %10:{scalar} = add %8 %9
                        in (%4, %10)
                    }},
                ]
                in (%4, %5)"},
        );
    }

    #[test]
    fn test_shard_map_boundary_pruning_reaches_nested_shard_maps() {
        // Pruning reaches a `shard_map` nested in the body of another `shard_map`: the nested boundary drops the
        // outputs that the enclosing body does not use, together with the body computation that only they use.
        let mesh = manual_mesh_2x2();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let array_type = f32_scalar_type();
        let inner_shard_map = ShardMap::from_shardings(
            mesh.clone(),
            vec![replicated.clone(); 2],
            vec![replicated.clone(); 3],
            vec!["y".to_string()],
        );
        let (inner_operation, inner_body) = mixed_known_unknown_shard_map_body();
        let inner_operation = ShardMapOperation::from_boundary(
            inner_shard_map,
            inner_operation.global_input_types().to_vec(),
            inner_operation.global_output_types().to_vec(),
        );
        let outer_body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let known_input = builder.add_input(array_type.clone().into());
            let runtime_input = builder.add_input(array_type.clone().into());
            let inner_body = builder.import_program(inner_body);
            let outputs = builder
                .add_instruction(inner_operation, vec![inner_body], vec![known_input, runtime_input], None)
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![outputs[1]], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let outer_operation = ShardMapOperation::from_boundary(
            ShardMap::from_shardings(mesh, vec![replicated.clone(); 2], vec![replicated], vec!["x".to_string()]),
            vec![array_type.clone(); 2],
            vec![array_type],
        );
        let program = shard_map_program(outer_operation, outer_body).unwrap();
        assert_eq!(
            program.into_pruned().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = shard_map [
                    mesh=['x'=2:manual, 'y'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, []}, {mesh<['x'=2:manual, 'y'=2:manual]>, []}],
                    out_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, []}],
                    manual_axes=['x'],
                    global_input_types=[f32[], f32[]],
                    global_output_types=[f32[]],
                ] %0 %1 [
                    body={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = shard_map [
                            mesh=['x'=2:manual, 'y'=2:manual],
                            in_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, []}, {mesh<['x'=2:manual, \
                    'y'=2:manual]>, []}],
                            out_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, []}],
                            manual_axes=['y'],
                            global_input_types=[f32[], f32[]],
                            global_output_types=[f32[]],
                        ] %0 %1 [
                            body={
                                lambda %0:f32[], %1:f32[] .
                                let %2:f32[] = mul %0 %1
                                in (%2)
                            },
                        ]
                        in (%2)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_shard_map_boundary_pruning_in_residual_policy_programs() {
        // Placing residuals with a policy prunes both resulting programs, so a `shard_map` that the residual program
        // replays and the known program keeps for another output computes only the outputs that each program uses.

        // `f(a, y) = (sin(a) * y, cos(a))`, where one replicated `shard_map` computes both `sin(a)` and `cos(a)`, with
        // `a` known and `y` unknown. Saving nothing recomputes `sin(a)` in the residual program.
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let array_type = f32_scalar_type();
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(array_type.clone().into());
            let sine = builder
                .add_instruction(ArrayOperation::Sin(SinOperation::new()), Vec::new(), vec![input], None)
                .unwrap()[0];
            let cosine = builder
                .add_instruction(ArrayOperation::Cos(CosOperation::new()), Vec::new(), vec![input], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![sine, cosine], vec![Placeholder], vec![Placeholder; 2])
                .unwrap()
        };
        let operation = ShardMapOperation::from_boundary(
            ShardMap::from_shardings(mesh, vec![replicated.clone()], vec![replicated; 2], vec!["x".to_string()]),
            vec![array_type.clone()],
            vec![array_type.clone(); 2],
        );
        let program = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let known_input = builder.add_input(array_type.clone().into());
            let runtime_input = builder.add_input(array_type.into());
            let body = builder.import_program(body);
            let outputs = builder.add_instruction(operation, vec![body], vec![known_input], None).unwrap().to_vec();
            let product = builder
                .add_instruction(
                    ArrayOperation::Mul(MulOperation::new()),
                    Vec::new(),
                    vec![outputs[0], runtime_input],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![product, outputs[1]],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let partition = program
            .partition(&[true, false])
            .unwrap()
            .with_residual_policy(&ResidualPolicyReference::new(NothingSavable))
            .unwrap();
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0), Known(0)],
                ]
                known={
                    lambda %0:f32[] .
                    let %1:f32[] = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{mesh<['x'=2:manual]>, []}],
                        out_shardings=[{mesh<['x'=2:manual]>, []}],
                        manual_axes=['x'],
                        global_input_types=[f32[]],
                        global_output_types=[f32[]],
                    ] %0 [
                        body={
                            lambda %0:f32[] .
                            let %1:f32[] = cos %0
                            in (%1)
                        },
                    ]
                    in (%1, %0)
                }
                residual={
                    lambda %0:f32[], %1:f32[] .
                    let %2:f32[] = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{mesh<['x'=2:manual]>, []}],
                        out_shardings=[{mesh<['x'=2:manual]>, []}],
                        manual_axes=['x'],
                        global_input_types=[f32[]],
                        global_output_types=[f32[]],
                    ] %1 [
                        body={
                            lambda %0:f32[] .
                            let %1:f32[] = sin %0
                            in (%1)
                        },
                    ]
                        %3:f32[] = mul %2 %0
                    in (%3)
                }"},
        );
    }

    #[test]
    fn test_shard_map_boundary_pruning_keeps_reference_boundaries_whole() {
        // A boundary with a reference position keeps its whole boundary, because the reference contract was validated
        // against it, even when none of its outputs is used.
        let sharded = sharded_along_x();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded, true);
        let program = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let inputs = operation
                .global_input_types()
                .iter()
                .cloned()
                .map(|r#type| builder.add_input(r#type))
                .collect::<Vec<_>>();
            let body = builder.import_program(body);
            builder.add_instruction(operation, vec![body], inputs, None).unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
                .unwrap()
        };
        assert_eq!(
            program.into_pruned().unwrap().to_string(),
            indoc! {"
                lambda %0:ref<f32[4]>, %1:f32[4] .
                let %2:f32[4] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[ref<f32[4]>, f32[4]],
                    global_output_types=[f32[4]],
                ] %0 %1 [
                    body={
                        lambda %0:ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]>, \
                    %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let () = reference_add_update %0 %1
                            %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reference_read %0
                        in (%2)
                    },
                ]
                in ()"},
        );
    }

    #[test]
    fn test_shard_map_reference_discharge() {
        // Discharging a shard map with a sharded reference input threads the state carry with the local shard type
        // inside the body and with the global referent at the boundary: the reference position becomes the global
        // referent under its input sharding, the mutated input publishes a hidden final-state output under that same
        // sharding, and the body's state carry is the `f32[2]` shard the device owns.
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded_along_x(), true);
        let program = shard_map_program(operation, body).unwrap();
        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Input { index: 0 });
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(1));
        assert!(!discharged.program().entry_region_ref().contains_references_in_closure());

        // The reference position carries the global referent, the value output keeps its position, and the final state
        // follows it under the reference input's sharding, while the body threads the local `f32[2]` shards.
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4], %3:f32[4] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4], f32[4]],
                    global_output_types=[f32[4], f32[4]],
                ] %0 %1 [
                    body={
                        lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = add %0 %1
                        in (%2, %2)
                    },
                ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_shard_map_reference_discharge_types_final_state_by_the_input_referent() {
        // The hidden final-state output is typed by the input that the mutated reference position replaces: when the
        // staged input carries sharding metadata that the declared global referent omits, the final state carries the
        // input's type, since the discharged state must match the allocation's referent exactly.
        let sharded = sharded_along_x();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded.clone(), true);
        let sharded_type = f32_vector_type(4).with_sharding(sharded).unwrap();
        let program = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(sharded_type.clone())));
            let update = builder.add_input(ArrayIrType::Array(sharded_type.clone()));
            let body_region = builder.import_program(body);
            let operation = TestOperation::ShardMap(Box::new(operation));
            let outputs = builder
                .add_instruction(operation, vec![body_region], vec![reference, update], None)
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda \
                %0:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}], %1:f32[4][sharding={mesh<['x'=2:manual]>, \
                [{'x'}]}] .
                let %2:f32[4], %3:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4], f32[4]],
                    global_output_types=[f32[4], f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ] %0 %1 [
                    body={
                        lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = add %0 %1
                        in (%2, %2)
                    },
                ]
                in (%2, %3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_shard_map_reference_discharge_of_replicated_references() {
        // Every device holds its own copy of a reference input that is replicated along the manual axis. Reading it
        // leaves it unmutated, so it has no final-state output.
        let replicated = Sharding::replicated(manual_mesh(), 1);
        let (operation, body) = reference_shard_map(f32_vector_type(2), replicated.clone(), false);
        let program = shard_map_program(operation, body).unwrap();
        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[4] .
                let %2:f32[4] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[2], f32[4]],
                    global_output_types=[f32[4]],
                ] %0 %1 [
                    body={
                        lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}], \
                    %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary \
                    [axis_name=\"x\"] %0
                            %3:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = add %2 %1
                        in (%3)
                    },
                ]
                in (%2)"},
        );

        // Mutating it with a value of its own (invariant) referent type at invariant indices keeps all copies
        // identical, so its final state is invariant along `x` and leaves the map as an output under its replicated
        // input sharding, which the reference contract accepts. The output read varies along `x` explicitly, as its
        // output sharding tiles that axis.
        let (operation, body) = reference_shard_map(f32_vector_type(2), replicated, false);
        let mutating_body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(body.input_types()[0].clone());
            builder.add_input(body.input_types()[1].clone());
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, state], None)
                .unwrap();
            let output =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let output = builder
                .add_instruction(ParallelVaryOperation::new("x".to_string()), Vec::new(), vec![output], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let program = shard_map_program(operation, mutating_body).unwrap();
        let discharged = program.clone().discharge_references(0).unwrap();
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert!(discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[4] .
                let %2:f32[4], %3:f32[2] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[2], f32[4]],
                    global_output_types=[f32[4], f32[2]],
                ] %0 %1 [
                    body={
                        lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}], \
                    %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = add %0 %0
                            %3:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary \
                    [axis_name=\"x\"] %2
                        in (%3, %2)
                    },
                ]
                in (%2, %3)"},
        );

        // The reference backend writes back the copy of the first device, which every device agrees with.
        let global_type = f32_vector_type(2);
        let reference = ArrayReference::new(Array::from_elements(global_type.clone(), &[1.0f32, 2.0]).unwrap());
        let value = TestValue::Array(Array::from_elements(f32_vector_type(4), &[0.0f32; 4]).unwrap());
        let outputs = program.interpret(vec![TestValue::Reference(reference.clone()), value]).unwrap();
        assert_eq!(arrays_f64(outputs), vec![vec![2.0, 4.0, 2.0, 4.0]]);
        assert_eq!(reference.read(), Ok(Array::from_elements(global_type, &[2.0f32, 4.0]).unwrap()));
    }

    #[test]
    fn test_shard_map_reference_reads_at_varying_indices() {
        // Reading a reference that is replicated along `x` at the coordinate of the executing device along `x` selects
        // a different element on every device, so the value read varies along `x`, and a replicated output sharding,
        // which would claim that the output is identical across devices, is rejected when the body is attached.
        let mesh = manual_mesh();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![Sharding::replicated(mesh.clone(), 1)],
            vec![Sharding::replicated(mesh.clone(), 0)],
            Vec::new(),
        )
        .unwrap();
        let body = varying_index_read_body(shard_map.local_input_type(0, &f32_vector_type(2)).unwrap());
        assert_eq!(
            body.to_string(),
            indoc! {"
                lambda %0:ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]> .
                let %1:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = axis_index \
                [axis_name=\"x\", mesh=['x'=2:manual]]
                    %2:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reference_read \
                [transforms=[index(axis=0, index=dynamic)]] %0 %1
                in (%2)"},
        );
        assert_eq!(
            ShardMapOperation::from_program(&body, vec![ReferenceType::new(f32_vector_type(2)).into()], shard_map)
                .map(|_| ()),
            Err(ShardMapError::OutputVaryingAlongUntiledManualAxis { output_index: 0, axis_name: "x".to_string() }),
        );

        // Under an output sharding that tiles `x`, each device contributes the row that it reads. Discharge varies the
        // replicated state along `x` before slicing it at the varying index, like the value-level `dynamic_slice`.
        let table_type = ArrayType::new_static(DataType::F32, [2, 1]);
        let shard_map =
            ShardMap::new(mesh.clone(), vec![Sharding::replicated(mesh, 2)], vec![sharded_along_x()], Vec::new())
                .unwrap();
        let body = varying_index_read_body(shard_map.local_input_type(0, &table_type).unwrap());
        let operation =
            ShardMapOperation::from_program(&body, vec![ReferenceType::new(table_type.clone()).into()], shard_map)
                .unwrap();
        let program = shard_map_program(operation, body).unwrap();
        assert_eq!(
            program.clone().discharge_references(0).unwrap().program().to_string(),
            indoc! {"
                lambda %0:f32[2, 1][sharding={mesh<['x'=2:manual]>, [{}, {}]}] .
                let %1:f32[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{}, {}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[2, 1][sharding={mesh<['x'=2:manual]>, [{}, {}]}]],
                    global_output_types=[f32[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ] %0 [
                    body={
                        lambda %0:f32[2, 1][sharding={mesh<['x'=2:manual]>, [{}, {}]}] .
                        let %1:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = axis_index \
                    [axis_name=\"x\", mesh=['x'=2:manual]]
                            %2:f32[2, 1][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                    parallel_vary [axis_name=\"x\"] %0
                            %3:f32[1, 1][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                    dynamic_slice [sizes=[1, 1]] %2 %1 %1
                            %4:f32[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reshape \
                    [shape=[1]] %3
                        in (%4)
                    },
                ]
                in (%1)"},
        );
        let table_type = table_type.with_sharding(Sharding::replicated(manual_mesh(), 2)).unwrap();
        let table = ArrayReference::new(Array::from_elements(table_type.clone(), &[10.0f32, 20.0]).unwrap());
        let outputs = program.interpret(vec![TestValue::Reference(table.clone())]).unwrap();
        assert_eq!(arrays_f64(outputs), vec![vec![10.0, 20.0]]);
        assert_eq!(table.read(), Ok(Array::from_elements(table_type, &[10.0f32, 20.0]).unwrap()));
    }

    #[test]
    fn test_shard_map_rejects_writes_into_replicated_references_under_varying_predicates() {
        // A `condition` whose predicate varies along `x` lets the devices take different branches, so a branch that
        // writes a reference replicated along `x` would leave its copies different. The branches are well-typed on
        // their own, so the write is rejected when the map is discharged, and the reference backend rejects the
        // diverged copies after running the devices.
        let mesh = manual_mesh();
        let shard_map =
            ShardMap::new(mesh.clone(), vec![Sharding::replicated(mesh, 1)], Vec::new(), Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &f32_vector_type(2)).unwrap();
        let branch = |write: bool| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(ReferenceType::new(local_type.clone()).into());
            if write {
                let state = builder
                    .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
                    .unwrap()[0];
                let one = builder
                    .add_instruction(ArrayOperation::OneLike(OneLikeOperation::new()), Vec::new(), vec![state], None)
                    .unwrap()[0];
                builder
                    .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, one], None)
                    .unwrap();
            }
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(ReferenceType::new(local_type.clone()).into());
            let index = add_axis_index(&mut builder);
            let zero = builder
                .add_instruction(ArrayOperation::ZeroLike(ZeroLikeOperation::new()), Vec::new(), vec![index], None)
                .unwrap()[0];
            let compare = ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::Equal));
            let predicate = builder.add_instruction(compare, Vec::new(), vec![index, zero], None).unwrap()[0];
            let branches = vec![builder.import_program(branch(true)), builder.import_program(branch(false))];
            builder
                .add_instruction(
                    TestOperation::Condition(ConditionOperation::new()),
                    branches,
                    vec![predicate, reference],
                    None,
                )
                .unwrap();
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap()
        };
        let operation =
            ShardMapOperation::from_program(&body, vec![ReferenceType::new(f32_vector_type(2)).into()], shard_map)
                .unwrap();
        let program = shard_map_program(operation, body).unwrap();
        assert_eq!(
            program.clone().discharge_references(0).map(|_| ()),
            Err(TypeError::invalid(
                "`condition` branch 0 mutates reference input 1 of type \
                 `ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]>`, whose referent does not vary over every manual \
                 axis that the predicate `bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]` varies \
                 over, so devices that take different branches would leave it different across devices; vary the \
                 referent over those axes",
            )
            .into()),
        );
        let global_type = f32_vector_type(2).with_sharding(Sharding::replicated(manual_mesh(), 1)).unwrap();
        let reference = ArrayReference::new(Array::from_elements(global_type, &[10.0f32, 20.0]).unwrap());
        assert_eq!(
            program.interpret(vec![TestValue::Reference(reference)]),
            Err(ProgramError::MalformedProgram(
                "`shard_map` reference input #0 is replicated along manual axis `x`, but its devices along that axis \
                 finished with different states"
                    .to_string(),
            )),
        );
    }

    #[test]
    fn test_shard_map_reference_discharge_rejects_repeated_reference_input_allocations() {
        // Each reference input reaches the body as independently owned local state, so two reference inputs denoting
        // one allocation cannot keep their aliasing and are rejected before the body is rebuilt.
        let sharded = sharded_along_x();
        let shard_map = ShardMap::from_shardings(
            manual_mesh(),
            vec![sharded.clone(), sharded.clone(), sharded.clone()],
            vec![sharded],
            vec!["x".to_string()],
        );
        let reference_type = ArrayIrType::Reference(ReferenceType::new(f32_vector_type(4)));
        let value_type = ArrayIrType::Array(f32_vector_type(4));
        let local_value_type = ArrayIrType::Array(shard_map.local_input_type(2, &f32_vector_type(4)).unwrap());
        let local_reference_type =
            ArrayIrType::Reference(ReferenceType::new(shard_map.local_input_type(0, &f32_vector_type(4)).unwrap()));
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let written = builder.add_input(local_reference_type.clone());
            let read = builder.add_input(local_reference_type);
            let update = builder.add_input(local_value_type);
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![written, update], None)
                .unwrap();
            let output =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![read], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let operation = ShardMapOperation::from_boundary(
            shard_map,
            vec![reference_type.clone(), reference_type.clone(), value_type.clone()],
            vec![value_type.clone()],
        );
        let program = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type);
            let update = builder.add_input(value_type);
            let body = builder.import_program(body);
            let outputs = builder
                .add_instruction(
                    TestOperation::ShardMap(Box::new(operation)),
                    vec![body],
                    vec![reference, reference, update],
                    None,
                )
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        let expected = ShardMapError::RepeatedReferenceInputAllocation { first_input_index: 0, second_input_index: 1 };
        assert_eq!(
            expected.to_string(),
            "`shard_map` reference inputs #0 and #1 denote the same allocation; pass distinct allocations",
        );
        assert!(matches!(
            program.discharge_references(0),
            Err(error) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
    }

    #[test]
    fn test_shard_map_reference_discharge_inherits_targets_in_its_body() {
        // The body is rebuilt under the caller's discharge targets, which name its allocations by their source
        // instruction identities. Discharging the external reference input and one of two allocations made inside the
        // body therefore discharges exactly those two, even though the body forwards the reference input and so is
        // normalized before it is rebuilt: the selected allocation becomes explicit local state, the unselected one
        // keeps its `reference_new` in the rebuilt body, the mutated external view publishes its final local state, and
        // a read through the forwarded output observes that state.
        let sharded = sharded_along_x();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded.clone(), true);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(f32_vector_type(4)));
        let value_type = ArrayIrType::Array(f32_vector_type(4));
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let inputs = body.input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
            let (reference, update) = (inputs[0], inputs[1]);
            let selected =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![update], None).unwrap()[0];
            let unselected =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![update], None).unwrap()[0];
            for target in [reference, selected, unselected] {
                builder
                    .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![target, update], None)
                    .unwrap();
            }
            let selected =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![selected], None).unwrap()[0];
            let unselected =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![unselected], None).unwrap()[0];
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![selected, unselected], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![output, reference],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let operation = ShardMapOperation::from_boundary(
            ShardMap::from_shardings(
                manual_mesh(),
                operation.shard_map().in_shardings().to_vec(),
                vec![sharded.clone(), sharded],
                vec!["x".to_string()],
            ),
            operation.global_input_types().to_vec(),
            vec![value_type.clone(), reference_type.clone()],
        )
        .with_output_forwarding(vec![None, Some(0)])
        .unwrap();
        let program = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type);
            let update = builder.add_input(value_type);
            let body = builder.import_program(body);
            let outputs = builder
                .add_instruction(
                    TestOperation::ShardMap(Box::new(operation)),
                    vec![body],
                    vec![reference, update],
                    None,
                )
                .unwrap()
                .to_vec();
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![outputs[1]], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![outputs[0], state],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let body = program.entry_region_ref().instructions()[0].regions()[0];
        let targets = [
            ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 }),
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(body, 0), output_index: 0 },
        ];
        assert_eq!(
            program.reference_discharge_targets(0).unwrap(),
            vec![
                targets[0],
                targets[1],
                ReferenceDischargeTarget::Internal { instruction: InstructionId::new(body, 1), output_index: 0 },
            ],
        );

        let discharged = program.partially_discharge_references(0, &targets).unwrap();
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Input { index: 0 });
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(2));
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4], %3:f32[4] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4], f32[4]],
                    global_output_types=[f32[4], f32[4]],
                ] %0 %1 [
                    body={
                        lambda \
                        %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                        %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let \
                        %2:ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]> = reference_new %1
                            %3:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = add %0 %1
                            %4:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = add %1 %1
                            () = reference_add_update %2 %1
                            %5:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reference_read %2
                            %6:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = add %4 %5
                        in (%6, %3)
                    },
                ]
                in (%2, %3, %3)"
            },
        );
    }

    #[test]
    fn test_shard_map_reference_discharge_rejects_preserved_references() {
        // A manual region threads state rather than destination references, so a reference input whose allocation the
        // caller does not discharge is rejected instead of being passed into the body.
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded_along_x(), true);
        let program = shard_map_program(operation, body).unwrap();
        let Err(ProgramError::UnsupportedOperation { message }) = program.partially_discharge_references(0, &[]) else {
            panic!("expected the preserved reference input to be rejected");
        };

        // The environment identity of an allocation is process-global, so only its position is pinned.
        let environment = message
            .strip_prefix(
                "`shard_map` does not thread a preserved reference through its body, but input #0 denotes preserved \
                 reference allocation ",
            )
            .and_then(|message| message.strip_suffix(":0; discharge it or pass a value"));
        assert!(environment.is_some_and(|environment| environment.parse::<usize>().is_ok()), "{message}");
    }

    #[test]
    fn test_shard_map_reference_discharge_rejects_captured_references() {
        // A reference reaching the body as a captured constant has no input sharding and therefore no owner.
        type CapturedValue = CaptureReference<ArrayIrType>;
        type CapturedOperation = ArrayIrOperation<CaptureReference<ArrayType>>;

        let sharded = sharded_along_x();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded, true);
        let local_reference_type = body.input_types()[0].clone();
        let capturing_body = {
            let mut builder = ProgramBuilder::<CapturedValue, CapturedOperation>::new();
            let reference = builder.add_input(local_reference_type.clone());
            let update = builder.add_input(body.input_types()[1].clone());
            let captured = builder.add_constant(CaptureReference::new(0, local_reference_type));
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![captured, update], None)
                .unwrap();
            let output =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<CapturedValue>, Vec<CapturedValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let program = {
            let mut builder = ProgramBuilder::<CapturedValue, CapturedOperation>::new();
            let inputs = operation
                .global_input_types()
                .iter()
                .cloned()
                .map(|r#type| builder.add_input(r#type))
                .collect::<Vec<_>>();
            let body_region = builder.import_program(capturing_body);
            let outputs = builder
                .add_instruction(CapturedOperation::ShardMap(Box::new(operation)), vec![body_region], inputs, None)
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<CapturedValue>, Vec<CapturedValue>>(outputs, vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let body = program.entry_region_ref().instructions()[0].regions()[0];
        let captured = program.region_ref(body).unwrap().instructions()[0].inputs()[0];
        assert_eq!(
            program.discharge_references(0).unwrap_err(),
            ReferenceAnalysisError::InvalidReferenceCapture {
                region: body,
                atom: captured,
                capture_index: 0,
                capture_count: 0,
            }
            .into(),
        );
    }

    #[test]
    fn test_shard_map_reference_discharge_keeps_the_forwarded_root() {
        // A reference output forwards a reference input by identity under an equal sharding; the caller keeps the
        // handle it passed, a mismatched output sharding is rejected at construction, and an allocation made inside the
        // body cannot escape.
        let sharded = sharded_along_x();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded.clone(), true);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(f32_vector_type(4)));
        let forwarding_body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let inputs = body.input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
            let mut outputs = builder.splice_program(&body, inputs.as_slice()).unwrap();
            outputs.push(inputs[0]);
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let forwarding_operation = |output_sharding: Sharding| {
            ShardMapOperation::from_boundary(
                ShardMap::from_shardings(
                    manual_mesh(),
                    operation.shard_map().in_shardings().to_vec(),
                    vec![sharded.clone(), output_sharding],
                    vec!["x".to_string()],
                ),
                operation.global_input_types().to_vec(),
                vec![ArrayIrType::Array(f32_vector_type(4)), reference_type.clone()],
            )
            .with_output_forwarding(vec![None, Some(0)])
            .unwrap()
        };

        // Equal shardings: the outer program reads through the forwarded output, which resolves to the same root as
        // the reference input, so discharge publishes exactly one final state and the read observes it.
        let program = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(ArrayIrType::Array(f32_vector_type(4)));
            let body_region = builder.import_program(forwarding_body.clone());
            let outputs = builder
                .add_instruction(
                    TestOperation::ShardMap(Box::new(forwarding_operation(sharded.clone()))),
                    vec![body_region],
                    vec![reference, update],
                    None,
                )
                .unwrap()
                .to_vec();
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![outputs[1]], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![outputs[0], state],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(discharged.output_count(), 2);
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(2));
        // The read through the forwarded handle observes the final state published by the shard map, which is also
        // the published final state of the reference input.
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4], %3:f32[4] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4], f32[4]],
                    global_output_types=[f32[4], f32[4]],
                ] %0 %1 [
                    body={
                        lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = add %0 %1
                        in (%2, %2)
                    },
                ]
                in (%2, %3, %3)"},
        );

        // Unequal shardings are rejected when the instruction is built.
        let replicated = Sharding::replicated(manual_mesh(), 1);
        assert!(matches!(
            shard_map_program(forwarding_operation(replicated.clone()), forwarding_body),
            Err(ProgramError::Type(error))
                if error == TypeError::invalid(format!(
                    "`shard_map` output #1 forwards reference input #0 but its output sharding `{replicated}` differs \
                     from the input sharding `{sharded}`",
                )),
        ));

        // An allocation made inside the body cannot escape through a reference output, even when the operation
        // declares it as forwarding an input of the same local type.
        let escaping_body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let inputs = body.input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
            let mut outputs = builder.splice_program(&body, inputs.as_slice()).unwrap();
            let allocation =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![inputs[1]], None).unwrap()[0];
            outputs.push(allocation);
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        assert!(matches!(
            forwarding_operation(sharded.clone()).validate_reference_body(escaping_body.entry_region_ref()),
            Err(ProgramError::MalformedProgram(message))
                if message == "`shard_map` output #1 is a reference allocated inside the shard-map body, which cannot \
                               escape; a reference output must forward a reference input",
        ));
    }

    #[test]
    fn test_shard_map_partial_evaluation() {
        // Online partial evaluation of a mixed `shard_map` against a live outer trace: the known half of the local body
        // is rewrapped as a known-side `shard_map` staged into the outer program over the symbolic known input, the
        // unknown half stays behind a residual `shard_map`, the known-to-unknown residual edges flow between them, and
        // the mesh and shardings are threaded onto both boundaries.
        let array_type = f32_scalar_type();
        let (operation, body) = mixed_known_unknown_shard_map_body();
        let program = shard_map_program(operation, body).unwrap();

        let outer = TestContext::new();
        let known = outer.input(ArrayIrType::Array(array_type.clone()));
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(known), PartialValue::Unknown(ArrayIrType::Array(array_type))],
            )
            .unwrap();

        // The boundary descriptors: the unknown enclosing input feeds the residual side, the residual edge is a known
        // feeder, and the outputs reassemble in original order.
        let [PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(residual)] = evaluation.inputs() else {
            panic!("expected the unknown input followed by the residual edge");
        };
        let [
            PartialEvaluationOutput::Known(known_output),
            PartialEvaluationOutput::Unknown(0),
            PartialEvaluationOutput::Unknown(1),
        ] = evaluation.outputs()
        else {
            panic!("expected one known output followed by two unknown outputs");
        };

        // The known half landed in the outer program as one known-side `shard_map` over the symbolic known input,
        // producing only the fully known boundary output (`a + a`). The residual edge is `a` itself, which the known
        // half merely forwards, so it is fed by the known boundary input directly (JAX's `in_fwd`) rather than routed
        // through the known-side `shard_map` as an extra output.
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![known_output.atom_id().unwrap(), residual.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, []}],
                    out_shardings=[{mesh<['x'=2:manual]>, []}],
                    manual_axes=['x'],
                    global_input_types=[f32[]],
                    global_output_types=[f32[]],
                ] %0 [
                    body={
                        lambda %0:f32[] .
                        let %1:f32[] = add %0 %0
                        in (%1)
                    },
                ]
                in (%1, %0)"},
        );

        // The unknown half stayed behind one residual `shard_map` over the unknown boundary input plus the residual
        // edge, with the shardings threaded per input.
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[], %3:f32[] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, []}, {mesh<['x'=2:manual]>, []}],
                    out_shardings=[{mesh<['x'=2:manual]>, []}, {mesh<['x'=2:manual]>, []}],
                    manual_axes=['x'],
                    global_input_types=[f32[], f32[]],
                    global_output_types=[f32[], f32[]],
                ] %0 %1 [
                    body={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = mul %1 %0
                            %3:f32[] = add %0 %1
                        in (%2, %3)
                    },
                ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_shard_map_partial_evaluation_splits_concrete_known_inputs() {
        // Partial evaluation splits a mixed `shard_map` whose known inputs are concrete values of an eager known-side
        // context exactly as it splits one whose known inputs are symbolic, as JAX's `_shard_map_partial_eval` does:
        // the known-side `shard_map` executes immediately (here, through the reference backend's emulation), so the
        // output `a + a` is known and concrete, and the residual `shard_map` keeps only the work that needs `x`, fed by
        // the known input `a` directly (JAX's `in_fwd`).
        let (operation, body) = mixed_known_unknown_shard_map_body();
        let program = shard_map_program(operation, body).unwrap();
        let a = ArrayIrValue::Array(Array::scalar(2.0f32).unwrap());
        let x = ArrayIrValue::Array(Array::scalar(3.0f32).unwrap());
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(a.clone()), PartialValue::Unknown(f32_scalar_type().into())])
            .unwrap();
        assert!(matches!(
            evaluation.inputs(),
            [PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(value)] if value == &a,
        ));
        assert!(matches!(
            evaluation.outputs(),
            [
                PartialEvaluationOutput::Known(value),
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Unknown(1),
            ] if value == &ArrayIrValue::Array(Array::scalar(4.0f32).unwrap()),
        ));
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[], %3:f32[] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, []}, {mesh<['x'=2:manual]>, []}],
                    out_shardings=[{mesh<['x'=2:manual]>, []}, {mesh<['x'=2:manual]>, []}],
                    manual_axes=['x'],
                    global_input_types=[f32[], f32[]],
                    global_output_types=[f32[], f32[]],
                ] %0 %1 [
                    body={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = mul %1 %0
                            %3:f32[] = add %0 %1
                        in (%2, %3)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(
            evaluation.interpret(&TestEagerContext::new(), std::slice::from_ref(&x)),
            program.interpret(vec![a, x]),
        );
    }

    #[test]
    fn test_shard_map_partial_evaluation_executes_known_collectives() {
        // The known-side `shard_map` of a split over concrete known inputs executes per device, so a collective over a
        // manual axis in the known half combines the shards of every device: with `a = [1, 2, 3, 4]` sharded along
        // `x`, the known output `parallel_reduce(a)` is `[1 + 3, 2 + 4]`, and only `a * x` stays behind the residual
        // `shard_map`.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let global_type = f32_vector_type(4).with_sharding(sharded.clone()).unwrap();
        let program = trace_test_program(
            |inputs| {
                let (sum, product) = shard_map(
                    |(a, x): (TestTracer, TestTracer)| (a.parallel_reduce(ReductionKind::Sum, "x").unwrap(), a * x),
                    (inputs[0].clone(), inputs[1].clone()),
                    mesh.clone(),
                    (sharded.clone(), sharded.clone()),
                    (replicated.clone(), sharded.clone()),
                )
                .unwrap();
                vec![sum, product]
            },
            vec![global_type.clone(), global_type.clone()],
            Vec::new(),
        );
        let a = ArrayIrValue::Array(Array::from_elements(global_type.clone(), &[1f32, 2.0, 3.0, 4.0]).unwrap());
        let x = ArrayIrValue::Array(Array::from_elements(global_type.clone(), &[5f32, 6.0, 7.0, 8.0]).unwrap());
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(a.clone()), PartialValue::Unknown(global_type.clone().into())])
            .unwrap();
        let sum_type = f32_vector_type(2).with_sharding(replicated).unwrap();
        assert!(matches!(
            evaluation.outputs(),
            [PartialEvaluationOutput::Known(value), PartialEvaluationOutput::Unknown(0)]
                if value == &ArrayIrValue::Array(Array::from_elements(sum_type, &[4f32, 6.0]).unwrap()),
        ));
        let local_type = f32_vector_type(2)
            .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:{global_type}, %1:{global_type} .
                let %2:{global_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharded}, {sharded}],
                    out_shardings=[{sharded}],
                    manual_axes=['x'],
                    global_input_types=[{global_type}, {global_type}],
                    global_output_types=[{global_type}],
                ] %0 %1 [
                    body={{
                        lambda %0:{local_type}, %1:{local_type} .
                        let %2:{local_type} = mul %1 %0
                        in (%2)
                    }},
                ]
                in (%2)"},
        );
        assert_eq!(
            evaluation.interpret(&TestEagerContext::new(), std::slice::from_ref(&x)),
            program.interpret(vec![a, x]),
        );
    }

    #[test]
    fn test_shard_map_partial_evaluation_of_jvp_programs_at_concrete_primals() {
        // Partially evaluating a forward-mode program at a concrete primal splits its fused `shard_map`: the known
        // half, which computes the output and the residual `cos(x)`, executes, and the residual `shard_map`, which
        // computes the output tangent, is the only instruction of the residual program.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let global_type = f32_vector_type(4).with_sharding(sharded.clone()).unwrap();
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.clone().sin().unwrap() * x,
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![global_type.clone()],
            Vec::new(),
        );
        let jvp = program.jvp().unwrap();
        let x = ArrayIrValue::Array(Array::from_elements(global_type.clone(), &[0.5f32, 1.0, 1.5, 2.0]).unwrap());
        let tangent = ArrayIrValue::Array(Array::from_elements(global_type, &[1f32, -1.0, 2.0, 0.5]).unwrap());
        let tangent_type = jvp.input_types()[1].clone();
        let evaluation = jvp
            .partially_evaluate(&[PartialValue::Known(x.clone()), PartialValue::Unknown(tangent_type)])
            .unwrap();
        assert!(matches!(
            evaluation.outputs(),
            [PartialEvaluationOutput::Known(_), PartialEvaluationOutput::Unknown(0)],
        ));
        let [instruction] = evaluation.program().instructions() else {
            panic!("expected the tangent `shard_map` as the only residual instruction");
        };
        assert_eq!(instruction.operation().name(), SHARD_MAP_OPERATION_NAME);
        assert_eq!(
            evaluation.interpret(&TestEagerContext::new(), std::slice::from_ref(&tangent)),
            jvp.interpret(vec![x, tangent]),
        );
    }

    #[test]
    fn test_shard_map_partial_evaluation_splits_fused_jvp_maps() {
        // Partially evaluating the fused forward-mode `shard_map` of `sin(x) * w` with known primals and unknown
        // tangents splits it into a known `shard_map`, which computes the output and the residuals `cos(x)` and
        // `sin(x)` (tiled along `x`), and a residual `shard_map`, which computes the output tangent. The residual `w`
        // is a known input, so the residual `shard_map` receives it from the known boundary input (JAX's `in_fwd`).
        let program = sine_product_shard_map_program();
        let jvp = program.jvp().unwrap();
        let sharding = "{mesh<['x'=2:manual]>, [{'x'}]}";
        let global = "f32[8][sharding={mesh<['x'=2:manual]>, [{'x'}]}]";
        let local = "f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        assert_eq!(
            jvp.to_string(),
            formatdoc! {"
                lambda %0:{global}, %1:{global}, %2:{global}, %3:{global} .
                let %4:{global}, %5:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}, {sharding}, {sharding}],
                    out_shardings=[{sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}, {global}, {global}],
                    global_output_types=[{global}, {global}],
                ] %0 %1 %2 %3 [
                    body={{
                        lambda %0:{local}, %1:{local}, %2:{local}, %3:{local} .
                        let %4:{local} = sin %0
                            %5:{local} = cos %0
                            %6:{local} = mul %5 %2
                            %7:{local} = mul %4 %1
                            %8:{local} = mul %1 %6
                            %9:{local} = mul %4 %3
                            %10:{local} = add %8 %9
                        in (%7, %10)
                    }},
                ]
                in (%4, %5)"},
        );
        let outer = TestContext::new();
        let input_types = jvp.input_types();
        let x = outer.input(input_types[0].clone());
        let w = outer.input(input_types[1].clone());
        let evaluation = jvp
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(x),
                    PartialValue::Known(w),
                    PartialValue::Unknown(input_types[2].clone()),
                    PartialValue::Unknown(input_types[3].clone()),
                ],
            )
            .unwrap();
        let mut known_outputs = Vec::new();
        for output in evaluation.outputs() {
            if let PartialEvaluationOutput::Known(value) = output {
                known_outputs.push(value.atom_id().unwrap());
            }
        }
        for input in evaluation.inputs() {
            if let PartialEvaluationInput::Known(value) = input {
                known_outputs.push(value.atom_id().unwrap());
            }
        }
        let output_count = known_outputs.len();
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                known_outputs,
                vec![Placeholder; 2],
                vec![Placeholder; output_count],
            )
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            formatdoc! {"
                lambda %0:{global}, %1:{global} .
                let %2:{global}, %3:{global}, %4:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}],
                    out_shardings=[{sharding}, {sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}],
                    global_output_types=[{global}, {global}, {global}],
                ] %0 %1 [
                    body={{
                        lambda %0:{local}, %1:{local} .
                        let %2:{local} = sin %0
                            %3:{local} = mul %2 %1
                            %4:{local} = cos %0
                        in (%3, %4, %2)
                    }},
                ]
                in (%2, %3, %1, %4)"},
        );
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:{global}, %1:{global}, %2:{global}, %3:{global}, %4:{global} .
                let %5:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}, {sharding}, {sharding}, {sharding}],
                    out_shardings=[{sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}, {global}, {global}, {global}],
                    global_output_types=[{global}],
                ] %0 %1 %2 %3 %4 [
                    body={{
                        lambda %0:{local}, %1:{local}, %2:{local}, %3:{local}, %4:{local} .
                        let %5:{local} = mul %2 %0
                            %6:{local} = mul %3 %5
                            %7:{local} = mul %4 %1
                            %8:{local} = add %6 %7
                        in (%8)
                    }},
                ]
                in (%5)"},
        );
    }

    #[test]
    fn test_shard_map_partial_evaluation_keeps_trivial_partitions_whole() {
        // A split whose known half has no instructions and determines no output would hoist no work, so the `shard_map`
        // stays whole and its symbolic known input feeds it as a residual.
        let replicated = Sharding::replicated(manual_mesh(), 0);
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |(a, x): (TestTracer, TestTracer)| a * x,
                        (inputs[0].clone(), inputs[1].clone()),
                        manual_mesh(),
                        (replicated.clone(), replicated.clone()),
                        replicated.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_scalar_type(), f32_scalar_type()],
            Vec::new(),
        );
        let outer = TestContext::new();
        let known = outer.input(f32_scalar_type().into());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(known.clone()), PartialValue::Unknown(f32_scalar_type().into())],
            )
            .unwrap();
        assert!(outer.builder().borrow().instructions().is_empty());
        assert!(matches!(
            evaluation.inputs(),
            [PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(value)]
                if value.atom_id() == known.atom_id(),
        ));
        assert!(matches!(evaluation.outputs(), [PartialEvaluationOutput::Unknown(0)]));
        let global_type = f32_scalar_type().with_sharding(replicated.clone()).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:{global_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}, {replicated}],
                    out_shardings=[{replicated}],
                    manual_axes=['x'],
                    global_input_types=[{global_type}, {global_type}],
                    global_output_types=[{global_type}],
                ] %1 %0 [
                    body={{
                        lambda %0:{global_type}, %1:{global_type} .
                        let %2:{global_type} = mul %0 %1
                        in (%2)
                    }},
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_shard_map_partial_evaluation_forwards_known_inputs() {
        // Online partial evaluation feeds a known input that the unknown half needs directly to the residual
        // `shard_map` under its own global type and input sharding (JAX's `in_fwd`), instead of routing it through the
        // known-side `shard_map` as a packed residual, and the known-side `shard_map` drops a known input that it no
        // longer reads.

        // The body over `[a, b, x]`, each a `f32[4]` sharded along `x`, computes `(sin(b), a * x)` with `a` and `b`
        // known and `x` unknown, so the unknown half needs `a` unchanged.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let global_type = f32_vector_type(4).with_sharding(sharded.clone()).unwrap();
        let program = trace_test_program(
            |inputs| {
                let (sine, product) = shard_map(
                    |(a, b, x): (TestTracer, TestTracer, TestTracer)| (b.sin().unwrap(), a * x),
                    (inputs[0].clone(), inputs[1].clone(), inputs[2].clone()),
                    mesh.clone(),
                    (sharded.clone(), sharded.clone(), sharded.clone()),
                    (sharded.clone(), sharded.clone()),
                )
                .unwrap();
                vec![sine, product]
            },
            vec![global_type.clone(), global_type.clone(), global_type.clone()],
            Vec::new(),
        );
        let outer = TestContext::new();
        let a = outer.input(ArrayIrType::Array(global_type.clone()));
        let b = outer.input(ArrayIrType::Array(global_type.clone()));
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(a), PartialValue::Known(b), PartialValue::Unknown(global_type.into())],
            )
            .unwrap();
        let [PartialEvaluationInput::Unknown(2), PartialEvaluationInput::Known(forwarded)] = evaluation.inputs() else {
            panic!("expected the unknown input followed by the forwarded known input");
        };
        let [PartialEvaluationOutput::Known(known_output), PartialEvaluationOutput::Unknown(0)] = evaluation.outputs()
        else {
            panic!("expected one known output followed by one unknown output");
        };
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![known_output.atom_id().unwrap(), forwarded.atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}], \
                    %1:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] .
                let %2:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                    global_output_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ] %1 [
                    body={
                        lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sin %0
                        in (%1)
                    },
                ]
                in (%2, %0)"
            },
        );
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}], \
                    %1:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] .
                let %2:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}], \
                        f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                    global_output_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ] %0 %1 [
                    body={
                        lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                            %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%2)"
            },
        );
    }

    #[test]
    fn test_shard_map_partial_evaluation_forwards_known_outputs() {
        // Online partial evaluation returns an output that merely forwards a known input as that input itself, when the
        // output sharding equals the input sharding and the input has the declared output type, so the output stays
        // known without any known-side work, and only the remaining outputs are split.

        // The body over `[a, x]`, each a `f32[4]` sharded along `x`, computes `(a, a * x)` with `a` known and `x`
        // unknown. The known half of `a * x` has no instructions, so the remaining map stays whole.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let global_type = f32_vector_type(4).with_sharding(sharded.clone()).unwrap();
        let function = |inputs: Vec<TestTracer>| {
            let (forwarded, product) = shard_map(
                |(a, x): (TestTracer, TestTracer)| (a.clone(), a * x),
                (inputs[0].clone(), inputs[1].clone()),
                mesh.clone(),
                (sharded.clone(), sharded.clone()),
                (sharded.clone(), sharded.clone()),
            )
            .unwrap();
            vec![forwarded, product]
        };
        let program = trace_test_program(function, vec![global_type.clone(), global_type.clone()], Vec::new());
        let outer = TestContext::new();
        let a = outer.input(ArrayIrType::Array(global_type.clone()));
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(a.clone()), PartialValue::Unknown(global_type.clone().into())],
            )
            .unwrap();
        let [PartialEvaluationOutput::Known(forwarded), PartialEvaluationOutput::Unknown(0)] = evaluation.outputs()
        else {
            panic!("expected one known output followed by one unknown output");
        };
        assert_eq!(forwarded.atom_id(), a.atom_id());
        assert!(outer.builder().borrow().instructions().is_empty());
        let local_type = f32_vector_type(2)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:{global_type}, %1:{global_type} .
                let %2:{global_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharded}, {sharded}],
                    out_shardings=[{sharded}],
                    manual_axes=['x'],
                    global_input_types=[{global_type}, {global_type}],
                    global_output_types=[{global_type}],
                ] %1 %0 [
                    body={{
                        lambda %0:{local_type}, %1:{local_type} .
                        let %2:{local_type} = mul %0 %1
                        in (%2)
                    }},
                ]
                in (%2)"},
        );

        // An output that forwards a known input of another type (here, an unsharded `f32[4]`) is not that input
        // itself, but it still stays known through a known-side `shard_map` without instructions in its body.
        let program = trace_test_program(function, vec![f32_vector_type(4), f32_vector_type(4)], Vec::new());
        let outer = TestContext::new();
        let a = outer.input(ArrayIrType::Array(f32_vector_type(4)));
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(a), PartialValue::Unknown(f32_vector_type(4).into())],
            )
            .unwrap();
        let [PartialEvaluationOutput::Known(known_output), PartialEvaluationOutput::Unknown(0)] = evaluation.outputs()
        else {
            panic!("expected one known output followed by one unknown output");
        };
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![known_output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            formatdoc! {"
                lambda %0:f32[4] .
                let %1:{global_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharded}],
                    out_shardings=[{sharded}],
                    manual_axes=['x'],
                    global_input_types=[{global_type}],
                    global_output_types=[{global_type}],
                ] %0 [
                    body={{
                        lambda %0:{local_type} .
                        in (%0)
                    }},
                ]
                in (%1)"},
        );
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:{global_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharded}, {sharded}],
                    out_shardings=[{sharded}],
                    manual_axes=['x'],
                    global_input_types=[{global_type}, {global_type}],
                    global_output_types=[{global_type}],
                ] %0 %1 [
                    body={{
                        lambda %0:{local_type}, %1:{local_type} .
                        let %2:{local_type} = mul %1 %0
                        in (%2)
                    }},
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_shard_map_partial_evaluation_forwards_known_output_residuals() {
        // Online partial evaluation feeds a residual that is also a known output to the residual `shard_map` from that
        // known boundary output, under its global type and with its output sharding as the input sharding (JAX's
        // `out_fwd`), instead of returning it a second time from the known-side `shard_map`.

        // The body over `[a, x]`, each a `f32[4]` sharded along `x`, computes `(sin(a), sin(a) * x)` with `a` known and
        // `x` unknown, so the unknown half needs the known output `sin(a)`.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let global_type = f32_vector_type(4).with_sharding(sharded.clone()).unwrap();
        let program = trace_test_program(
            |inputs| {
                let (sine, product) = shard_map(
                    |(a, x): (TestTracer, TestTracer)| {
                        let sine = a.sin().unwrap();
                        (sine.clone(), sine * x)
                    },
                    (inputs[0].clone(), inputs[1].clone()),
                    mesh.clone(),
                    (sharded.clone(), sharded.clone()),
                    (sharded.clone(), sharded.clone()),
                )
                .unwrap();
                vec![sine, product]
            },
            vec![global_type.clone(), global_type.clone()],
            Vec::new(),
        );
        let outer = TestContext::new();
        let a = outer.input(ArrayIrType::Array(global_type.clone()));
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(a), PartialValue::Unknown(global_type.clone().into())],
            )
            .unwrap();
        let [PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(residual)] = evaluation.inputs() else {
            panic!("expected the unknown input followed by the forwarded known output");
        };
        let [PartialEvaluationOutput::Known(known_output), PartialEvaluationOutput::Unknown(0)] = evaluation.outputs()
        else {
            panic!("expected one known output followed by one unknown output");
        };
        assert_eq!(residual.atom_id(), known_output.atom_id());
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![known_output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let local_type = f32_vector_type(2)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            formatdoc! {"
                lambda %0:{global_type} .
                let %1:{global_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharded}],
                    out_shardings=[{sharded}],
                    manual_axes=['x'],
                    global_input_types=[{global_type}],
                    global_output_types=[{global_type}],
                ] %0 [
                    body={{
                        lambda %0:{local_type} .
                        let %1:{local_type} = sin %0
                        in (%1)
                    }},
                ]
                in (%1)"},
        );
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:{global_type}, %1:{global_type} .
                let %2:{global_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharded}, {sharded}],
                    out_shardings=[{sharded}],
                    manual_axes=['x'],
                    global_input_types=[{global_type}, {global_type}],
                    global_output_types=[{global_type}],
                ] %0 %1 [
                    body={{
                        lambda %0:{local_type}, %1:{local_type} .
                        let %2:{local_type} = mul %1 %0
                        in (%2)
                    }},
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_shard_map_partial_evaluation_forwards_unreduced_known_output_residuals() {
        // A known output whose output sharding is unreduced along `x` crosses into the residual `shard_map` under that
        // sharding, so each device receives a partial sum of the known output rather than its own local value. That is
        // sound because the residual body may use an unreduced value only linearly: here, `s = -a` is unreduced along
        // `x` and the residual body multiplies it by `x`, which is reduced along `x`, so the product is again a pending
        // sum along `x` whose total does not depend on how the residual is split across the devices.
        let mesh = manual_mesh();
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let reduced = Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["x"]).unwrap();
        let unreduced_type = f32_vector_type(2).with_sharding(unreduced.clone()).unwrap();
        let reduced_type = f32_vector_type(2).with_sharding(reduced.clone()).unwrap();
        let program = trace_test_program(
            |inputs| {
                let (negation, product) = shard_map(
                    |(a, x): (TestTracer, TestTracer)| {
                        let negation = -a;
                        (negation.clone(), negation * x)
                    },
                    (inputs[0].clone(), inputs[1].clone()),
                    mesh.clone(),
                    (unreduced.clone(), reduced.clone()),
                    (unreduced.clone(), unreduced.clone()),
                )
                .unwrap();
                vec![negation, product]
            },
            vec![unreduced_type.clone(), reduced_type.clone()],
            Vec::new(),
        );
        let a = ArrayIrValue::Array(Array::from_elements(unreduced_type.clone(), &[1f32, 2.0]).unwrap());
        let x = ArrayIrValue::Array(Array::from_elements(reduced_type.clone(), &[3f32, 4.0]).unwrap());
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(a.clone()), PartialValue::Unknown(reduced_type.clone().into())])
            .unwrap();
        let negation = ArrayIrValue::Array(Array::from_elements(unreduced_type.clone(), &[-1f32, -2.0]).unwrap());
        assert!(matches!(
            evaluation.inputs(),
            [PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(value)] if value == &negation,
        ));
        assert!(matches!(
            evaluation.outputs(),
            [PartialEvaluationOutput::Known(value), PartialEvaluationOutput::Unknown(0)] if value == &negation,
        ));
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:{reduced_type}, %1:{unreduced_type} .
                let %2:{unreduced_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{reduced}, {unreduced}],
                    out_shardings=[{unreduced}],
                    manual_axes=['x'],
                    global_input_types=[{reduced_type}, {unreduced_type}],
                    global_output_types=[{unreduced_type}],
                ] %0 %1 [
                    body={{
                        lambda %0:{reduced_type}, %1:{unreduced_type} .
                        let %2:{unreduced_type} = mul %1 %0
                        in (%2)
                    }},
                ]
                in (%2)"},
        );
        let product = ArrayIrValue::Array(Array::from_elements(unreduced_type, &[-3f32, -8.0]).unwrap());
        assert_eq!(program.interpret(vec![a, x.clone()]), Ok(vec![negation.clone(), product.clone()]));
        assert_eq!(
            evaluation.interpret(&TestEagerContext::new(), std::slice::from_ref(&x)),
            Ok(vec![negation, product]),
        );
    }

    #[test]
    fn test_shard_map_partial_evaluation_forwards_reduced_known_output_residuals() {
        // A known output whose output sharding is reduced along `x` is invariant along `x`, so it crosses into the
        // residual `shard_map` under that sharding as the same local value on every device.
        let mesh = manual_mesh();
        let reduced = Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["x"]).unwrap();
        let reduced_type = f32_vector_type(2).with_sharding(reduced.clone()).unwrap();
        let program = trace_test_program(
            |inputs| {
                let (sine, product) = shard_map(
                    |(a, x): (TestTracer, TestTracer)| {
                        let sine = a.sin().unwrap();
                        (sine.clone(), sine * x)
                    },
                    (inputs[0].clone(), inputs[1].clone()),
                    mesh.clone(),
                    (reduced.clone(), reduced.clone()),
                    (reduced.clone(), reduced.clone()),
                )
                .unwrap();
                vec![sine, product]
            },
            vec![reduced_type.clone(), reduced_type.clone()],
            Vec::new(),
        );
        let a = ArrayIrValue::Array(Array::from_elements(reduced_type.clone(), &[1f32, 2.0]).unwrap());
        let x = ArrayIrValue::Array(Array::from_elements(reduced_type.clone(), &[3f32, 4.0]).unwrap());
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(a.clone()), PartialValue::Unknown(reduced_type.clone().into())])
            .unwrap();
        let [PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(residual)] = evaluation.inputs() else {
            panic!("expected the unknown input followed by the forwarded known output");
        };
        let [PartialEvaluationOutput::Known(sine), PartialEvaluationOutput::Unknown(0)] = evaluation.outputs() else {
            panic!("expected one known output followed by one unknown output");
        };
        assert_eq!(residual, sine);
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:{reduced_type}, %1:{reduced_type} .
                let %2:{reduced_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{reduced}, {reduced}],
                    out_shardings=[{reduced}],
                    manual_axes=['x'],
                    global_input_types=[{reduced_type}, {reduced_type}],
                    global_output_types=[{reduced_type}],
                ] %0 %1 [
                    body={{
                        lambda %0:{reduced_type}, %1:{reduced_type} .
                        let %2:{reduced_type} = mul %1 %0
                        in (%2)
                    }},
                ]
                in (%2)"},
        );
        let outputs = program.interpret(vec![a, x.clone()]).unwrap();
        assert_eq!(outputs[0], *sine);
        assert_eq!(evaluation.interpret(&TestEagerContext::new(), std::slice::from_ref(&x)), Ok(outputs));
    }

    #[test]
    fn test_shard_map_partial_evaluation_threads_unreduced_residual_edges() {
        // A residual that is neither a known input nor a known output crosses as a residual edge under the boundary
        // that `residual_boundary` derives from its local type. Here, `s = -a` is unreduced along `x` and only the
        // residual body uses it, multiplying it by `x`, which is reduced along `x`, so the edge keeps its pending sum
        // across the boundary, and the split program computes the same values as the original one.
        let mesh = manual_mesh();
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let reduced = Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["x"]).unwrap();
        let unreduced_type = f32_vector_type(2).with_sharding(unreduced.clone()).unwrap();
        let reduced_type = f32_vector_type(2).with_sharding(reduced.clone()).unwrap();
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |(a, x): (TestTracer, TestTracer)| -a * x,
                        (inputs[0].clone(), inputs[1].clone()),
                        mesh.clone(),
                        (unreduced.clone(), reduced.clone()),
                        unreduced.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![unreduced_type.clone(), reduced_type.clone()],
            Vec::new(),
        );
        let a = ArrayIrValue::Array(Array::from_elements(unreduced_type.clone(), &[1f32, 2.0]).unwrap());
        let x = ArrayIrValue::Array(Array::from_elements(reduced_type.clone(), &[3f32, 4.0]).unwrap());
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(a.clone()), PartialValue::Unknown(reduced_type.clone().into())])
            .unwrap();
        let negation = ArrayIrValue::Array(Array::from_elements(unreduced_type.clone(), &[-1f32, -2.0]).unwrap());
        assert!(matches!(
            evaluation.inputs(),
            [PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(value)] if value == &negation,
        ));
        assert!(matches!(evaluation.outputs(), [PartialEvaluationOutput::Unknown(0)]));
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:{reduced_type}, %1:{unreduced_type} .
                let %2:{unreduced_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{reduced}, {unreduced}],
                    out_shardings=[{unreduced}],
                    manual_axes=['x'],
                    global_input_types=[{reduced_type}, {unreduced_type}],
                    global_output_types=[{unreduced_type}],
                ] %0 %1 [
                    body={{
                        lambda %0:{reduced_type}, %1:{unreduced_type} .
                        let %2:{unreduced_type} = mul %1 %0
                        in (%2)
                    }},
                ]
                in (%2)"},
        );
        let product = ArrayIrValue::Array(Array::from_elements(unreduced_type, &[-3f32, -8.0]).unwrap());
        assert_eq!(program.interpret(vec![a, x.clone()]), Ok(vec![product.clone()]));
        assert_eq!(evaluation.interpret(&TestEagerContext::new(), std::slice::from_ref(&x)), Ok(vec![product]));
    }

    #[test]
    fn test_shard_map_partial_evaluation_keeps_effects_of_bodies_with_only_forwarded_outputs() {
        // When every output forwards a known input, the outputs are those inputs, but a body whose effects are retained
        // when unused (here, a print of the unknown input `x`) still leaves a `shard_map` without outputs behind.
        let replicated = Sharding::replicated(manual_mesh(), 0);
        let scalar = f32_scalar_type().with_sharding(replicated.clone()).unwrap();
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |(a, x): (TestTracer, TestTracer)| {
                            x.print_with_effect_class("x", EffectClass::DeviceOrderedIo).unwrap();
                            a
                        },
                        (inputs[0].clone(), inputs[1].clone()),
                        manual_mesh(),
                        (replicated.clone(), replicated.clone()),
                        replicated.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![scalar.clone(), scalar.clone()],
            Vec::new(),
        );
        let outer = TestContext::new();
        let a = outer.input(ArrayIrType::Array(scalar.clone()));
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(a.clone()), PartialValue::Unknown(scalar.clone().into())],
            )
            .unwrap();
        let [PartialEvaluationOutput::Known(forwarded)] = evaluation.outputs() else {
            panic!("expected one known output");
        };
        assert_eq!(forwarded.atom_id(), a.atom_id());
        assert!(outer.builder().borrow().instructions().is_empty());
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:{scalar}, %1:{scalar} .
                let () = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}, {replicated}],
                    out_shardings=[],
                    manual_axes=['x'],
                    global_input_types=[{scalar}, {scalar}],
                    global_output_types=[],
                ] %1 %0 [
                    body={{
                        lambda %0:{scalar}, %1:{scalar} .
                        let %2:{scalar} = print [label=x, effect_class=device_ordered_io] %1
                        in ()
                    }},
                ]
                in ()"},
        );
    }

    #[test]
    fn test_shard_map_partial_evaluation_retains_deferred_work_of_bodies_with_only_forwarded_outputs() {
        // A body whose outputs all forward known inputs is kept exactly when its remaining work is retained when
        // unused. Deferred work has no effect class but must survive: here, a reverse-mode carrier of a custom function
        // applied to the unknown input `x`, whose backward rule prints its cotangent, stays in a `shard_map` without
        // outputs.
        let replicated = Sharding::replicated(manual_mesh(), 0);
        let scalar = f32_scalar_type().with_sharding(replicated.clone()).unwrap();
        let shard_map =
            ShardMap::new(manual_mesh(), vec![replicated.clone(); 2], vec![replicated.clone()], Vec::new()).unwrap();
        let backward = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let cotangent = builder.add_input(scalar.clone().into());
            let print = PrintOperation::<ArrayIrType>::new("cotangent").with_effect_class(EffectClass::DeviceOrderedIo);
            builder.add_instruction(print, Vec::new(), vec![cotangent], None).unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![cotangent], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let a = builder.add_input(scalar.clone().into());
            let x = builder.add_input(scalar.clone().into());
            let backward = builder.import_program(backward);
            let carrier = CustomFunctionTransposeOperation::from_backward_region(
                0,
                vec![scalar.clone().into()],
                vec![scalar.clone().into()],
            );
            builder
                .add_instruction(TestOperation::CustomFunctionTranspose(carrier), vec![backward], vec![x], None)
                .unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![a], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        assert!(body.effects().classes().is_empty() && body.effects().is_retained_when_unused());
        let input_types = vec![f32_scalar_type().into(); 2];
        let operation = ShardMapOperation::from_program(&body, input_types, shard_map.clone()).unwrap();
        let program = shard_map_program(operation, body).unwrap();
        let outer = TestContext::new();
        let a = outer.input(ArrayIrType::Array(scalar.clone()));
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(a.clone()), PartialValue::Unknown(scalar.clone().into())],
            )
            .unwrap();
        let [PartialEvaluationOutput::Known(forwarded)] = evaluation.outputs() else {
            panic!("expected one known output");
        };
        assert_eq!(forwarded.atom_id(), a.atom_id());
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:{scalar}, %1:{scalar} .
                let () = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}, {replicated}],
                    out_shardings=[],
                    manual_axes=['x'],
                    global_input_types=[{scalar}, {scalar}],
                    global_output_types=[],
                ] %1 %0 [
                    body={{
                        lambda %0:{scalar}, %1:{scalar} .
                        let %2:{scalar} = custom_function_transpose [leading_input_count=0] %1 [
                            backward={{
                                lambda %0:{scalar} .
                                let %1:{scalar} = print [label=cotangent, effect_class=device_ordered_io] %0
                                in (%0)
                            }},
                        ]
                        in ()
                    }},
                ]
                in ()"},
        );

        // An unused local allocation has an effect class but is not retained when unused, so it leaves no work behind
        // and no `shard_map` is staged at all.
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let a = builder.add_input(scalar.clone().into());
            let x = builder.add_input(scalar.clone().into());
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![x], None).unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![a], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        assert!(!body.effects().classes().is_empty() && !body.effects().is_retained_when_unused());
        let input_types = vec![f32_scalar_type().into(); 2];
        let operation = ShardMapOperation::from_program(&body, input_types, shard_map).unwrap();
        let program = shard_map_program(operation, body).unwrap();
        let outer = TestContext::new();
        let a = outer.input(ArrayIrType::Array(scalar.clone()));
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(a.clone()), PartialValue::Unknown(scalar.clone().into())],
            )
            .unwrap();
        let [PartialEvaluationOutput::Known(forwarded)] = evaluation.outputs() else {
            panic!("expected one known output");
        };
        assert_eq!(forwarded.atom_id(), a.atom_id());
        assert!(outer.builder().borrow().instructions().is_empty());
        assert!(evaluation.program().instructions().is_empty());
    }

    #[test]
    fn test_shard_map_partial_evaluation_orders_effects_across_the_split() {
        // A split keeps the per-device order of the effects of both halves: the print of the known input `a` runs in
        // the known-side `shard_map`, which is staged before the residual `shard_map` that runs the print of the
        // unknown input `x`, matching their order in the body.
        let replicated = Sharding::replicated(manual_mesh(), 0);
        let program = trace_test_program(
            |inputs| {
                let (sine, product) = shard_map(
                    |(a, x): (TestTracer, TestTracer)| {
                        let a = a.print_with_effect_class("a", EffectClass::DeviceOrderedIo).unwrap();
                        let x = x.print_with_effect_class("x", EffectClass::DeviceOrderedIo).unwrap();
                        (a.sin().unwrap(), a * x)
                    },
                    (inputs[0].clone(), inputs[1].clone()),
                    manual_mesh(),
                    (replicated.clone(), replicated.clone()),
                    (replicated.clone(), replicated.clone()),
                )
                .unwrap();
                vec![sine, product]
            },
            vec![f32_scalar_type(), f32_scalar_type()],
            Vec::new(),
        );
        let outer = TestContext::new();
        let known = outer.input(f32_scalar_type().into());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(known), PartialValue::Unknown(f32_scalar_type().into())],
            )
            .unwrap();
        let known_outputs = evaluation
            .outputs()
            .iter()
            .filter_map(|output| match output {
                PartialEvaluationOutput::Known(value) => Some(value.atom_id().unwrap()),
                PartialEvaluationOutput::Unknown(_) => None,
            })
            .chain(evaluation.inputs().iter().filter_map(|input| match input {
                PartialEvaluationInput::Known(value) => Some(value.atom_id().unwrap()),
                PartialEvaluationInput::Unknown(_) => None,
            }))
            .collect::<Vec<_>>();
        let known_output_count = known_outputs.len();
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                known_outputs,
                vec![Placeholder],
                vec![Placeholder; known_output_count],
            )
            .unwrap();
        let scalar = f32_scalar_type().with_sharding(replicated.clone()).unwrap();
        assert_eq!(
            known_program.to_string(),
            formatdoc! {"
                lambda %0:f32[] .
                let %1:{scalar}, %2:{scalar} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}],
                    out_shardings=[{replicated}, {replicated}],
                    manual_axes=['x'],
                    global_input_types=[{scalar}],
                    global_output_types=[{scalar}, {scalar}],
                ] %0 [
                    body={{
                        lambda %0:{scalar} .
                        let %1:{scalar} = print [label=a, effect_class=device_ordered_io] %0
                            %2:{scalar} = sin %1
                        in (%2, %1)
                    }},
                ]
                in (%1, %2)"},
        );
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:f32[], %1:{scalar} .
                let %2:{scalar} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}, {replicated}],
                    out_shardings=[{replicated}],
                    manual_axes=['x'],
                    global_input_types=[{scalar}, {scalar}],
                    global_output_types=[{scalar}],
                ] %0 %1 [
                    body={{
                        lambda %0:{scalar}, %1:{scalar} .
                        let %2:{scalar} = print [label=x, effect_class=device_ordered_io] %0
                            %3:{scalar} = mul %1 %2
                        in (%3)
                    }},
                ]
                in (%2)"},
        );

        // When the body prints the unknown input `x` before the known input `a`, hoisting the print of `a` into the
        // known-side `shard_map` would run it first, so the partition keeps it (and the `sin` that reads its result) on
        // the residual side, and the map, whose known half would then be trivial, stays whole.
        let program = trace_test_program(
            |inputs| {
                let (sine, product) = shard_map(
                    |(a, x): (TestTracer, TestTracer)| {
                        let x = x.print_with_effect_class("x", EffectClass::DeviceOrderedIo).unwrap();
                        let a = a.print_with_effect_class("a", EffectClass::DeviceOrderedIo).unwrap();
                        (a.sin().unwrap(), a * x)
                    },
                    (inputs[0].clone(), inputs[1].clone()),
                    manual_mesh(),
                    (replicated.clone(), replicated.clone()),
                    (replicated.clone(), replicated.clone()),
                )
                .unwrap();
                vec![sine, product]
            },
            vec![f32_scalar_type(), f32_scalar_type()],
            Vec::new(),
        );
        let outer = TestContext::new();
        let known = outer.input(f32_scalar_type().into());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(known), PartialValue::Unknown(f32_scalar_type().into())],
            )
            .unwrap();
        assert!(outer.builder().borrow().instructions().is_empty());
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:{scalar}, %3:{scalar} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}, {replicated}],
                    out_shardings=[{replicated}, {replicated}],
                    manual_axes=['x'],
                    global_input_types=[{scalar}, {scalar}],
                    global_output_types=[{scalar}, {scalar}],
                ] %1 %0 [
                    body={{
                        lambda %0:{scalar}, %1:{scalar} .
                        let %2:{scalar} = print [label=x, effect_class=device_ordered_io] %1
                            %3:{scalar} = print [label=a, effect_class=device_ordered_io] %0
                            %4:{scalar} = sin %3
                            %5:{scalar} = mul %3 %2
                        in (%4, %5)
                    }},
                ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_shard_map_partial_evaluation_threads_dimension_residuals() {
        // Online partial evaluation of a `shard_map` whose unknown half consumes a first-class dimension computed by
        // its known half: the dimension crosses the split as its integer scalar value, packed per device by the
        // known-side `shard_map`, and the residual `shard_map` redefines the same dimension from that scalar.
        let array_type = f32_scalar_type();
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let shard_map = ShardMap::from_shardings(
            mesh,
            vec![replicated.clone(), replicated.clone()],
            vec![replicated],
            vec!["x".to_string()],
        );
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(5)).unwrap());
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let known_input = builder.add_input(array_type.clone().into());
        let runtime_input = builder.add_input(array_type.clone().into());
        let count = builder
            .add_instruction(
                ArrayOperation::ConvertElementType(ConvertElementTypeOperation::new(DataType::I64, false)),
                Vec::new(),
                vec![known_input],
                None,
            )
            .unwrap()[0];
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![count], None)
            .unwrap()[0];
        let broadcast = builder
            .add_instruction(
                DynamicBroadcastOperation::new(Vec::new()),
                Vec::new(),
                vec![runtime_input, dimension],
                None,
            )
            .unwrap()[0];
        let sum = builder
            .add_instruction(
                ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Sum)),
                Vec::new(),
                vec![broadcast],
                None,
            )
            .unwrap()[0];
        let body = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let operation = ShardMapOperation::from_boundary(
            shard_map,
            vec![array_type.clone(), array_type.clone()],
            vec![array_type.clone()],
        );
        let program = shard_map_program(operation, body).unwrap();

        let outer = TestContext::new();
        let known = outer.input(ArrayIrType::Array(array_type.clone()));
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(known), PartialValue::Unknown(ArrayIrType::Array(array_type))],
            )
            .unwrap();
        let PartialEvaluationInput::Known(residual) = &evaluation.inputs()[1] else {
            panic!("expected the dimension residual to be fed by the known-side `shard_map`");
        };
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![residual.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:i64[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, []}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[]],
                    global_output_types=[i64[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ] %0 [
                    body={
                        lambda %0:f32[] .
                        let %1:i64[] = convert_element_type [data_type=i64] %0
                            %2:dimension<extent ∈ [0, 5)> = dimension_from_scalar [bounds=[0, 5)] %1
                            %3:i64[] = dimension_to_scalar %2
                            %4:i64[][sharding={mesh<['x'=2:manual]>, []}] = broadcast \
                                [output_type=i64[][sharding={mesh<['x'=2:manual]>, []}], output_axes=[]] %3
                            %5:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = parallel_vary \
                                [axis_name=\"x\"] %4
                            %6:i64[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reshape \
                                [shape=[1], output_sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] %5
                        in (%6)
                    },
                ]
                in (%1)"
            },
        );
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:i64[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}] .
                let %2:f32[] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, []}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, []}],
                    manual_axes=['x'],
                    global_input_types=[f32[], i64[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                    global_output_types=[f32[]],
                ] %0 %1 [
                    body={
                        lambda %0:f32[], %1:i64[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let %2:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reshape [shape=[], \
                            output_sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] %1
                            %3:dimension<extent ∈ [0, 5)> = dimension_from_scalar [bounds=[0, 5)] %2
                            %4:f32[extent] = broadcast [output_axes=[]] %0 %3
                            %5:f32[] = reduce [kind=sum, axes=[0]] %4
                        in (%5)
                    },
                ]
                in (%2)"
            },
        );

        // Partially evaluating the residual program again with its residual edge known (as a tangent `shard_map` is
        // partially evaluated during linearization) keeps the map whole: unpacking the dimension from the edge is the
        // only work that depends on the edge alone, so splitting would only unpack and repack the edge.
        let residual_program = evaluation.program().clone();
        let outer = TestContext::new();
        let known = outer.input(residual_program.input_types()[1].clone());
        let evaluation = residual_program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Unknown(residual_program.input_types()[0].clone()), PartialValue::Known(known)],
            )
            .unwrap();
        assert!(outer.builder().borrow().instructions().is_empty());
        assert_eq!(evaluation.program().to_string(), residual_program.to_string());
    }

    #[test]
    fn test_shard_map_partial_evaluation_keeps_dynamic_residual_boundaries_whole() {
        // Online partial evaluation keeps a `shard_map` whole when splitting it would need a dynamically shaped
        // residual, whose dimension identity is defined inside the known half and therefore cannot cross the static
        // boundary.
        let program = dynamic_residual_shard_map_program();
        let array_type = f32_scalar_type();
        let outer = TestContext::new();
        let known = outer.input(ArrayIrType::Array(array_type.clone()));
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(known), PartialValue::Unknown(ArrayIrType::Array(array_type))],
            )
            .unwrap();

        // Nothing is hoisted into the outer program: the whole `shard_map` is residualized over the unknown input and
        // the known input, which feeds it as a residual, so the dynamic intermediate never leaves the body.
        assert!(outer.builder().borrow().instructions().is_empty());
        assert_eq!(evaluation.inputs().len(), 2);
        assert!(matches!(&evaluation.inputs()[0], PartialEvaluationInput::Unknown(1)));
        assert!(matches!(&evaluation.inputs()[1], PartialEvaluationInput::Known(_)));
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[][sharding={mesh<['x'=2:manual]>, []}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, []}, {mesh<['x'=2:manual]>, []}],
                    out_shardings=[{mesh<['x'=2:manual]>, []}],
                    manual_axes=['x'],
                    global_input_types=[f32[][sharding={mesh<['x'=2:manual]>, []}], \
                        f32[][sharding={mesh<['x'=2:manual]>, []}]],
                    global_output_types=[f32[][sharding={mesh<['x'=2:manual]>, []}]],
                ] %1 %0 [
                    body={
                        lambda %0:f32[], %1:f32[] .
                        let %2:i64[] = convert_element_type [data_type=i64] %0
                            %3:dimension<extent ∈ [0, 5)> = dimension_from_scalar [bounds=[0, 5)] %2
                            %4:f32[extent] = broadcast [output_axes=[]] %0 %3
                            %5:f32[extent] = sin %4
                            %6:f32[extent] = mul %5 %1
                            %7:f32[] = reduce [kind=sum, axes=[0]] %6
                        in (%7)
                    },
                ]
                in (%2)"
            },
        );
    }

    #[test]
    fn test_shard_map_partial_evaluation_keeps_reference_bearing_bodies_whole() {
        // Partial evaluation keeps a reference-bearing shard map whole even when its known inputs are symbolic.
        let sharded = sharded_along_x();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded, true);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(f32_vector_type(4)));
        let value_type = ArrayIrType::Array(f32_vector_type(4));
        let program = shard_map_program(operation, body).unwrap();

        let outer = TestContext::new();
        let known = outer.input(reference_type);
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(value_type)])
            .unwrap();
        assert!(outer.builder().borrow().instructions().is_empty());
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[4], %1:ref<f32[4]> .
                let %2:f32[4] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[ref<f32[4]>, f32[4]],
                    global_output_types=[f32[4]],
                ] %1 %0 [
                    body={
                        lambda %0:ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]>, \
                    %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let () = reference_add_update %0 %1
                            %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reference_read %0
                        in (%2)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_shard_map_partial_evaluation_of_bodies_without_inputs() {
        // A `shard_map` without inputs has only (vacuously) known inputs, so partial evaluation against a live outer
        // trace binds it whole into the known-side trace, where its device-dependent output stays a known tracer.
        let program = axis_index_shard_map_program();
        let outer = TestContext::new();
        let evaluation = program.partially_evaluate_in_context(&outer, &[]).unwrap();
        let [PartialEvaluationOutput::Known(output)] = evaluation.outputs() else {
            panic!("expected one known output");
        };
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output.atom_id().unwrap()], Vec::new(), vec![Placeholder])
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda  .
                let %0:u64[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[],
                    global_output_types=[u64[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ] [
                    body={
                        lambda  .
                        let %0:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = axis_index \
                            [axis_name=\"x\", mesh=['x'=2:manual]]
                            %1:u64[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reshape \
                                [shape=[1]] %0
                        in (%1)
                    },
                ]
                in (%0)"
            },
        );
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda  .
                in ()"
            },
        );
    }

    #[test]
    fn test_shard_map_batching() {
        // A `shard_map` over `x` that doubles its `f32[4]` input, which is sharded along `x`.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.clone() + x,
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_vector_type(4)],
            Vec::new(),
        );
        let extent_type = DimensionValue::constant(3).unwrap().r#type().into_owned();
        let batched = |input_axes: &[BatchAxis]| {
            program
                .batched_with_threaded_extent(
                    extent_type.clone(),
                    ShardingDimension::Replicated,
                    input_axes,
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
                .0
                .to_string()
        };

        // Replicated batching retains the boundary.
        assert_eq!(
            batched(&[BatchAxis::replicated()]),
            indoc! {"
                lambda %0:dimension<3>, %1:f32[4] .
                let %2:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                    global_output_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ] %1 [
                    body={
                        lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = add %0 %0
                        in (%1)
                    },
                ]
                in (%0, %2)"
            },
        );

        // Mapped batching inserts the batch dimension at its batch axis into the input and output shardings, as an
        // unpartitioned dimension, and into the global and local boundary types, with the static batch extent, and it
        // batches the body structurally.
        assert_eq!(
            batched(&[BatchAxis::new(1)]),
            indoc! {"
                lambda %0:dimension<3>, %1:f32[4, 3] .
                let %2:f32[4, 3][sharding={mesh<['x'=2:manual]>, [{'x'}, {}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}, {}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}, {}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4, 3][sharding={mesh<['x'=2:manual]>, [{'x'}, {}]}]],
                    global_output_types=[f32[4, 3][sharding={mesh<['x'=2:manual]>, [{'x'}, {}]}]],
                ] %1 [
                    body={
                        lambda %0:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                        let %1:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = add %0 %0
                        in (%1)
                    },
                ]
                in (%0, %2)"
            },
        );
    }

    #[test]
    fn test_shard_map_batching_of_invocations() {
        // Batching an invocation of `shard_map` on values (i.e., `vmap` of `shard_map`) stages the batched boundary of
        // the mapped batching rule into the enclosing trace.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let program = trace_test_program(
            |inputs| {
                let output = batch(
                    |item| {
                        let item = ValueProjection::<ArrayType>::into_projected(item)?;
                        let output = shard_map(
                            |x: ShardMapTracer<BatchingContext<TestContext, ArrayIrBatchingPolicy>>| x.clone() + x,
                            item,
                            mesh.clone(),
                            sharded.clone(),
                            sharded.clone(),
                        )?;
                        Ok(output.into_value())
                    },
                    inputs[0].value().clone(),
                    BatchAxis::new(1),
                    BatchAxis::new(1),
                    BatchAxisSpecification::default(),
                )
                .unwrap();
                vec![ValueProjection::<ArrayType>::into_projected(output).unwrap()]
            },
            vec![ArrayType::new_static(DataType::F32, [4, 3])],
            Vec::new(),
        );
        let sharding = "{mesh<['x'=2:manual]>, [{'x'}, {}]}";
        let global = "f32[4, 3][sharding={mesh<['x'=2:manual]>, [{'x'}, {}]}]";
        let local = "f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}]";
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:f32[4, 3] .
                let %1:dimension<3> = constant [value=3]
                    %2:{global} = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{sharding}],
                        out_shardings=[{sharding}],
                        manual_axes=['x'],
                        global_input_types=[{global}],
                        global_output_types=[{global}],
                    ] %0 [
                        body={{
                            lambda %0:{local} .
                            let %1:{local} = add %0 %0
                            in (%1)
                        }},
                    ]
                in (%2)"
            },
        );
    }

    #[test]
    fn test_shard_map_batching_mixes_mapped_and_replicated_inputs() {
        // Mapped and replicated inputs mix: only the mapped input and the outputs that depend on it gain the batch
        // dimension, while the replicated input keeps its boundary.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let program = trace_test_program(
            |inputs| {
                let (sum, doubled) = shard_map(
                    |(x, w): (TestTracer, TestTracer)| (x + w.clone(), w.clone() + w),
                    (inputs[0].clone(), inputs[1].clone()),
                    mesh.clone(),
                    (sharded.clone(), sharded.clone()),
                    (sharded.clone(), sharded.clone()),
                )
                .unwrap();
                vec![sum, doubled]
            },
            vec![f32_vector_type(4), f32_vector_type(4)],
            Vec::new(),
        );
        let batched = program
            .batched_with_threaded_extent(
                DimensionValue::constant(3).unwrap().r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0), BatchAxis::replicated()]);
        assert_eq!(
            batched.into_parts().0.to_string(),
            indoc! {"
                lambda %0:dimension<3>, %1:f32[3, 4], %2:f32[4] .
                let %3:f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}], \
                    %4:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}], \
                        f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                    global_output_types=[f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}], \
                        f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ] %1 %2 [
                    body={
                        lambda %0:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                            %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let %2:dimension<3> = constant [value=3]
                            %3:dimension<2> = constant [value=2]
                            %4:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = broadcast [
                                output_axes=[1],
                                output_sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}},
                            ] %1 %2 %3
                            %5:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = add %0 %4
                            %6:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = add %1 %1
                        in (%5, %6)
                    },
                ]
                in (%0, %3, %4)"
            },
        );
    }

    #[test]
    fn test_shard_map_batching_passes_batch_dimensions_through_collectives() {
        // A collective over the manual axis inside the body combines the local shards elementwise, so it passes the
        // batch dimension through untouched.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.parallel_reduce(ReductionKind::Sum, "x").unwrap(),
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        replicated.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_vector_type(4)],
            Vec::new(),
        );
        assert_eq!(
            program
                .batched_with_threaded_extent(
                    DimensionValue::constant(3).unwrap().r#type().into_owned(),
                    ShardingDimension::Replicated,
                    &[BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
                .0
                .to_string(),
            indoc! {"
                lambda %0:dimension<3>, %1:f32[3, 4] .
                let %2:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{}, {}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}]],
                    global_output_types=[f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}]}]],
                ] %1 [
                    body={
                        lambda %0:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                        let %1:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}]}] = parallel_reduce [kind=sum, \
                            axis_name=\"x\", mesh=['x'=2:manual]] %0
                        in (%1)
                    },
                ]
                in (%0, %2)"
            },
        );
    }

    #[test]
    fn test_shard_map_batching_rejects_ragged_inputs() {
        // A bounded ragged batch would lose its per-item extents at the static boundary, so it is rejected.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let operation = ShardMapOperation::from_boundary(
            ShardMap::from_shardings(
                manual_mesh(),
                vec![Sharding::replicated(manual_mesh(), 1)],
                vec![Sharding::replicated(manual_mesh(), 1)],
                vec!["x".to_string()],
            ),
            vec![f32_vector_type(3)],
            vec![f32_vector_type(3)],
        );
        let input = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 3, vec![2.0f32, 1.0, 0.0, 3.0, 0.0, 0.0]).unwrap()),
            BatchAxis::new(0),
        )
        .unwrap()
        .with_ragged_axes(vec![RaggedAxis::new(
            1,
            ArrayIrValue::Array(Array::vector(vec![2i32, 1]).unwrap()),
            length,
            vec![0],
        )])
        .unwrap();
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            TestEagerContext::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        assert!(matches!(
            operation.batch(&context, &EmptyRegionDriver, &[input]),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`shard_map` does not support bounded ragged dimension `length` on input 0",
        ));
    }

    #[test]
    fn test_shard_map_batching_of_mapped_reference_inputs() {
        // A mapped reference input crosses the boundary with the batch dimension inserted into its referent at its
        // batch axis, as an unpartitioned dimension, and the body reads and writes per-item values through it. The
        // reference is mapped along its second axis and the value along its first, so the body moves the batch axis of
        // the update to that of the reference before adding it.
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded_along_x(), true);
        let batched = shard_map_program(operation, body)
            .unwrap()
            .batched_with_threaded_extent(
                DimensionValue::constant(3).unwrap().r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(1), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(1)]);
        assert_eq!(
            batched.into_parts().0.to_string(),
            indoc! {"
                lambda %0:dimension<3>, %1:ref<f32[4, 3]>, %2:f32[3, 4] .
                let %3:f32[4, 3] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}, {}]}, {mesh<['x'=2:manual]>, [{}, {'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}, {}]}],
                    manual_axes=['x'],
                    global_input_types=[ref<f32[4, 3]>, f32[3, 4]],
                    global_output_types=[f32[4, 3]],
                ] %1 %2 [
                    body={
                        lambda %0:ref<f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}]>, \
                            %1:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                        let %2:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = transpose \
                            [permutation=[1, 0]] %1
                            () = reference_add_update %0 %2
                            %3:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                                reference_read %0
                        in (%3)
                    },
                ]
                in (%0, %3)"
            },
        );
    }

    #[test]
    fn test_shard_map_batching_keeps_replicated_reference_inputs() {
        // A reference input that is not mapped keeps its boundary and is shared by every batch item, while the mapped
        // value input gains the batch dimension, so the body reads the shared referent once for all batch items.
        let (operation, body) = reference_shard_map(f32_vector_type(2), Sharding::replicated(manual_mesh(), 1), false);
        let batched = shard_map_program(operation, body)
            .unwrap()
            .batched_with_threaded_extent(
                DimensionValue::constant(3).unwrap().r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0)]);
        assert_eq!(
            batched.into_parts().0.to_string(),
            indoc! {"
                lambda %0:dimension<3>, %1:ref<f32[2]>, %2:f32[3, 4] .
                let %3:f32[3, 4] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{}]}, {mesh<['x'=2:manual]>, [{}, {'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[ref<f32[2]>, f32[3, 4]],
                    global_output_types=[f32[3, 4]],
                ] %1 %2 [
                    body={
                        lambda %0:ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]>, \
                            %1:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                        let %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = reference_read %0
                            %3:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary \
                                [axis_name=\"x\"] %2
                            %4:dimension<3> = constant [value=3]
                            %5:dimension<2> = constant [value=2]
                            %6:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = broadcast [
                                output_axes=[1],
                                output_sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}},
                            ] %3 %4 %5
                            %7:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = add %6 %1
                        in (%7)
                    },
                ]
                in (%0, %3)"
            },
        );
    }

    #[test]
    fn test_shard_map_batching_rejects_batched_writes_into_replicated_references() {
        // A reference input that is not mapped is one holder shared by every batch item, so the body cannot write a
        // batched value into it (as JAX's `_swap_vmap` rejects); the caller must pass the reference mapped instead.
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded_along_x(), true);
        let result = shard_map_program(operation, body).unwrap().batched_with_threaded_extent(
            DimensionValue::constant(3).unwrap().r#type().into_owned(),
            ShardingDimension::Replicated,
            &[BatchAxis::replicated(), BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        );
        assert!(matches!(
            result,
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`reference_add_update` cannot store a batched value into an unbatched reference; pass \
                               the reference as a batched input instead",
        ));
    }

    #[test]
    fn test_shard_map_batching_of_forwarded_reference_outputs() {
        // A forwarded reference output is the mapped reference input that it forwards, so it carries that input's
        // batch axis and its output sharding gains the batch dimension at the same position as the input sharding.
        let (operation, body) = forwarding_reference_shard_map();
        let batched = shard_map_program(operation, body)
            .unwrap()
            .batched_with_threaded_extent(
                DimensionValue::constant(3).unwrap().r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0), BatchAxis::new(0)]);
        let sharding = "{mesh<['x'=2:manual]>, [{}, {'x'}]}";
        let packed = format!("f32[3, 4][sharding={sharding}]");
        let local = "f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}]";
        assert_eq!(
            batched.into_parts().0.to_string(),
            formatdoc! {"
                lambda %0:dimension<3>, %1:ref<{packed}>, %2:{packed} .
                let %3:{packed}, %4:ref<{packed}> = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}],
                    out_shardings=[{sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[ref<{packed}>, {packed}],
                    global_output_types=[{packed}, ref<{packed}>],
                    output_forwarding=[_, 0],
                ] %1 %2 [
                    body={{
                        lambda %0:ref<{local}>, %1:{local} .
                        let () = reference_add_update %0 %1
                            %2:{local} = reference_read %0
                        in (%2, %0)
                    }},
                ]
                in (%0, %3, %4)"
            },
        );
    }

    #[test]
    fn test_shard_map_batching_commutes_with_reference_discharge() {
        // Batching a reference-bearing `shard_map` and then discharging its references yields the same program as
        // discharging them first and then batching the resulting state-passing `shard_map`.
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded_along_x(), true);
        let program = shard_map_program(operation, body).unwrap();
        let batched = |program: TestProgram| {
            program
                .batched_with_threaded_extent(
                    DimensionValue::constant(3).unwrap().r#type().into_owned(),
                    ShardingDimension::Replicated,
                    &[BatchAxis::new(0); 2],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
                .0
        };
        let batched_then_discharged = batched(program.clone()).discharge_references(0).unwrap();
        let discharged_then_batched = batched(program.discharge_references(0).unwrap().program().clone());
        assert_eq!(batched_then_discharged.program().to_string(), discharged_then_batched.to_string());
        assert_eq!(
            discharged_then_batched.to_string(),
            indoc! {"
                lambda %0:dimension<3>, %1:f32[3, 4], %2:f32[3, 4] .
                let %3:f32[3, 4], %4:f32[3, 4] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}, {mesh<['x'=2:manual]>, [{}, {'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}, {mesh<['x'=2:manual]>, [{}, {'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[3, 4], f32[3, 4]],
                    global_output_types=[f32[3, 4], f32[3, 4]],
                ] %1 %2 [
                    body={
                        lambda %0:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                            %1:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                        let %2:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = add %0 %1
                        in (%2, %2)
                    },
                ]
                in (%0, %3, %4)"
            },
        );
    }

    #[test]
    fn test_shard_map_batching_places_batch_axes_on_free_mesh_axes() {
        // A batch axis placed on free mesh axes partitions the batch dimension of the boundary along them, while the
        // body keeps the whole batch extent, because free axes do not partition local values (as JAX keeps a batch axis
        // placed on explicit mesh axes in the sharding of the boundary values only).
        let batched = |mesh: LogicalMesh, manual_axes: Vec<String>| {
            let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let program = trace_test_program(
                |inputs| {
                    vec![
                        shard_map_with_options(
                            |x: TestTracer| x.clone() + x,
                            inputs[0].clone(),
                            mesh,
                            sharded.clone(),
                            sharded,
                            manual_axes,
                        )
                        .unwrap(),
                    ]
                },
                vec![f32_vector_type(4)],
                Vec::new(),
            );
            program
                .batched_with_threaded_extent(
                    DimensionValue::constant(3).unwrap().r#type().into_owned(),
                    ShardingDimension::sharded(["y"]),
                    &[BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
                .0
                .to_string()
        };

        // An explicit axis `y` also places the batch dimension of the local body values.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        assert_eq!(
            batched(mesh, Vec::new()),
            indoc! {"
            lambda %0:dimension<3>, %1:f32[3, 4] .
            let %2:f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'y'}, {'x'}]}] = shard_map [
                mesh=['x'=2:manual, 'y'=2:explicit],
                in_shardings=[{mesh<['x'=2:manual, 'y'=2:explicit]>, [{'y'}, {'x'}]}],
                out_shardings=[{mesh<['x'=2:manual, 'y'=2:explicit]>, [{'y'}, {'x'}]}],
                manual_axes=['x'],
                global_input_types=[f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'y'}, {'x'}]}]],
                global_output_types=[f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'y'}, {'x'}]}]],
            ] %1 [
                body={
                    lambda \
                            %0:f32[3, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'y'}, {}], \
                            varying_manual={'x'}}] .
                    let %1:f32[3, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'y'}, {}], \
                        varying_manual={'x'}}] = add %0 %0
                    in (%1)
                },
            ]
            in (%0, %2)"
            }
        );

        // A manual axis `y` that the map does not make manual places only the boundary values, because a local value
        // placed on a manual axis would vary along it.
        assert_eq!(
            batched(manual_mesh_2x2(), vec!["x".to_string()]),
            indoc! {"
            lambda %0:dimension<3>, %1:f32[3, 4] .
            let %2:f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}, {'x'}]}] = shard_map [
                mesh=['x'=2:manual, 'y'=2:manual],
                in_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}, {'x'}]}],
                out_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}, {'x'}]}],
                manual_axes=['x'],
                global_input_types=[f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}, {'x'}]}]],
                global_output_types=[f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}, {'x'}]}]],
            ] %1 [
                body={
                    lambda %0:f32[3, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                    let %1:f32[3, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], \
                        varying_manual={'x'}}] = add %0 %0
                    in (%1)
                },
            ]
            in (%0, %2)"
            }
        );
    }

    #[test]
    fn test_shard_map_batching_places_batch_axes_on_unused_manual_axes() {
        // A batch axis placed on an active manual axis `y` that no sharding names and that the body does not use (the
        // analogue of JAX's `spmd_axis_name`) makes `y` free in the batched map. Every device along `y` runs the same
        // body on the same inputs, so the batched map, which partitions the batch dimension along `y` at its boundary,
        // computes the same values.
        let mesh = manual_mesh_2x2();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.clone() + x,
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_vector_type(4)],
            Vec::new(),
        );
        let batched = program
            .batched_with_threaded_extent(
                DimensionValue::constant(3).unwrap().r#type().into_owned(),
                ShardingDimension::sharded(["y"]),
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(
            batched.into_parts().0.to_string(),
            indoc! {"
                lambda %0:dimension<3>, %1:f32[3, 4] .
                let %2:f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}, {'x'}]}] = shard_map [
                    mesh=['x'=2:manual, 'y'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}, {'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}, {'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}, {'x'}]}]],
                    global_output_types=[f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}, {'x'}]}]],
                ] %1 [
                    body={
                        lambda %0:f32[3, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], \
                            varying_manual={'x'}}] .
                        let %1:f32[3, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], \
                            varying_manual={'x'}}] = add %0 %0
                        in (%1)
                    },
                ]
                in (%0, %2)"
            }
        );
    }

    #[test]
    fn test_shard_map_batching_drops_batch_axes_placed_on_auto_mesh_axes() {
        // The boundary drops `Auto` axes from every sharding, so it also drops the `Auto` axes of the batch placement:
        // a placement on the `Auto` axis `a` alone leaves the batch dimension replicated, and a placement on `a` and
        // the explicit axis `y` keeps only `y`, both at the boundary and inside the body.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(|x: TestTracer| x.clone() + x, inputs[0].clone(), mesh.clone(), sharded.clone(), sharded)
                        .unwrap(),
                ]
            },
            vec![f32_vector_type(4)],
            Vec::new(),
        );
        let batched = |placement_axes: &[&str]| {
            program
                .batched_with_threaded_extent(
                    DimensionValue::constant(3).unwrap().r#type().into_owned(),
                    ShardingDimension::sharded(placement_axes.iter().copied()),
                    &[BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
                .0
                .to_string()
        };
        assert_eq!(
            batched(&["a"]),
            indoc! {"
            lambda %0:dimension<3>, %1:f32[3, 4] .
            let %2:f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:explicit, 'a'=2:auto]>, [{}, {'x'}]}] = shard_map [
                mesh=['x'=2:manual, 'y'=2:explicit, 'a'=2:auto],
                in_shardings=[{mesh<['x'=2:manual, 'y'=2:explicit, 'a'=2:auto]>, [{}, {'x'}]}],
                out_shardings=[{mesh<['x'=2:manual, 'y'=2:explicit, 'a'=2:auto]>, [{}, {'x'}]}],
                manual_axes=['x'],
                global_input_types=[f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:explicit, 'a'=2:auto]>, \
                    [{}, {'x'}]}]],
                global_output_types=[f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:explicit, 'a'=2:auto]>, \
                    [{}, {'x'}]}]],
            ] %1 [
                body={
                    lambda %0:f32[3, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit, 'a'=2:auto]>, [{}, {}], \
                        varying_manual={'x'}}] .
                    let %1:f32[3, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit, 'a'=2:auto]>, [{}, {}], \
                        varying_manual={'x'}}] = add %0 %0
                    in (%1)
                },
            ]
            in (%0, %2)"
            }
        );
        assert_eq!(
            batched(&["a", "y"]),
            indoc! {"
            lambda %0:dimension<3>, %1:f32[3, 4] .
            let %2:f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:explicit, 'a'=2:auto]>, [{'y'}, {'x'}]}] = shard_map [
                mesh=['x'=2:manual, 'y'=2:explicit, 'a'=2:auto],
                in_shardings=[{mesh<['x'=2:manual, 'y'=2:explicit, 'a'=2:auto]>, [{'y'}, {'x'}]}],
                out_shardings=[{mesh<['x'=2:manual, 'y'=2:explicit, 'a'=2:auto]>, [{'y'}, {'x'}]}],
                manual_axes=['x'],
                global_input_types=[f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:explicit, 'a'=2:auto]>, \
                    [{'y'}, {'x'}]}]],
                global_output_types=[f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:explicit, 'a'=2:auto]>, \
                    [{'y'}, {'x'}]}]],
            ] %1 [
                body={
                    lambda %0:f32[3, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit, 'a'=2:auto]>, [{'y'}, {}], \
                        varying_manual={'x'}}] .
                    let %1:f32[3, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit, 'a'=2:auto]>, [{'y'}, {}], \
                        varying_manual={'x'}}] = add %0 %0
                    in (%1)
                },
            ]
            in (%0, %2)"
            }
        );
    }

    #[test]
    fn test_shard_map_batching_rejects_batch_axes_placed_on_specified_mesh_axes() {
        // A batch axis placed on a mesh axis that a sharding names is rejected, as JAX rejects a `spmd_axis_name` that
        // its specifications mention, because the batch dimension cannot be partitioned along an axis that already
        // partitions another dimension.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.clone() + x,
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_vector_type(4)],
            Vec::new(),
        );
        let result = program.batched_with_threaded_extent(
            DimensionValue::constant(3).unwrap().r#type().into_owned(),
            ShardingDimension::sharded(["x"]),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        );
        let expected = ShardMapError::BatchAxisPlacedOnSpecifiedAxis { axis_name: "x".to_string() };
        assert!(matches!(
            &result,
            Err(BatchingError::Program(error)) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
        assert_eq!(
            expected.to_string(),
            "batching a `shard_map` places the batch axis on mesh axis `x`, which an input or output sharding of the \
             `shard_map` names",
        );
    }

    #[test]
    fn test_shard_map_batching_rejects_batch_axes_placed_on_axes_that_the_body_uses() {
        // A batch axis placed on an active manual axis `y` along which a body value varies (here, the input of a
        // reduction over `y`, which the reduction makes varying first) is rejected, because the body computes different
        // values on different devices along `y`, so `y` cannot become free.
        let mesh = manual_mesh_2x2();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.parallel_reduce(ReductionKind::Sum, "y").unwrap(),
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_vector_type(4)],
            Vec::new(),
        );
        let result = program.batched_with_threaded_extent(
            DimensionValue::constant(3).unwrap().r#type().into_owned(),
            ShardingDimension::sharded(["y"]),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        );
        let expected = ShardMapError::BatchAxisPlacedOnUsedManualAxis { axis_name: "y".to_string() };
        assert!(matches!(
            &result,
            Err(BatchingError::Program(error)) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
        assert_eq!(
            expected.to_string(),
            "batching a `shard_map` places the batch axis on manual axis `y`, which the `shard_map` body uses",
        );
    }

    #[test]
    fn test_shard_map_batching_rejects_batch_axes_placed_on_axes_that_the_body_communicates_along() {
        // A batch axis placed on an active manual axis `y` over which the body communicates is rejected even when no
        // body value varies along `y`. Here, the body permutes its local values along `y` through the meshless form of
        // `parallel_permute`, whose output type is that of its input, which does not vary along `y`, although the
        // devices along `y` exchange their values.
        let mesh = manual_mesh_2x2();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let shard_map = ShardMap::new(mesh, vec![sharded.clone()], vec![sharded], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &f32_vector_type(4)).unwrap();
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(local_type.into());
            let permute = ParallelPermuteOperation::new("y".to_string(), 2, vec![(0, 1), (1, 0)]);
            let output = builder.add_instruction(permute, Vec::new(), vec![input], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let operation = ShardMapOperation::from_program(&body, vec![f32_vector_type(4).into()], shard_map).unwrap();
        let program = shard_map_program(operation, body).unwrap();
        let result = program.batched_with_threaded_extent(
            DimensionValue::constant(3).unwrap().r#type().into_owned(),
            ShardingDimension::sharded(["y"]),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        );
        let expected = ShardMapError::BatchAxisPlacedOnUsedManualAxis { axis_name: "y".to_string() };
        assert!(matches!(
            &result,
            Err(BatchingError::Program(error)) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
    }

    #[test]
    fn test_shard_map_batching_rejects_batch_axes_placed_on_every_manual_axis() {
        // A batch axis placed on every active manual axis would leave the batched map without a manual axis, so it is
        // rejected even though no sharding names `x` and the body does not use it.
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.clone() + x,
                        inputs[0].clone(),
                        mesh.clone(),
                        replicated.clone(),
                        replicated.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_vector_type(4)],
            Vec::new(),
        );
        let result = program.batched_with_threaded_extent(
            DimensionValue::constant(3).unwrap().r#type().into_owned(),
            ShardingDimension::sharded(["x"]),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        );
        let expected = ShardMapError::BatchAxisPlacedOnEveryManualAxis;
        assert!(matches!(
            &result,
            Err(BatchingError::Program(error)) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
        assert_eq!(
            expected.to_string(),
            "batching a `shard_map` places the batch axis on every manual axis of the `shard_map`, so no manual axis \
             would remain",
        );
    }

    #[test]
    fn test_shard_map_batching_rejects_batch_axes_placed_on_manual_axes_of_device_ordered_bodies() {
        // A batch axis placed on an active manual axis `y` of a body with the `DeviceOrderedIo` effect is rejected,
        // because making `y` free would change the devices that execute the effect.
        let mesh = manual_mesh_2x2();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.print_with_effect_class("body", EffectClass::DeviceOrderedIo).unwrap(),
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_vector_type(4)],
            Vec::new(),
        );
        let result = program.batched_with_threaded_extent(
            DimensionValue::constant(3).unwrap().r#type().into_owned(),
            ShardingDimension::sharded(["y"]),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        );
        let expected = ShardMapError::BatchAxisPlacedOnManualAxisWithDeviceOrderedIo { axis_name: "y".to_string() };
        assert!(matches!(
            &result,
            Err(BatchingError::Program(error)) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
        assert_eq!(
            expected.to_string(),
            "batching a `shard_map` places the batch axis on manual axis `y`, but the `shard_map` body has the \
             `DeviceOrderedIo` effect, which requires every manual axis to remain manual",
        );
    }

    #[test]
    fn test_shard_map_batching_validates_boundaries_against_enclosing_manual_axes() {
        // Inside a manual region over `y`, a map over `x` receives an input that varies along `y`, from which the
        // batching rule infers that `y` is manual in an enclosing region and validates the batched boundary against
        // it. A replicated batch axis keeps the boundary valid, and the batched map still varies along `y`.
        let mesh = manual_mesh_2x2();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let varying = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["y"]).unwrap();
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.clone() + x,
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_vector_type(4).with_sharding(varying).unwrap()],
            vec![("y".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 1, size: 2 })],
        );
        let batched = |placement: ShardingDimension| {
            program.batched_with_threaded_extent(
                DimensionValue::constant(3).unwrap().r#type().into_owned(),
                placement,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
        };
        let sharding = "{mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {'x'}]}";
        let input = "f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'y'}}]";
        let global = "f32[3, 4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {'x'}], varying_manual={'y'}}]";
        let local = "f32[3, 2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}, {}], varying_manual={'x', 'y'}}]";
        assert_eq!(
            batched(ShardingDimension::Replicated).unwrap().into_parts().0.to_string(),
            formatdoc! {"
                lambda %0:dimension<3>, %1:{input} .
                let %2:{global} = shard_map [
                    mesh=['x'=2:manual, 'y'=2:manual],
                    in_shardings=[{sharding}],
                    out_shardings=[{sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}],
                    global_output_types=[{global}],
                ] %1 [
                    body={{
                        lambda %0:{local} .
                        let %1:{local} = add %0 %0
                        in (%1)
                    }},
                ]
                in (%0, %2)"},
        );

        // A batch axis placed on `y` would partition the boundary values along an axis along which they are already
        // the per-device shards of the enclosing region.
        let expected = ShardMapError::SpecificationNamesEnclosingManualAxis {
            value_kind: "input",
            value_index: 0,
            axis_name: "y".to_string(),
        };
        assert!(matches!(
            &batched(ShardingDimension::sharded(["y"])),
            Err(BatchingError::Program(error)) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
    }

    #[test]
    fn test_shard_map_batching_rejects_dynamic_batch_extents() {
        // Mapped batching rejects a dynamic batch extent, which cannot become a dimension of the static boundary types.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.clone() + x,
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_vector_type(4)],
            Vec::new(),
        );
        let extent_type = DimensionType::new("items", DimensionBounds::new(0, Some(5)).unwrap());
        let result = program.batched_with_threaded_extent(
            extent_type.clone(),
            ShardingDimension::Replicated,
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        );
        let expected = ShardMapError::DynamicBatchExtentNotSupported { extent_type };
        assert!(matches!(
            &result,
            Err(BatchingError::Program(error)) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
        assert_eq!(
            expected.to_string(),
            "batching a `shard_map` over mapped inputs requires a static batch extent, but it has type \
             `dimension<items ∈ [0, 5)>`",
        );
    }

    #[test]
    fn test_shard_map_batching_of_bodies_without_inputs() {
        // A `shard_map` without inputs has no batched input, so batching keeps its boundary and its output unbatched.
        let extent_type = DimensionValue::constant(3).unwrap().r#type().into_owned();
        let batched = axis_index_shard_map_program()
            .batched_with_threaded_extent(
                extent_type,
                ShardingDimension::Replicated,
                &[],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<3> .
                let %1:u64[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[],
                    global_output_types=[u64[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ] [
                    body={
                        lambda  .
                        let %0:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = axis_index \
                            [axis_name=\"x\", mesh=['x'=2:manual]]
                            %1:u64[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reshape \
                                [shape=[1]] %0
                        in (%1)
                    },
                ]
                in (%0, %1)"
            },
        );
    }

    #[test]
    fn test_shard_map_batching_consumes_collectives_over_the_batch_axis() {
        // The body inherits the name `items` of the enclosing `batch` level, which batches the body and so consumes its
        // `parallel_reduce` over `items` (JAX's `_batched_reduction_collective`): every batch item receives the sum of
        // the local shards of all items, so the `shard_map` output is unbatched and `batch` broadcasts it to the
        // requested output axis.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let program = trace_test_program(
            |inputs| {
                let output = batch(
                    |item| {
                        let item = ValueProjection::<ArrayType>::into_projected(item)?;
                        let output = shard_map(
                            |x: TestTracer| x.parallel_reduce(ReductionKind::Sum, "items").unwrap(),
                            item,
                            mesh.clone(),
                            sharded.clone(),
                            sharded.clone(),
                        )?;
                        Ok(output.into_value())
                    },
                    inputs[0].value().clone(),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("items"),
                )
                .unwrap();
                vec![ValueProjection::<ArrayType>::into_projected(output).unwrap()]
            },
            vec![ArrayType::new_static(DataType::F32, [3, 4])],
            Vec::new(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3, 4] .
                let %1:dimension<3> = constant [value=3]
                    %2:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}],
                        out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                        manual_axes=['x'],
                        global_input_types=[f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}]],
                        global_output_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                    ] %0 [
                        body={
                            lambda %0:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                            let %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reduce \
                                [kind=sum, axes=[0]] %0
                            in (%1)
                        },
                    ]
                    %3:dimension<4> = constant [value=4]
                    %4:f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}] = broadcast \
                        [output_axes=[1], output_sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}] %2 %1 %3
                in (%4)"
            },
        );
    }

    #[test]
    fn test_shard_map_batching_reduces_over_batch_axes_placed_on_mesh_axes() {
        // The batch axis `items` of the input is placed on the explicit mesh axis `y`, which the boundary keeps, while
        // the body keeps the whole batch extent, so the `parallel_reduce` over `items` inside the body still sums all
        // batch items.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4, 4])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"]), ShardingDimension::sharded(["x"])])
                    .unwrap(),
            )
            .unwrap();
        let program = trace_test_program(
            |inputs| {
                let output = batch(
                    |item| {
                        let item = ValueProjection::<ArrayType>::into_projected(item)?;
                        let output = shard_map(
                            |x: TestTracer| x.parallel_reduce(ReductionKind::Sum, "items").unwrap(),
                            item,
                            mesh.clone(),
                            sharded.clone(),
                            sharded.clone(),
                        )?;
                        Ok(output.into_value())
                    },
                    inputs[0].value().clone(),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("items"),
                )
                .unwrap();
                vec![ValueProjection::<ArrayType>::into_projected(output).unwrap()]
            },
            vec![input_type],
            Vec::new(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4, 4][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'y'}, {'x'}]}] .
                let %1:dimension<4> = constant [value=4]
                    %2:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'x'}]}] = shard_map [
                        mesh=['x'=2:manual, 'y'=2:explicit],
                        in_shardings=[{mesh<['x'=2:manual, 'y'=2:explicit]>, [{'y'}, {'x'}]}],
                        out_shardings=[{mesh<['x'=2:manual, 'y'=2:explicit]>, [{'x'}]}],
                        manual_axes=['x'],
                        global_input_types=[f32[4, 4][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'y'}, {'x'}]}]],
                        global_output_types=[f32[4][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'x'}]}]],
                    ] %0 [
                        body={
                            lambda \
                                %0:f32[4, 2][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'y'}, {}], \
                                    varying_manual={'x'}}] .
                            let %1:f32[2][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{}], \
                                varying_manual={'x'}}] = reduce \
                                [kind=sum, axes=[0]] %0
                            in (%1)
                        },
                    ]
                    %3:dimension<4> = constant [value=4]
                    %4:f32[4, 4][sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'y'}, {'x'}]}] = broadcast [
                        output_axes=[1],
                        output_sharding={mesh<['x'=2:manual, 'y'=2:explicit]>, [{'y'}, {'x'}]},
                    ] %2 %1 %3
                in (%4)"
            }
        );
    }

    #[test]
    fn test_shard_map_batching_of_axis_indices_over_the_batch_axis() {
        // A body without mapped inputs still depends on the batch item when it reads the index of an enclosing named
        // `batch` level, so that level batches the body instead of keeping the boundary of the call: the consumed
        // `axis_index` becomes the vector of item indices, and the `shard_map` output gains the batch dimension.
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let program = trace_test_program(
            |inputs| {
                let output = batch(
                    |item| {
                        let output = shard_map_in_context(
                            &item.domain(),
                            |context: &ShardMapContext<TestContext>, ()| context.axis_index("items").unwrap(),
                            (),
                            mesh.clone(),
                            (),
                            replicated.clone(),
                            Vec::new(),
                        )?;
                        Ok(output.into_value())
                    },
                    inputs[0].value().clone(),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("items"),
                )
                .unwrap();
                vec![ValueProjection::<ArrayType>::into_projected(output).unwrap()]
            },
            vec![f32_vector_type(3)],
            Vec::new(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:dimension<3> = constant [value=3]
                    %2:u64[3][sharding={mesh<['x'=2:manual]>, [{}]}] = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[],
                        out_shardings=[{mesh<['x'=2:manual]>, [{}]}],
                        manual_axes=['x'],
                        global_input_types=[],
                        global_output_types=[u64[3][sharding={mesh<['x'=2:manual]>, [{}]}]],
                    ] [
                        body={
                            lambda  .
                            let %0:u64[3] = iota [type=u64[3], dimension=0]
                            in (%0)
                        },
                    ]
                in (%2)"
            },
        );
    }

    #[test]
    fn test_shard_map_batching_forwards_collectives_over_outer_batch_axes() {
        // An anonymous inner `batch` level batches the body but forwards its `parallel_reduce` over the name `items` of
        // the outer level, which consumes it when it batches the `shard_map` that the inner level staged.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let program = trace_test_program(
            |inputs| {
                let output = batch(
                    |items| {
                        Ok(batch(
                            |item| {
                                let item = ValueProjection::<ArrayType>::into_projected(item)?;
                                let output = shard_map(
                                    |x: TestTracer| x.parallel_reduce(ReductionKind::Sum, "items").unwrap(),
                                    item,
                                    mesh.clone(),
                                    sharded.clone(),
                                    sharded.clone(),
                                )?;
                                Ok(output.into_value())
                            },
                            items,
                            BatchAxis::new(0),
                            BatchAxis::new(0),
                            BatchAxisSpecification::default(),
                        )?)
                    },
                    inputs[0].value().clone(),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("items"),
                )
                .unwrap();
                vec![ValueProjection::<ArrayType>::into_projected(output).unwrap()]
            },
            vec![ArrayType::new_static(DataType::F32, [3, 2, 4])],
            Vec::new(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3, 2, 4] .
                let %1:dimension<3> = constant [value=3]
                    %2:dimension<2> = constant [value=2]
                    %3:f32[2, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}] = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{mesh<['x'=2:manual]>, [{}, {}, {'x'}]}],
                        out_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}],
                        manual_axes=['x'],
                        global_input_types=[f32[3, 2, 4][sharding={mesh<['x'=2:manual]>, [{}, {}, {'x'}]}]],
                        global_output_types=[f32[2, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}]],
                    ] %0 [
                        body={
                            lambda \
                                %0:f32[3, 2, 2][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}] .
                            let %1:f32[2, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = reduce \
                                [kind=sum, axes=[0]] %0
                            in (%1)
                        },
                    ]
                    %4:dimension<2> = constant [value=2]
                    %5:dimension<4> = constant [value=4]
                    %6:f32[3, 2, 4][sharding={mesh<['x'=2:manual]>, [{}, {}, {'x'}]}] = broadcast \
                        [output_axes=[1, 2], output_sharding={mesh<['x'=2:manual]>, [{}, {}, {'x'}]}] %3 %1 %4 %5
                in (%6)"
            },
        );
    }

    #[test]
    fn test_shard_map_batching_of_gradients_with_collectives_over_the_batch_axis() {
        // `batch` over `items` of the gradient of `sum(shard_map(|x| parallel_reduce(x, "items") * x))`: the gradient
        // stages the primal and transposed maps inside the batch level, whose batching rule then consumes the
        // collectives over `items` in their bodies.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let program = trace_test_program(
            |inputs| {
                let gradients = batch(
                    |item| {
                        differentiate_at(item)
                            .gradient(|x| {
                                let x = ValueProjection::<ArrayType>::into_projected(x)?;
                                let y = shard_map(
                                    |x: TestTracer| x.parallel_reduce(ReductionKind::Sum, "items").unwrap() * x,
                                    x,
                                    mesh.clone(),
                                    sharded.clone(),
                                    sharded.clone(),
                                )?;
                                Ok::<_, ProgramError>(y.reduce(&[0], ReductionKind::Sum)?.into_value())
                            })
                            .map_err(ProgramError::from)
                    },
                    inputs[0].value().clone(),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("items"),
                )
                .unwrap();
                vec![ValueProjection::<ArrayType>::into_projected(gradients).unwrap()]
            },
            vec![ArrayType::new_static(DataType::F32, [3, 4])],
            Vec::new(),
        );
        // The batched `sum` of the primal output is dead: `gradient` computes the primal outputs to linearize and then
        // discards them, and tracing does not eliminate dead code (simplification, e.g., `Program::into_pruned`, does).
        // The residual `parallel_reduce(x, "items")` crosses from the primal map to the transposed map tiled along
        // `x`, so the transposed map consumes it without any conversion.
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3, 4] .
                let %1:dimension<3> = constant [value=3]
                    %2:f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}], \
                        %3:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}],
                        out_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                        manual_axes=['x'],
                        global_input_types=[f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}]],
                        global_output_types=[f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}], \
                            f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                    ] %0 [
                        body={
                            lambda %0:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                            let %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reduce \
                                [kind=sum, axes=[0]] %0
                                %2:dimension<3> = constant [value=3]
                                %3:dimension<2> = constant [value=2]
                                %4:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                                    broadcast [
                                    output_axes=[1],
                                    output_sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}},
                                ] %1 %2 %3
                                %5:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = mul %4 \
                                    %0
                            in (%5, %1)
                        },
                    ]
                    %4:f32[3][sharding={mesh<['x'=2:manual]>, [{}]}] = reduce [kind=sum, axes=[1]] %2
                    %5:f32[][sharding={mesh<['x'=2:manual]>, []}] = one \
                        [type=f32[][sharding={mesh<['x'=2:manual]>, []}]]
                    %6:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = broadcast \
                        [output_type=f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}], output_axes=[]] %5
                    %7:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{}, {'x'}]}, \
                            {mesh<['x'=2:manual]>, [{'x'}]}],
                        out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                        manual_axes=['x'],
                        global_input_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}], \
                            f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}], \
                            f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                        global_output_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                    ] %6 %0 %3 [
                        body={
                            lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                                %1:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                                %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                            let %3:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %2 %0
                                %4:dimension<3> = constant [value=3]
                                %5:dimension<2> = constant [value=2]
                                %6:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                                    broadcast [
                                    output_axes=[1],
                                    output_sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}},
                                ] %0 %4 %5
                                %7:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = mul %1 \
                                    %6
                                %8:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reduce \
                                    [kind=sum, axes=[0]] %7
                                %9:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = add %3 %8
                            in (%9)
                        },
                    ]
                    %8:f32[4] = broadcast [output_type=f32[4], output_axes=[0]] %7
                    %9:dimension<4> = constant [value=4]
                    %10:f32[3, 4] = broadcast [output_axes=[1]] %8 %1 %9
                in (%10)"
            },
        );
    }

    #[test]
    fn test_shard_map_differentiation() {
        // Forward-mode differentiation binds one fused `shard_map` over the primal and the tangent of its input, whose
        // body is the fused JVP program of the body and whose tangent boundary types are the tangent descriptors of
        // the primal ones, which may have other element types.
        let mesh = manual_mesh();
        let boundary_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(4)]));
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(boundary_type.clone().into());
            let output = builder
                .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![input, input], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let shard_map = ShardMap::from_shardings(
            mesh.clone(),
            vec![Sharding::replicated(mesh.clone(), 1)],
            vec![Sharding::replicated(mesh, 1)],
            vec!["x".to_string()],
        );
        let operation =
            ShardMapOperation::from_boundary(shard_map, vec![boundary_type.clone()], vec![boundary_type.clone()]);
        let program = shard_map_program(operation, body).unwrap();

        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[4], %1:f32[4] .
                let %2:f8e8m0fnu[4], %3:f32[4] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{}]}, {mesh<['x'=2:manual]>, [{}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{}]}, {mesh<['x'=2:manual]>, [{}]}],
                    manual_axes=['x'],
                    global_input_types=[f8e8m0fnu[4], f32[4]],
                    global_output_types=[f8e8m0fnu[4], f32[4]],
                ] %0 %1 [
                    body={
                        lambda %0:f8e8m0fnu[4], %1:f32[4] .
                        let %2:f8e8m0fnu[4] = mul %0 %0
                            %3:f32[4] = convert_element_type [data_type=f32] %0
                            %4:f32[4] = mul %3 %1
                            %5:f32[4] = convert_element_type [data_type=f32] %0
                            %6:f32[4] = mul %5 %1
                            %7:f32[4] = add %4 %6
                        in (%2, %7)
                    },
                ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_shard_map_differentiation_treats_structurally_zero_tangents_as_inactive() {
        // Forward-mode differentiation treats a structurally zero input tangent as inactive, like JAX's `which_nz`: the
        // fused `shard_map` receives no tangent slot for it, and an output that depends only on such inputs receives a
        // structurally zero tangent instead of a tangent computed from materialized zeros (JAX's `which_nz_out`).

        // The body maps `(x, y)` to `(sin(x), cos(y))`, and only `x` receives a live tangent.
        let array_type = f32_scalar_type();
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let x = builder.add_input(array_type.clone().into());
            let y = builder.add_input(array_type.clone().into());
            let sine = builder
                .add_instruction(ArrayOperation::Sin(SinOperation::new()), Vec::new(), vec![x], None)
                .unwrap()[0];
            let cosine = builder
                .add_instruction(ArrayOperation::Cos(CosOperation::new()), Vec::new(), vec![y], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![sine, cosine], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let operation = ShardMapOperation::from_boundary(
            ShardMap::from_shardings(mesh, vec![replicated.clone(); 2], vec![replicated; 2], vec!["x".to_string()]),
            vec![array_type.clone(); 2],
            vec![array_type; 2],
        );
        let program = shard_map_program(operation, body).unwrap();
        // The fused `shard_map` consumes only the tangent of `x` besides the primals, and it produces only the tangent
        // of `sin(x)` besides the outputs. The tangent of `cos(y)` is a structural zero, which the program boundary
        // materializes outside the manual region.
        assert_eq!(
            program.entry_region_ref().jvp(&[0]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[] .
                let %3:f32[], %4:f32[], %5:f32[] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, []}, {mesh<['x'=2:manual]>, []}, {mesh<['x'=2:manual]>, \
                    []}],
                    out_shardings=[{mesh<['x'=2:manual]>, []}, {mesh<['x'=2:manual]>, []}, {mesh<['x'=2:manual]>, \
                    []}],
                    manual_axes=['x'],
                    global_input_types=[f32[], f32[], f32[]],
                    global_output_types=[f32[], f32[], f32[]],
                ] %0 %1 %2 [
                    body={
                        lambda %0:f32[], %1:f32[], %2:f32[] .
                        let %3:f32[] = sin %0
                            %4:f32[] = cos %1
                            %5:f32[] = cos %0
                            %6:f32[] = mul %5 %2
                        in (%3, %4, %6)
                    },
                ]
                    %6:f32[] = zero [type=f32[]]
                in (%3, %4, %5, %6)"},
        );
    }

    #[test]
    fn test_shard_map_differentiation_preserves_sparse_output_tangent_order() {
        // The boolean output has no tangent slot, while `cos(y)` has a slot but a structurally zero tangent. Only
        // `x` is active, so the two live tangents, which the fused `shard_map` returns after its four outputs, must
        // still line up with `sin(x)` and the final forwarded `x`.
        let replicated = Sharding::replicated(manual_mesh(), 0);
        let program = trace_test_program(
            |inputs| {
                let (flag, sine, cosine, forwarded) = shard_map(
                    |(flag, x, y): (TestTracer, TestTracer, TestTracer)| (flag, x.sin().unwrap(), y.cos().unwrap(), x),
                    (inputs[0].clone(), inputs[1].clone(), inputs[2].clone()),
                    manual_mesh(),
                    (replicated.clone(), replicated.clone(), replicated.clone()),
                    (replicated.clone(), replicated.clone(), replicated.clone(), replicated.clone()),
                )
                .unwrap();
                vec![flag, sine, cosine, forwarded]
            },
            vec![ArrayType::scalar(DataType::Boolean), f32_scalar_type(), f32_scalar_type()],
            Vec::new(),
        );
        let differentiated = program.entry_region_ref().jvp(&[1]).unwrap();
        let [fused_map] = differentiated
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), ArrayIrOperation::ShardMap(_)))
            .collect::<Vec<_>>()[..]
        else {
            panic!("expected one fused `shard_map`");
        };
        assert_eq!(differentiated.output_count(), 7);
        assert_eq!(fused_map.outputs().len(), 6);
        assert_eq!(differentiated.output_ids()[4], fused_map.outputs()[4]);
        assert_eq!(differentiated.output_ids()[6], fused_map.outputs()[5]);
        assert!(differentiated.instructions().iter().any(|instruction| {
            matches!(instruction.operation(), ArrayIrOperation::Array(ArrayOperation::Zero(_)))
                && instruction.outputs() == &differentiated.output_ids()[5..6]
        }));
    }

    #[test]
    fn test_shard_map_differentiation_forwards_residuals() {
        // Residuals that already cross the boundary are not carried by residual edges (JAX's `in_fwd` and `out_fwd`):
        // under linearization, which binds a primal and a tangent `shard_map`, the input `x`, which linearization
        // saves once and the tangent of `x * x` reads twice, reaches the tangent `shard_map` as the primal input under
        // its input sharding, and `exp(x)` reaches it as the primal output. Only `cos(x)`, which is computed inside the
        // body, is a residual edge, tiled along `x` so that its local shard is `cos(x)` itself.
        let sharded = sharded_along_x();
        let global_type = f32_vector_type(4).with_sharding(sharded.clone()).unwrap();
        let program = trace_test_program(
            |inputs| {
                let (square, exponential, sine) = shard_map(
                    |x: TestTracer| (x.clone() * x.clone(), x.exp().unwrap(), x.sin().unwrap()),
                    inputs[0].clone(),
                    manual_mesh(),
                    sharded.clone(),
                    (sharded.clone(), sharded.clone(), sharded.clone()),
                )
                .unwrap();
                vec![square, exponential, sine]
            },
            vec![global_type],
            Vec::new(),
        );
        let linearization = program.linearize().unwrap();
        let sharding = "{mesh<['x'=2:manual]>, [{'x'}]}";
        let global = "f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]";
        let local = "f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        assert_eq!(
            linearization.primal().to_string(),
            formatdoc! {"
                lambda %0:{global} .
                let %1:{global}, %2:{global}, %3:{global}, %4:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}],
                    out_shardings=[{sharding}, {sharding}, {sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}],
                    global_output_types=[{global}, {global}, {global}, {global}],
                ] %0 [
                    body={{
                        lambda %0:{local} .
                        let %1:{local} = mul %0 %0
                            %2:{local} = exp %0
                            %3:{local} = sin %0
                            %4:{local} = cos %0
                        in (%1, %2, %3, %4)
                    }},
                ]
                in (%1, %2, %3, %0, %2, %4)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            formatdoc! {"
                lambda %0:{global}, %1:{global}, %2:{global}, %3:{global} .
                let %4:{global}, %5:{global}, %6:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}, {sharding}, {sharding}],
                    out_shardings=[{sharding}, {sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}, {global}, {global}],
                    global_output_types=[{global}, {global}, {global}],
                ] %0 %1 %2 %3 [
                    body={{
                        lambda %0:{local}, %1:{local}, %2:{local}, %3:{local} .
                        let %4:{local} = mul %1 %0
                            %5:{local} = mul %1 %0
                            %6:{local} = add %4 %5
                            %7:{local} = mul %2 %0
                            %8:{local} = mul %3 %0
                        in (%6, %7, %8)
                    }},
                ]
                in (%4, %5, %6)"},
        );
    }

    #[test]
    fn test_shard_map_differentiation_deduplicates_residual_edges() {
        // Under linearization, `c * c` for `c = cos(x)` needs `c` and `sin(x)` as residuals; `c`, which the tangent of
        // `c * c` uses twice, crosses as one residual edge.
        let sharded = sharded_along_x();
        let global_type = f32_vector_type(4).with_sharding(sharded.clone()).unwrap();
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| {
                            let cosine = x.cos().unwrap();
                            cosine.clone() * cosine
                        },
                        inputs[0].clone(),
                        manual_mesh(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![global_type],
            Vec::new(),
        );
        let linearization = program.linearize().unwrap();
        let sharding = "{mesh<['x'=2:manual]>, [{'x'}]}";
        let global = "f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]";
        let local = "f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        assert_eq!(
            linearization.primal().to_string(),
            formatdoc! {"
                lambda %0:{global} .
                let %1:{global}, %2:{global}, %3:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}],
                    out_shardings=[{sharding}, {sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}],
                    global_output_types=[{global}, {global}, {global}],
                ] %0 [
                    body={{
                        lambda %0:{local} .
                        let %1:{local} = cos %0
                            %2:{local} = mul %1 %1
                            %3:{local} = sin %0
                        in (%2, %3, %1)
                    }},
                ]
                in (%1, %2, %3)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            formatdoc! {"
                lambda %0:{global}, %1:{global}, %2:{global} .
                let %3:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}, {sharding}],
                    out_shardings=[{sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}, {global}],
                    global_output_types=[{global}],
                ] %0 %1 %2 [
                    body={{
                        lambda %0:{local}, %1:{local}, %2:{local} .
                        let %3:{local} = mul %1 %0
                            %4:{local} = neg %3
                            %5:{local} = mul %2 %4
                            %6:{local} = mul %2 %4
                            %7:{local} = add %5 %6
                        in (%7)
                    }},
                ]
                in (%3)"},
        );
    }

    #[test]
    fn test_shard_map_linearization_crosses_residual_edges_without_conversions() {
        // Linearization binds a primal and a tangent `shard_map`, and the residual edges `cos(x)` and `sin(x)` cross
        // between them tiled along `x`, so that their local shards are the residuals themselves. Neither body converts
        // them, so the partial evaluation of the tangent `shard_map` (whose residual inputs are known) finds no work
        // that depends on the residuals alone and keeps it whole, instead of splitting off a known `shard_map` that
        // only converts the residual edges back and forth. The linearization of the fused forward-mode program binds
        // one map on each side as well.
        let program = sine_product_shard_map_program();
        let linearization = program.linearize().unwrap();
        let sharding = "{mesh<['x'=2:manual]>, [{'x'}]}";
        let global = "f32[8][sharding={mesh<['x'=2:manual]>, [{'x'}]}]";
        let local = "f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        assert_eq!(
            linearization.primal().to_string(),
            formatdoc! {"
                lambda %0:{global}, %1:{global} .
                let %2:{global}, %3:{global}, %4:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}],
                    out_shardings=[{sharding}, {sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}],
                    global_output_types=[{global}, {global}, {global}],
                ] %0 %1 [
                    body={{
                        lambda %0:{local}, %1:{local} .
                        let %2:{local} = sin %0
                            %3:{local} = mul %2 %1
                            %4:{local} = cos %0
                        in (%3, %4, %2)
                    }},
                ]
                in (%2, %3, %1, %4)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            formatdoc! {"
                lambda %0:{global}, %1:{global}, %2:{global}, %3:{global}, %4:{global} .
                let %5:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}, {sharding}, {sharding}, {sharding}],
                    out_shardings=[{sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}, {global}, {global}, {global}],
                    global_output_types=[{global}],
                ] %0 %1 %2 %3 %4 [
                    body={{
                        lambda %0:{local}, %1:{local}, %2:{local}, %3:{local}, %4:{local} .
                        let %5:{local} = mul %2 %0
                            %6:{local} = mul %3 %5
                            %7:{local} = mul %4 %1
                            %8:{local} = add %6 %7
                        in (%8)
                    }},
                ]
                in (%5)"},
        );
        let nested = program.jvp().unwrap().linearize().unwrap();
        assert_eq!(
            nested.primal().to_string(),
            formatdoc! {"
                lambda %0:{global}, %1:{global}, %2:{global}, %3:{global} .
                let %4:{global}, %5:{global}, %6:{global}, %7:{global}, %8:{global}, %9:{global}, \
                    %10:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}, {sharding}, {sharding}],
                    out_shardings=[{sharding}, {sharding}, {sharding}, {sharding}, {sharding}, {sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}, {global}, {global}],
                    global_output_types=[{global}, {global}, {global}, {global}, {global}, {global}, {global}],
                ] %0 %1 %2 %3 [
                    body={{
                        lambda %0:{local}, %1:{local}, %2:{local}, %3:{local} .
                        let %4:{local} = sin %0
                            %5:{local} = mul %4 %1
                            %6:{local} = cos %0
                            %7:{local} = mul %6 %2
                            %8:{local} = mul %1 %7
                            %9:{local} = mul %4 %3
                            %10:{local} = add %8 %9
                            %11:{local} = cos %0
                            %12:{local} = sin %0
                        in (%5, %10, %11, %12, %6, %4, %7)
                    }},
                ]
                in (%4, %5, %6, %7, %2, %8, %1, %9, %10, %3)"},
        );
        let tangent_inputs = (0..12).map(|index| format!("%{index}:{global}")).collect::<Vec<_>>().join(", ");
        assert_eq!(
            nested.tangent().to_string(),
            formatdoc! {"
                lambda {tangent_inputs} .
                let %12:{global}, %13:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}, {sharding}, {sharding}, {sharding}, {sharding}, {sharding}, \
                        {sharding}, {sharding}, {sharding}, {sharding}, {sharding}],
                    out_shardings=[{sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}, {global}, {global}, {global}, {global}, {global}, \
                        {global}, {global}, {global}, {global}, {global}],
                    global_output_types=[{global}, {global}],
                ] %0 %1 %2 %3 %4 %5 %6 %7 %8 %9 %10 %11 [
                    body={{
                        lambda %0:{local}, %1:{local}, %2:{local}, %3:{local}, %4:{local}, %5:{local}, %6:{local}, \
                            %7:{local}, %8:{local}, %9:{local}, %10:{local}, %11:{local} .
                        let %12:{local} = mul %4 %0
                            %13:{local} = mul %8 %12
                            %14:{local} = mul %9 %1
                            %15:{local} = add %13 %14
                            %16:{local} = mul %10 %1
                            %17:{local} = mul %5 %0
                            %18:{local} = neg %17
                            %19:{local} = mul %6 %18
                            %20:{local} = mul %7 %2
                            %21:{local} = add %19 %20
                            %22:{local} = mul %8 %21
                            %23:{local} = add %16 %22
                            %24:{local} = mul %11 %12
                            %25:{local} = mul %9 %3
                            %26:{local} = add %24 %25
                            %27:{local} = add %23 %26
                        in (%15, %27)
                    }},
                ]
                in (%12, %13)"},
        );
    }

    #[test]
    fn test_shard_map_linearization_crosses_varying_scalar_residual_edges_without_conversion_maps() {
        // The body reshapes its `f32[1]` shard to the varying scalar `x` and returns `sin(x)`, so the residual `cos(x)`
        // is a varying scalar, which crosses as a single-element vector tiled along `x`, and the tangent body starts by
        // unpacking it. The tangent `shard_map` is partially evaluated with its residual edge known, but that unpacking
        // is the only work that depends on the residual alone, so the map is kept whole instead of splitting off a
        // known `shard_map` that only unpacks and repacks the edge. This holds for forward linearization, for reverse
        // mode, and for the linearization of the fused forward-mode program.
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.reshape(Shape::new(Vec::new())).unwrap().sin().unwrap().reshape([1]).unwrap(),
                        inputs[0].clone(),
                        manual_mesh(),
                        sharded_along_x(),
                        sharded_along_x(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_vector_type(2)],
            Vec::new(),
        );
        let linearization = program.linearize().unwrap();
        let sharding = "{mesh<['x'=2:manual]>, [{'x'}]}";
        let global = "f32[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}]";
        let promoted_sharding = "{mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}";
        let promoted = format!("f32[1][sharding={promoted_sharding}]");
        let scalar_sharding = "{mesh<['x'=2:manual]>, [], varying_manual={'x'}}";
        let scalar = format!("f32[][sharding={scalar_sharding}]");
        assert_eq!(
            linearization.primal().to_string(),
            formatdoc! {"
                lambda %0:f32[2] .
                let %1:{global}, %2:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}],
                    out_shardings=[{sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}],
                    global_output_types=[{global}, {global}],
                ] %0 [
                    body={{
                        lambda %0:{promoted} .
                        let %1:{scalar} = reshape [shape=[]] %0
                            %2:{scalar} = sin %1
                            %3:{promoted} = reshape [shape=[1]] %2
                            %4:{scalar} = cos %1
                            %5:{promoted} = reshape [shape=[1], output_sharding={promoted_sharding}] %4
                        in (%3, %5)
                    }},
                ]
                in (%1, %2)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            formatdoc! {"
                lambda %0:f32[2], %1:{global} .
                let %2:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}],
                    out_shardings=[{sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}],
                    global_output_types=[{global}],
                ] %0 %1 [
                    body={{
                        lambda %0:{promoted}, %1:{promoted} .
                        let %2:{scalar} = reshape [shape=[], output_sharding={scalar_sharding}] %1
                            %3:{scalar} = reshape [shape=[]] %0
                            %4:{scalar} = mul %2 %3
                            %5:{promoted} = reshape [shape=[1]] %4
                        in (%5)
                    }},
                ]
                in (%2)"},
        );
        let reverse = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        assert_eq!(reverse.primal().to_string(), linearization.primal().to_string());
        assert_eq!(reverse.tangent().to_string(), linearization.tangent().to_string());
        let nested = program.jvp().unwrap().linearize().unwrap();
        assert_eq!(
            nested.primal().to_string(),
            formatdoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:{global}, %3:{global}, %4:{global}, %5:{global}, %6:{global}, %7:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}],
                    out_shardings=[{sharding}, {sharding}, {sharding}, {sharding}, {sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}],
                    global_output_types=[{global}, {global}, {global}, {global}, {global}, {global}],
                ] %0 %1 [
                    body={{
                        lambda %0:{promoted}, %1:{promoted} .
                        let %2:{scalar} = reshape [shape=[]] %0
                            %3:{scalar} = sin %2
                            %4:{promoted} = reshape [shape=[1]] %3
                            %5:{scalar} = cos %2
                            %6:{scalar} = reshape [shape=[]] %1
                            %7:{scalar} = mul %5 %6
                            %8:{promoted} = reshape [shape=[1]] %7
                            %9:{scalar} = cos %2
                            %10:{promoted} = reshape [shape=[1], output_sharding={promoted_sharding}] %9
                            %11:{scalar} = sin %2
                            %12:{promoted} = reshape [shape=[1], output_sharding={promoted_sharding}] %11
                            %13:{promoted} = reshape [shape=[1], output_sharding={promoted_sharding}] %6
                            %14:{promoted} = reshape [shape=[1], output_sharding={promoted_sharding}] %5
                        in (%4, %8, %10, %12, %13, %14)
                    }},
                ]
                in (%2, %3, %4, %5, %6, %7)"},
        );
        assert_eq!(
            nested.tangent().to_string(),
            formatdoc! {"
                lambda %0:f32[2], %1:f32[2], %2:{global}, %3:{global}, %4:{global}, %5:{global} .
                let %6:{global}, %7:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}, {sharding}, {sharding}, {sharding}, {sharding}],
                    out_shardings=[{sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}, {global}, {global}, {global}, {global}],
                    global_output_types=[{global}, {global}],
                ] %0 %1 %2 %3 %4 %5 [
                    body={{
                        lambda %0:{promoted}, %1:{promoted}, %2:{promoted}, %3:{promoted}, %4:{promoted}, \
                            %5:{promoted} .
                        let %6:{scalar} = reshape [shape=[], output_sharding={scalar_sharding}] %2
                            %7:{scalar} = reshape [shape=[]] %0
                            %8:{scalar} = mul %6 %7
                            %9:{promoted} = reshape [shape=[1]] %8
                            %10:{scalar} = reshape [shape=[], output_sharding={scalar_sharding}] %4
                            %11:{scalar} = reshape [shape=[], output_sharding={scalar_sharding}] %3
                            %12:{scalar} = mul %11 %7
                            %13:{scalar} = neg %12
                            %14:{scalar} = mul %10 %13
                            %15:{scalar} = reshape [shape=[], output_sharding={scalar_sharding}] %5
                            %16:{scalar} = reshape [shape=[]] %1
                            %17:{scalar} = mul %15 %16
                            %18:{scalar} = add %14 %17
                            %19:{promoted} = reshape [shape=[1]] %18
                        in (%9, %19)
                    }},
                ]
                in (%6, %7)"},
        );

        // The linearization matches the same function outside `shard_map`.
        let unmapped =
            trace_test_program(|inputs| vec![inputs[0].clone().sin().unwrap()], vec![f32_vector_type(2)], Vec::new());
        let x = TestValue::Array(Array::vector(vec![0.5f32, 1.5]).unwrap());
        let tangent = TestValue::Array(Array::vector(vec![1.0f32, -2.0]).unwrap());
        assert_eq!(
            evaluate_linearization(&program, vec![x.clone()], vec![tangent.clone()]),
            evaluate_linearization(&unmapped, vec![x], vec![tangent]),
        );
    }

    #[test]
    fn test_shard_map_linearization_of_while_loops_over_fused_maps() {
        // The forward-mode rule of an unbounded `while` loop differentiates its body with the fused policy, so a
        // `shard_map` in the body becomes one fused map, and linearization then partitions the fused loop, whose body
        // the partial-evaluation rule of `shard_map` splits. The result matches that of the same loop without the map.
        let program = sine_while_program(true);
        let sharding = "{mesh<['x'=2:manual]>, [{'x'}]}";
        let global = "f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]";
        let local = "f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        assert_eq!(
            program.jvp().unwrap().to_string(),
            formatdoc! {"
                lambda %0:f32[], %1:{global}, %2:f32[], %3:{global} .
                let %4:f32[], %5:{global}, %6:f32[], %7:{global} = while %0 %1 %2 %3 [
                    condition={{
                        lambda %0:f32[], %1:{global}, %2:f32[], %3:{global} .
                        let %4:f32[] = zero_like %0
                            %5:bool[] = compare [direction=GreaterThan] %0 %4
                        in (%5)
                    }},
                    body={{
                        lambda %0:f32[], %1:{global}, %2:f32[], %3:{global} .
                        let %4:f32[] = one_like %0
                            %5:f32[] = sub %0 %4
                            %6:{global}, %7:{global} = shard_map [
                                mesh=['x'=2:manual],
                                in_shardings=[{sharding}, {sharding}],
                                out_shardings=[{sharding}, {sharding}],
                                manual_axes=['x'],
                                global_input_types=[{global}, {global}],
                                global_output_types=[{global}, {global}],
                            ] %1 %3 [
                                body={{
                                    lambda %0:{local}, %1:{local} .
                                    let %2:{local} = sin %0
                                        %3:{local} = cos %0
                                        %4:{local} = mul %3 %1
                                    in (%2, %4)
                                }},
                            ]
                        in (%5, %6, %2, %7)
                    }},
                ]
                in (%4, %5, %6, %7)"},
        );
        let sharded = sharded_along_x();
        let global_type = f32_vector_type(4).with_sharding(sharded).unwrap();
        let counter = ArrayIrValue::Array(Array::from_elements(f32_scalar_type(), &[3f32]).unwrap());
        let x = ArrayIrValue::Array(Array::from_elements(global_type.clone(), &[0.5f32, 1.0, 1.5, 2.0]).unwrap());
        let counter_tangent = ArrayIrValue::Array(Array::from_elements(f32_scalar_type(), &[0f32]).unwrap());
        let tangent = ArrayIrValue::Array(Array::from_elements(global_type, &[1f32, -1.0, 2.0, 0.5]).unwrap());
        let primals = vec![counter, x];
        let tangents = vec![counter_tangent, tangent];
        let expected = evaluate_linearization(&sine_while_program(false), primals.clone(), tangents.clone());
        assert_eq!(evaluate_linearization(&program, primals, tangents), expected);
    }

    #[test]
    fn test_shard_map_linearization_of_rematerialized_maps() {
        // The partitioned forward-mode rule of `rematerialize` partitions the fused derivative of its body, whose
        // `shard_map` becomes one fused map that the partial-evaluation rule of `shard_map` splits. The result matches
        // that of the same function without the map.
        type BodyTracer = Tracer<TestContext>;
        let sharded = sharded_along_x();
        let global_type = f32_vector_type(4).with_sharding(sharded.clone()).unwrap();
        let program = |through_shard_map: bool| {
            let sharded = sharded.clone();
            let function = rematerialize(move |x: BodyTracer| {
                let x = ValueProjection::<ArrayType>::into_projected(x)?;
                let output = if through_shard_map {
                    shard_map(
                        |x: TestTracer| x.clone().sin().unwrap() * x,
                        x,
                        manual_mesh(),
                        sharded.clone(),
                        sharded.clone(),
                    )?
                } else {
                    x.clone().sin()? * x
                };
                Ok(output.into_value())
            });
            trace_test_program(
                |inputs| {
                    let output = function.call(inputs[0].value().clone()).unwrap();
                    vec![ValueProjection::<ArrayType>::into_projected(output).unwrap()]
                },
                vec![global_type.clone()],
                Vec::new(),
            )
        };
        let x = ArrayIrValue::Array(Array::from_elements(global_type.clone(), &[0.5f32, 1.0, 1.5, 2.0]).unwrap());
        let tangent = ArrayIrValue::Array(Array::from_elements(global_type.clone(), &[1f32, -1.0, 2.0, 0.5]).unwrap());
        let expected = evaluate_linearization(&program(false), vec![x.clone()], vec![tangent.clone()]);
        assert_eq!(evaluate_linearization(&program(true), vec![x], vec![tangent]), expected);
    }

    #[test]
    fn test_shard_map_differentiation_deduplicates_repeated_residuals() {
        // The in-repo linearization saves distinct residual atoms, so this test injects a linearization of `sin(x)`
        // whose primal program returns each of its residuals `cos(x)`, `x`, and `sin(x)` twice. Each repeated residual
        // shares one tangent residual input, whether it is a residual edge (`cos(x)`), a forwarded primal input (`x`),
        // or a forwarded primal output (`sin(x)`), and the edge crosses the primal boundary once.
        let replicated = Sharding::replicated(manual_mesh(), 0);
        let scalar = f32_scalar_type().with_sharding(replicated.clone()).unwrap();
        let source = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let x = builder.add_input(scalar.clone().into());
            let sine = builder
                .add_instruction(ArrayOperation::Sin(SinOperation::new()), Vec::new(), vec![x], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![sine], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let shard_map =
            ShardMap::new(manual_mesh(), vec![replicated.clone()], vec![replicated.clone()], Vec::new()).unwrap();
        let operation = ShardMapOperation::from_program(&source, vec![f32_scalar_type().into()], shard_map).unwrap();
        let primal = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let x = builder.add_input(scalar.clone().into());
            let sine = builder
                .add_instruction(ArrayOperation::Sin(SinOperation::new()), Vec::new(), vec![x], None)
                .unwrap()[0];
            let cosine = builder
                .add_instruction(ArrayOperation::Cos(CosOperation::new()), Vec::new(), vec![x], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![sine, cosine, cosine, x, x, sine, sine],
                    vec![Placeholder],
                    vec![Placeholder; 7],
                )
                .unwrap()
        };
        // The tangent program reads every residual once, multiplying them into the input tangent.
        let tangent = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let mut output = builder.add_input(scalar.clone().into());
            let residuals = (0..6).map(|_| builder.add_input(scalar.clone().into())).collect::<Vec<_>>();
            for residual in residuals {
                output = builder
                    .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![residual, output], None)
                    .unwrap()[0];
            }
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 7], vec![Placeholder])
                .unwrap()
        };
        let driver = TestDifferentiationDriver { source: source.clone(), primal, tangent, residual_count: 6 };
        let bodies =
            shard_map_bodies::<TestContext, _>(&operation, &driver, source.entry_region_ref(), &[true]).unwrap();
        let tangent = bodies.tangent.unwrap();
        assert_eq!(
            tangent.residuals,
            vec![ShardMapResidual::Edge(0), ShardMapResidual::Input(0), ShardMapResidual::Output(0)],
        );

        // The primal `shard_map` returns its output followed by the single residual edge `cos(x)`.
        assert_eq!(
            bodies.primal_operation,
            ShardMapOperation::from_boundary(
                ShardMap::from_shardings(
                    manual_mesh(),
                    vec![replicated.clone()],
                    vec![replicated.clone(); 2],
                    vec!["x".to_string()],
                ),
                vec![scalar.clone()],
                vec![scalar.clone(); 2],
            ),
        );
        assert_eq!(
            bodies.primal_body.to_string(),
            formatdoc! {"
                lambda %0:{scalar} .
                let %1:{scalar} = sin %0
                    %2:{scalar} = cos %0
                in (%1, %2)"},
        );

        // The tangent `shard_map` takes `ẋ` followed by one input per distinct residual: the edge `cos(x)`, the primal
        // input `x`, and the primal output `sin(x)`, each of which feeds both of its uses.
        assert_eq!(
            tangent.operation,
            ShardMapOperation::from_boundary(
                ShardMap::from_shardings(
                    manual_mesh(),
                    vec![replicated.clone(); 4],
                    vec![replicated],
                    vec!["x".to_string()],
                ),
                vec![scalar.clone(); 4],
                vec![scalar.clone()],
            ),
        );
        assert_eq!(
            tangent.body.to_string(),
            formatdoc! {"
                lambda %0:{scalar}, %1:{scalar}, %2:{scalar}, %3:{scalar} .
                let %4:{scalar} = mul %1 %0
                    %5:{scalar} = mul %1 %4
                    %6:{scalar} = mul %2 %5
                    %7:{scalar} = mul %2 %6
                    %8:{scalar} = mul %3 %7
                    %9:{scalar} = mul %3 %8
                in (%9)"},
        );
    }

    #[test]
    fn test_shard_map_differentiation_forwards_reference_input_residuals() {
        // A reference residual preserves the identity of the primal reference, so a linearization whose tangent reads
        // the reference input `r` forwards `r` itself to the tangent `shard_map` under its input sharding, rather than
        // packing it into a residual edge. The in-repo linearization saves the read value instead (refer to
        // `test_shard_map_differentiation_saves_reference_reads_as_array_residuals`), so this test injects the
        // linearization of the body `(r, x) -> read(r) * x` with only `x` active, whose tangent reads `r` again.
        let (operation, source) = reference_product_shard_map();
        let reference_type = source.input_types()[0].clone();
        let value_type = source.input_types()[1].clone();
        let primal = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone());
            let value = builder.add_input(value_type.clone());
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let product = builder
                .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![state, value], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![product, reference],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let tangent = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let value_tangent = builder.add_input(value_type);
            let reference = builder.add_input(reference_type);
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let product = builder
                .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![state, value_tangent], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let driver = TestDifferentiationDriver { source: source.clone(), primal, tangent, residual_count: 1 };
        let bodies =
            shard_map_bodies::<TestContext, _>(&operation, &driver, source.entry_region_ref(), &[false, true]).unwrap();
        assert_eq!(bodies.tangent.unwrap().residuals, vec![ShardMapResidual::Input(0)]);

        // Binding both maps through the rule transfers the primal reference to the tangent context by identity and
        // attaches the tangent body, whose boundary takes `ẋ` followed by the reference input itself under its input
        // sharding. The primal `shard_map` is the source boundary, since its only residual is forwarded.
        let context = TestContext::new();
        let reference = context.input(operation.global_input_types()[0].clone());
        let value = context.input(operation.global_input_types()[1].clone());
        let value_tangent = context.input(operation.global_input_types()[1].tangent().unwrap());
        let inputs = [
            DifferentiationDual::new_with_zero_tangent(reference).unwrap(),
            DifferentiationDual::new(value, value_tangent).unwrap(),
        ];
        let outputs = operation.jvp(&DifferentiationContext::fused(context.clone()), &driver, &inputs).unwrap();
        let [output] = &outputs[..] else {
            panic!("expected one output dual");
        };
        let MaybeZero::Value(output_tangent) = output.tangent() else {
            panic!("expected a live output tangent");
        };
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![output.primal().atom_id().unwrap(), output_tangent.atom_id().unwrap()],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let sharding = "{mesh<['x'=2:manual]>, [{'x'}]}";
        let global = "f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]";
        let local = "f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:ref<{global}>, %1:{global}, %2:{global} .
                let %3:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}],
                    out_shardings=[{sharding}],
                    manual_axes=['x'],
                    global_input_types=[ref<{global}>, {global}],
                    global_output_types=[{global}],
                ] %0 %1 [
                    body={{
                        lambda %0:ref<{local}>, %1:{local} .
                        let %2:{local} = reference_read %0
                            %3:{local} = mul %2 %1
                        in (%3)
                    }},
                ]
                    %4:{global} = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{sharding}, {sharding}],
                        out_shardings=[{sharding}],
                        manual_axes=['x'],
                        global_input_types=[{global}, ref<{global}>],
                        global_output_types=[{global}],
                    ] %2 %0 [
                        body={{
                            lambda %0:{local}, %1:ref<{local}> .
                            let %2:{local} = reference_read %1
                                %3:{local} = mul %2 %0
                            in (%3)
                        }},
                    ]
                in (%3, %4)"},
        );
    }

    #[test]
    fn test_shard_map_differentiation_snapshots_reference_residuals_allocated_in_the_body() {
        // A reference residual allocated inside the body cannot cross the boundary between the primal and tangent maps
        // by identity, so it crosses as a snapshot of its final state: the primal body reads its complete referent
        // after its last instruction into a residual edge (tiled along `x`, since the referent varies along `x`), and
        // the tangent body allocates a fresh reference from that edge before its first instruction. As in
        // `test_shard_map_differentiation_forwards_reference_input_residuals`, the injected linearization belongs to
        // the body `(r, x) -> read(r) * x` with only `x` active, but it saves a fresh reference to the read value.
        let (operation, source) = reference_product_shard_map();
        let reference_type = source.input_types()[0].clone();
        let value_type = source.input_types()[1].clone();
        let primal = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone());
            let value = builder.add_input(value_type.clone());
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let local =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            let product = builder
                .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![state, value], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![product, local],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let tangent = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let value_tangent = builder.add_input(value_type);
            let local = builder.add_input(reference_type);
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![local], None).unwrap()[0];
            let product = builder
                .add_instruction(ArrayOperation::Mul(MulOperation::new()), Vec::new(), vec![state, value_tangent], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let driver = TestDifferentiationDriver { source: source.clone(), primal, tangent, residual_count: 1 };
        let bodies =
            shard_map_bodies::<TestContext, _>(&operation, &driver, source.entry_region_ref(), &[false, true]).unwrap();
        let tangent = bodies.tangent.unwrap();
        assert_eq!(tangent.residuals, vec![ShardMapResidual::ReferenceSnapshot(0)]);
        let global = ArrayIrType::Array(f32_vector_type(4).with_sharding(sharded_along_x()).unwrap());
        assert_eq!(bodies.primal_operation.global_output_types(), &[global.clone(), global.clone()]);
        assert_eq!(tangent.operation.global_input_types(), &[global.clone(), global]);
        let local = "f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        assert_eq!(
            bodies.primal_body.to_string(),
            formatdoc! {"
                lambda %0:ref<{local}>, %1:{local} .
                let %2:{local} = reference_read %0
                    %3:ref<{local}> = reference_new %2
                    %4:{local} = mul %2 %1
                    %5:{local} = reference_read %3
                in (%4, %5)"},
        );
        assert_eq!(
            tangent.body.to_string(),
            formatdoc! {"
                lambda %0:{local}, %1:{local} .
                let %2:ref<{local}> = reference_new %1
                    %3:{local} = reference_read %2
                    %4:{local} = mul %3 %0
                in (%4)"},
        );
    }

    #[test]
    fn test_shard_map_differentiation_of_custom_functions_over_body_allocated_references() {
        // A body may allocate a reference and pass it to a `custom_function` as a leading non-differentiated (plumbing)
        // input. The explicit JVP rule replays inline, so the reference is not a residual. The body accesses no
        // reference, so forward mode binds one fused map, whose dead work (here, the allocations of the reference and
        // its tangent) is removed before binding, and in reverse mode, only `cos(x)` crosses from the primal map to the
        // tangent map.
        type BodyTracer = Tracer<TestContext>;
        let function = custom_function(|(_, x): (BodyTracer, BodyTracer)| {
            Ok(ValueProjection::<ArrayType>::into_projected(x)?.sin()?.into_value())
        })
        .with_non_differentiated_count(1)
        .with_jvp(|(_, x): (BodyTracer, BodyTracer), (_, tangent): (BodyTracer, BodyTracer)| {
            let x = ValueProjection::<ArrayType>::into_projected(x)?;
            let tangent = ValueProjection::<ArrayType>::into_projected(tangent)?;
            Ok((x.sin()?.into_value(), (x.cos()? * tangent).into_value()))
        });
        let replicated = Sharding::replicated(manual_mesh(), 0);
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| {
                            let x = x.into_value();
                            let reference = x.reference_new().unwrap();
                            ValueProjection::<ArrayType>::into_projected(function.call((reference, x)).unwrap())
                                .unwrap()
                        },
                        inputs[0].clone(),
                        manual_mesh(),
                        replicated.clone(),
                        replicated.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_scalar_type()],
            Vec::new(),
        );
        let scalar = f32_scalar_type().with_sharding(replicated.clone()).unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            formatdoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:{scalar}, %3:{scalar} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}, {replicated}],
                    out_shardings=[{replicated}, {replicated}],
                    manual_axes=['x'],
                    global_input_types=[{scalar}, {scalar}],
                    global_output_types=[{scalar}, {scalar}],
                ] %0 %1 [
                    body={{
                        lambda %0:{scalar}, %1:{scalar} .
                        let %2:{scalar} = sin %0
                            %3:{scalar} = cos %0
                            %4:{scalar} = mul %3 %1
                        in (%2, %4)
                    }},
                ]
                in (%2, %3)"},
        );
        let linearization = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            formatdoc! {"
                lambda %0:{scalar}, %1:{scalar} .
                let %2:{scalar} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}, {replicated}],
                    out_shardings=[{replicated}],
                    manual_axes=['x'],
                    global_input_types=[{scalar}, {scalar}],
                    global_output_types=[{scalar}],
                ] %0 %1 [
                    body={{
                        lambda %0:{scalar}, %1:{scalar} .
                        let %2:{scalar} = mul %1 %0
                        in (%2)
                    }},
                ]
                    %3:f32[] = broadcast [output_type=f32[], output_axes=[]] %2
                in (%3)"},
        );
    }

    #[test]
    fn test_shard_map_linearization_snapshots_body_allocated_references_of_derived_custom_function_calls() {
        // A `custom_function` with a batching rule linearizes through a derived `pushforward` call, which takes the
        // body-allocated plumbing reference `r` as a residual. The primal map snapshots the final state of `r` into a
        // residual edge tiled along `x`, and the tangent map allocates its own `r` from that edge. Forward mode binds
        // one fused map, whose derived `jvp` call takes the primal `r` directly (the dead allocation of the tangent of
        // `r` is removed before binding). Both match the same function outside `shard_map` on the reference backend.
        type BodyTracer = Tracer<TestContext>;
        let function = custom_function(custom_sine).with_non_differentiated_count(1).with_batching(
            |_: BatchingLevelExtent<BodyTracer>,
             (r, x): (BodyTracer, BodyTracer),
             (_, axis): (BatchAxis, BatchAxis)| { Ok((custom_sine((r, x))?, axis)) },
        );
        let call = |r, x| function.call((r, x)).unwrap();
        let program = body_allocated_reference_program(f32_vector_type(4), Some(sharded_along_x()), call);
        let unmapped = body_allocated_reference_program(f32_vector_type(4), None, call);
        let sharding = "{mesh<['x'=2:manual]>, [{'x'}]}";
        let global = "f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]";
        let local = "f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        let primal = "custom_function [name=\"custom_function\", non_differentiated_count=1]";
        let pushforward = "custom_function [name=\"pushforward(custom_function)\", non_differentiated_count=1]";
        let jvp = "custom_function [name=\"jvp(custom_function)\", non_differentiated_count=1]";
        assert_eq!(
            program.jvp().unwrap().to_string(),
            formatdoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:{global}, %3:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}],
                    out_shardings=[{sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}],
                    global_output_types=[{global}, {global}],
                ] %0 %1 [
                    body={{
                        lambda %0:{local}, %1:{local} .
                        let %2:ref<{local}> = reference_new %0
                            %3:{local}, %4:{local} = {jvp} %2 %0 %1 [
                                primal={{
                                    lambda %0:ref<{local}>, %1:{local}, %2:{local} .
                                    let %3:{local} = sin %1
                                        %4:{local} = cos %1
                                        %5:{local} = mul %4 %2
                                    in (%3, %5)
                                }},
                            ]
                        in (%3, %4)
                    }},
                ]
                in (%2, %3)"},
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            formatdoc! {"
                lambda %0:f32[4] .
                let %1:{global}, %2:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}],
                    out_shardings=[{sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}],
                    global_output_types=[{global}, {global}],
                ] %0 [
                    body={{
                        lambda %0:{local} .
                        let %1:ref<{local}> = reference_new %0
                            %2:{local} = reference_read %1
                            %3:{local} = {primal} %1 %0 [
                                primal={{
                                    lambda %0:ref<{local}>, %1:{local} .
                                    let %2:{local} = sin %1
                                    in (%2)
                                }},
                            ]
                        in (%3, %2)
                    }},
                ]
                in (%1, %2, %0)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            formatdoc! {"
                lambda %0:f32[4], %1:{global}, %2:f32[4] .
                let %3:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}, {sharding}],
                    out_shardings=[{sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}, {global}],
                    global_output_types=[{global}],
                ] %0 %1 %2 [
                    body={{
                        lambda %0:{local}, %1:{local}, %2:{local} .
                        let %3:ref<{local}> = reference_new %1
                            %4:{local} = {pushforward} %3 %2 %0 [
                                primal={{
                                    lambda %0:ref<{local}>, %1:{local}, %2:{local} .
                                    let %3:{local} = cos %1
                                        %4:{local} = mul %3 %2
                                    in (%4)
                                }},
                            ]
                        in (%4)
                    }},
                ]
                in (%3)"},
        );
        let x = TestValue::Array(Array::vector(vec![0.25f32, 0.5, 1.0, 2.0]).unwrap());
        let tangent = TestValue::Array(Array::vector(vec![1.0f32, -2.0, 3.0, 0.5]).unwrap());
        assert_eq!(
            evaluate_linearization(&program, vec![x.clone()], vec![tangent.clone()]),
            evaluate_linearization(&unmapped, vec![x.clone()], vec![tangent.clone()]),
        );
        let jvp = |program: &TestProgram| {
            let outputs = program.jvp().unwrap().interpret(vec![x.clone(), tangent.clone()]).unwrap();
            outputs
                .into_iter()
                .map(|output| match output {
                    ArrayIrValue::Array(array) => array.to_f64s(),
                    _ => panic!("expected an array value"),
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(jvp(&program), jvp(&unmapped));
    }

    #[test]
    fn test_shard_map_reverse_mode_differentiation_snapshots_body_allocated_residuals_of_custom_vjp_rules() {
        // A reverse-mode rule may forward the body-allocated plumbing reference `r` as one of its residuals, alone or
        // next to a JVP rule (which reverse mode does not use). The primal map snapshots the final state of `r`, which
        // is replicated, into an untiled residual edge, and the tangent map allocates its own `r` from that edge before
        // the transposed backward rule receives it. Both match the same function outside `shard_map`.
        type BodyTracer = Tracer<TestContext>;
        let forward = |(r, x): (BodyTracer, BodyTracer)| {
            let x = ValueProjection::<ArrayType>::into_projected(x)?;
            Ok((x.sin()?.into_value(), (r, x.cos()?.into_value())))
        };
        let backward = |(r, cosine): (BodyTracer, BodyTracer), cotangent: BodyTracer| {
            let cosine = ValueProjection::<ArrayType>::into_projected(cosine)?;
            let cotangent = ValueProjection::<ArrayType>::into_projected(cotangent)?;
            Ok((r, (cosine * cotangent).into_value()))
        };
        let reverse = custom_function(custom_sine).with_non_differentiated_count(1).with_vjp(forward, backward);
        let combined = custom_function(custom_sine)
            .with_non_differentiated_count(1)
            .with_jvp(|(_, x): (BodyTracer, BodyTracer), (_, tangent): (BodyTracer, BodyTracer)| {
                let x = ValueProjection::<ArrayType>::into_projected(x)?;
                let tangent = ValueProjection::<ArrayType>::into_projected(tangent)?;
                Ok((x.sin()?.into_value(), (x.cos()? * tangent).into_value()))
            })
            .with_vjp(forward, backward);
        let replicated = Sharding::replicated(manual_mesh(), 0);
        let x = TestValue::Array(Array::scalar(0.5f32).unwrap());
        let expected = vec![vec![f64::from(f32::sin(0.5))], vec![f64::from(f32::cos(0.5))]];
        let scalar = f32_scalar_type().with_sharding(replicated.clone()).unwrap();
        let check = |call: &dyn Fn(BodyTracer, BodyTracer) -> BodyTracer| {
            let program = body_allocated_reference_program(f32_scalar_type(), Some(replicated.clone()), call);
            let unmapped = body_allocated_reference_program(f32_scalar_type(), None, call);
            let linearization = program
                .entry_region_ref()
                .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
                .unwrap();
            assert_eq!(
                linearization.primal().to_string(),
                formatdoc! {"
                    lambda %0:f32[] .
                    let %1:{scalar}, %2:{scalar}, %3:{scalar} = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{replicated}],
                        out_shardings=[{replicated}, {replicated}, {replicated}],
                        manual_axes=['x'],
                        global_input_types=[{scalar}],
                        global_output_types=[{scalar}, {scalar}, {scalar}],
                    ] %0 [
                        body={{
                            lambda %0:{scalar} .
                            let %1:ref<{scalar}> = reference_new %0
                                %2:{scalar} = reference_read %1
                                %3:{scalar} = sin %0
                                %4:{scalar} = cos %0
                            in (%3, %2, %4)
                        }},
                    ]
                    in (%1, %2, %3)"},
            );
            assert_eq!(
                linearization.pullback().unwrap().to_string(),
                formatdoc! {"
                    lambda %0:{scalar}, %1:{scalar}, %2:{scalar} .
                    let %3:{scalar} = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{replicated}, {replicated}, {replicated}],
                        out_shardings=[{replicated}],
                        manual_axes=['x'],
                        global_input_types=[{scalar}, {scalar}, {scalar}],
                        global_output_types=[{scalar}],
                    ] %0 %1 %2 [
                        body={{
                            lambda %0:{scalar}, %1:{scalar}, %2:{scalar} .
                            let %3:{scalar} = mul %2 %0
                            in (%3)
                        }},
                    ]
                        %4:f32[] = broadcast [output_type=f32[], output_axes=[]] %3
                    in (%4)"},
            );
            assert_eq!(evaluate_pullback(&program, x.clone(), 1), expected);
            assert_eq!(evaluate_pullback(&unmapped, x.clone(), 1), expected);
        };
        check(&|r, x| reverse.call((r, x)).unwrap());
        check(&|r, x| combined.call((r, x)).unwrap());
    }

    #[test]
    fn test_shard_map_reverse_mode_differentiation_observes_the_final_state_of_body_allocated_residuals() {
        // The backward rule reads the plumbing reference `r` that its forward rule saved, and the body writes `x * x`
        // into `r` after the call. Outside `shard_map`, the pullback runs after the whole primal program, so it reads
        // `x * x`, and the snapshot that crosses from the primal map to the tangent map holds that same final state,
        // because the primal body reads it after its last instruction. The gradient is therefore `cos(x) * x * x` in
        // both cases.
        type BodyTracer = Tracer<TestContext>;
        let function = custom_function(custom_sine).with_non_differentiated_count(1).with_vjp(
            |(r, x): (BodyTracer, BodyTracer)| {
                let x = ValueProjection::<ArrayType>::into_projected(x)?;
                Ok((x.sin()?.into_value(), (r, x.cos()?.into_value())))
            },
            |(r, cosine): (BodyTracer, BodyTracer), cotangent: BodyTracer| {
                let state = ValueProjection::<ArrayType>::into_projected(r.read()?)?;
                let cosine = ValueProjection::<ArrayType>::into_projected(cosine)?;
                let cotangent = ValueProjection::<ArrayType>::into_projected(cotangent)?;
                Ok((r, (cosine * cotangent * state).into_value()))
            },
        );
        let call = |r: BodyTracer, x: BodyTracer| {
            let output = function.call((r.clone(), x.clone())).unwrap();
            let x = ValueProjection::<ArrayType>::into_projected(x).unwrap();
            r.write(&(x.clone() * x).into_value()).unwrap();
            output
        };
        let program = body_allocated_reference_program(f32_vector_type(4), Some(sharded_along_x()), call);
        let unmapped = body_allocated_reference_program(f32_vector_type(4), None, call);
        let linearization = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        let sharding = "{mesh<['x'=2:manual]>, [{'x'}]}";
        let global = "f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]";
        let local = "f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        assert_eq!(
            linearization.primal().to_string(),
            formatdoc! {"
                lambda %0:f32[4] .
                let %1:{global}, %2:{global}, %3:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}],
                    out_shardings=[{sharding}, {sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}],
                    global_output_types=[{global}, {global}, {global}],
                ] %0 [
                    body={{
                        lambda %0:{local} .
                        let %1:ref<{local}> = reference_new %0
                            %2:{local} = mul %0 %0
                            () = reference_write %1 %2
                            %3:{local} = reference_read %1
                            %4:{local} = sin %0
                            %5:{local} = cos %0
                        in (%4, %3, %5)
                    }},
                ]
                in (%1, %0, %2, %3)"},
        );
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            formatdoc! {"
                lambda %0:{global}, %1:f32[4], %2:{global}, %3:{global} .
                let %4:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}, {sharding}, {sharding}],
                    out_shardings=[{sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}, {global}, {global}],
                    global_output_types=[{global}],
                ] %0 %1 %2 %3 [
                    body={{
                        lambda %0:{local}, %1:{local}, %2:{local}, %3:{local} .
                        let %4:ref<{local}> = reference_new %2
                            %5:{local} = reference_read %4
                            %6:{local} = mul %3 %0
                            %7:{local} = mul %6 %5
                        in (%7)
                    }},
                ]
                    %5:f32[4] = broadcast [output_type=f32[4], output_axes=[0]] %4
                in (%5)"},
        );
        let values = [0.25f32, 0.5, 1.0, 2.0];
        let x = TestValue::Array(Array::vector(values.to_vec()).unwrap());
        let expected = vec![
            values.iter().map(|&value| f64::from(f32::sin(value))).collect::<Vec<_>>(),
            values.iter().map(|&value| f64::from(f32::cos(value) * value * value)).collect(),
        ];
        assert_eq!(evaluate_pullback(&program, x.clone(), 1), expected);
        assert_eq!(evaluate_pullback(&unmapped, x, 1), expected);
    }

    #[test]
    fn test_shard_map_reverse_mode_differentiation_restarts_every_pullback_from_the_final_primal_state() {
        // The backward rule reads the plumbing reference `r` that its forward rule saved, doubles its state, and scales
        // the cotangent by the state that it read. Every application of the pullback runs the tangent map, which
        // allocates its own `r` from the snapshot of the final primal state `x`, so every application returns
        // `cos(x) * x`. Outside `shard_map`, the backward rule would update the primal allocation itself instead, so
        // that a later application would observe the doubled state. The body also reshapes its `f32[1]` shard to the
        // varying scalar `x`, whose snapshot edge is promoted to `f32[1]` so that it can be tiled along `x`.
        type BodyTracer = Tracer<TestContext>;
        let function = custom_function(custom_sine).with_non_differentiated_count(1).with_vjp(
            |(r, x): (BodyTracer, BodyTracer)| {
                let x = ValueProjection::<ArrayType>::into_projected(x)?;
                Ok((x.sin()?.into_value(), (r, x.cos()?.into_value())))
            },
            |(r, cosine): (BodyTracer, BodyTracer), cotangent: BodyTracer| {
                let state = ValueProjection::<ArrayType>::into_projected(r.read()?)?;
                r.write(&(state.clone() + state.clone()).into_value())?;
                let cosine = ValueProjection::<ArrayType>::into_projected(cosine)?;
                let cotangent = ValueProjection::<ArrayType>::into_projected(cotangent)?;
                Ok((r, (cosine * cotangent * state).into_value()))
            },
        );
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| {
                            let x = x.reshape(Shape::new(Vec::new())).unwrap().into_value();
                            let reference = x.reference_new().unwrap();
                            let output = function.call((reference, x)).unwrap();
                            ValueProjection::<ArrayType>::into_projected(output).unwrap().reshape([1]).unwrap()
                        },
                        inputs[0].clone(),
                        manual_mesh(),
                        sharded_along_x(),
                        sharded_along_x(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_vector_type(2)],
            Vec::new(),
        );
        let linearization = program
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        let sharding = "{mesh<['x'=2:manual]>, [{'x'}]}";
        let global = "f32[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}]";
        let promoted_sharding = "{mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}";
        let promoted = format!("f32[1][sharding={promoted_sharding}]");
        let scalar_sharding = "{mesh<['x'=2:manual]>, [], varying_manual={'x'}}";
        let scalar = format!("f32[][sharding={scalar_sharding}]");
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            formatdoc! {"
                lambda %0:{global}, %1:{global}, %2:{global} .
                let %3:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}, {sharding}],
                    out_shardings=[{sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global}, {global}, {global}],
                    global_output_types=[{global}],
                ] %0 %1 %2 [
                    body={{
                        lambda %0:{promoted}, %1:{promoted}, %2:{promoted} .
                        let %3:{scalar} = reshape [shape=[], output_sharding={scalar_sharding}] %0
                            %4:{scalar} = reshape [shape=[], output_sharding={scalar_sharding}] %1
                            %5:ref<{scalar}> = reference_new %4
                            %6:{scalar} = reshape [shape=[], output_sharding={scalar_sharding}] %2
                            %7:{scalar} = reference_read %5
                            %8:{scalar} = add %7 %7
                            () = reference_write %5 %8
                            %9:{scalar} = mul %6 %3
                            %10:{scalar} = mul %9 %7
                            %11:{promoted} = reshape [shape=[1], output_sharding={promoted_sharding}] %10
                        in (%11)
                    }},
                ]
                    %4:f32[2] = broadcast [output_type=f32[2], output_axes=[0]] %3
                in (%4)"},
        );
        let values = [0.5f32, 1.5];
        let x = TestValue::Array(Array::vector(values.to_vec()).unwrap());
        let gradient = values.iter().map(|&value| f64::from(f32::cos(value) * value)).collect::<Vec<_>>();
        assert_eq!(
            evaluate_pullback(&program, x, 2),
            vec![values.iter().map(|&value| f64::from(f32::sin(value))).collect(), gradient.clone(), gradient],
        );
    }

    #[test]
    fn test_shard_map_differentiation_saves_reference_reads_as_array_residuals() {
        // End to end, linearizing the body `(r, x) -> read(r) * x` with only `x` active saves the value read from `r`
        // rather than `r` itself, so that value crosses as an ordinary residual edge, tiled along `x` (so that its
        // local shard is the value itself), and the tangent `shard_map` does not take the reference. The reference
        // input keeps the two-map form even in forward mode.
        let (operation, body) = reference_product_shard_map();
        let program = shard_map_program(operation, body).unwrap();
        let sharding = "{mesh<['x'=2:manual]>, [{'x'}]}";
        let global = "f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]";
        let local = "f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        assert_eq!(
            program.entry_region_ref().jvp(&[1]).unwrap().to_string(),
            formatdoc! {"
                lambda %0:ref<{global}>, %1:{global}, %2:{global} .
                let %3:{global}, %4:{global} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}, {sharding}],
                    out_shardings=[{sharding}, {sharding}],
                    manual_axes=['x'],
                    global_input_types=[ref<{global}>, {global}],
                    global_output_types=[{global}, {global}],
                ] %0 %1 [
                    body={{
                        lambda %0:ref<{local}>, %1:{local} .
                        let %2:{local} = reference_read %0
                            %3:{local} = mul %2 %1
                        in (%3, %2)
                    }},
                ]
                    %5:{global} = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{sharding}, {sharding}],
                        out_shardings=[{sharding}],
                        manual_axes=['x'],
                        global_input_types=[{global}, {global}],
                        global_output_types=[{global}],
                    ] %2 %4 [
                        body={{
                            lambda %0:{local}, %1:{local} .
                            let %2:{local} = mul %1 %0
                            in (%2)
                        }},
                    ]
                in (%3, %5)"},
        );
    }

    #[test]
    fn test_shard_map_differentiation_skips_tangent_maps_without_live_output_tangents() {
        // An active input whose tangent reaches no output (here, through `stop_gradient`) leaves no live output
        // tangent, and the linear tangent body has no effects, so no tangent `shard_map` is staged: the source
        // `shard_map` is the primal one, and the output tangent is a structural zero.
        let replicated = Sharding::replicated(manual_mesh(), 0);
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| x.stop_gradient().unwrap(),
                        inputs[0].clone(),
                        manual_mesh(),
                        replicated.clone(),
                        replicated.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_scalar_type()],
            Vec::new(),
        );
        let scalar = f32_scalar_type().with_sharding(replicated.clone()).unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            formatdoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:{scalar} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}],
                    out_shardings=[{replicated}],
                    manual_axes=['x'],
                    global_input_types=[{scalar}],
                    global_output_types=[{scalar}],
                ] %0 [
                    body={{
                        lambda %0:{scalar} .
                        let %1:{scalar} = stop_gradient %0
                        in (%1)
                    }},
                ]
                    %3:{scalar} = zero [type={scalar}]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_shard_map_differentiation_rejects_dynamic_residuals() {
        // Forward-mode differentiation splits a body into primal and tangent maps, so it rejects a body whose residuals
        // include a dynamically shaped intermediate, which cannot cross the static boundary between them.
        let program = dynamic_residual_shard_map_program();
        let expected = ShardMapError::DynamicResidualNotSupported { residual_index: 1, dimension: 0 };
        assert!(matches!(
            program.jvp(),
            Err(DifferentiationError::Program(error)) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
        assert!(matches!(
            program.entry_region_ref().jvp(&[1]),
            Err(DifferentiationError::Program(error)) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
    }

    #[test]
    fn test_shard_map_differentiation_keeps_bodies_with_nested_dynamic_values_unfused() {
        // Forward mode fuses no body that has a dynamically shaped value, also when that value is in a nested region.
        // Here, the body `(a, x) -> condition(a > 0, sum(sin(broadcast(a, n)) * x), x)` with `n` derived from `a` has
        // only statically shaped values outside its `condition`, but the residual `cos(broadcast(a, n))` of the true
        // branch is dynamically shaped. A fused map would compute the output and its tangent together, and splitting it
        // into its known and unknown halves (e.g., when a `while` or `rematerialize` rule partitions it) would have to
        // keep the `condition` on the unknown side, leaving the primal output unknown. Forward mode therefore uses the
        // two-map form, which rejects the residual, exactly as linearization does.
        let scalar_type = f32_scalar_type();
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let local_type = ArrayIrType::Array(scalar_type.clone().with_sharding(replicated.clone()).unwrap());
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(5)).unwrap());
        let dimension_type = ArrayIrType::Dimension(DimensionType::from(extent.clone()));
        let true_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let a = builder.add_input(local_type.clone());
            let x = builder.add_input(local_type.clone());
            let dimension = builder.add_input(dimension_type.clone());
            let broadcast = DynamicBroadcastOperation::new(Vec::new());
            let broadcast = builder.add_instruction(broadcast, Vec::new(), vec![a, dimension], None).unwrap()[0];
            let sine = ArrayOperation::Sin(SinOperation::new());
            let sine = builder.add_instruction(sine, Vec::new(), vec![broadcast], None).unwrap()[0];
            let product = ArrayOperation::Mul(MulOperation::new());
            let product = builder.add_instruction(product, Vec::new(), vec![sine, x], None).unwrap()[0];
            let sum = ArrayOperation::Reduce(ReduceOperation::new(vec![0], ReductionKind::Sum));
            let sum = builder.add_instruction(sum, Vec::new(), vec![product], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            builder.add_input(local_type.clone());
            let x = builder.add_input(local_type.clone());
            builder.add_input(dimension_type.clone());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![x], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let a = builder.add_input(local_type.clone());
            let x = builder.add_input(local_type);
            let convert = ArrayOperation::ConvertElementType(ConvertElementTypeOperation::new(DataType::I64, false));
            let count = builder.add_instruction(convert, Vec::new(), vec![a], None).unwrap()[0];
            let dimension = DimensionFromScalarOperation::new(extent);
            let dimension = builder.add_instruction(dimension, Vec::new(), vec![count], None).unwrap()[0];
            let zero = ArrayOperation::ZeroLike(ZeroLikeOperation::new());
            let zero = builder.add_instruction(zero, Vec::new(), vec![a], None).unwrap()[0];
            let compare = ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::GreaterThan));
            let predicate = builder.add_instruction(compare, Vec::new(), vec![a, zero], None).unwrap()[0];
            let branches = vec![builder.import_program(true_branch), builder.import_program(false_branch)];
            let condition = ConditionOperation::<ArrayIrType>::new();
            let inputs = vec![predicate, a, x, dimension];
            let output = builder.add_instruction(condition, branches, inputs, None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        assert!(!body.atoms().iter().any(|atom| {
            matches!(atom.r#type().as_ref(), ArrayIrType::Array(r#type) if r#type.static_shape().is_none())
        }));
        let shard_map =
            ShardMap::new(mesh, vec![replicated.clone(), replicated.clone()], vec![replicated], Vec::new()).unwrap();
        let operation =
            ShardMapOperation::from_program(&body, vec![scalar_type.clone().into(), scalar_type.into()], shard_map);
        let program = shard_map_program(operation.unwrap(), body).unwrap();
        let expected = ShardMapError::DynamicResidualNotSupported { residual_index: 2, dimension: 0 };
        assert!(matches!(
            program.jvp(),
            Err(DifferentiationError::Program(error)) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
        assert!(matches!(
            program.linearize(),
            Err(DifferentiationError::Program(error)) if error.downcast_custom::<ShardMapError>() == Some(&expected),
        ));
    }

    #[test]
    fn test_shard_map_differentiation_threads_tangent_references() {
        // Forward mode threads a tangent reference beside an active reference input under the primal's input sharding,
        // and a plumbing reference input (one without a tangent reference) has no tangent boundary slot.
        let sharded = sharded_along_x();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded.clone(), true);
        let program = shard_map_program(operation, body).unwrap();

        // Every input active: the fused program consumes `[r, x, ṙ, ẋ]`, the primal shard map keeps its boundary
        // and gains the residual edges, and the tangent shard map consumes the tangent reference under the primal's
        // input sharding followed by the tangent value and the residuals.
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:ref<f32[4]>, %1:f32[4], %2:ref<f32[4]>, %3:f32[4] .
                let %4:f32[4] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[ref<f32[4]>, f32[4]],
                    global_output_types=[f32[4]],
                ] %0 %1 [
                    body={
                        lambda %0:ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]>, \
                    %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let () = reference_add_update %0 %1
                            %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reference_read %0
                        in (%2)
                    },
                ]
                    %5:f32[4] = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                        out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                        manual_axes=['x'],
                        global_input_types=[ref<f32[4]>, f32[4]],
                        global_output_types=[f32[4]],
                    ] %2 %3 [
                        body={
                            lambda %0:ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]>, \
                    %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                            let () = reference_add_update %0 %1
                                %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                    reference_read %0
                            in (%2)
                        },
                    ]
                in (%4, %5)"},
        );

        // A plumbing reference (no tangent reference supplied) is inactive: the tangent shard map consumes only the
        // tangent value followed by the residuals. The body reads the reference without mutating it, since writing an
        // active tangent into a plumbing reference is rejected by the reference rules themselves.
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded, false);
        let program = shard_map_program(operation, body).unwrap();
        assert_eq!(
            program.entry_region_ref().jvp(&[1]).unwrap().to_string(),
            indoc! {"
                lambda %0:ref<f32[4]>, %1:f32[4], %2:f32[4] .
                let %3:f32[4] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[ref<f32[4]>, f32[4]],
                    global_output_types=[f32[4]],
                ] %0 %1 [
                    body={
                        lambda %0:ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]>, \
                    %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reference_read %0
                            %3:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = add %2 %1
                        in (%3)
                    },
                ]
                    %4:f32[4] = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                        out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                        manual_axes=['x'],
                        global_input_types=[f32[4]],
                        global_output_types=[f32[4]],
                    ] %2 [
                        body={
                            lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                            in (%0)
                        },
                    ]
                in (%3, %4)"},
        );
    }

    #[test]
    fn test_shard_map_differentiation_of_reads_of_replicated_references_at_varying_indices() {
        // Every device reads the row of a table that is replicated along `x` at its own coordinate along `x`, so the
        // map gathers the first two rows. Forward mode reads the tangent reference at the same device-varying index.
        // Reverse mode cannot add the varying cotangent of the read directly into the replicated cotangent reference of
        // the table at the varying index, which would update a different row on every device of a referent that is
        // typed as identical across them. The read implicitly varies the table along `x` before selecting a row, so its
        // transpose scatters the cotangent into a local zero buffer that varies along `x`, sums the buffer over `x`
        // with `parallel_reduce`, and adds that invariant sum into the replicated cotangent reference. The gradient of
        // the table thus collects the contributions of all devices, and its unread row receives none.
        let mesh = manual_mesh();
        let table_type = ArrayType::new_static(DataType::F32, [3, 1]);
        let shard_map =
            ShardMap::new(mesh.clone(), vec![Sharding::replicated(mesh, 2)], vec![sharded_along_x()], Vec::new())
                .unwrap();
        let body = varying_index_read_body(shard_map.local_input_type(0, &table_type).unwrap());
        let operation =
            ShardMapOperation::from_program(&body, vec![ReferenceType::new(table_type.clone()).into()], shard_map)
                .unwrap();
        let program = |mapped: bool| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(table_type.clone().into());
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
            let output = if mapped {
                let body = builder.import_program(body.clone());
                let operation = TestOperation::ShardMap(Box::new(operation.clone()));
                builder.add_instruction(operation, vec![body], vec![reference], None).unwrap()[0]
            } else {
                // The unmapped program reads the same two rows through a static slice and flattens them.
                let axes = vec![ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(0, 1, 1)];
                let rows = ArrayReferenceTransform::Slice { axes };
                let read = ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new()
                    .with_transforms(vec![rows]);
                let rows = builder.add_instruction(read, Vec::new(), vec![reference], None).unwrap()[0];
                let reshape = ArrayOperation::Reshape(ReshapeOperation::new(f32_vector_type(2).shape().clone()));
                builder.add_instruction(reshape, Vec::new(), vec![rows], None).unwrap()[0]
            };
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let (mapped, unmapped) = (program(true), program(false));
        let input = TestValue::Array(Array::from_elements(table_type, &[10.0f32, 20.0, 30.0]).unwrap());
        assert_eq!(evaluate_pushforward(&mapped, input.clone()), vec![vec![10.0, 20.0], vec![1.0, 1.0]]);
        assert_eq!(evaluate_pushforward(&mapped, input.clone()), evaluate_pushforward(&unmapped, input.clone()));

        let linearization = mapped
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        let replicated = "f32[3, 1][sharding={mesh<['x'=2:manual]>, [{}, {}]}]";
        let varying = "f32[3, 1][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}]";
        let seed = "f32[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}]";
        let index = "u64[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}]";
        let local_seed = "f32[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        let local_index = "u64[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            formatdoc! {"
                lambda %0:{seed}, %1:{index} .
                let %2:f32[3, 1] = zero [type=f32[3, 1]]
                    %3:ref<f32[3, 1]> = reference_new %2
                    %4:{replicated} = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{{mesh<['x'=2:manual]>, [{{'x'}}]}}, {{mesh<['x'=2:manual]>, [{{'x'}}]}}],
                        out_shardings=[{{mesh<['x'=2:manual]>, [{{}}, {{}}]}}],
                        manual_axes=['x'],
                        global_input_types=[{seed}, {index}],
                        global_output_types=[{replicated}],
                    ] %0 %1 [
                        body={{
                            lambda %0:{local_seed}, %1:{local_index} .
                            let %2:{varying} = zero [type={varying}]
                                %3:ref<{varying}> = reference_new %2
                                %4:u64[][sharding={{mesh<['x'=2:manual]>, [], varying_manual={{'x'}}}}] = reshape \
                [shape=[], output_sharding={{mesh<['x'=2:manual]>, [], varying_manual={{'x'}}}}] %1
                                () = reference_add_update [transforms=[index(axis=0, index=dynamic)]] %3 %0 %4
                                %5:{varying} = reference_freeze %3
                                %6:{replicated} = zero [type={replicated}]
                                %7:ref<{replicated}> = reference_new %6
                                %8:{replicated} = parallel_reduce [kind=sum, axis_name=\"x\", mesh=['x'=2:manual]] %5
                                () = reference_add_update %7 %8
                                %9:{replicated} = reference_freeze %7
                            in (%9)
                        }},
                    ]
                    %5:f32[3, 1] = broadcast [output_type=f32[3, 1], output_axes=[0, 1]] %4
                    () = reference_add_update %3 %5
                    %6:f32[3, 1] = reference_freeze %3
                in (%6)"},
        );

        // Repeated applications of the pullback restart from zero, and a nonuniform seed lands in the rows read.
        let expected = vec![vec![10.0, 20.0], vec![1.0, 1.0, 0.0], vec![1.0, 1.0, 0.0]];
        assert_eq!(evaluate_pullback(&mapped, input.clone(), 2), expected);
        assert_eq!(evaluate_pullback(&unmapped, input.clone(), 2), expected);
        let pullback = |program: &TestProgram| {
            let linearization = program
                .entry_region_ref()
                .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
                .unwrap();
            let mut outputs = linearization.primal().interpret(vec![input.clone()]).unwrap();
            let residuals = outputs.split_off(1);
            let seed_type = outputs[0].r#type().into_owned();
            let ArrayIrType::Array(seed_type) = seed_type else {
                panic!("expected an array output");
            };
            let mut inputs = vec![TestValue::Array(Array::from_elements(seed_type, &[3.0f32, 7.0]).unwrap())];
            inputs.extend(residuals);
            arrays_f64(linearization.pullback().unwrap().interpret(inputs).unwrap())
        };
        assert_eq!(pullback(&mapped), vec![vec![3.0, 7.0, 0.0]]);
        assert_eq!(pullback(&mapped), pullback(&unmapped));
    }

    #[test]
    fn test_shard_map_differentiation_of_writes_into_replicated_references() {
        // Overwriting a replicated reference with a constant kills the dependence of its state on the input, so the
        // input receives a zero cotangent: the cotangent reference of a mutated replicated reference crosses the
        // transposed map by identity (it is forwarded rather than frozen), so the transposed write zeroes the caller's
        // destination itself (an accumulator that starts from zero would instead return the incoming cotangent
        // unchanged).
        let write_one = |builder: &mut ProgramBuilder<TestValue, TestOperation>, reference: AtomId| {
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let one = builder
                .add_instruction(ArrayOperation::OneLike(OneLikeOperation::new()), Vec::new(), vec![state], None)
                .unwrap()[0];
            builder
                .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, one], None)
                .unwrap();
        };
        let input = TestValue::Array(Array::from_elements(f32_vector_type(2), &[3.0f32, 5.0]).unwrap());
        let mapped = replicated_reference_update_program(true, write_one);
        let unmapped = replicated_reference_update_program(false, write_one);
        let replicated = "f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let linearization = mapped
            .entry_region_ref()
            .linearize_shared_for_rule(&[0], DifferentiationRule::JvpForTranspose)
            .unwrap();
        assert_eq!(
            linearization.pullback().unwrap().to_string(),
            formatdoc! {"
                lambda %0:f32[2], %1:{replicated} .
                let %2:f32[2] = zero [type=f32[2]]
                    %3:ref<f32[2]> = reference_new %2
                    () = reference_add_update %3 %0
                    %4:ref<f32[2]> = shard_map [
                        mesh=['x'=2:manual],
                        in_shardings=[{{mesh<['x'=2:manual]>, [{{}}]}}, {{mesh<['x'=2:manual]>, [{{}}]}}],
                        out_shardings=[{{mesh<['x'=2:manual]>, [{{}}]}}],
                        manual_axes=['x'],
                        global_input_types=[ref<{replicated}>, {replicated}],
                        global_output_types=[ref<{replicated}>],
                        output_forwarding=[0],
                    ] %3 %1 [
                        body={{
                            lambda %0:ref<{replicated}>, %1:{replicated} .
                            let %2:{replicated} = zero [type={replicated}]
                                %3:{replicated} = reference_swap %0 %2
                            in (%0)
                        }},
                    ]
                    %5:f32[2] = reference_freeze %3
                in (%5)"},
        );
        assert_eq!(evaluate_pullback(&mapped, input.clone(), 2), vec![vec![1.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]]);
        assert_eq!(evaluate_pullback(&mapped, input.clone(), 2), evaluate_pullback(&unmapped, input.clone(), 2));
        assert_eq!(evaluate_pushforward(&mapped, input.clone()), vec![vec![1.0, 1.0], vec![0.0, 0.0]]);
        assert_eq!(evaluate_pushforward(&mapped, input.clone()), evaluate_pushforward(&unmapped, input));

        // A read-modify-write that doubles the replicated reference doubles the cotangent of its incoming state.
        let double = |builder: &mut ProgramBuilder<TestValue, TestOperation>, reference: AtomId| {
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let doubled = builder
                .add_instruction(ArrayOperation::Add(AddOperation::new()), Vec::new(), vec![state, state], None)
                .unwrap()[0];
            builder
                .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, doubled], None)
                .unwrap();
        };
        let input = TestValue::Array(Array::from_elements(f32_vector_type(2), &[3.0f32, 5.0]).unwrap());
        let mapped = replicated_reference_update_program(true, double);
        let unmapped = replicated_reference_update_program(false, double);
        assert_eq!(evaluate_pullback(&mapped, input.clone(), 2), vec![vec![6.0, 10.0], vec![2.0, 2.0], vec![2.0, 2.0]]);
        assert_eq!(evaluate_pullback(&mapped, input.clone(), 2), evaluate_pullback(&unmapped, input.clone(), 2));
        assert_eq!(evaluate_pushforward(&mapped, input.clone()), vec![vec![6.0, 10.0], vec![2.0, 2.0]]);
        assert_eq!(evaluate_pushforward(&mapped, input.clone()), evaluate_pushforward(&unmapped, input));
    }

    #[test]
    fn test_shard_map_differentiation_of_bodies_without_inputs() {
        // A `shard_map` without inputs has no input tangents, so forward-mode differentiation keeps its primal
        // boundary and gives its output a zero tangent without staging a tangent `shard_map`. The body lifts a constant
        // through its context, which is invariant and is therefore marked as varying along the tiled axis `x`.
        let context = TestContext::new();
        let output = shard_map_in_context(
            &context,
            |context: &ShardMapContext<TestContext>, ()| {
                context.lift(Array::scalar(3.0f32).unwrap()).unwrap().reshape([1]).unwrap()
            },
            (),
            manual_mesh(),
            (),
            sharded_along_x(),
            Vec::new(),
        )
        .unwrap();
        let builder = context.builder().borrow().clone();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![output.value().atom_id().unwrap()],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda  .
                let %0:f32[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[],
                    global_output_types=[f32[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ] [
                    body={
                        lambda  .
                        let %0:f32[] = const 3.0
                            %1:f32[1] = reshape [shape=[1]] %0
                            %2:f32[1][sharding={mesh<['x'=2:manual]>, [{}]}] = broadcast \
                                [output_type=f32[1][sharding={mesh<['x'=2:manual]>, [{}]}], output_axes=[0]] %1
                            %3:f32[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary \
                                [axis_name=\"x\"] %2
                        in (%3)
                    },
                ]
                    %1:f32[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = zero \
                        [type=f32[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}]]
                in (%0, %1)"
            },
        );
    }

    #[test]
    fn test_shard_map_transposition() {
        // Reverse mode through a `shard_map` reaches the transposition rule through the composite operation family's
        // derived transposition dispatcher, which stages a transposed `shard_map` instead of reporting the operation as
        // non-transposable, so the manual region survives reverse mode.

        // The transposed boundary dualizes the shardings and boundary types of the source boundary, so a tangent that
        // is unreduced along the manual axis `x` receives a cotangent that is reduced along `x`.
        let tangent_sharding = Sharding::replicated(manual_mesh(), 1).with_unreduced_axes(["x"]).unwrap();
        let tangent_type = f32_vector_type(4).with_sharding(tangent_sharding.clone()).unwrap();
        let cotangent_type = tangent_type.cotangent().unwrap();
        let driver = TestTranspositionDriver {
            source: identity_body(tangent_type.clone()),
            transposed: identity_body(cotangent_type.clone()),
        };
        let shard_map =
            ShardMap::new(manual_mesh(), vec![tangent_sharding.clone()], vec![tangent_sharding], Vec::new()).unwrap();
        let operation =
            ShardMapOperation::from_boundary(shard_map, vec![tangent_type.clone()], vec![tangent_type.clone()]);
        let context = TestContext::new();
        let output_cotangent = context.input(ArrayIrType::Array(cotangent_type));
        let mut transposition_context = TranspositionContext::new(context.clone());
        let inputs = [PartialValue::Unknown(ArrayIrType::Array(tangent_type))];
        let accumulators = transposition_context.cotangent_accumulators(&inputs, &[]).unwrap();
        TestOperation::ShardMap(Box::new(operation))
            .transpose(
                &mut transposition_context,
                &driver,
                &inputs,
                &[MaybeZero::Value(output_cotangent)],
                &accumulators,
            )
            .unwrap();
        let cotangents = transposition_context.take_cotangents(&accumulators).unwrap();
        let [MaybeZero::Value(cotangent)] = &cotangents[..] else {
            panic!("expected one materialized input cotangent");
        };
        let pullback = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![cotangent.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{}], reduced={'x'}}] .
                let %1:f32[4][sharding={mesh<['x'=2:manual]>, [{}], reduced={'x'}}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{}], reduced={'x'}}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{}], reduced={'x'}}],
                    manual_axes=['x'],
                    global_input_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{}], reduced={'x'}}]],
                    global_output_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{}], reduced={'x'}}]],
                ] %0 [
                    body={
                        lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{}], reduced={'x'}}] .
                        in (%0)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_dualizes_reference_destination_shardings() {
        // The configuration of `test_shard_map_transposition` (a tangent that is unreduced along the manual axis `x`)
        // with a reference cotangent destination: the input is replicated along `x`, so its contributions accumulate in
        // a fresh local buffer whose frozen value leaves the map under the cotangent dual of the input sharding
        // (reduced along `x`) and is added into the destination once outside the map.
        let tangent_sharding = Sharding::replicated(manual_mesh(), 1).with_unreduced_axes(["x"]).unwrap();
        let tangent_type = f32_vector_type(4).with_sharding(tangent_sharding.clone()).unwrap();
        let shard_map =
            ShardMap::new(manual_mesh(), vec![tangent_sharding.clone()], vec![tangent_sharding.clone()], Vec::new())
                .unwrap();
        let operation =
            ShardMapOperation::from_boundary(shard_map, vec![tangent_type.clone()], vec![tangent_type.clone()]);
        let cotangent_type = tangent_type.cotangent().unwrap();
        let cotangent_sharding = tangent_sharding.cotangent();
        let program = shard_map_program(operation, identity_body(tangent_type)).unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(
            transposed.to_string(),
            formatdoc! {"
                lambda %0:{cotangent_type}, %1:ref<{cotangent_type}> .
                let %2:{cotangent_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{cotangent_sharding}],
                    out_shardings=[{cotangent_sharding}],
                    manual_axes=['x'],
                    global_input_types=[{cotangent_type}],
                    global_output_types=[{cotangent_type}],
                ] %0 [
                    body={{
                        lambda %0:{cotangent_type} .
                        let %1:{cotangent_type} = zero [type={cotangent_type}]
                            %2:ref<{cotangent_type}> = reference_new %1
                            () = reference_add_update %2 %0
                            %3:{cotangent_type} = reference_freeze %2
                        in (%3)
                    }},
                ]
                    () = reference_add_update %1 %2
                in ()"},
        );

        // A tangent that is sharded along the manual axis `x` and unreduced along the explicit axis `y` is owned by
        // each device, so its cotangent reference crosses the map under the cotangent dual of the input sharding
        // (reduced along `y`), and each device accumulates into the shard that it owns in place.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let tangent_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_unreduced_axes(["y"])
            .unwrap();
        let tangent_type = f32_vector_type(4).with_sharding(tangent_sharding.clone()).unwrap();
        let shard_map =
            ShardMap::new(mesh, vec![tangent_sharding.clone()], vec![tangent_sharding.clone()], Vec::new()).unwrap();
        let local_cotangent_type = shard_map.local_input_type(0, &tangent_type).unwrap().cotangent().unwrap();
        let cotangent_type = tangent_type.cotangent().unwrap();
        let cotangent_sharding = tangent_sharding.cotangent();
        let body = identity_body(shard_map.local_input_type(0, &tangent_type).unwrap());
        let operation =
            ShardMapOperation::from_boundary(shard_map, vec![tangent_type.clone()], vec![tangent_type.clone()]);
        let program = shard_map_program(operation, body).unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(
            transposed.to_string(),
            formatdoc! {"
                lambda %0:{cotangent_type}, %1:ref<{cotangent_type}> .
                let () = shard_map [
                    mesh=['x'=2:manual, 'y'=2:explicit],
                    in_shardings=[{cotangent_sharding}, {cotangent_sharding}],
                    out_shardings=[],
                    manual_axes=['x'],
                    global_input_types=[{cotangent_type}, ref<{cotangent_type}>],
                    global_output_types=[],
                ] %0 %1 [
                    body={{
                        lambda %0:{local_cotangent_type}, %1:ref<{local_cotangent_type}> .
                        let () = reference_add_update %1 %0
                        in ()
                    }},
                ]
                in ()"},
        );
    }

    #[test]
    fn test_shard_map_transposition_rejects_irreconcilable_cotangent_types() {
        // Transposition reconciles only the placement and memory kind of an assembled input cotangent with the caller's
        // cotangent type. Boundary validation admits only inputs whose other type properties agree with the declared
        // global input types, so no well-formed boundary reaches a mismatch, and this test calls the reconciliation
        // helper directly with assembled cotangents that a malformed boundary could produce.
        let context = TestContext::new();
        let sharded = f32_vector_type(4).with_sharding(sharded_along_x()).unwrap();

        // A cotangent whose reduction state differs is rejected instead of being passed through mistyped.
        let reduced = f32_vector_type(4)
            .with_sharding(Sharding::replicated(manual_mesh(), 1).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        let result =
            reconcile_input_cotangent(&context, 0, context.input(sharded.clone().into()), &reduced.clone().into());
        let expected = ShardMapError::CotangentTypeMismatch {
            input_index: 0,
            expected: reduced.clone().into(),
            actual: sharded.clone().into(),
        };
        assert!(matches!(result, Err(error) if error.downcast_custom::<ShardMapError>() == Some(&expected)));
        assert_eq!(
            expected.to_string(),
            format!(
                "`shard_map` transposition produced cotangent type `{sharded}` for input #0, which cannot be \
                 reconciled with the expected cotangent type `{reduced}`",
            ),
        );

        // So are cotangents of another shape or placed over another mesh.
        let longer = f32_vector_type(8).with_sharding(sharded_along_x()).unwrap();
        let result =
            reconcile_input_cotangent(&context, 1, context.input(sharded.clone().into()), &longer.clone().into());
        let expected = ShardMapError::CotangentTypeMismatch {
            input_index: 1,
            expected: longer.into(),
            actual: sharded.clone().into(),
        };
        assert!(matches!(result, Err(error) if error.downcast_custom::<ShardMapError>() == Some(&expected)));
        let other_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let other = f32_vector_type(4)
            .with_sharding(Sharding::new(other_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let result =
            reconcile_input_cotangent(&context, 2, context.input(sharded.clone().into()), &other.clone().into());
        let expected =
            ShardMapError::CotangentTypeMismatch { input_index: 2, expected: other.into(), actual: sharded.into() };
        assert!(matches!(result, Err(error) if error.downcast_custom::<ShardMapError>() == Some(&expected)));
    }

    #[test]
    fn test_shard_map_transposition_reshards_free_axis_placements() {
        // The forward boundary places a caller's input by its input sharding in place of the caller's own placement,
        // so transposition reshards the assembled input cotangent back to the caller's placement over the explicit
        // axis `y` instead of rejecting the placement difference.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let along_x = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let transposed = |caller_type: ArrayType| {
            let program = trace_test_program(
                |inputs| {
                    let input = inputs[0].clone();
                    vec![shard_map(|x: TestTracer| x, input, mesh.clone(), along_x.clone(), along_x.clone()).unwrap()]
                },
                vec![caller_type],
                Vec::new(),
            );
            program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap().to_string()
        };
        let along_y = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let along_x_and_y = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let global_type = f32_vector_type(8).with_sharding(along_x.clone()).unwrap();
        let local_type = f32_vector_type(4)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let transposed_shard_map = formatdoc! {"
            let %1:{global_type} = shard_map [
                mesh=['x'=2:manual, 'y'=2:explicit],
                in_shardings=[{along_x}],
                out_shardings=[{along_x}],
                manual_axes=['x'],
                global_input_types=[{global_type}],
                global_output_types=[{global_type}],
            ] %0 [
                body={{
                    lambda %0:{local_type} .
                    in (%0)
                }},
            ]"};
        let along_y_type = f32_vector_type(8).with_sharding(along_y.clone()).unwrap();
        assert_eq!(
            transposed(along_y_type.clone()),
            formatdoc! {"
                lambda %0:{global_type} .
                {transposed_shard_map}
                    %2:{along_y_type} = reshard [sharding={along_y}] %1
                in (%2)"},
        );

        // A caller placement over both the manual axis `x` and the explicit axis `y` is restored with one
        // placement-only `broadcast`. `reshard` cannot target the manual axis `x`, so resharding over `y` first would
        // replicate the cotangent along `x` only for the `broadcast` to re-place it along `x`, which compiles to an
        // extra round trip of collectives, while the `broadcast` alone compiles to local slicing.
        let along_x_and_y_type = f32_vector_type(8).with_sharding(along_x_and_y.clone()).unwrap();
        assert_eq!(
            transposed(along_x_and_y_type.clone()),
            formatdoc! {"
                lambda %0:{global_type} .
                {transposed_shard_map}
                    %2:{along_x_and_y_type} = broadcast [
                        output_type={along_x_and_y_type},
                        output_axes=[0],
                    ] %1
                in (%2)"},
        );

        // A caller without a sharding is treated as replicated over the mesh, so an assembled cotangent placed over the
        // explicit axis `y` is first resharded to replicated, and the `broadcast` then drops the remaining placement.
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let replicated_type = f32_vector_type(8).with_sharding(replicated.clone()).unwrap();
        let program = trace_test_program(
            |inputs| {
                let input = inputs[0].clone();
                vec![
                    shard_map(|x: TestTracer| x, input, mesh.clone(), along_x_and_y.clone(), along_x_and_y.clone())
                        .unwrap(),
                ]
            },
            vec![f32_vector_type(8)],
            Vec::new(),
        );
        let local_type =
            f32_vector_type(4).with_sharding(along_y.clone().with_varying_manual_axes(["x"]).unwrap()).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{along_x_and_y_type} .
                let %1:{along_x_and_y_type} = shard_map [
                    mesh=['x'=2:manual, 'y'=2:explicit],
                    in_shardings=[{along_x_and_y}],
                    out_shardings=[{along_x_and_y}],
                    manual_axes=['x'],
                    global_input_types=[{along_x_and_y_type}],
                    global_output_types=[{along_x_and_y_type}],
                ] %0 [
                    body={{
                        lambda %0:{local_type} .
                        in (%0)
                    }},
                ]
                    %2:{replicated_type} = reshard [sharding={replicated}] %1
                    %3:f32[8] = broadcast [output_type=f32[8], output_axes=[0]] %2
                in (%3)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_keeps_auto_axis_placements_untracked() {
        // A caller placement over an auto axis is untracked, so it is not a `reshard` target: the assembled cotangent
        // is re-placed along the auto axis `y` with a placement-only `broadcast` only.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let along_x = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let along_y = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let program = trace_test_program(
            |inputs| {
                let input = inputs[0].clone();
                vec![shard_map(|x: TestTracer| x, input, mesh.clone(), along_x.clone(), along_x.clone()).unwrap()]
            },
            vec![f32_vector_type(8).with_sharding(along_y.clone()).unwrap()],
            Vec::new(),
        );
        let global_type = f32_vector_type(8).with_sharding(along_x.clone()).unwrap();
        let local_type = f32_vector_type(4)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let along_y_type = f32_vector_type(8).with_sharding(along_y).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{global_type} .
                let %1:{global_type} = shard_map [
                    mesh=['x'=2:manual, 'y'=2:auto],
                    in_shardings=[{along_x}],
                    out_shardings=[{along_x}],
                    manual_axes=['x'],
                    global_input_types=[{global_type}],
                    global_output_types=[{global_type}],
                ] %0 [
                    body={{
                        lambda %0:{local_type} .
                        in (%0)
                    }},
                ]
                    %2:{along_y_type} = broadcast [
                        output_type={along_y_type},
                        output_axes=[0],
                    ] %1
                in (%2)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_reshards_explicit_reduction_states() {
        // A caller tangent that is unreduced along the explicit axis `z` has a cotangent that is reduced along `z`, and
        // the `reshard` that restores its placement along the explicit axis `y` carries that reduction state.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let along_x = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_unreduced_axes(["z"])
            .unwrap();
        let along_y = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])])
            .unwrap()
            .with_unreduced_axes(["z"])
            .unwrap();
        let program = trace_test_program(
            |inputs| {
                let input = inputs[0].clone();
                vec![shard_map(|x: TestTracer| x, input, mesh.clone(), along_x.clone(), along_x.clone()).unwrap()]
            },
            vec![f32_vector_type(8).with_sharding(along_y.clone()).unwrap()],
            Vec::new(),
        );
        let cotangent_sharding = along_x.cotangent();
        let cotangent_type = f32_vector_type(8).with_sharding(cotangent_sharding.clone()).unwrap();
        let local_type = f32_vector_type(4)
            .with_sharding(
                Sharding::replicated(mesh.clone(), 1)
                    .with_reduced_axes(["z"])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let caller_cotangent_sharding = along_y.cotangent();
        let caller_cotangent_type = f32_vector_type(8).with_sharding(caller_cotangent_sharding.clone()).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{cotangent_type} .
                let %1:{cotangent_type} = shard_map [
                    mesh=['x'=2:manual, 'y'=2:explicit, 'z'=2:explicit],
                    in_shardings=[{cotangent_sharding}],
                    out_shardings=[{cotangent_sharding}],
                    manual_axes=['x'],
                    global_input_types=[{cotangent_type}],
                    global_output_types=[{cotangent_type}],
                ] %0 [
                    body={{
                        lambda %0:{local_type} .
                        in (%0)
                    }},
                ]
                    %2:{caller_cotangent_type} = reshard [
                        sharding={caller_cotangent_sharding},
                    ] %1
                in (%2)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_reconciles_placements_in_enclosing_manual_regions() {
        // Inside a manual region over `x`, a nested map over `y` receives an input that varies along `x` and that the
        // caller places along the explicit axis `z`. Transposition reshards the assembled cotangent to that placement,
        // which preserves its variation along the enclosing manual axis `x`.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let along_y = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let along_z = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["z"])])
            .unwrap()
            .with_varying_manual_axes(["x"])
            .unwrap();
        let program = trace_test_program(
            |inputs| {
                let input = inputs[0].clone();
                vec![shard_map(|x: TestTracer| x, input, mesh.clone(), along_y.clone(), along_y.clone()).unwrap()]
            },
            vec![f32_vector_type(8).with_sharding(along_z.clone()).unwrap()],
            vec![("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 })],
        );
        let boundary_sharding = along_y.clone().with_varying_manual_axes(["x"]).unwrap();
        let global_type = f32_vector_type(8).with_sharding(boundary_sharding.clone()).unwrap();
        let local_type = f32_vector_type(4)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x", "y"]).unwrap())
            .unwrap();
        let along_z_type = f32_vector_type(8).with_sharding(along_z.clone()).unwrap();
        let target_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["z"])]).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{global_type} .
                let %1:{global_type} = shard_map [
                    mesh=['x'=2:manual, 'y'=2:manual, 'z'=2:explicit],
                    in_shardings=[{along_y}],
                    out_shardings=[{along_y}],
                    manual_axes=['y'],
                    global_input_types=[{global_type}],
                    global_output_types=[{global_type}],
                ] %0 [
                    body={{
                        lambda %0:{local_type} .
                        in (%0)
                    }},
                ]
                    %2:{along_z_type} = reshard [sharding={target_sharding}] %1
                in (%2)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_reconciles_placements_across_axis_type_variants() {
        // A caller may place its input along the axis `x` of a mesh in which `x` is explicit, while the shard map uses
        // the same mesh axes with `x` manual. The assembled input cotangent is reconciled in the frame of the caller's
        // mesh, in which it already has the caller's placement along `x`, so no `reshard` is needed, and a
        // placement-only `broadcast` re-places the cotangent over the caller's mesh.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let caller_type = f32_vector_type(8)
            .with_sharding(Sharding::new(explicit_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let program = trace_test_program(
            |inputs| {
                let input = inputs[0].clone();
                vec![shard_map(|x: TestTracer| x, input, manual_mesh(), sharded_along_x(), sharded_along_x()).unwrap()]
            },
            vec![caller_type.clone()],
            Vec::new(),
        );
        let global_type = f32_vector_type(8).with_sharding(sharded_along_x()).unwrap();
        let local_type = f32_vector_type(4)
            .with_sharding(Sharding::replicated(manual_mesh(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{global_type} .
                let %1:{global_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharding}],
                    out_shardings=[{sharding}],
                    manual_axes=['x'],
                    global_input_types=[{global_type}],
                    global_output_types=[{global_type}],
                ] %0 [
                    body={{
                        lambda %0:{local_type} .
                        in (%0)
                    }},
                ]
                    %2:{caller_type} = broadcast [output_type={caller_type}, output_axes=[0]] %1
                in (%2)",
                sharding = sharded_along_x(),
            },
        );
    }

    #[test]
    fn test_shard_map_transposition_reconciles_replicated_reference_destinations() {
        // The frozen local accumulator of a replicated reference destination leaves the map under the input sharding,
        // so it is reconciled with the referent type of the caller's destination before it is added into it, exactly
        // like a returned cotangent. Here, the caller's input carries no sharding, so a placement-only `broadcast`
        // drops the placement of the accumulated cotangent.
        let replicated = Sharding::replicated(manual_mesh(), 1);
        let program = trace_test_program(
            |inputs| {
                let input = inputs[0].clone();
                vec![shard_map(|x: TestTracer| x, input, manual_mesh(), replicated.clone(), sharded_along_x()).unwrap()]
            },
            vec![f32_vector_type(2)],
            Vec::new(),
        );
        let global = f32_vector_type(4).with_sharding(sharded_along_x()).unwrap();
        let local = f32_vector_type(2)
            .with_sharding(Sharding::replicated(manual_mesh(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let replicated_type = f32_vector_type(2).with_sharding(replicated.clone()).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{global}, %1:ref<f32[2]> .
                let %2:{replicated_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharded}],
                    out_shardings=[{replicated}],
                    manual_axes=['x'],
                    global_input_types=[{global}],
                    global_output_types=[{replicated_type}],
                ] %0 [
                    body={{
                        lambda %0:{local} .
                        let %1:{replicated_type} = zero [type={replicated_type}]
                            %2:ref<{replicated_type}> = reference_new %1
                            %3:{replicated_type} = parallel_reduce [kind=sum, axis_name=\"x\", mesh=['x'=2:manual]] %0
                            () = reference_add_update %2 %3
                            %4:{replicated_type} = reference_freeze %2
                        in (%4)
                    }},
                ]
                    %3:f32[2] = broadcast [output_type=f32[2], output_axes=[0]] %2
                    () = reference_add_update %1 %3
                in ()",
                sharded = sharded_along_x(),
            },
        );

        // A caller placement along the explicit axis `y` is restored with `reshard` before the accumulated cotangent is
        // added into the caller's destination, instead of relying on the addition to re-place it implicitly.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let along_x = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let along_y = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let program = trace_test_program(
            |inputs| {
                let input = inputs[0].clone();
                vec![shard_map(|x: TestTracer| x, input, mesh.clone(), replicated.clone(), along_x.clone()).unwrap()]
            },
            vec![f32_vector_type(4).with_sharding(along_y.clone()).unwrap()],
            Vec::new(),
        );
        let global = f32_vector_type(8).with_sharding(along_x.clone()).unwrap();
        let local = f32_vector_type(4)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let replicated_type = f32_vector_type(4).with_sharding(replicated.clone()).unwrap();
        let along_y_type = f32_vector_type(4).with_sharding(along_y.clone()).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{global}, %1:ref<{along_y_type}> .
                let %2:{replicated_type} = shard_map [
                    mesh=['x'=2:manual, 'y'=2:explicit],
                    in_shardings=[{along_x}],
                    out_shardings=[{replicated}],
                    manual_axes=['x'],
                    global_input_types=[{global}],
                    global_output_types=[{replicated_type}],
                ] %0 [
                    body={{
                        lambda %0:{local} .
                        let %1:{replicated_type} = zero [type={replicated_type}]
                            %2:ref<{replicated_type}> = reference_new %1
                            %3:{replicated_type} = parallel_reduce [kind=sum, axis_name=\"x\", mesh=['x'=2:manual, \
                    'y'=2:explicit]] %0
                            () = reference_add_update %2 %3
                            %4:{replicated_type} = reference_freeze %2
                        in (%4)
                    }},
                ]
                    %3:{along_y_type} = reshard [sharding={along_y}] %2
                    () = reference_add_update %1 %3
                in ()"},
        );
    }

    #[test]
    fn test_shard_map_transposition_transfers_cotangents_to_the_caller_memory() {
        // The forward boundary moves a caller's input from pinned host memory into device memory, so transposition
        // transfers the assembled cotangent back after reconciling its placement in device memory, whether the caller's
        // input carries the input sharding itself or no sharding at all.
        let host = Memory::Host { pinned: true };
        let global = f32_vector_type(4).with_sharding(sharded_along_x()).unwrap();
        let local = f32_vector_type(2)
            .with_sharding(Sharding::replicated(manual_mesh(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let transposed = |caller_type: ArrayType, kind: CotangentDestinationKind| {
            let program = trace_test_program(
                |inputs| {
                    let input = inputs[0].clone();
                    vec![
                        shard_map(|x: TestTracer| x, input, manual_mesh(), sharded_along_x(), sharded_along_x())
                            .unwrap(),
                    ]
                },
                vec![caller_type],
                Vec::new(),
            );
            program.transpose_with_respect_to(&[0], &[kind]).unwrap().to_string()
        };
        let transposed_shard_map = formatdoc! {"
            let %1:{global} = shard_map [
                mesh=['x'=2:manual],
                in_shardings=[{sharded}],
                out_shardings=[{sharded}],
                manual_axes=['x'],
                global_input_types=[{global}],
                global_output_types=[{global}],
            ] %0 [
                body={{
                    lambda %0:{local} .
                    in (%0)
                }},
            ]",
            sharded = sharded_along_x(),
        };
        let host_global = global.clone().with_memory(host);
        assert_eq!(
            transposed(host_global.clone(), CotangentDestinationKind::Return),
            formatdoc! {"
                lambda %0:{global} .
                {transposed_shard_map}
                    %2:{host_global} = transfer_to_memory [destination=Host[Pinned]] %1
                in (%2)"},
        );
        let host_unplaced = f32_vector_type(4).with_memory(host);
        assert_eq!(
            transposed(host_unplaced.clone(), CotangentDestinationKind::Return),
            formatdoc! {"
                lambda %0:{global} .
                {transposed_shard_map}
                    %2:f32[4] = broadcast [output_type=f32[4], output_axes=[0]] %1
                    %3:{host_unplaced} = transfer_to_memory [destination=Host[Pinned]] %2
                in (%3)"},
        );

        // The accumulator of a replicated reference destination in host memory is transferred before it is added into
        // that destination.
        let replicated = Sharding::replicated(manual_mesh(), 1);
        let program = trace_test_program(
            |inputs| {
                let input = inputs[0].clone();
                vec![shard_map(|x: TestTracer| x, input, manual_mesh(), replicated.clone(), sharded_along_x()).unwrap()]
            },
            vec![f32_vector_type(2).with_memory(host)],
            Vec::new(),
        );
        let replicated_type = f32_vector_type(2).with_sharding(replicated.clone()).unwrap();
        let host_replicated = f32_vector_type(2).with_memory(host);
        let local_replicated = f32_vector_type(2)
            .with_sharding(Sharding::replicated(manual_mesh(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{global}, %1:ref<{host_replicated}> .
                let %2:{replicated_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{sharded}],
                    out_shardings=[{replicated}],
                    manual_axes=['x'],
                    global_input_types=[{global}],
                    global_output_types=[{replicated_type}],
                ] %0 [
                    body={{
                        lambda %0:{local_replicated} .
                        let %1:{replicated_type} = zero [type={replicated_type}]
                            %2:ref<{replicated_type}> = reference_new %1
                            %3:{replicated_type} = parallel_reduce [kind=sum, axis_name=\"x\", mesh=['x'=2:manual]] %0
                            () = reference_add_update %2 %3
                            %4:{replicated_type} = reference_freeze %2
                        in (%4)
                    }},
                ]
                    %3:f32[2] = broadcast [output_type=f32[2], output_axes=[0]] %2
                    %4:{host_replicated} = transfer_to_memory [destination=Host[Pinned]] %3
                    () = reference_add_update %1 %4
                in ()",
                sharded = sharded_along_x(),
            },
        );
    }

    #[test]
    fn test_shard_map_transposition_leaves_replica_aggregation_to_the_body() {
        // A replicated input enters the body invariant and reaches an output tiled along `x` through `parallel_vary`,
        // whose transpose is the cross-device sum `parallel_reduce`. That sum aggregates the contributions of all
        // devices to the replicated input cotangent, so the transposed boundary performs no reduction of its own and
        // only a placement-only `broadcast` reconciles the assembled cotangent with the caller's unplaced input type.
        let replicated = Sharding::replicated(manual_mesh(), 1);
        let program = trace_test_program(
            |inputs| {
                let input = inputs[0].clone();
                vec![shard_map(|x: TestTracer| x, input, manual_mesh(), replicated.clone(), sharded_along_x()).unwrap()]
            },
            vec![f32_vector_type(2)],
            Vec::new(),
        );
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] .
                let %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                    global_output_types=[f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]],
                ] %0 [
                    body={
                        lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = parallel_reduce [kind=sum, \
                            axis_name=\"x\", mesh=['x'=2:manual]] %0
                        in (%1)
                    },
                ]
                    %2:f32[2] = broadcast [output_type=f32[2], output_axes=[0]] %1
                in (%2)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_preserves_zero_space_seed_positions() {
        // Transposing a map whose outputs are a zero-space integer, a value, and a forwarded reference keeps every
        // output cotangent seed at its program position, although only the value cotangent crosses the transposed
        // boundary.
        let mesh = manual_mesh();
        let sharding = Sharding::replicated(mesh.clone(), 0);
        let value_type = f32_scalar_type().with_sharding(sharding.clone()).unwrap();
        let integer_type = ArrayType::scalar(DataType::I32).with_sharding(sharding.clone()).unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(value_type.clone()));
        let mut body = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body.add_input(reference_type.clone());
        let value = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let integer = body
            .add_instruction(
                ArrayOperation::Zero(ZeroOperation::new(integer_type.clone())),
                Vec::new(),
                Vec::new(),
                None,
            )
            .unwrap()[0];
        let body = body
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![integer, value, reference],
                vec![Placeholder],
                vec![Placeholder; 3],
            )
            .unwrap();
        let operation = ShardMapOperation::from_program(
            &body,
            vec![reference_type.clone()],
            ShardMap::new(mesh, vec![sharding.clone()], vec![sharding; 3], vec!["x".to_string()]).unwrap(),
        )
        .unwrap();
        let program = shard_map_program(operation, body).unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda \
                %0:zero[][sharding={mesh<['x'=2:manual]>, []}], %1:f32[][sharding={mesh<['x'=2:manual]>, []}], \
                %2:ref<f32[][sharding={mesh<['x'=2:manual]>, []}]> .
                let %3:f32[][sharding={mesh<['x'=2:manual]>, []}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, []}],
                    out_shardings=[{mesh<['x'=2:manual]>, []}],
                    manual_axes=['x'],
                    global_input_types=[f32[][sharding={mesh<['x'=2:manual]>, []}]],
                    global_output_types=[f32[][sharding={mesh<['x'=2:manual]>, []}]],
                ] %1 [
                    body={
                        lambda %0:f32[][sharding={mesh<['x'=2:manual]>, []}] .
                        let %1:f32[][sharding={mesh<['x'=2:manual]>, []}] = zero \
                            [type=f32[][sharding={mesh<['x'=2:manual]>, []}]]
                            %2:ref<f32[][sharding={mesh<['x'=2:manual]>, []}]> = reference_new %1
                            () = reference_add_update %2 %0
                            %3:f32[][sharding={mesh<['x'=2:manual]>, []}] = reference_freeze %2
                        in (%3)
                    },
                ]
                    () = reference_add_update %2 %3
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_shard_map_transposition_of_zero_cotangents_uses_cotangent_descriptors() {
        // Zero cotangents skip a pure body only after its effects have been inspected, so the driver supplies the real
        // identity-region boundary rather than an empty driver that cannot answer that query.
        let tangent_sharding = Sharding::replicated(manual_mesh(), 1).with_unreduced_axes(["x"]).unwrap();
        let tangent_type = f32_vector_type(4).with_sharding(tangent_sharding.clone()).unwrap();
        let cotangent_type = tangent_type.cotangent().unwrap();
        let driver = TestTranspositionDriver {
            source: identity_body(tangent_type.clone()),
            transposed: identity_body(cotangent_type.clone()),
        };
        let shard_map =
            ShardMap::new(manual_mesh(), vec![tangent_sharding.clone()], vec![tangent_sharding], Vec::new()).unwrap();
        let operation =
            ShardMapOperation::from_boundary(shard_map, vec![tangent_type.clone()], vec![tangent_type.clone()]);
        let mut context = TestContext::new();
        let known = context.input(ArrayIrType::Array(tangent_type.clone()));
        let cotangents = transpose_primal_shard_map(
            &operation,
            &mut context,
            &driver,
            &[PartialValue::Known(known)],
            &[MaybeZero::Zero(ArrayIrType::Array(tangent_type))],
            &CotangentDestinations::without_references([true]),
        )
        .unwrap();
        assert!(matches!(&cotangents[..], [MaybeZero::Zero(actual)] if actual == &ArrayIrType::Array(cotangent_type)));
    }

    #[test]
    fn test_shard_map_transposition_projects_zero_output_cotangents_out_of_the_body() {
        // A structurally zero output cotangent does not cross the transposed boundary (as in JAX, which binds the
        // transposed map over the nonzero output cotangents only). The body `x -> (x, x)` is transposed projected onto
        // its first output, so the transposed body takes the one nonzero output cotangent and materializes no zero.
        let sharded = sharded_along_x();
        let global_type = f32_vector_type(4).with_sharding(sharded.clone()).unwrap();
        let shard_map =
            ShardMap::new(manual_mesh(), vec![sharded.clone()], vec![sharded.clone(), sharded], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &global_type).unwrap();
        let source = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(local_type.clone().into());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![input, input], vec![Placeholder], vec![Placeholder; 2])
                .unwrap()
        };
        let driver = TestTranspositionDriver { source, transposed: identity_body(local_type) };
        let operation = ShardMapOperation::from_boundary(
            shard_map,
            vec![global_type.clone()],
            vec![global_type.clone(), global_type.clone()],
        );
        let mut context = TestContext::new();
        let output_cotangent = context.input(ArrayIrType::Array(global_type.clone()));
        let cotangents = transpose_primal_shard_map(
            &operation,
            &mut context,
            &driver,
            &[PartialValue::Unknown(ArrayIrType::Array(global_type.clone()))],
            &[MaybeZero::Value(output_cotangent), MaybeZero::Zero(ArrayIrType::Array(global_type))],
            &CotangentDestinations::without_references([true]),
        )
        .unwrap();
        let [MaybeZero::Value(cotangent)] = &cotangents[..] else {
            panic!("expected one materialized input cotangent");
        };
        let pullback = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![cotangent.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                    global_output_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ] %0 [
                    body={
                        lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        in (%0)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_projects_zero_space_outputs_out_of_the_body() {
        // The structurally zero cotangent of a zero-space output (a boolean predicate) does not cross the transposed
        // boundary: the body is transposed projected onto its value output, so only the value cotangent crosses the
        // boundary and the transposed body takes no predicate cotangent.
        let sharding = Sharding::replicated(manual_mesh(), 1);
        let value_type = f32_vector_type(4).with_sharding(sharding.clone()).unwrap();
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let source = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let value = builder.add_input(value_type.clone().into());
            let predicate = builder
                .add_instruction(
                    ArrayOperation::Zero(ZeroOperation::new(predicate_type.clone())),
                    Vec::new(),
                    Vec::new(),
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![value, predicate],
                    vec![Placeholder],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let driver = TestTranspositionDriver { source, transposed: identity_body(value_type.clone()) };
        let shard_map =
            ShardMap::new(manual_mesh(), vec![sharding.clone()], vec![sharding.clone(), sharding], Vec::new()).unwrap();
        let operation = ShardMapOperation::from_boundary(
            shard_map,
            vec![value_type.clone()],
            vec![value_type.clone(), predicate_type.clone()],
        );
        let mut context = TestContext::new();
        let value_cotangent = context.input(ArrayIrType::Array(value_type.clone()));

        let contributions = transpose_primal_shard_map(
            &operation,
            &mut context,
            &driver,
            &[PartialValue::Unknown(ArrayIrType::Array(value_type.clone()))],
            &[
                MaybeZero::Value(value_cotangent),
                MaybeZero::Zero(ArrayIrType::Array(predicate_type.cotangent().unwrap())),
            ],
            &CotangentDestinations::without_references([true]),
        )
        .unwrap();
        let [MaybeZero::Value(contribution)] = &contributions[..] else {
            panic!("expected one materialized input cotangent");
        };
        let pullback = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![contribution.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:manual]>, [{}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{}]}]],
                    global_output_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{}]}]],
                ] %0 [
                    body={
                        lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{}]}] .
                        in (%0)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_returns_structurally_zero_input_cotangents() {
        // An input whose cotangent the transposed body produces as a structural zero is not an output of the transposed
        // map (as in JAX's `_shard_map_transpose`), so its cotangent stays a symbolic zero outside the map instead of
        // being assembled from materialized zeros. The body maps `(x, y)` to `x`, so the cotangent of `y` is zero.
        let replicated = Sharding::replicated(manual_mesh(), 1);
        let input_type = f32_vector_type(2);
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let x = builder.add_input(input_type.clone().into());
            let _y = builder.add_input(input_type.clone().into());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![x], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let shard_map =
            ShardMap::new(manual_mesh(), vec![replicated.clone(); 2], vec![replicated.clone()], Vec::new()).unwrap();
        let operation =
            ShardMapOperation::from_program(&body, vec![input_type.clone().into(), input_type.into()], shard_map)
                .unwrap();
        let program = shard_map_program(operation, body).unwrap();
        let destination_kinds = [CotangentDestinationKind::Return, CotangentDestinationKind::Return];
        let global_type = f32_vector_type(2).with_sharding(replicated.clone()).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{global_type} .
                let %1:{global_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}],
                    out_shardings=[{replicated}],
                    manual_axes=['x'],
                    global_input_types=[{global_type}],
                    global_output_types=[{global_type}],
                ] %0 [
                    body={{
                        lambda %0:f32[2] .
                        in (%0)
                    }},
                ]
                    %2:{global_type} = zero [type={global_type}]
                in (%1, %2)"},
        );

        // When every input cotangent is a structural zero, the transposed map has no outputs and no effects, so it is
        // not staged at all.
        assert_eq!(
            program.transpose_with_respect_to(&[1], &[CotangentDestinationKind::Return]).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{global_type}, %1:{global_type} .
                let %2:{global_type} = zero [type={global_type}]
                in (%2)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_returns_input_cotangents_of_zero_output_cotangents_as_structural_zeros() {
        // The body `(x, w) -> (x, w + w)` receives a structurally zero cotangent for its first output, so it is
        // transposed projected onto its second output, from which `x` is disconnected. The transposed body therefore
        // produces the cotangent of `x` as a canonical `zero`, which is returned as a structural zero outside the map
        // instead of crossing the transposed boundary as materialized zeros.
        let replicated = Sharding::replicated(manual_mesh(), 1);
        let global_type = f32_vector_type(2).with_sharding(replicated.clone()).unwrap();
        let shard_map =
            ShardMap::new(manual_mesh(), vec![replicated.clone(); 2], vec![replicated.clone(); 2], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &global_type).unwrap();
        let source = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let x = builder.add_input(local_type.clone().into());
            let w = builder.add_input(local_type.clone().into());
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![w, w], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![x, sum], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let transposed = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let cotangent = builder.add_input(local_type.clone().into());
            let zero = builder
                .add_instruction(
                    ArrayOperation::Zero(ZeroOperation::new(local_type.clone())),
                    Vec::new(),
                    Vec::new(),
                    None,
                )
                .unwrap()[0];
            let sum =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![cotangent, cotangent], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![zero, sum], vec![Placeholder], vec![Placeholder; 2])
                .unwrap()
        };
        let driver = TestTranspositionDriver { source, transposed };
        let operation = ShardMapOperation::from_boundary(
            shard_map,
            vec![global_type.clone(), global_type.clone()],
            vec![global_type.clone(), global_type.clone()],
        );
        let mut context = TestContext::new();
        let output_cotangent = context.input(ArrayIrType::Array(global_type.clone()));
        let cotangents = transpose_primal_shard_map(
            &operation,
            &mut context,
            &driver,
            &[
                PartialValue::Unknown(ArrayIrType::Array(global_type.clone())),
                PartialValue::Unknown(ArrayIrType::Array(global_type.clone())),
            ],
            &[MaybeZero::Zero(ArrayIrType::Array(global_type.clone())), MaybeZero::Value(output_cotangent)],
            &CotangentDestinations::without_references([true, true]),
        )
        .unwrap();
        let [MaybeZero::Zero(zero_type), MaybeZero::Value(cotangent)] = &cotangents[..] else {
            panic!("expected a structurally zero input cotangent followed by a materialized one");
        };
        assert_eq!(zero_type, &ArrayIrType::Array(global_type.clone()));
        let pullback = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![cotangent.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        // The transposed `shard_map` takes and returns only the cotangents of `w`.
        assert_eq!(
            pullback.to_string(),
            formatdoc! {"
                lambda %0:{global_type} .
                let %1:{global_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}],
                    out_shardings=[{replicated}],
                    manual_axes=['x'],
                    global_input_types=[{global_type}],
                    global_output_types=[{global_type}],
                ] %0 [
                    body={{
                        lambda %0:{local_type} .
                        let %1:{local_type} = add %0 %0
                        in (%1)
                    }},
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_returns_zero_constant_input_cotangents_as_structural_zeros() {
        // A transposed body that returns a zero constant as an input cotangent (rather than the output of a
        // zero-producing instruction) produces a structural zero as well, so the cotangent is returned as a symbolic
        // zero, and the transposed map, which is then left without outputs and effects, is not staged.
        let sharded = sharded_along_x();
        let global_type = f32_vector_type(4).with_sharding(sharded.clone()).unwrap();
        let shard_map = ShardMap::new(manual_mesh(), vec![sharded.clone()], vec![sharded], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &global_type).unwrap();
        let transposed = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            builder.add_input(local_type.clone().into());
            let zero = Array::from_elements(local_type.clone(), &[0f32; 2]).unwrap();
            let zero = builder.add_constant(TestValue::Array(zero));
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![zero], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let driver = TestTranspositionDriver { source: identity_body(local_type), transposed };
        let operation =
            ShardMapOperation::from_boundary(shard_map, vec![global_type.clone()], vec![global_type.clone()]);
        let mut context = TestContext::new();
        let output_cotangent = context.input(ArrayIrType::Array(global_type.clone()));
        let cotangents = transpose_primal_shard_map(
            &operation,
            &mut context,
            &driver,
            &[PartialValue::Unknown(ArrayIrType::Array(global_type.clone()))],
            &[MaybeZero::Value(output_cotangent)],
            &CotangentDestinations::without_references([true]),
        )
        .unwrap();
        assert!(matches!(&cotangents[..], [MaybeZero::Zero(actual)] if actual == &ArrayIrType::Array(global_type)));
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_shard_map_transposition_returns_structural_zeros_for_inputs_of_unused_outputs() {
        // The outer program uses only the second output of the map `(x, w) -> (neg(x), w + w)`, so the cotangent of the
        // first output is a structural zero. The body is transposed projected onto its second output, from which `x`
        // is disconnected, so the transposed map takes and returns only the cotangents of `w` and the cotangent of `x`
        // is a structural zero, which the outer pullback materializes as a global `zero` at its boundary.
        let replicated = Sharding::replicated(manual_mesh(), 1);
        let input_type = f32_vector_type(2);
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let x = builder.add_input(input_type.clone().into());
            let w = builder.add_input(input_type.clone().into());
            let negation = builder
                .add_instruction(ArrayOperation::Neg(NegOperation::new()), Vec::new(), vec![x], None)
                .unwrap()[0];
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![w, w], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![negation, sum],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let shard_map =
            ShardMap::new(manual_mesh(), vec![replicated.clone(); 2], vec![replicated.clone(); 2], Vec::new()).unwrap();
        let operation =
            ShardMapOperation::from_program(&body, vec![input_type.clone().into(), input_type.into()], shard_map)
                .unwrap();
        let program = shard_map_program(operation, body).unwrap().with_outputs(&[1]).unwrap();
        let destination_kinds = [CotangentDestinationKind::Return, CotangentDestinationKind::Return];
        let global_type = f32_vector_type(2).with_sharding(replicated.clone()).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{global_type} .
                let %1:{global_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}],
                    out_shardings=[{replicated}],
                    manual_axes=['x'],
                    global_input_types=[{global_type}],
                    global_output_types=[{global_type}],
                ] %0 [
                    body={{
                        lambda %0:f32[2] .
                        let %1:f32[2] = add %0 %0
                        in (%1)
                    }},
                ]
                    %2:{global_type} = zero [type={global_type}]
                in (%2, %1)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_returns_structural_zeros_for_inputs_reaching_only_unused_outputs() {
        // The outer program uses only the second output of the map `(x, w) -> (x + w, w)`. Both inputs reach the first
        // output, but only `w` reaches the second one, so the cotangent of `x` is a structural zero and the cotangent
        // of `w` is the output cotangent itself, without a materialized zero added to it.
        let replicated = Sharding::replicated(manual_mesh(), 1);
        let input_type = f32_vector_type(2);
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let x = builder.add_input(input_type.clone().into());
            let w = builder.add_input(input_type.clone().into());
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![x, w], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum, w], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let shard_map =
            ShardMap::new(manual_mesh(), vec![replicated.clone(); 2], vec![replicated.clone(); 2], Vec::new()).unwrap();
        let operation =
            ShardMapOperation::from_program(&body, vec![input_type.clone().into(), input_type.into()], shard_map)
                .unwrap();
        let program = shard_map_program(operation, body).unwrap().with_outputs(&[1]).unwrap();
        let destination_kinds = [CotangentDestinationKind::Return, CotangentDestinationKind::Return];
        let global_type = f32_vector_type(2).with_sharding(replicated.clone()).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{global_type} .
                let %1:{global_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}],
                    out_shardings=[{replicated}],
                    manual_axes=['x'],
                    global_input_types=[{global_type}],
                    global_output_types=[{global_type}],
                ] %0 [
                    body={{
                        lambda %0:f32[2] .
                        in (%0)
                    }},
                ]
                    %2:{global_type} = zero [type={global_type}]
                in (%2, %1)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_drops_zero_cotangents_of_repeated_outputs() {
        // The map `x -> (x, x)` returns its input twice, and the outer program uses only the second output. The body is
        // transposed projected onto that output, so the transposed body is the identity on the one nonzero output
        // cotangent instead of adding a materialized zero to it.
        let replicated = Sharding::replicated(manual_mesh(), 1);
        let input_type = f32_vector_type(2);
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let x = builder.add_input(input_type.clone().into());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![x, x], vec![Placeholder], vec![Placeholder; 2])
                .unwrap()
        };
        let shard_map =
            ShardMap::new(manual_mesh(), vec![replicated.clone()], vec![replicated.clone(); 2], Vec::new()).unwrap();
        let operation = ShardMapOperation::from_program(&body, vec![input_type.into()], shard_map).unwrap();
        let program = shard_map_program(operation, body).unwrap().with_outputs(&[1]).unwrap();
        let global_type = f32_vector_type(2).with_sharding(replicated.clone()).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap().to_string(),
            formatdoc! {"
                lambda %0:{global_type} .
                let %1:{global_type} = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{replicated}],
                    out_shardings=[{replicated}],
                    manual_axes=['x'],
                    global_input_types=[{global_type}],
                    global_output_types=[{global_type}],
                ] %0 [
                    body={{
                        lambda %0:f32[2] .
                        in (%0)
                    }},
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_keeps_effects_of_bodies_without_output_cotangents() {
        // A body whose backward work prints (here, a reverse-mode carrier of a custom function applied to `x`, whose
        // backward rule prints its cotangent) must be transposed even when every value output cotangent is a structural
        // zero, as here, where the outer program uses no output of the map. The body is projected onto no outputs,
        // which keeps the carrier because it carries deferred work, so the transposed map still runs the print on a
        // zero seed. The backward rule returns that zero seed as the cotangent of `x`, which is therefore a structural
        // zero, and the transposed map without outputs is staged for its effect alone.
        let replicated = Sharding::replicated(manual_mesh(), 0);
        let scalar = f32_scalar_type().with_sharding(replicated.clone()).unwrap();
        let backward = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let cotangent = builder.add_input(scalar.clone().into());
            let print = PrintOperation::<ArrayIrType>::new("cotangent").with_effect_class(EffectClass::DeviceOrderedIo);
            builder.add_instruction(print, Vec::new(), vec![cotangent], None).unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![cotangent], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let x = builder.add_input(scalar.clone().into());
            let backward = builder.import_program(backward);
            let carrier = CustomFunctionTransposeOperation::from_backward_region(
                0,
                vec![scalar.clone().into()],
                vec![scalar.clone().into()],
            );
            builder
                .add_instruction(TestOperation::CustomFunctionTranspose(carrier), vec![backward], vec![x], None)
                .unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![x], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let shard_map =
            ShardMap::new(manual_mesh(), vec![replicated.clone()], vec![replicated.clone()], Vec::new()).unwrap();
        let operation = ShardMapOperation::from_program(&body, vec![f32_scalar_type().into()], shard_map).unwrap();
        let program = shard_map_program(operation, body).unwrap().with_outputs(&[]).unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap().to_string(),
            formatdoc! {"
                lambda  .
                let () = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[],
                    out_shardings=[],
                    manual_axes=['x'],
                    global_input_types=[],
                    global_output_types=[],
                ] [
                    body={{
                        lambda  .
                        let %0:{scalar} = zero [type={scalar}]
                            %1:{scalar} = print [label=cotangent, effect_class=device_ordered_io] %0
                        in ()
                    }},
                ]
                    %0:{scalar} = zero [type={scalar}]
                in (%0)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_keeps_known_residual_shardings() {
        // A tangent `shard_map` whose residual is a forwarded primal output under an output sharding that is unreduced
        // along the explicit axis `y` keeps that residual's sharding on the transposed boundary, where it is a known
        // input, while the linear cotangents cross under the cotangent duals of their shardings (reduced along `y`).
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let unreduced = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_unreduced_axes(["y"])
            .unwrap();
        let global_type = f32_vector_type(4).with_sharding(unreduced.clone()).unwrap();
        let shard_map = ShardMap::new(mesh, vec![unreduced.clone(); 2], vec![unreduced.clone()], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &global_type).unwrap();
        let local_cotangent_type = local_type.cotangent().unwrap();
        let source = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let tangent = builder.add_input(local_type.clone().into());
            let _residual = builder.add_input(local_type.clone().into());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![tangent], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let transposed = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let cotangent = builder.add_input(local_cotangent_type.clone().into());
            let _residual = builder.add_input(local_type.clone().into());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![cotangent], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let driver = TestTranspositionDriver { source, transposed };
        let operation = ShardMapOperation::from_boundary(
            shard_map,
            vec![global_type.clone(), global_type.clone()],
            vec![global_type.clone()],
        );
        let mut context = TestContext::new();
        let output_cotangent = context.input(ArrayIrType::Array(global_type.cotangent().unwrap()));
        let residual = context.input(ArrayIrType::Array(global_type.clone()));
        let cotangents = transpose_primal_shard_map(
            &operation,
            &mut context,
            &driver,
            &[PartialValue::Unknown(ArrayIrType::Array(global_type.clone())), PartialValue::Known(residual)],
            &[MaybeZero::Value(output_cotangent)],
            &CotangentDestinations::without_references([true, false]),
        )
        .unwrap();
        let [MaybeZero::Value(cotangent), MaybeZero::Zero(_)] = &cotangents[..] else {
            panic!("expected one materialized input cotangent followed by the known residual's zero");
        };
        let pullback = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![cotangent.atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let cotangent_type = global_type.cotangent().unwrap();
        let cotangent_sharding = unreduced.cotangent();
        assert_eq!(
            pullback.to_string(),
            formatdoc! {"
                lambda %0:{cotangent_type}, %1:{global_type} .
                let %2:{cotangent_type} = shard_map [
                    mesh=['x'=2:manual, 'y'=2:explicit],
                    in_shardings=[{cotangent_sharding}, {unreduced}],
                    out_shardings=[{cotangent_sharding}],
                    manual_axes=['x'],
                    global_input_types=[{cotangent_type}, {global_type}],
                    global_output_types=[{cotangent_type}],
                ] %0 %1 [
                    body={{
                        lambda %0:{local_cotangent_type}, %1:{local_type} .
                        in (%0)
                    }},
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_shard_map_transposition_threads_cotangent_references() {
        // Reverse mode threads a cotangent reference at the position of a live linear reference input under the
        // primal's input sharding: the transposed shard map consumes `[ȳ, r̄]`, accumulates into `r̄` in place, returns
        // it by identity ahead of `x̄`, and its body is exactly the transposition of the local body.
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded_along_x(), true);
        let program = shard_map_program(operation, body.clone()).unwrap();
        let destination_kinds = [CotangentDestinationKind::Reference, CotangentDestinationKind::Return];
        let transposed = program.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[4], %1:ref<f32[4]> .
                let %2:ref<f32[4]>, %3:f32[4] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4], ref<f32[4]>],
                    global_output_types=[ref<f32[4]>, f32[4]],
                    output_forwarding=[1, _],
                ] %0 %1 [
                    body={
                        lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                            %1:ref<f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]> .
                        let () = reference_add_update %1 %0
                            %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reference_read %1
                        in (%1, %2)
                    },
                ]
                in (%1, %3)"},
        );
        let transposed_body = transposed.region_ref(transposed.instructions()[0].regions()[0]).unwrap().to_program();
        let inlined = body.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(transposed_body.to_string(), inlined.to_string());
    }

    #[test]
    fn test_shard_map_function() {
        // This is the primary test of the free function `shard_map`. Its conventional name, `test_shard_map`, is the
        // primary test of the operation itself (i.e., of `ShardMapOperation`, whose tests are named after the
        // operation), so the function's tests use the `test_shard_map_function` prefix instead.

        // Invoking a shard map on tracers of an enclosing trace binds the checked operation in that trace, with the
        // traced local body attached as its region, and rebuilds the structured outputs from the operation's outputs.
        // The replicated input enters the body invariant, so combining it with the varying local shard aligns it first.
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let program = trace_test_program(
            |inputs| {
                let (sum, product) = shard_map(
                    |(lhs, rhs): (TestTracer, TestTracer)| (lhs.clone() + rhs.clone(), lhs * rhs),
                    (inputs[0].clone(), inputs[1].clone()),
                    mesh.clone(),
                    (sharded.clone(), replicated.clone()),
                    (sharded.clone(), sharded.clone()),
                )
                .unwrap();
                vec![sum, product]
            },
            vec![f32_vector_type(4), f32_vector_type(2)],
            Vec::new(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[2] .
                let %2:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}], \
                    %3:f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}], \
                        f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]],
                    global_output_types=[f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}], \
                        f32[4][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ] %0 %1 [
                    body={
                        lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                            %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] .
                        let %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                                parallel_vary [axis_name=\"x\"] %1
                            %3:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = add %0 %2
                            %4:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                                parallel_vary [axis_name=\"x\"] %1
                            %5:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %0 %4
                        in (%3, %5)
                    },
                ]
                in (%2, %3)"
            },
        );
    }

    #[test]
    fn test_shard_map_function_rejects_mesh_without_manual_axes() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        trace_test_program(
            |inputs| {
                let result =
                    shard_map(|x: TestTracer| x, inputs[0].clone(), mesh.clone(), sharded.clone(), sharded.clone());
                assert_eq!(result, Err(ShardMapError::MeshHasNoManualAxes));
                assert_eq!(
                    result.unwrap_err().to_string(),
                    "`shard_map` requires at least one mesh axis with type `manual`",
                );
                inputs
            },
            vec![f32_vector_type(4)],
            Vec::new(),
        );
    }

    #[test]
    fn test_shard_map_function_rejects_collectives_over_unbound_axes() {
        // The body binds only the active manual axis `x`, so a collective over `y` fails while the body is traced
        // instead of tracing as a silent identity.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        trace_test_program(
            |inputs| {
                vec![
                    shard_map(
                        |x: TestTracer| {
                            assert_eq!(
                                x.parallel_reduce(ReductionKind::Sum, "y").unwrap_err(),
                                ProgramError::Axis(AxisError::UnboundAxisName { name: "y".to_string() }),
                            );
                            x
                        },
                        inputs[0].clone(),
                        mesh.clone(),
                        sharded.clone(),
                        sharded.clone(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_vector_type(8)],
            Vec::new(),
        );
    }

    #[test]
    fn test_shard_map_function_without_inputs() {
        // Without inputs, no input supplies the invocation context. An invocation without outputs still traces its
        // body, which validates it, and returns an empty result. A body traced without input values and without a
        // context cannot create any value, so it cannot have effects either (`shard_map_in_context` binds bodies
        // without inputs in an explicit context instead).
        let mesh = manual_mesh();
        let traced = Cell::new(false);
        let result = shard_map(
            |inputs: Vec<TestTracer>| -> Vec<TestTracer> {
                traced.set(true);
                inputs
            },
            Vec::<TestTracer>::new(),
            mesh.clone(),
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(result, Ok(Vec::new()));
        assert!(traced.get());

        // The body is traced over the boundary that it declares, so an invalid boundary is reported, not ignored.
        let result = shard_map(
            |_: Vec<TestTracer>| -> Vec<TestTracer> { unreachable!("an invalid boundary is never traced") },
            Vec::<TestTracer>::new(),
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(result, Err(ShardMapError::MeshHasNoManualAxes));

        // An invocation with outputs cannot bind them without a context, so it fails before tracing its body.
        let result = shard_map(
            |_: Vec<TestTracer>| -> Vec<TestTracer> { unreachable!("an invocation without a context is never traced") },
            Vec::<TestTracer>::new(),
            mesh.clone(),
            Vec::new(),
            vec![Sharding::replicated(mesh.clone(), 1)],
        );
        assert_eq!(result, Err(ShardMapError::MissingTracedInvocationDomain));
        assert_eq!(
            result.unwrap_err().to_string(),
            "`shard_map` with non-empty outputs requires at least one input leaf; use `shard_map_in_context` to \
             provide the context explicitly",
        );

        // The input specifications must still match the inputs, even though no input supplies a context.
        let result = shard_map(
            |_: Vec<TestTracer>| -> Vec<TestTracer> { unreachable!("an invalid invocation is never traced") },
            Vec::<TestTracer>::new(),
            mesh.clone(),
            vec![Sharding::replicated(mesh, 1)],
            Vec::new(),
        );
        assert_eq!(result, Err(ShardMapError::InputTypeCountMismatch { expected: 1, actual: 0 }));
        assert_eq!(result.unwrap_err().to_string(), "got 0 global input type(s), but `shard_map` expects 1");
    }

    #[test]
    fn test_shard_map_with_options() {
        // A shard map over a manual-axis subset nests inside the body of an enclosing shard map over the other axis.
        // The inner body inherits the enclosing binding of `x` and binds its own active manual axis `y`, and the
        // traced outer body attaches the inner body as the region of its nested `shard_map` instruction.
        let mesh = manual_mesh_2x2();
        let outer_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let inner_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let inner_named_axes = RefCell::new(None);
        let traced: TracedShardMap<TestEagerContext, ArrayType, ArrayType> = trace_shard_map_with_options(
            |x: TestTracer| {
                let nested = shard_map_with_options(
                    |y: TestTracer| {
                        inner_named_axes.replace(Some(y.value().context().named_axes()));
                        y.clone() + y
                    },
                    x.clone(),
                    mesh.clone(),
                    inner_sharding.clone(),
                    inner_sharding.clone(),
                    vec!["y".to_string()],
                )
                .unwrap();
                nested + x
            },
            f32_vector_type(8),
            mesh.clone(),
            outer_sharding.clone(),
            outer_sharding,
            vec!["x".to_string()],
        )
        .unwrap();
        assert_eq!(
            inner_named_axes.into_inner(),
            Some(vec![
                ("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 }),
                ("y".to_string(), NamedAxis::Mesh { mesh, axis: 1, size: 2 }),
            ]),
        );
        assert_eq!(
            traced.body().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}], varying_manual={'x'}}] = \
                        shard_map [
                    mesh=['x'=2:manual, 'y'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}]}],
                    out_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}]}],
                    manual_axes=['y'],
                    global_input_types=[\
                        f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}], varying_manual={'x'}}]],
                    global_output_types=[\
                        f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}], varying_manual={'x'}}]],
                ] %0 [
                    body={
                        lambda \
                            %0:f32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] .
                        let \
                            %1:f32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] \
                            = add %0 %0
                        in (%1)
                    },
                ]
                    %2:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}], varying_manual={'x'}}] = add %1 %0
                in (%2)"
            },
        );
    }

    #[test]
    fn test_shard_map_with_options_inherits_enclosing_named_axes() {
        // The body inherits exactly the enclosing mesh bindings whose names are axes of the shard map's mesh and the
        // enclosing batched bindings whose names are not, together with its own active manual axes: `x` is inherited,
        // `items` is the axis of an enclosing `batch` level, `y` is the shard map's own, the batched `z` is named like
        // an axis of the mesh, and `w` is not an axis of the mesh.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let other_mesh = LogicalMesh::new(vec![
            MeshAxis::new("w", 4, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 4, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let body_named_axes = RefCell::new(None);
        trace_test_program(
            |inputs| {
                vec![
                    shard_map_with_options(
                        |x: TestTracer| {
                            body_named_axes.replace(Some(x.value().context().named_axes()));
                            x
                        },
                        inputs[0].clone(),
                        mesh.clone(),
                        replicated.clone(),
                        replicated.clone(),
                        vec!["y".to_string()],
                    )
                    .unwrap(),
                ]
            },
            vec![f32_vector_type(4)],
            vec![
                ("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 }),
                ("items".to_string(), NamedAxis::Batched { size: Some(3) }),
                ("z".to_string(), NamedAxis::Batched { size: Some(2) }),
                ("w".to_string(), NamedAxis::Mesh { mesh: other_mesh, axis: 0, size: 4 }),
            ],
        );
        assert_eq!(
            body_named_axes.into_inner(),
            Some(vec![
                ("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 }),
                ("items".to_string(), NamedAxis::Batched { size: Some(3) }),
                ("y".to_string(), NamedAxis::Mesh { mesh, axis: 1, size: 2 }),
            ]),
        );
    }

    #[test]
    fn test_shard_map_with_options_excludes_axes_manual_in_enclosing_regions() {
        // Inside a manual region over `x`, an empty manual-axis selection selects only the remaining manual axis `y`,
        // so the nested map keeps the variation of its input along `x`, and its output still varies along `x`.
        let mesh = manual_mesh_2x2();
        let outer_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let inner_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let traced: TracedShardMap<TestEagerContext, ArrayType, ArrayType> = trace_shard_map_with_options(
            |x: TestTracer| {
                shard_map(
                    |y: TestTracer| y.clone() + y,
                    x,
                    mesh.clone(),
                    inner_sharding.clone(),
                    inner_sharding.clone(),
                )
                .unwrap()
            },
            f32_vector_type(8),
            mesh.clone(),
            outer_sharding.clone(),
            outer_sharding.clone(),
            vec!["x".to_string()],
        )
        .unwrap();
        assert_eq!(
            traced.body().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}], varying_manual={'x'}}] = \
                        shard_map [
                    mesh=['x'=2:manual, 'y'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}]}],
                    out_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}]}],
                    manual_axes=['y'],
                    global_input_types=[\
                        f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}], varying_manual={'x'}}]],
                    global_output_types=[\
                        f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}], varying_manual={'x'}}]],
                ] %0 [
                    body={
                        lambda \
                            %0:f32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] .
                        let \
                            %1:f32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] \
                            = add %0 %0
                        in (%1)
                    },
                ]
                in (%1)"
            },
        );

        // Specifications over the already manual axis `x` are rejected instead of making `x` manual again, which would
        // drop the variation of the input along `x` and type the per-device output as replicated.
        let traced: TracedShardMap<TestEagerContext, ArrayType, ArrayType> = trace_shard_map_with_options(
            |x: TestTracer| {
                let result = shard_map(
                    |x: TestTracer| x.clone() + x,
                    x.clone(),
                    mesh.clone(),
                    outer_sharding.clone(),
                    outer_sharding.clone(),
                );
                let expected = ShardMapError::SpecificationNamesEnclosingManualAxis {
                    value_kind: "input",
                    value_index: 0,
                    axis_name: "x".to_string(),
                };
                assert_eq!(result, Err(expected));
                assert_eq!(
                    result.unwrap_err().to_string(),
                    "input sharding #0 names mesh axis `x`, which is already manual in an enclosing manual region",
                );
                x
            },
            f32_vector_type(8),
            mesh.clone(),
            outer_sharding.clone(),
            outer_sharding.clone(),
            vec!["x".to_string()],
        )
        .unwrap();
        assert_eq!(traced.body().instructions().len(), 0);

        // A mesh whose manual axes are all already manual leaves no axis for the nested map.
        let mesh = manual_mesh();
        let sharding = sharded_along_x();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let _: TracedShardMap<TestEagerContext, ArrayType, ArrayType> = trace_shard_map(
            |x: TestTracer| {
                let result =
                    shard_map(|x: TestTracer| x, x.clone(), mesh.clone(), replicated.clone(), replicated.clone());
                assert_eq!(result, Err(ShardMapError::AllManualAxesAlreadyManual));
                assert_eq!(
                    result.unwrap_err().to_string(),
                    "every manual mesh axis is already manual in an enclosing manual region, so `shard_map` has no \
                     mesh axis left to make manual",
                );
                x
            },
            f32_vector_type(8),
            mesh.clone(),
            sharding.clone(),
            sharding,
        )
        .unwrap();
    }

    #[test]
    fn test_shard_map_with_options_rejects_axes_manual_in_enclosing_regions() {
        // Explicitly requesting an axis that an enclosing manual region already made manual is rejected. (An enclosing
        // binding of that name over another device mesh is rejected as a mesh mismatch instead; refer to
        // `test_shard_map_with_options_rejects_meshes_that_differ_from_enclosing_manual_meshes`.)
        let mesh = manual_mesh_2x2();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        trace_test_program(
            |inputs| {
                for axis_name in ["x", "y"] {
                    let result = shard_map_with_options(
                        |x: TestTracer| x,
                        inputs[0].clone(),
                        mesh.clone(),
                        replicated.clone(),
                        replicated.clone(),
                        vec![axis_name.to_string()],
                    );
                    assert_eq!(result, Err(ShardMapError::AxisAlreadyManual { axis_name: axis_name.to_string() }));
                }
                let result = shard_map_with_options(
                    |x: TestTracer| x,
                    inputs[0].clone(),
                    mesh.clone(),
                    replicated.clone(),
                    replicated.clone(),
                    vec!["x".to_string()],
                );
                assert_eq!(
                    result.unwrap_err().to_string(),
                    "mesh axis `x` is already manual in an enclosing manual region, so `shard_map` cannot make it \
                     manual again",
                );
                inputs
            },
            vec![f32_vector_type(4)],
            vec![
                ("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 }),
                ("y".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 1, size: 2 }),
            ],
        );
    }

    #[test]
    fn test_shard_map_with_options_rejects_meshes_that_retype_enclosing_manual_axes() {
        // Inside a manual region over `x`, a nested map whose mesh types `x` as `auto` or `explicit` is rejected up
        // front, whether it selects its manual axes implicitly or explicitly, instead of failing later while deriving
        // its boundary types. The nested map must use the enclosing mesh, in which `x` is still `manual`.
        let mesh = manual_mesh_2x2();
        let auto_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let explicit_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        trace_test_program(
            |inputs| {
                let result = shard_map_with_options(
                    |x: TestTracer| x,
                    inputs[0].clone(),
                    auto_mesh.clone(),
                    Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap(),
                    Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap(),
                    Vec::new(),
                );
                assert_eq!(result, Err(ShardMapError::EnclosingManualAxisNotManual { axis_name: "x".to_string() }));
                assert_eq!(
                    result.unwrap_err().to_string(),
                    "mesh axis `x` is manual in an enclosing manual region, but the `shard_map` mesh does not type it \
                     `manual`",
                );
                let result = shard_map_with_options(
                    |x: TestTracer| x,
                    inputs[0].clone(),
                    explicit_mesh.clone(),
                    Sharding::replicated(explicit_mesh.clone(), 1),
                    Sharding::replicated(explicit_mesh.clone(), 1),
                    vec!["y".to_string()],
                );
                assert_eq!(result, Err(ShardMapError::EnclosingManualAxisNotManual { axis_name: "x".to_string() }));
                inputs
            },
            vec![f32_vector_type(4)],
            vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2 })],
        );
    }

    #[test]
    fn test_shard_map_with_options_rejects_meshes_that_differ_from_enclosing_manual_meshes() {
        // Inside a manual region over the axis `x` of one device mesh, a nested map over another device mesh that also
        // has an axis named `x` (here, with another sibling axis or another size) is rejected, because names alone
        // cannot tell whether its `x` is the enclosing manual axis. A nested map over a mesh without an axis named
        // `x` does not see the enclosing binding at all and is accepted.
        let mesh = manual_mesh_2x2();
        let sibling_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let resized_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let unrelated_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        trace_test_program(
            |inputs| {
                let result = shard_map_with_options(
                    |x: TestTracer| x,
                    inputs[0].clone(),
                    sibling_mesh.clone(),
                    Sharding::new(sibling_mesh.clone(), vec![ShardingDimension::sharded(["z"])]).unwrap(),
                    Sharding::new(sibling_mesh.clone(), vec![ShardingDimension::sharded(["z"])]).unwrap(),
                    Vec::new(),
                );
                assert_eq!(
                    result,
                    Err(ShardMapError::EnclosingManualAxisMeshMismatch {
                        axis_name: "x".to_string(),
                        enclosing_mesh: mesh.clone(),
                        mesh: sibling_mesh.clone(),
                    }),
                );
                assert_eq!(
                    result.unwrap_err().to_string(),
                    "mesh axis `x` is manual in an enclosing manual region over mesh `['x'=2:manual, 'y'=2:manual]`, \
                     which differs from the `shard_map` mesh `['x'=2:manual, 'z'=2:manual]`",
                );
                let result = shard_map_with_options(
                    |x: TestTracer| x,
                    inputs[0].clone(),
                    resized_mesh.clone(),
                    Sharding::replicated(resized_mesh.clone(), 1),
                    Sharding::replicated(resized_mesh.clone(), 1),
                    Vec::new(),
                );
                assert_eq!(
                    result,
                    Err(ShardMapError::EnclosingManualAxisMeshMismatch {
                        axis_name: "x".to_string(),
                        enclosing_mesh: mesh.clone(),
                        mesh: resized_mesh.clone(),
                    }),
                );
                vec![
                    shard_map_with_options(
                        |x: TestTracer| x,
                        inputs[0].clone(),
                        unrelated_mesh.clone(),
                        Sharding::replicated(unrelated_mesh.clone(), 1),
                        Sharding::replicated(unrelated_mesh.clone(), 1),
                        Vec::new(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_vector_type(4)],
            vec![("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 })],
        );
    }

    #[test]
    fn test_shard_map_with_options_rejects_outputs_with_other_structures() {
        // Outputs with as many leaves as the output specifications but a different structure are rejected instead of
        // being silently restructured into the structure of the specifications.
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        trace_test_program(
            |inputs| {
                let result = shard_map(
                    |x: TestTracer| vec![vec![x.clone()], vec![x.clone(), x]],
                    inputs[0].clone(),
                    mesh.clone(),
                    replicated.clone(),
                    vec![vec![replicated.clone(), replicated.clone()], vec![replicated.clone()]],
                );
                assert_eq!(
                    result,
                    Err(ShardMapError::Parameter(ParameterError::MismatchedParameterStructures {
                        left_structure: "[$[0][0], $[0][1], $[1][0]]".to_string(),
                        right_structure: "[$[0][0], $[1][0], $[1][1]]".to_string(),
                    })),
                );
                assert_eq!(
                    result.unwrap_err().to_string(),
                    "mismatched parameter structures: [$[0][0], $[0][1], $[1][0]] and [$[0][0], $[1][0], $[1][1]]",
                );
                inputs
            },
            vec![f32_vector_type(4)],
            Vec::new(),
        );
    }

    #[test]
    fn test_shard_map_in_context() {
        // A body without inputs receives its context, through which it reads the coordinate of each device along `x`.
        // The `shard_map` is bound in the provided context without inputs, and its output, which the output sharding
        // tiles along `x`, assembles the coordinates of the two devices (JAX's `test_axis_index`).
        assert_eq!(
            axis_index_shard_map_program().to_string(),
            indoc! {"
                lambda  .
                let %0:u64[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}] = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[],
                    global_output_types=[u64[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ] [
                    body={
                        lambda  .
                        let %0:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = axis_index \
                            [axis_name=\"x\", mesh=['x'=2:manual]]
                            %1:u64[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reshape \
                                [shape=[1]] %0
                        in (%1)
                    },
                ]
                in (%0)"
            },
        );
    }

    #[test]
    fn test_shard_map_in_context_rejects_inputs_of_other_contexts() {
        // The operation is bound in the provided context, which rejects an input that is a tracer of another trace.
        let input_context = TestContext::new();
        let input = input_context.input(ArrayIrType::Array(f32_vector_type(4)));
        let input = ValueProjection::<ArrayType>::into_projected(input).unwrap();
        let sharded = sharded_along_x();
        let result = shard_map_in_context(
            &TestContext::new(),
            |_: &ShardMapContext<TestContext>, x: TestTracer| x,
            input,
            manual_mesh(),
            sharded.clone(),
            sharded,
            Vec::new(),
        );
        let error = result.map(|_| ()).unwrap_err();
        assert_eq!(error, ShardMapError::Program(ProgramError::MismatchedProgramBuilders));
        assert_eq!(error.to_string(), "values used in the same operation must share the same program builder");
    }

    #[test]
    fn test_shard_map_in_context_inherits_enclosing_manual_axes() {
        // A body without inputs inherits the enclosing binding of `x`, through which it reads its coordinate along the
        // enclosing manual axis, while the empty manual-axis selection selects only the remaining manual axis `y`. The
        // output therefore varies along `x` and gains the variation along `y` that its output sharding tiles.
        let mesh = manual_mesh_2x2();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let body_named_axes = RefCell::new(None);
        let program = trace_test_program(
            |inputs| {
                vec![
                    shard_map_in_context(
                        inputs[0].value().context(),
                        |context: &ShardMapContext<TestContext>, ()| {
                            body_named_axes.replace(Some(context.named_axes()));
                            context.axis_index("x").unwrap().reshape([1]).unwrap()
                        },
                        (),
                        mesh.clone(),
                        (),
                        sharding.clone(),
                        Vec::new(),
                    )
                    .unwrap(),
                ]
            },
            vec![f32_scalar_type()],
            vec![("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 })],
        );
        assert_eq!(
            body_named_axes.into_inner(),
            Some(vec![
                ("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 }),
                ("y".to_string(), NamedAxis::Mesh { mesh, axis: 1, size: 2 }),
            ]),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:u64[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}], varying_manual={'x'}}] = \
                    shard_map [
                    mesh=['x'=2:manual, 'y'=2:manual],
                    in_shardings=[],
                    out_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}]}],
                    manual_axes=['y'],
                    global_input_types=[],
                    global_output_types=[u64[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}], \
                        varying_manual={'x'}}]],
                ] [
                    body={
                        lambda  .
                        let %0:u64[][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [], varying_manual={'x'}}] = \
                            axis_index [axis_name=\"x\", mesh=['x'=2:manual, 'y'=2:manual]]
                            %1:u64[1][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x'}}] = \
                                reshape [shape=[1]] %0
                            %2:u64[1][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] \
                                = parallel_vary [axis_name=\"y\"] %1
                        in (%2)
                    },
                ]
                in (%1)"
            },
        );
    }

    #[test]
    fn test_shard_map_in_context_keeps_effects_of_bodies_without_inputs() {
        // A body without inputs and outputs can still stage effects through its context, so the `shard_map` is bound
        // without inputs and outputs and its effect survives.
        let context = TestContext::new();
        shard_map_in_context(
            &context,
            |context: &ShardMapContext<TestContext>, ()| {
                context
                    .axis_index("x")
                    .unwrap()
                    .print_with_effect_class("index", EffectClass::DeviceOrderedIo)
                    .unwrap();
            },
            (),
            manual_mesh(),
            (),
            (),
            Vec::new(),
        )
        .unwrap();
        let builder = context.builder().borrow().clone();
        let program = builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), Vec::new(), Vec::new()).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let () = shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[],
                    out_shardings=[],
                    manual_axes=['x'],
                    global_input_types=[],
                    global_output_types=[],
                ] [
                    body={
                        lambda  .
                        let %0:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = axis_index \
                            [axis_name=\"x\", mesh=['x'=2:manual]]
                            %1:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = print [label=index, \
                                effect_class=device_ordered_io] %0
                        in ()
                    },
                ]
                in ()"
            },
        );
    }

    #[test]
    fn test_trace_shard_map() {
        // Descriptor-only tracing names its domain explicitly, here through the annotated result, which also lets the
        // body closure infer the type of its input. The traced boundary derives the global and local types of both
        // sides and checks the simplified body.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedShardMap<TestEagerContext, ArrayType, ArrayType> =
            trace_shard_map(|x| x.clone() + x, f32_vector_type(8), mesh.clone(), sharded.clone(), sharded.clone())
                .unwrap();
        let global_type = f32_vector_type(8).with_sharding(sharded.clone()).unwrap();
        let local_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(traced.global_input_types(), &global_type);
        assert_eq!(traced.local_input_types(), &local_type);
        assert_eq!(traced.local_output_types(), &local_type);
        assert_eq!(traced.global_output_types(), &global_type);
        assert_eq!(
            traced.operation().shard_map(),
            &ShardMap::new(mesh, vec![sharded.clone()], vec![sharded], Vec::new()).unwrap(),
        );
        assert_eq!(traced.operation().global_input_types(), &[ArrayIrType::Array(global_type.clone())]);
        assert_eq!(traced.operation().global_output_types(), &[ArrayIrType::Array(global_type)]);
        assert_eq!(
            traced.body().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=4:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[2][sharding={mesh<['x'=4:manual]>, [{}], varying_manual={'x'}}] = add %0 %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_trace_shard_map_inserts_output_variation() {
        // An invariant body output whose output sharding tiles an active manual axis is marked as varying along it.
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let sharded = sharded_along_x();
        let traced: TracedShardMap<TestEagerContext, ArrayType, ArrayType> =
            trace_shard_map(|x| x, f32_vector_type(1), mesh, replicated.clone(), sharded.clone()).unwrap();
        assert_eq!(
            traced.local_output_types(),
            &f32_vector_type(1).with_sharding(replicated.with_varying_manual_axes(["x"]).unwrap()).unwrap(),
        );
        assert_eq!(traced.global_output_types(), &f32_vector_type(2).with_sharding(sharded).unwrap());
        assert_eq!(
            traced.body().to_string(),
            indoc! {"
                lambda %0:f32[1][sharding={mesh<['x'=2:manual]>, [{}]}] .
                let %1:f32[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_trace_shard_map_hides_auto_axes_in_boundary_types() {
        // Auto axes are placed by backend compilers, so the boundary types carry the specifications without them.
        let mesh = data_model_mesh();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data", "model"])]).unwrap();
        let traced: TracedShardMap<TestEagerContext, ArrayType, ArrayType> =
            trace_shard_map(|x| x.clone() + x, f32_vector_type(16), mesh.clone(), sharding.clone(), sharding.clone())
                .unwrap();
        let global_type = f32_vector_type(16).with_sharding(sharding.without_auto_axes()).unwrap();
        assert_eq!(traced.global_input_types(), &global_type);
        assert_eq!(
            traced.local_input_types(),
            &f32_vector_type(8)
                .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["data"]).unwrap())
                .unwrap(),
        );
        assert_eq!(traced.global_output_types(), &global_type);
    }

    #[test]
    fn test_trace_shard_map_structured_inputs() {
        // Structured inputs and outputs keep their structure on both sides of the boundary, and the body receives its
        // local inputs with the structure of the global input types.
        let mesh = manual_mesh();
        let sharding = sharded_along_x();
        let input_types = vec![
            ArrayType::new_static(DataType::F32, [6]),
            ArrayType::new_static(DataType::F32, [8]),
            ArrayType::new_static(DataType::I32, [4]),
            ArrayType::new_static(DataType::I32, [4]),
            ArrayType::new_static(DataType::I32, [4]),
            ArrayType::new_static(DataType::I32, [4]),
        ];
        let traced: TracedShardMap<TestEagerContext, Vec<ArrayType>, ArrayType> = trace_shard_map(
            |inputs: Vec<TestTracer>| {
                inputs[0]
                    .parallel_ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
                    .unwrap()
            },
            input_types.clone(),
            mesh.clone(),
            vec![sharding.clone(); 6],
            sharding.clone(),
        )
        .unwrap();
        assert_eq!(
            traced.global_input_types(),
            &input_types
                .into_iter()
                .map(|input_type| input_type.with_sharding(sharding.clone()).unwrap())
                .collect::<Vec<_>>(),
        );
        let local_sharding = Sharding::replicated(mesh, 1).with_varying_manual_axes(["x"]).unwrap();
        assert_eq!(
            traced.local_input_types(),
            &vec![
                ArrayType::new_static(DataType::F32, [3]).with_sharding(local_sharding.clone()).unwrap(),
                ArrayType::new_static(DataType::F32, [4]).with_sharding(local_sharding.clone()).unwrap(),
                ArrayType::new_static(DataType::I32, [2]).with_sharding(local_sharding.clone()).unwrap(),
                ArrayType::new_static(DataType::I32, [2]).with_sharding(local_sharding.clone()).unwrap(),
                ArrayType::new_static(DataType::I32, [2]).with_sharding(local_sharding.clone()).unwrap(),
                ArrayType::new_static(DataType::I32, [2]).with_sharding(local_sharding).unwrap(),
            ],
        );
        assert_eq!(traced.global_output_types(), &f32_vector_type(8).with_sharding(sharding).unwrap());
        assert_eq!(
            traced.body().to_string(),
            indoc! {"
                lambda \
                %0:f32[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                %1:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                %2:i32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                %3:i32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                %4:i32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                %5:i32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                let \
                %6:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                parallel_ragged_all_to_all [axis_name=\"x\", axis_size=2, mesh=['x'=2:manual]] %0 %1 %2 %3 %4 %5
                in (%6)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_trace_shard_map_rejects_dynamic_input_types() {
        let dynamic_input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("dynamic", DimensionBounds::unbounded()).into()]),
        );
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let result: Result<TracedShardMap<TestEagerContext, ArrayType, ArrayType>, _> =
            trace_shard_map(|x| x, dynamic_input_type, mesh, sharded.clone(), sharded);
        let error = result.map(|_| ()).unwrap_err();
        assert_eq!(
            error,
            ShardMapError::DynamicShapeNotSupported { value_kind: "input", value_index: 0, dimension: 0 },
        );
        assert_eq!(error.to_string(), "input type #0 dimension #0 must be static at a `shard_map` boundary");
    }

    #[test]
    fn test_trace_shard_map_rejects_ordered_io_bodies() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // `OrderedIo` demands one order across devices, which independent per-device execution cannot provide.
        let result: Result<TracedShardMap<TestEagerContext, ArrayType, ArrayType>, _> = trace_shard_map(
            |x: TestTracer| x.print("body").unwrap(),
            f32_vector_type(8),
            mesh.clone(),
            sharded.clone(),
            sharded.clone(),
        );
        let error = result.map(|_| ()).unwrap_err();
        assert_eq!(error, ShardMapError::OrderedIoNotSupported);
        assert_eq!(
            error.to_string(),
            "`shard_map` bodies require `DeviceOrderedIo` because `OrderedIo` demands one order across devices",
        );

        // `DeviceOrderedIo` only orders effects on each device, which a manual computation preserves.
        let traced: TracedShardMap<TestEagerContext, ArrayType, ArrayType> = trace_shard_map(
            |x: TestTracer| x.print_with_effect_class("body", EffectClass::DeviceOrderedIo).unwrap(),
            f32_vector_type(8),
            mesh,
            sharded.clone(),
            sharded,
        )
        .unwrap();
        assert_eq!(
            traced.body().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=4:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[2][sharding={mesh<['x'=4:manual]>, [{}], varying_manual={'x'}}] = print [label=body, \
                    effect_class=device_ordered_io] %0
                in (%1)"},
        );
    }

    #[test]
    fn test_trace_shard_map_rejects_captures_registered_in_its_body() {
        // The body is traced in a fresh trace whose capture table is discarded, so registering a capture rejects the
        // body regardless of whether the returned reference is staged (refer to
        // `TracingContext::trace_with_named_axes` for the silent-aliasing rationale). The composite test family does
        // not embed capture references, so this test traces over a constant family that does.
        type CapturingDomain = TracingContext<CaptureReference<ArrayIrType>, TestOperation>;
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        let result: Result<TracedShardMap<CapturingDomain, ArrayType, ArrayType>, _> = trace_shard_map(
            |x: ShardMapTracer<CapturingDomain>| {
                x.value()
                    .context()
                    .capture(CaptureReference::new(0, ArrayIrType::Array(f32_vector_type(2))))
                    .unwrap();
                x
            },
            f32_vector_type(4),
            mesh,
            sharded.clone(),
            sharded,
        );
        let error = result.map(|_| ()).unwrap_err();
        assert_eq!(error, ShardMapError::Program(ProgramError::DiscardedCaptures { count: 1 }));
        assert_eq!(
            error.to_string(),
            "trace registered 1 runtime capture(s) through a trace entry point that discards its capture table; \
             capture-owning traces must construct their `TracingContext` directly and pair the traced program with \
             that context's capture table",
        );
    }

    #[test]
    fn test_trace_shard_map_with_options() {
        // Tracing over the manual-axis subset `y` of the 2x2 mesh shrinks the local shard along `y` only and marks it
        // as varying along `y` only, while `x` stays a free axis whose placement the body does not see.
        let mesh = manual_mesh_2x2();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let traced: TracedShardMap<TestEagerContext, ArrayType, ArrayType> = trace_shard_map_with_options(
            |x| x.clone() + x,
            f32_vector_type(8),
            mesh,
            sharding.clone(),
            sharding,
            vec!["y".to_string()],
        )
        .unwrap();
        assert_eq!(
            traced.operation().to_string(),
            indoc! {"
                shard_map [
                    mesh=['x'=2:manual, 'y'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}]}],
                    out_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}]}],
                    manual_axes=['y'],
                    global_input_types=[f32[8][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}]}]],
                    global_output_types=[f32[8][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}]}]],
                ]"},
        );
        assert_eq!(
            traced.body().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'y'}}] .
                let %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'y'}}] = add %0 %0
                in (%1)"},
        );
    }

    #[test]
    fn test_trace_shard_map_with_named_axes() {
        // A body traced for the named-axis scope of an enclosing manual region over `x` inherits the binding of `x`, so
        // a collective over `x` resolves inside it, and the empty manual-axis selection selects only the remaining
        // manual axis `y`. The global input carries the variation along `x` of the enclosing region, which the
        // reduction over `x` removes from the output.
        let mesh = manual_mesh_2x2();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let input_type = f32_vector_type(4)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let body_named_axes = RefCell::new(None);
        let traced: TracedShardMap<TestEagerContext, ArrayType, ArrayType> = trace_shard_map_with_named_axes(
            |_, y: TestTracer| {
                body_named_axes.replace(Some(y.value().context().named_axes()));
                y.parallel_reduce(ReductionKind::Sum, "x").unwrap()
            },
            input_type,
            mesh.clone(),
            sharding.clone(),
            sharding,
            Vec::new(),
            vec![("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 })],
        )
        .unwrap();
        assert_eq!(
            body_named_axes.into_inner(),
            Some(vec![
                ("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 }),
                ("y".to_string(), NamedAxis::Mesh { mesh, axis: 1, size: 2 }),
            ]),
        );
        assert_eq!(
            traced.operation().to_string(),
            indoc! {"
                shard_map [
                    mesh=['x'=2:manual, 'y'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}]}],
                    out_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}]}],
                    manual_axes=['y'],
                    global_input_types=[f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}], \
                        varying_manual={'x'}}]],
                    global_output_types=[f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}]}]],
                ]"
            },
        );
        assert_eq!(
            traced.body().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] .
                let %1:f32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'y'}}] = \
                    parallel_reduce [kind=sum, axis_name=\"x\", mesh=['x'=2:manual, 'y'=2:manual]] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_trace_shard_map_with_named_axes_excludes_axes_manual_in_enclosing_regions() {
        // Inside the named-axis scope of a manual region over `x`, an empty manual-axis selection selects only the
        // remaining manual axis `y`, so the traced map keeps the variation of its input along `x`, and its output still
        // varies along `x`.
        let mesh = manual_mesh_2x2();
        let enclosing_named_axes = vec![("x".to_string(), NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 })];
        let outer_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let inner_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let input_type = f32_vector_type(4)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let traced: TracedShardMap<TestEagerContext, ArrayType, ArrayType> = trace_shard_map_with_named_axes(
            |_, y: TestTracer| y.clone() + y,
            input_type.clone(),
            mesh.clone(),
            inner_sharding.clone(),
            inner_sharding,
            Vec::new(),
            enclosing_named_axes.clone(),
        )
        .unwrap();
        assert_eq!(
            traced.operation().to_string(),
            indoc! {"
                shard_map [
                    mesh=['x'=2:manual, 'y'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}]}],
                    out_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}]}],
                    manual_axes=['y'],
                    global_input_types=[f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}], \
                        varying_manual={'x'}}]],
                    global_output_types=[f32[4][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{'y'}], \
                        varying_manual={'x'}}]],
                ]"
            },
        );
        assert_eq!(
            traced.body().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] .
                let %1:f32[2][sharding={mesh<['x'=2:manual, 'y'=2:manual]>, [{}], varying_manual={'x', 'y'}}] = add %0 \
                    %0
                in (%1)"
            },
        );

        // Specifications over the already manual axis `x` are rejected instead of making `x` manual again, which would
        // drop the variation of the input along `x` and type the per-device output as replicated.
        let result: Result<TracedShardMap<TestEagerContext, ArrayType, ArrayType>, _> = trace_shard_map_with_named_axes(
            |_, x: TestTracer| x.clone() + x,
            input_type,
            mesh.clone(),
            outer_sharding.clone(),
            outer_sharding,
            Vec::new(),
            enclosing_named_axes,
        );
        assert_eq!(
            result.map(|_| ()),
            Err(ShardMapError::SpecificationNamesEnclosingManualAxis {
                value_kind: "input",
                value_index: 0,
                axis_name: "x".to_string(),
            }),
        );

        // A mesh whose manual axes are all already manual leaves no axis for the traced map.
        let mesh = manual_mesh();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let result: Result<TracedShardMap<TestEagerContext, ArrayType, ArrayType>, _> = trace_shard_map_with_named_axes(
            |_, x: TestTracer| x,
            f32_vector_type(4),
            mesh.clone(),
            replicated.clone(),
            replicated,
            Vec::new(),
            vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2 })],
        );
        assert_eq!(result.map(|_| ()), Err(ShardMapError::AllManualAxesAlreadyManual));
    }

    #[test]
    fn test_trace_shard_map_with_named_axes_traces_bodies_without_inputs() {
        // A descriptor-only body without inputs creates its values through its body context, here the coordinate of
        // each device along `x`, which the output sharding tiles into the global vector of device coordinates (JAX's
        // `shard_map(lambda: axis_index('x'), in_specs=(), out_specs=P('x'))`).
        let traced: TracedShardMap<TestEagerContext, (), ArrayType> = trace_shard_map_with_named_axes(
            |context: &ShardMapContext<TestEagerContext>, ()| context.axis_index("x").unwrap().reshape([1]).unwrap(),
            (),
            manual_mesh(),
            (),
            sharded_along_x(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            traced.operation().to_string(),
            indoc! {"
                shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[],
                    global_output_types=[u64[2][sharding={mesh<['x'=2:manual]>, [{'x'}]}]],
                ]"},
        );
        assert_eq!(
            traced.body().to_string(),
            indoc! {"
                lambda  .
                let %0:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = axis_index \
                    [axis_name=\"x\", mesh=['x'=2:manual]]
                    %1:u64[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = reshape [shape=[1]] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_traced_shard_map_operation() {
        assert_eq!(
            traced_test_shard_map().operation().to_string(),
            indoc! {"
                shard_map [
                    mesh=['x'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{}]}],
                    out_shardings=[{mesh<['x'=2:manual]>, [{'x'}]}, {mesh<['x'=2:manual]>, [{}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[8][sharding={mesh<['x'=2:manual]>, [{'x'}]}], \
                    f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]],
                    global_output_types=[f32[8][sharding={mesh<['x'=2:manual]>, [{'x'}]}], \
                    f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]],
                ]"},
        );
    }

    #[test]
    fn test_traced_shard_map_global_input_types() {
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        assert_eq!(
            traced_test_shard_map().global_input_types(),
            &vec![
                f32_vector_type(8).with_sharding(sharded).unwrap(),
                f32_vector_type(2).with_sharding(Sharding::replicated(mesh, 1)).unwrap(),
            ],
        );
    }

    #[test]
    fn test_traced_shard_map_local_input_types() {
        // The sharded input shrinks along `x` and varies along it, while the replicated input keeps its shape.
        let replicated = Sharding::replicated(manual_mesh(), 1);
        assert_eq!(
            traced_test_shard_map().local_input_types(),
            &vec![
                f32_vector_type(4)
                    .with_sharding(replicated.clone().with_varying_manual_axes(["x"]).unwrap())
                    .unwrap(),
                f32_vector_type(2).with_sharding(replicated).unwrap(),
            ],
        );
    }

    #[test]
    fn test_traced_shard_map_global_output_types() {
        let mesh = manual_mesh();
        let sharded = sharded_along_x();
        assert_eq!(
            traced_test_shard_map().global_output_types(),
            &vec![
                f32_vector_type(8).with_sharding(sharded).unwrap(),
                f32_vector_type(2).with_sharding(Sharding::replicated(mesh, 1)).unwrap(),
            ],
        );
    }

    #[test]
    fn test_traced_shard_map_local_output_types() {
        let replicated = Sharding::replicated(manual_mesh(), 1);
        assert_eq!(
            traced_test_shard_map().local_output_types(),
            &vec![
                f32_vector_type(4)
                    .with_sharding(replicated.clone().with_varying_manual_axes(["x"]).unwrap())
                    .unwrap(),
                f32_vector_type(2).with_sharding(replicated).unwrap(),
            ],
        );
    }

    #[test]
    fn test_traced_shard_map_body() {
        assert_eq!(
            traced_test_shard_map().body().to_string(),
            indoc! {"
                lambda \
                %0:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] .
                let %2:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = add %0 %0
                in (%2, %1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_residual_boundary() {
        // A residual that varies along the active manual axis `y` crosses with its leading dimension tiled along `y`
        // ahead of its explicit placement over `z`, so that its local shard is the residual itself, while the variation
        // along the outer manual axis `x` and the unreduced axis `u` survive. A varying scalar is promoted to a
        // single-element vector first, and a replicated residual crosses as itself.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("u", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let local_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["z"])])
            .unwrap()
            .with_unreduced_axes(["u"])
            .unwrap()
            .with_varying_manual_axes(["x", "y"])
            .unwrap();
        let local_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(local_sharding.clone()).unwrap();
        let expected_sharding = local_sharding
            .with_dimensions(vec![ShardingDimension::sharded(["y", "z"])])
            .unwrap()
            .with_varying_manual_axes(["x"])
            .unwrap();
        let expected_type =
            ArrayType::new_static(DataType::F32, [12]).with_sharding(expected_sharding.clone()).unwrap();
        let shard_map = ShardMap::from_shardings(mesh.clone(), Vec::new(), Vec::new(), vec!["y".to_string()]);
        assert_eq!(residual_boundary(0, &local_type, &shard_map), Ok((expected_type.clone(), expected_sharding)));
        let in_sharding = expected_type.sharding().unwrap().clone();
        let boundary = ShardMap::from_shardings(mesh.clone(), vec![in_sharding], Vec::new(), vec!["y".to_string()]);
        assert_eq!(boundary.local_input_type(0, &expected_type), Ok(local_type));

        let scalar_sharding = Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["y"]).unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32).with_sharding(scalar_sharding).unwrap();
        let expected_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let expected_type = ArrayType::new_static(DataType::F32, [3]).with_sharding(expected_sharding.clone()).unwrap();
        assert_eq!(residual_boundary(1, &scalar_type, &shard_map), Ok((expected_type, expected_sharding)));
        assert_eq!(
            promoted_residual_type(&scalar_type, &shard_map).unwrap().to_string(),
            "f32[1][sharding={mesh<['x'=2:manual, 'y'=3:manual, 'z'=2:explicit, 'u'=2:explicit]>, [{}], \
             varying_manual={'y'}}]",
        );

        let replicated_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["z"])]).unwrap())
            .unwrap();
        let result = residual_boundary(2, &replicated_type, &shard_map);
        assert_eq!(result, Ok((replicated_type.clone(), replicated_type.sharding().unwrap().clone())));
    }

    #[test]
    fn test_residual_boundary_rejects_dynamic_residuals() {
        // A residual with a dynamic dimension cannot cross a static shard-map boundary.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(5)).unwrap());
        let local_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), extent.into()]));
        let shard_map = ShardMap::from_shardings(manual_mesh(), Vec::new(), Vec::new(), vec!["x".to_string()]);
        let result = residual_boundary(3, &local_type, &shard_map);
        assert_eq!(result, Err(ShardMapError::DynamicResidualNotSupported { residual_index: 3, dimension: 1 }));
        assert_eq!(
            result.unwrap_err().to_string(),
            "residual #3 dimension #1 must be static to cross a `shard_map` boundary; dynamically shaped \
             intermediates cannot be shared between split `shard_map` bodies",
        );
    }
}
