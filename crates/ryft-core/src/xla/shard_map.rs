//! Manual SPMD metadata for Shardy `sdy.manual_computation`.
//!
//! This module provides the tracing-backed `shard_map` surface for `ryft-core::xla`.
//!
//! The default public entry point is [`shard_map`], which stages a Rust closure over shard-local
//! tensor types derived from global [`ArrayType`] metadata, lowers the resulting `tracing_v2`
//! graph to StableHLO/Shardy MLIR, and returns a [`TracedShardMap`] handle for inspection and
//! lowering.
//!
//! Internally, the module keeps a small metadata model of JAX's `shard_map`: a logical mesh,
//! per-input and per-output shardings, the active set of manual mesh axes for the region, and the
//! `check_vma` mode. That internal metadata is responsible for:
//!
//! - validating manual-axis usage against a [`LogicalMesh`],
//! - deriving body-local shapes from global shapes, and
//! - rendering the Shardy attributes needed by `sdy.manual_computation`.
//!
//! [`TracedShardMap`] extends that metadata with a staged `tracing_v2` body graph. The traced path
//! can derive body-local input types from global input types, trace a Rust closure over those
//! local tensor types, and lower the resulting staged body to StableHLO/Shardy MLIR.
//!
//! # Relationship to existing sharding types
//!
//! The public [`shard_map`] function accepts structured mesh-bound [`Sharding`] values.
//!
//! By default the public [`shard_map`] helper treats every mesh axis whose type is
//! [`Manual`](crate::sharding::MeshAxisType::Manual) as manual for the
//! `sdy.manual_computation` region. The more configurable [`shard_map_with_options`] helper can
//! instead activate only a subset of those manual mesh axes, mirroring JAX's `axis_names`
//! parameter.
//!
//! This matters because the surrounding mesh still determines which axes are free versus manual
//! inside the region. When a mesh also contains free axes, `ShardMap` renders the corresponding
//! `in_shardings` / `out_shardings` dimensions as open so that Shardy can propagate those free
//! axes the same way JAX does.
//!
//! # Shardy correspondence
//!
//! The internal metadata helpers render the three attributes attached to
//! `sdy.manual_computation`:
//!
//! | shard-map data       | Shardy attribute    |
//! | -------------------- | ------------------- |
//! | input shardings      | `in_shardings=[...]`  |
//! | output shardings     | `out_shardings=[...]` |
//! | manual mesh axes     | `manual_axes={...}`   |
//!
//! Refer to the [Shardy compiler API documentation](https://openxla.org/shardy/compiler_api) and
//! the [Shardy dialect documentation](https://openxla.org/shardy/sdy_dialect) for the IR-level
//! semantics of manual computation regions.

use std::ops::{Add, Mul, Neg};
use std::{
    collections::{BTreeSet, HashSet},
    fmt::Debug,
};

use ryft_macros::Parameter;
#[cfg(test)]
use ryft_mlir::Block;
use ryft_mlir::Context as MlirContext;
use ryft_mlir::dialects::shardy::{
    DimensionShardingAttributeRef, ManualAxesAttributeRef, TensorShardingAttributeRef,
    TensorShardingPerValueAttributeRef,
};
use thiserror::Error;

use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::sharding::{LogicalMesh, MeshAxisType, Sharding, ShardingDimension, ShardingError};
use crate::tracing_v2::{
    CompiledFunction, FloatExt, JitTracer, Linearized, MatrixOps, OneLike, TraceError, TraceValue, ZeroLike, jit,
    operations::WithShardingConstraintOp,
};
use crate::types::{ArrayType, Shape, Size, Typed};
use crate::xla::sharding::SHARDY_MESH_SYMBOL_NAME;

use super::lowering::LoweringError;

/// Error type for internal shard-map metadata validation and Shardy rendering.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ShardMapError {
    /// Underlying error returned by the mesh/sharding layer.
    #[error("{0}")]
    ShardingError(#[from] ShardingError),

    /// Error returned when a mesh used for `ShardMap` has no manual axes.
    #[error("shard_map requires at least one mesh axis with type manual")]
    MeshHasNoManualAxes,

    /// Error returned when a partitioned dimension uses a free axis more major than a manual axis.
    #[error(
        "{value_kind} sharding #{value_index} dimension #{dimension} uses free axis '{free_axis_name}' \
         more major than manual axis '{manual_axis_name}'"
    )]
    ManualAxisMustPrecedeFreeAxis {
        value_kind: &'static str,
        value_index: usize,
        dimension: usize,
        free_axis_name: String,
        manual_axis_name: String,
    },

    /// Error returned when a provided global shape rank does not match the sharding rank.
    #[error(
        "{value_kind} sharding #{value_index} has rank {partition_rank}, but the provided shape \
         has rank {shape_rank}"
    )]
    RankMismatch { value_kind: &'static str, value_index: usize, partition_rank: usize, shape_rank: usize },

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
}

/// Error type for tracing and lowering `xla::shard_map` bodies.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum ShardMapTraceError {
    /// Underlying error returned by the mesh/sharding layer.
    #[error("{0}")]
    ShardingError(#[from] ShardingError),

    /// Error returned when a mesh used for `shard_map` has no manual axes.
    #[error("shard_map requires at least one mesh axis with type manual")]
    MeshHasNoManualAxes,

    /// Error returned when a partitioned dimension uses a free axis more major than a manual axis.
    #[error(
        "{value_kind} sharding #{value_index} dimension #{dimension} uses free axis '{free_axis_name}' \
         more major than manual axis '{manual_axis_name}'"
    )]
    ManualAxisMustPrecedeFreeAxis {
        value_kind: &'static str,
        value_index: usize,
        dimension: usize,
        free_axis_name: String,
        manual_axis_name: String,
    },

    /// Error returned when a provided global shape rank does not match the sharding rank.
    #[error(
        "{value_kind} sharding #{value_index} has rank {partition_rank}, but the provided shape \
         has rank {shape_rank}"
    )]
    RankMismatch { value_kind: &'static str, value_index: usize, partition_rank: usize, shape_rank: usize },

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

    /// Underlying tracing error returned while staging a shard-map body.
    #[error("{0}")]
    TraceError(#[from] TraceError),

    /// Underlying parameter-structure error returned while reparameterizing traced values.
    #[error("{0}")]
    ParameterError(#[from] ParameterError),

    /// Error returned while building StableHLO/Shardy MLIR for a traced shard-map body.
    #[error("{message}")]
    LoweringFailure { message: String },

    /// Error returned when the number of global input types does not match the number of input shardings.
    #[error("got {actual} global input type(s), but shard_map expects {expected}")]
    InputTypeCountMismatch { expected: usize, actual: usize },

    /// Error returned when the number of traced output types does not match the number of output shardings.
    #[error("traced body produced {actual} output type(s), but shard_map expects {expected}")]
    OutputTypeCountMismatch { expected: usize, actual: usize },

    /// Error returned when a traced shard-map type contains a dynamic dimension that is not supported yet.
    #[error("{value_kind} type #{value_index} dimension #{dimension} must be static for traced shard_map")]
    DynamicShapeNotSupported { value_kind: &'static str, value_index: usize, dimension: usize },

    /// Error returned when `check_vma=true` and one output still varies along an omitted manual axis.
    #[error(
        "output type #{output_index} still varies along manual axis '{axis_name}', but out_specs does not mention it"
    )]
    OutputVaryingManualAxisNotInOutSpecs { output_index: usize, axis_name: String },

    /// Error returned when one input manual-axis state does not match the corresponding `in_specs`.
    #[error("{value_kind} type #{value_index} has {state_kind} {actual:?}, but shard_map expects {expected:?}")]
    ShardingStateMismatch {
        value_kind: &'static str,
        value_index: usize,
        state_kind: &'static str,
        expected: Vec<String>,
        actual: Vec<String>,
    },

    /// Error returned when reconstructing a global output shape overflows `usize`.
    #[error("overflow while {context}")]
    Overflow { context: String },
}

impl From<LoweringError> for ShardMapTraceError {
    fn from(error: LoweringError) -> Self {
        match error {
            LoweringError::ShardMapError(error) => Self::from(error),
            LoweringError::ShardingError(error) => Self::ShardingError(error),
            error => Self::LoweringFailure { message: error.to_string() },
        }
    }
}

impl From<ShardMapError> for ShardMapTraceError {
    fn from(error: ShardMapError) -> Self {
        match error {
            ShardMapError::ShardingError(error) => Self::ShardingError(error),
            ShardMapError::MeshHasNoManualAxes => Self::MeshHasNoManualAxes,
            ShardMapError::ManualAxisMustPrecedeFreeAxis {
                value_kind,
                value_index,
                dimension,
                free_axis_name,
                manual_axis_name,
            } => Self::ManualAxisMustPrecedeFreeAxis {
                value_kind,
                value_index,
                dimension,
                free_axis_name,
                manual_axis_name,
            },
            ShardMapError::RankMismatch { value_kind, value_index, partition_rank, shape_rank } => {
                Self::RankMismatch { value_kind, value_index, partition_rank, shape_rank }
            }
            ShardMapError::ManualAxisIntroducesPadding {
                value_kind,
                value_index,
                dimension,
                dimension_size,
                manual_partition_count,
            } => Self::ManualAxisIntroducesPadding {
                value_kind,
                value_index,
                dimension,
                dimension_size,
                manual_partition_count,
            },
        }
    }
}

/// Constant kind tracked for shard-map body tracing.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub(crate) enum ShardMapConstantKind {
    /// Zero splat constant.
    Zero,

    /// One splat constant.
    One,
}

/// Abstract tensor leaf used while tracing shard-map bodies.
#[derive(Clone, Debug, PartialEq, Eq, Parameter)]
pub(crate) struct ShardMapTensor {
    r#type: ArrayType,
    constant_kind: Option<ShardMapConstantKind>,
}

impl ShardMapTensor {
    /// Creates a traced shard-map tensor with the provided type.
    pub(crate) fn new(r#type: ArrayType) -> Self {
        Self { r#type, constant_kind: None }
    }

    fn constant(r#type: ArrayType, constant_kind: ShardMapConstantKind) -> Self {
        Self { r#type, constant_kind: Some(constant_kind) }
    }

    /// Returns the underlying array type.
    pub(crate) fn r#type(&self) -> &ArrayType {
        &self.r#type
    }

    /// Returns a copy of this tensor with the provided array type.
    pub(crate) fn with_type(&self, r#type: ArrayType) -> Self {
        Self { r#type, constant_kind: self.constant_kind }
    }

    /// Returns the tracked constant kind, if this value came from `zero_like()` or `one_like()`.
    pub(crate) fn constant_kind(&self) -> Option<ShardMapConstantKind> {
        self.constant_kind
    }
}

impl Typed<ArrayType> for ShardMapTensor {
    fn tpe(&self) -> ArrayType {
        self.r#type.clone()
    }
}

impl TraceValue for ShardMapTensor {}

impl crate::tracing_v2::TransformLeaf for ShardMapTensor {}

fn without_varying_manual_axes(r#type: &ArrayType) -> ArrayType {
    let mut r#type = r#type.clone();
    if let Some(sharding) = &mut r#type.sharding {
        sharding.varying_manual_axes.clear();
    }
    r#type
}

impl ZeroLike for ShardMapTensor {
    fn zero_like(&self) -> Self {
        Self::constant(without_varying_manual_axes(&self.r#type), ShardMapConstantKind::Zero)
    }
}

impl OneLike for ShardMapTensor {
    fn one_like(&self) -> Self {
        Self::constant(without_varying_manual_axes(&self.r#type), ShardMapConstantKind::One)
    }
}

impl Add for ShardMapTensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let output_type =
            crate::tracing_v2::operations::binary_same_abstract("add", &[self.r#type.clone(), rhs.r#type.clone()])
                .unwrap_or(self.r#type);
        Self::new(output_type)
    }
}

impl Mul for ShardMapTensor {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let output_type =
            crate::tracing_v2::operations::binary_same_abstract("mul", &[self.r#type.clone(), rhs.r#type.clone()])
                .unwrap_or(self.r#type);
        Self::new(output_type)
    }
}

impl Neg for ShardMapTensor {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(self.r#type)
    }
}

impl FloatExt for ShardMapTensor {
    fn sin(self) -> Self {
        Self::new(self.r#type)
    }

    fn cos(self) -> Self {
        Self::new(self.r#type)
    }
}

impl MatrixOps for ShardMapTensor {
    fn matmul(self, rhs: Self) -> Self {
        match (
            self.r#type.data_type,
            self.r#type.shape.dimensions.as_slice(),
            rhs.r#type.data_type,
            rhs.r#type.shape.dimensions.as_slice(),
        ) {
            (
                left_data_type,
                [Size::Static(left_rows), Size::Static(left_cols)],
                right_data_type,
                [Size::Static(right_rows), Size::Static(right_cols)],
            ) if left_data_type == right_data_type && left_cols == right_rows => Self::new(
                crate::tracing_v2::operations::matrix::matmul_abstract(&self.r#type, &rhs.r#type, "matmul")
                    .unwrap_or_else(|_| {
                        ArrayType::new(
                            left_data_type,
                            Shape::new(vec![Size::Static(*left_rows), Size::Static(*right_cols)]),
                            None,
                            None,
                        )
                        .expect("unsharded matrix result fallback should always be valid")
                    }),
            ),
            _ => Self::new(self.r#type),
        }
    }

    fn transpose_matrix(self) -> Self {
        match self.r#type.shape.dimensions.as_slice() {
            [_, _] => Self::new(
                crate::tracing_v2::operations::matrix::transpose_abstract(&self.r#type, "matrix_transpose")
                    .unwrap_or(self.r#type),
            ),
            _ => Self::new(self.r#type),
        }
    }
}

/// Tracer alias used while staging shard-map bodies.
pub(crate) type ShardMapTracer = JitTracer<ShardMapTensor>;

pub(crate) type ShardMapLocalTraceInput<Input> =
    <<Input as Parameterized<ArrayType>>::To<ShardMapTensor> as Parameterized<ShardMapTensor>>::To<ShardMapTracer>;

pub(crate) type ShardMapLocalTraceOutput<Output> =
    <<Output as Parameterized<ArrayType>>::To<ShardMapTensor> as Parameterized<ShardMapTensor>>::To<ShardMapTracer>;

type TracedXlaInput<Input> =
    <<Input as Parameterized<ArrayType>>::To<ShardMapTensor> as Parameterized<ShardMapTensor>>::To<ShardMapTracer>;

type TracedXlaOutput<Output> =
    <<Output as Parameterized<ArrayType>>::To<ShardMapTensor> as Parameterized<ShardMapTensor>>::To<ShardMapTracer>;

/// Dispatch trait used by [`shard_map`] to select the appropriate tracing regime from the input leaf type.
#[doc(hidden)]
pub(crate) trait ShardMapInvocationLeaf: Parameter + Sized {
    /// Return type produced by [`shard_map`] for the corresponding input leaf regime.
    type Return<
        Input: Parameterized<Self, ParameterStructure: Clone>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    >
    where
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Linearized<ShardMapTracer>>;

    /// Invokes [`shard_map`] for one specific tracing regime.
    fn invoke<
        F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>,
        Input: Parameterized<Self, ParameterStructure: Clone>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    >(
        function: F,
        inputs: Input,
        mesh: LogicalMesh,
        in_specs: Input::To<Sharding>,
        out_specs: Output::To<Sharding>,
        manual_axes: Vec<String>,
        check_vma: bool,
    ) -> Result<Self::Return<Input, Output>, ShardMapTraceError>
    where
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Linearized<ShardMapTracer>>;
}

/// Stages an arbitrary traced XLA function over global tensor types.
///
/// This is the general XLA tracing entry point used when callers want to compose `shard_map`
/// with other `tracing_v2` transforms such as `grad` and then lower the resulting whole graph to
/// StableHLO/Shardy MLIR.
///
/// # Parameters
///
///   - `function`: Function to trace over global XLA values.
///   - `global_input_types`: Global input array types passed to the traced function.
#[allow(private_bounds, private_interfaces)]
pub fn trace<
    F: FnOnce(TracedXlaInput<Input>) -> TracedXlaOutput<Output>,
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
>(
    function: F,
    global_input_types: Input,
) -> Result<TracedXlaProgram<Input, Output>, ShardMapTraceError>
where
    Input::Family: ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
    Output::Family: ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
{
    let (global_output_types, compiled) = trace_xla_function(function, &global_input_types)?;
    Ok(TracedXlaProgram { global_input_types, global_output_types, compiled })
}

/// Applies a strict sharding constraint to one traced XLA value tree.
///
/// This mirrors [`jax.lax.with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html):
/// it behaves like the identity at the value level while recording a concrete Shardy
/// `sdy.sharding_constraint` on each traced leaf.
///
/// # Parameters
///
///   - `input`: Structured traced XLA value whose leaves will be constrained.
///   - `shardings`: Structured shardings with the same leaf layout as `input`.
#[allow(private_bounds, private_interfaces)]
pub fn with_sharding_constraint<Input: Parameterized<ShardMapTracer, ParameterStructure: Clone>>(
    input: Input,
    shardings: Input::To<Sharding>,
) -> Result<Input, ShardMapTraceError>
where
    Input::Family: ParameterizedFamily<Sharding>,
{
    fn constrain_leaf(input: ShardMapTracer, sharding: Sharding) -> Result<ShardMapTracer, ShardMapTraceError> {
        let op = WithShardingConstraintOp::new(sharding.clone());
        let input_type = input.tpe();
        if op.sharding().rank() != input_type.rank() {
            return Err(ShardingError::ShardingRankMismatch {
                sharding_rank: op.sharding().rank(),
                array_rank: input_type.rank(),
            }
            .into());
        }
        let mut output_type = input.value.tpe();
        output_type.sharding =
            Some(sharding_with_varying_manual_axes(&sharding, varying_axes(output_type.sharding.as_ref())));
        let output_value = input.value.with_type(output_type);
        Ok(JitTracer::apply_staged_op(std::slice::from_ref(&input), std::sync::Arc::new(op), vec![output_value])?
            .into_iter()
            .next()
            .expect("with_sharding_constraint should produce one output per input leaf"))
    }

    let structure = input.parameter_structure();
    let constrained = input
        .into_parameters()
        .zip(shardings.into_parameters())
        .map(|(parameter, sharding)| constrain_leaf(parameter, sharding))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Input::from_parameters(structure, constrained)?)
}

/// Stages a traced shard-map body over the provided mesh and shardings.
///
/// This is the ergonomic public entry point for traced XLA shard-map staging. It mirrors the
/// function-first shape of JAX's `shard_map` while adapting it to Rust and `tracing_v2` by
/// requiring explicit `global_input_types`.
///
/// Mesh axes whose type is [`Manual`](crate::sharding::MeshAxisType::Manual) define the default
/// manual axes of the computation. Structured `in_specs` and `out_specs` follow the same
/// `Parameterized` layout as the corresponding input and output types. The body closure receives
/// only the traced local inputs, which lets common cases compile cleanly as `|x| ...` or
/// `|(lhs, rhs)| ...` without explicit tracer annotations.
///
/// # Parameters
///
///   - `function`: Body closure to trace over local shard-map values.
///   - `global_input_types`: Global input array types used to derive the local body argument types.
///   - `mesh`: Logical mesh that the manual computation is defined over.
///   - `in_specs`: Structured shardings for the global inputs.
///   - `out_specs`: Structured shardings for the global outputs.
#[allow(private_bounds, private_interfaces)]
pub fn shard_map<
    F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>,
    Input: Parameterized<Leaf, ParameterStructure: Clone>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    Leaf: ShardMapInvocationLeaf,
>(
    function: F,
    inputs: Input,
    mesh: LogicalMesh,
    in_specs: Input::To<Sharding>,
    out_specs: Output::To<Sharding>,
) -> Result<<Leaf as ShardMapInvocationLeaf>::Return<Input, Output>, ShardMapTraceError>
where
    Input::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Sharding>
        + ParameterizedFamily<ShardMapTensor>
        + ParameterizedFamily<ShardMapTracer>,
    Output::Family: ParameterizedFamily<Sharding>
        + ParameterizedFamily<ShardMapTensor>
        + ParameterizedFamily<ShardMapTracer>
        + ParameterizedFamily<Linearized<ShardMapTracer>>,
{
    shard_map_with_options(function, inputs, mesh, in_specs, out_specs, vec![], true)
}

/// Stages a traced shard-map body with one explicit manual-axis subset and `check_vma` mode.
///
/// `manual_axes` mirrors JAX's `axis_names`: when the list is empty, all mesh axes whose type is
/// [`Manual`](crate::sharding::MeshAxisType::Manual) are active for this shard-map. `check_vma`
/// mirrors JAX's default output-validity check for omitted manual axes.
///
/// # Parameters
///
///   - `function`: Body closure to trace over local shard-map values.
///   - `global_input_types`: Global input array types used to derive the local body argument types.
///   - `mesh`: Logical mesh that the manual computation is defined over.
///   - `in_specs`: Structured shardings for the global inputs.
///   - `out_specs`: Structured shardings for the global outputs.
///   - `manual_axes`: Active manual mesh axes for this shard-map. An empty list means "all manual
///     mesh axes".
///   - `check_vma`: Whether to reject outputs that still vary along active manual axes omitted from
///     `out_specs`.
#[allow(private_bounds, private_interfaces)]
pub fn shard_map_with_options<
    F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>,
    Input: Parameterized<Leaf, ParameterStructure: Clone>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    Leaf: ShardMapInvocationLeaf,
>(
    function: F,
    inputs: Input,
    mesh: LogicalMesh,
    in_specs: Input::To<Sharding>,
    out_specs: Output::To<Sharding>,
    manual_axes: Vec<String>,
    check_vma: bool,
) -> Result<<Leaf as ShardMapInvocationLeaf>::Return<Input, Output>, ShardMapTraceError>
where
    Input::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<Sharding>
        + ParameterizedFamily<ShardMapTensor>
        + ParameterizedFamily<ShardMapTracer>,
    Output::Family: ParameterizedFamily<Sharding>
        + ParameterizedFamily<ShardMapTensor>
        + ParameterizedFamily<ShardMapTracer>
        + ParameterizedFamily<Linearized<ShardMapTracer>>,
{
    Leaf::invoke(function, inputs, mesh, in_specs, out_specs, manual_axes, check_vma)
}

/// Traced shard-map program backed by a staged `tracing_v2` graph.
#[allow(private_bounds, private_interfaces)]
pub struct TracedShardMap<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>>
where
    Input::Family: ParameterizedFamily<ShardMapTensor>,
    Output::Family: ParameterizedFamily<ShardMapTensor>,
{
    shard_map: ShardMap,
    global_input_types: Input,
    local_input_types: Input,
    global_output_types: Output,
    local_output_types: Output,
    compiled: CompiledFunction<ShardMapTensor, Input::To<ShardMapTensor>, Output::To<ShardMapTensor>>,
}

/// Traced XLA program backed by a staged `tracing_v2` graph.
#[allow(private_bounds, private_interfaces)]
pub struct TracedXlaProgram<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>>
where
    Input::Family: ParameterizedFamily<ShardMapTensor>,
    Output::Family: ParameterizedFamily<ShardMapTensor>,
{
    global_input_types: Input,
    global_output_types: Output,
    compiled: CompiledFunction<ShardMapTensor, Input::To<ShardMapTensor>, Output::To<ShardMapTensor>>,
}

/// Metadata describing one manual SPMD computation over a mesh.
///
/// A `ShardMap` stores the mesh plus the validated per-input and per-output shardings, the active
/// manual-axis subset, and whether JAX-style `check_vma` validation is enabled.
///
/// The public constructors accept [`Sharding`] values and project them into
/// traced/type-level semantics, so `Auto` mesh axes remain hidden while `Manual` axes still
/// drive the manual-computation body.
///
/// Reference: https://docs.jax.dev/en/latest/notebooks/shard_map.html.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ShardMap {
    mesh: LogicalMesh,
    in_shardings: Vec<Sharding>,
    out_shardings: Vec<Sharding>,
    manual_axes: Vec<String>,
    check_vma: bool,
}

impl ShardMap {
    /// Creates a `ShardMap` with one explicit manual-axis selection and `check_vma` mode.
    ///
    /// When `manual_axes` is empty, every mesh axis with type
    /// [`Manual`](crate::sharding::MeshAxisType::Manual) is treated as manual inside the body.
    /// The constructor returns [`ShardMapError::MeshHasNoManualAxes`] if the resulting active set
    /// is empty.
    ///
    /// # Parameters
    ///
    ///   - `mesh`: Logical mesh that the manual computation is defined over.
    ///   - `in_specs`: Per-input shardings for the global inputs.
    ///   - `out_specs`: Per-output shardings for the global outputs.
    ///   - `manual_axes`: Active manual mesh axes for this shard-map. An empty list means "all
    ///     manual mesh axes".
    ///   - `check_vma`: Whether to reject outputs that still vary along active manual axes omitted
    ///     from `out_specs`.
    pub(crate) fn new(
        mesh: LogicalMesh,
        in_specs: Vec<Sharding>,
        out_specs: Vec<Sharding>,
        manual_axes: Vec<String>,
        check_vma: bool,
    ) -> Result<Self, ShardMapError> {
        let manual_axes = normalize_manual_axes(&mesh, manual_axes)?;
        let in_shardings = build_shardings(&mesh, manual_axes.as_slice(), in_specs, "input")?;
        let out_shardings = build_shardings(&mesh, manual_axes.as_slice(), out_specs, "output")?;
        Ok(Self { mesh, in_shardings, out_shardings, manual_axes, check_vma })
    }

    /// Builds a shard map directly from already-validated shardings.
    pub(crate) fn from_shardings(
        mesh: LogicalMesh,
        in_shardings: Vec<Sharding>,
        out_shardings: Vec<Sharding>,
        manual_axes: Vec<String>,
        check_vma: bool,
    ) -> Self {
        Self { mesh, in_shardings, out_shardings, manual_axes, check_vma }
    }

    /// Returns the logical mesh of this manual computation.
    pub(crate) fn mesh(&self) -> &LogicalMesh {
        &self.mesh
    }

    /// Returns the validated per-input shardings.
    pub(crate) fn in_shardings(&self) -> &[Sharding] {
        self.in_shardings.as_slice()
    }

    /// Returns the validated per-output shardings.
    pub(crate) fn out_shardings(&self) -> &[Sharding] {
        self.out_shardings.as_slice()
    }

    /// Returns the active manual mesh axes for this shard-map.
    pub(crate) fn manual_axes(&self) -> &[String] {
        self.manual_axes.as_slice()
    }

    fn manual_axis_names(&self) -> HashSet<&str> {
        self.manual_axes.iter().map(String::as_str).collect()
    }

    pub(crate) fn check_vma(&self) -> bool {
        self.check_vma
    }

    /// Returns the local body shape for input `input_index`.
    ///
    /// The returned shape is the tensor shape seen inside the manual computation body for the
    /// corresponding global input. Only manual axes reduce the local shape; free axes remain
    /// global from the body's point of view.
    ///
    /// # Parameters
    ///
    ///   - `input_index`: Index of the input sharding to use.
    ///   - `global_shape`: Global input shape associated with that input.
    fn local_input_shape(&self, input_index: usize, global_shape: &[usize]) -> Result<Vec<usize>, ShardMapError> {
        local_shape_for_sharding(
            &self.in_shardings[input_index],
            self.manual_axis_names(),
            global_shape,
            "input",
            input_index,
        )
    }

    /// Returns the local body shape for output `output_index`.
    ///
    /// # Parameters
    ///
    ///   - `output_index`: Index of the output sharding to use.
    ///   - `global_shape`: Global output shape associated with that output.
    #[cfg(test)]
    fn local_output_shape(&self, output_index: usize, global_shape: &[usize]) -> Result<Vec<usize>, ShardMapError> {
        local_shape_for_sharding(
            &self.out_shardings[output_index],
            self.manual_axis_names(),
            global_shape,
            "output",
            output_index,
        )
    }

    /// Renders the Shardy `in_shardings=[...]` attribute payload.
    ///
    /// The returned string is suitable for direct insertion into an `sdy.manual_computation`
    /// operation.
    ///
    #[cfg(test)]
    fn to_shardy_in_shardings_attribute(&self) -> String {
        render_shardy_sharding_list(self.in_shardings.as_slice(), self.manual_axes())
    }

    /// Renders the Shardy `out_shardings=[...]` attribute payload.
    ///
    #[cfg(test)]
    fn to_shardy_out_shardings_attribute(&self) -> String {
        render_shardy_sharding_list(self.out_shardings.as_slice(), self.manual_axes())
    }

    /// Renders the Shardy `manual_axes={...}` attribute payload.
    #[cfg(test)]
    fn to_shardy_manual_axes_attribute(&self) -> String {
        render_shardy_axes(self.manual_axes())
    }

    /// Renders the three Shardy attributes attached to `sdy.manual_computation`.
    ///
    #[cfg(test)]
    fn to_shardy_manual_computation_attributes(&self) -> String {
        let in_shardings = self.to_shardy_in_shardings_attribute();
        let out_shardings = self.to_shardy_out_shardings_attribute();
        format!(
            "in_shardings={in_shardings} out_shardings={out_shardings} manual_axes={}",
            self.to_shardy_manual_axes_attribute()
        )
    }

    /// Builds the typed Shardy `in_shardings` attribute used by `sdy.manual_computation`.
    pub(crate) fn to_shardy_in_shardings<'c, 't>(
        &self,
        context: &'c MlirContext<'t>,
    ) -> TensorShardingPerValueAttributeRef<'c, 't> {
        shardy_tensor_sharding_per_value(self.in_shardings.as_slice(), self.manual_axes(), context)
    }

    /// Builds the typed Shardy `out_shardings` attribute used by `sdy.manual_computation`.
    pub(crate) fn to_shardy_out_shardings<'c, 't>(
        &self,
        context: &'c MlirContext<'t>,
    ) -> TensorShardingPerValueAttributeRef<'c, 't> {
        shardy_tensor_sharding_per_value(self.out_shardings.as_slice(), self.manual_axes(), context)
    }

    /// Builds the typed Shardy `manual_axes` attribute used by `sdy.manual_computation`.
    pub(crate) fn to_shardy_manual_axes<'c, 't>(&self, context: &'c MlirContext<'t>) -> ManualAxesAttributeRef<'c, 't> {
        context.shardy_manual_axes(self.manual_axes())
    }

    /// Traces a shard-map body over local body tensor types using `tracing_v2::jit`.
    ///
    /// # Parameters
    ///
    ///   - `function`: Body closure to trace over local shard-map values.
    ///   - `global_input_types`: Global input array types in the same leaf order as the shard-map
    ///     input shardings.
    pub(crate) fn trace<
        F: FnOnce(ShardMapLocalTraceInput<Input>) -> ShardMapLocalTraceOutput<Output>,
        Input: Parameterized<ArrayType, ParameterStructure: Clone>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    >(
        &self,
        function: F,
        global_input_types: Input,
    ) -> Result<TracedShardMap<Input, Output>, ShardMapTraceError>
    where
        Input::Family: ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
    {
        let global_input_types = derive_global_input_types(self, &global_input_types)?;
        let local_input_types = derive_local_input_types(self, &global_input_types)?;
        let (local_output_types, compiled) = trace_xla_function(function, &local_input_types)?;
        let global_output_types = derive_global_output_types(self, &local_output_types)?;

        Ok(TracedShardMap {
            shard_map: self.clone(),
            global_input_types,
            local_input_types,
            global_output_types,
            local_output_types,
            compiled,
        })
    }
}

#[allow(private_bounds, private_interfaces)]
impl<
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
> TracedShardMap<Input, Output>
where
    Input::Family: ParameterizedFamily<ShardMapTensor>,
    Output::Family: ParameterizedFamily<ShardMapTensor>,
{
    /// Returns the global input types used to derive the traced local body inputs.
    pub fn global_input_types(&self) -> &Input {
        &self.global_input_types
    }

    /// Returns the traced local body input types.
    pub fn local_input_types(&self) -> &Input {
        &self.local_input_types
    }

    /// Returns the traced local body output types.
    pub fn local_output_types(&self) -> &Output {
        &self.local_output_types
    }

    /// Returns the reconstructed global output types implied by the traced body and output shardings.
    pub fn global_output_types(&self) -> &Output {
        &self.global_output_types
    }

    /// Renders a full StableHLO/Shardy MLIR module for this traced shard-map.
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name to use for the outer `func.func`.
    pub fn to_mlir_module<S: AsRef<str>>(&self, function_name: S) -> Result<String, ShardMapTraceError> {
        let simplified_program = self.compiled.program().simplify()?;
        super::lowering::to_mlir_module(
            &self.shard_map,
            simplified_program.graph(),
            &self.global_input_types,
            &self.local_input_types,
            &self.global_output_types,
            &self.local_output_types,
            function_name,
        )
        .map_err(ShardMapTraceError::from)
    }
}

#[allow(private_bounds, private_interfaces)]
impl<
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
> TracedXlaProgram<Input, Output>
where
    Input::Family: ParameterizedFamily<ShardMapTensor>,
    Output::Family: ParameterizedFamily<ShardMapTensor>,
{
    /// Returns the staged traced XLA program backing this handle.
    #[cfg(feature = "benchmarking")]
    pub(crate) fn compiled(
        &self,
    ) -> &CompiledFunction<ShardMapTensor, Input::To<ShardMapTensor>, Output::To<ShardMapTensor>> {
        &self.compiled
    }

    /// Returns the traced global input types.
    pub fn global_input_types(&self) -> &Input {
        &self.global_input_types
    }

    /// Returns the traced global output types.
    pub fn global_output_types(&self) -> &Output {
        &self.global_output_types
    }

    /// Renders a full StableHLO/Shardy MLIR module for this traced XLA program.
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name to use for the outer `func.func`.
    pub fn to_mlir_module<S: AsRef<str>>(&self, function_name: S) -> Result<String, ShardMapTraceError> {
        let simplified_program = self.compiled.program().simplify()?;
        super::lowering::to_mlir_module_for_graph(
            simplified_program.graph(),
            &self.global_input_types,
            &self.global_output_types,
            function_name,
        )
        .map_err(ShardMapTraceError::from)
    }
}

/// Erased shard-map body payload used by nested higher-order shard-map ops.
#[derive(Clone)]
pub(crate) struct FlatTracedShardMap {
    pub(crate) shard_map: ShardMap,
    pub(crate) global_input_types: Vec<ArrayType>,
    pub(crate) local_input_types: Vec<ArrayType>,
    pub(crate) global_output_types: Vec<ArrayType>,
    pub(crate) local_output_types: Vec<ArrayType>,
    pub(crate) compiled: CompiledFunction<ShardMapTensor, Vec<ShardMapTensor>, Vec<ShardMapTensor>>,
}

impl FlatTracedShardMap {
    /// Builds an erased shard-map body from explicit traced components.
    pub(crate) fn from_parts(
        shard_map: ShardMap,
        global_input_types: Vec<ArrayType>,
        local_input_types: Vec<ArrayType>,
        global_output_types: Vec<ArrayType>,
        local_output_types: Vec<ArrayType>,
        compiled: CompiledFunction<ShardMapTensor, Vec<ShardMapTensor>, Vec<ShardMapTensor>>,
    ) -> Self {
        Self { shard_map, global_input_types, local_input_types, global_output_types, local_output_types, compiled }
    }

    /// Builds an erased shard-map body from the typed traced representation.
    pub(crate) fn from_traced<
        Input: Parameterized<ArrayType, ParameterStructure: Clone>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    >(
        traced: &TracedShardMap<Input, Output>,
    ) -> Self
    where
        Input::Family: ParameterizedFamily<ShardMapTensor>,
        Output::Family: ParameterizedFamily<ShardMapTensor>,
    {
        let local_input_types = traced.local_input_types.parameters().cloned().collect::<Vec<_>>();
        let local_output_types = traced.local_output_types.parameters().cloned().collect::<Vec<_>>();
        let compiled = CompiledFunction::from_graph(
            traced.compiled.graph().clone_with_structures::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
                vec![Placeholder; local_input_types.len()],
                vec![Placeholder; local_output_types.len()],
            ),
        );
        Self::from_parts(
            traced.shard_map.clone(),
            traced.global_input_types.parameters().cloned().collect::<Vec<_>>(),
            local_input_types,
            traced.global_output_types.parameters().cloned().collect::<Vec<_>>(),
            local_output_types,
            compiled,
        )
    }

    /// Returns a copy of this erased shard-map body with dead staged work removed.
    pub(crate) fn simplified(&self) -> Result<Self, ShardMapTraceError> {
        Ok(Self::from_parts(
            self.shard_map.clone(),
            self.global_input_types.clone(),
            self.local_input_types.clone(),
            self.global_output_types.clone(),
            self.local_output_types.clone(),
            CompiledFunction::from_program(self.compiled.program().simplify()?),
        ))
    }
}

fn merge_unique_axes(left: &BTreeSet<String>, right: &BTreeSet<String>) -> BTreeSet<String> {
    left.union(right).cloned().collect()
}

fn axes_to_vec(axis_names: &BTreeSet<String>) -> Vec<String> {
    axis_names.iter().cloned().collect()
}

fn varying_axes(sharding: Option<&Sharding>) -> BTreeSet<String> {
    sharding.map(|sharding| sharding.varying_manual_axes.clone()).unwrap_or_default()
}

fn sharding_with_varying_manual_axes(sharding: &Sharding, varying_axes: BTreeSet<String>) -> Sharding {
    let varying_axes = varying_axes
        .into_iter()
        .filter(|axis_name| sharding.mesh.axis_type(axis_name) == Some(MeshAxisType::Manual))
        .collect::<BTreeSet<_>>();
    Sharding::with_manual_axes(
        sharding.mesh.clone(),
        sharding.dimensions.clone(),
        sharding.unreduced_axes.clone(),
        sharding.reduced_manual_axes.clone(),
        varying_axes,
    )
    .expect("derived shard_map sharding should preserve valid manual-axis metadata")
}

fn axes_match(left: &BTreeSet<String>, right: &BTreeSet<String>) -> bool {
    left == right
}

fn validate_input_sharding_state(
    actual: Option<&Sharding>,
    expected: &Sharding,
    input_index: usize,
) -> Result<(), ShardMapTraceError> {
    let Some(actual) = actual else {
        return Ok(());
    };
    if !axes_match(&actual.unreduced_axes, &expected.unreduced_axes) {
        return Err(ShardMapTraceError::ShardingStateMismatch {
            value_kind: "input",
            value_index: input_index,
            state_kind: "unreduced axes",
            expected: axes_to_vec(&expected.unreduced_axes),
            actual: axes_to_vec(&actual.unreduced_axes),
        });
    }
    if !axes_match(&actual.reduced_manual_axes, &expected.reduced_manual_axes) {
        return Err(ShardMapTraceError::ShardingStateMismatch {
            value_kind: "input",
            value_index: input_index,
            state_kind: "reduced axes",
            expected: axes_to_vec(&expected.reduced_manual_axes),
            actual: axes_to_vec(&actual.reduced_manual_axes),
        });
    }
    Ok(())
}

fn spec_varying_axes(sharding: &Sharding, manual_axis_names: &HashSet<&str>) -> BTreeSet<String> {
    let mut varying_axes = BTreeSet::new();
    for partition_dimension in &sharding.dimensions {
        if let ShardingDimension::Sharded(axis_names) = partition_dimension {
            for axis_name in axis_names {
                if manual_axis_names.contains(axis_name.as_str()) {
                    varying_axes.insert(axis_name.clone());
                }
            }
        }
    }
    varying_axes
}

fn derive_global_input_types<Input: Parameterized<ArrayType, ParameterStructure: Clone>>(
    shard_map: &ShardMap,
    global_input_types: &Input,
) -> Result<Input, ShardMapTraceError> {
    let global_input_type_count = global_input_types.parameter_count();
    if global_input_type_count != shard_map.in_shardings().len() {
        return Err(ShardMapTraceError::InputTypeCountMismatch {
            expected: shard_map.in_shardings().len(),
            actual: global_input_type_count,
        });
    }

    let structure = global_input_types.parameter_structure();
    let global_input_types = global_input_types
        .parameters()
        .cloned()
        .enumerate()
        .map(|(input_index, mut global_input_type)| {
            let sharding = shard_map.in_shardings()[input_index].clone();
            validate_input_sharding_state(global_input_type.sharding.as_ref(), &sharding, input_index)?;
            global_input_type.sharding =
                Some(sharding_with_varying_manual_axes(&sharding, varying_axes(global_input_type.sharding.as_ref())));
            Ok::<ArrayType, ShardMapTraceError>(global_input_type)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Input::from_parameters(structure, global_input_types)?)
}

fn derive_local_input_types<Input: Parameterized<ArrayType, ParameterStructure: Clone>>(
    shard_map: &ShardMap,
    global_input_types: &Input,
) -> Result<Input, ShardMapTraceError> {
    let global_input_type_count = global_input_types.parameter_count();
    if global_input_type_count != shard_map.in_shardings().len() {
        return Err(ShardMapTraceError::InputTypeCountMismatch {
            expected: shard_map.in_shardings().len(),
            actual: global_input_type_count,
        });
    }

    let manual_axis_names = shard_map.manual_axis_names();
    let structure = global_input_types.parameter_structure();
    let local_input_types = global_input_types
        .parameters()
        .cloned()
        .enumerate()
        .map(|(input_index, global_input_type)| {
            ensure_static_array_type(&global_input_type, "input", input_index)?;
            let global_shape = static_shape_values(&global_input_type, "input", input_index)?;
            let local_shape = shard_map.local_input_shape(input_index, &global_shape)?;
            let local_sharding = shard_map.in_shardings()[input_index].clone();
            let local_varying_axes = merge_unique_axes(
                &varying_axes(global_input_type.sharding.as_ref()),
                &spec_varying_axes(&local_sharding, &manual_axis_names),
            );
            Ok::<ArrayType, ShardMapTraceError>(ArrayType::new(
                global_input_type.data_type,
                Shape::new(local_shape.into_iter().map(Size::Static).collect()),
                global_input_type.layout.clone(),
                Some(sharding_with_varying_manual_axes(&local_sharding, local_varying_axes)),
            )?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Input::from_parameters(structure, local_input_types)?)
}

fn traced_input_tensors<Input: Parameterized<ArrayType, ParameterStructure: Clone>>(
    local_input_types: &Input,
) -> Result<Input::To<ShardMapTensor>, ShardMapTraceError>
where
    Input::Family: ParameterizedFamily<ShardMapTensor>,
{
    let structure = local_input_types.parameter_structure();
    let tensors = local_input_types.parameters().cloned().map(ShardMapTensor::new).collect::<Vec<_>>();
    Ok(Input::To::<ShardMapTensor>::from_parameters(structure, tensors)?)
}

fn trace_xla_function<
    F: FnOnce(TracedXlaInput<Input>) -> TracedXlaOutput<Output>,
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
>(
    function: F,
    input_types: &Input,
) -> Result<
    (Output, CompiledFunction<ShardMapTensor, Input::To<ShardMapTensor>, Output::To<ShardMapTensor>>),
    ShardMapTraceError,
>
where
    Input::Family: ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
    Output::Family: ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
{
    let traced_inputs = traced_input_tensors(input_types)?;
    let (output_tensors, compiled) =
        jit::<_, Input::To<ShardMapTensor>, Output::To<ShardMapTensor>, ShardMapTensor>(function, traced_inputs)?;
    let output_types = Output::from_parameters(
        output_tensors.parameter_structure(),
        output_tensors.into_parameters().map(|tensor| tensor.tpe()).collect::<Vec<_>>(),
    )?;
    Ok((output_types, compiled))
}

pub(crate) fn derive_global_output_types<Output: Parameterized<ArrayType, ParameterStructure: Clone>>(
    shard_map: &ShardMap,
    local_output_types: &Output,
) -> Result<Output, ShardMapTraceError> {
    let local_output_type_count = local_output_types.parameter_count();
    if local_output_type_count != shard_map.out_shardings().len() {
        return Err(ShardMapTraceError::OutputTypeCountMismatch {
            expected: shard_map.out_shardings().len(),
            actual: local_output_type_count,
        });
    }

    let manual_axis_names = shard_map.manual_axis_names();
    let structure = local_output_types.parameter_structure();
    let global_output_types = local_output_types
        .parameters()
        .cloned()
        .enumerate()
        .map(|(output_index, local_output_type)| {
            ensure_static_array_type(&local_output_type, "output", output_index)?;
            let output_sharding = &shard_map.out_shardings()[output_index];
            let expected_current_varying_axes = spec_varying_axes(output_sharding, &manual_axis_names);
            let effective_local_varying_axes =
                merge_unique_axes(&varying_axes(local_output_type.sharding.as_ref()), &expected_current_varying_axes);
            if shard_map.check_vma() {
                let local_unreduced_axes = local_output_type
                    .sharding
                    .as_ref()
                    .map(|sharding| sharding.unreduced_axes.clone())
                    .unwrap_or_default();
                let effective_local_unreduced_axes =
                    merge_unique_axes(&local_unreduced_axes, &output_sharding.unreduced_axes);
                if !axes_match(&effective_local_unreduced_axes, &output_sharding.unreduced_axes) {
                    return Err(ShardMapTraceError::ShardingStateMismatch {
                        value_kind: "output",
                        value_index: output_index,
                        state_kind: "unreduced axes",
                        expected: axes_to_vec(&output_sharding.unreduced_axes),
                        actual: axes_to_vec(&local_unreduced_axes),
                    });
                }

                let local_reduced_manual_axes = local_output_type
                    .sharding
                    .as_ref()
                    .map(|sharding| sharding.reduced_manual_axes.clone())
                    .unwrap_or_default();
                if !axes_match(&local_reduced_manual_axes, &output_sharding.reduced_manual_axes) {
                    return Err(ShardMapTraceError::ShardingStateMismatch {
                        value_kind: "output",
                        value_index: output_index,
                        state_kind: "reduced axes",
                        expected: axes_to_vec(&output_sharding.reduced_manual_axes),
                        actual: axes_to_vec(&local_reduced_manual_axes),
                    });
                }

                for axis_name in &effective_local_varying_axes {
                    if manual_axis_names.contains(axis_name.as_str())
                        && !expected_current_varying_axes.contains(axis_name)
                    {
                        return Err(ShardMapTraceError::OutputVaryingManualAxisNotInOutSpecs {
                            output_index,
                            axis_name: axis_name.clone(),
                        });
                    }
                }
            }
            let surviving_varying_axes = effective_local_varying_axes
                .into_iter()
                .filter(|axis_name| !manual_axis_names.contains(axis_name.as_str()))
                .collect::<BTreeSet<_>>();
            let global_shape = global_shape_for_sharding(
                output_sharding,
                &manual_axis_names,
                static_shape_values(&local_output_type, "output", output_index)?,
                output_index,
            )?;
            Ok::<ArrayType, ShardMapTraceError>(ArrayType::new(
                local_output_type.data_type,
                Shape::new(global_shape.into_iter().map(Size::Static).collect()),
                local_output_type.layout.clone(),
                Some(sharding_with_varying_manual_axes(output_sharding, surviving_varying_axes)),
            )?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Output::from_parameters(structure, global_output_types)?)
}

fn ensure_static_array_type(
    array_type: &ArrayType,
    value_kind: &'static str,
    value_index: usize,
) -> Result<(), ShardMapTraceError> {
    for (dimension, size) in array_type.shape.dimensions.iter().enumerate() {
        if !matches!(size, Size::Static(_)) {
            return Err(ShardMapTraceError::DynamicShapeNotSupported { value_kind, value_index, dimension });
        }
    }
    Ok(())
}

fn static_shape_values(
    array_type: &ArrayType,
    value_kind: &'static str,
    value_index: usize,
) -> Result<Vec<usize>, ShardMapTraceError> {
    array_type
        .shape
        .dimensions
        .iter()
        .enumerate()
        .map(|(dimension, size)| match size {
            Size::Static(value) => Ok(*value),
            Size::Dynamic(_) => {
                Err(ShardMapTraceError::DynamicShapeNotSupported { value_kind, value_index, dimension })
            }
        })
        .collect()
}

fn global_shape_for_sharding(
    sharding: &Sharding,
    manual_axis_names: &HashSet<&str>,
    local_shape: Vec<usize>,
    output_index: usize,
) -> Result<Vec<usize>, ShardMapTraceError> {
    if sharding.rank() != local_shape.len() {
        return Err(ShardMapTraceError::RankMismatch {
            value_kind: "output",
            value_index: output_index,
            partition_rank: sharding.rank(),
            shape_rank: local_shape.len(),
        });
    }

    sharding
        .dimensions
        .iter()
        .zip(local_shape)
        .enumerate()
        .map(|(dimension, (partition_dimension, local_dimension_size))| {
            let manual_partition_count = match partition_dimension {
                ShardingDimension::Sharded(axis_names) => axis_names
                    .iter()
                    .filter(|axis_name| manual_axis_names.contains(axis_name.as_str()))
                    .try_fold(1usize, |partition_count, axis_name| {
                        let axis_size = sharding.mesh.axis_size(axis_name).ok_or_else(|| {
                            ShardMapTraceError::ShardingError(ShardingError::UnknownMeshAxisName {
                                name: axis_name.clone(),
                            })
                        })?;
                        partition_count.checked_mul(axis_size).ok_or_else(|| ShardMapTraceError::Overflow {
                            context: format!(
                                "computing global output shape for output #{output_index} dimension #{dimension}"
                            ),
                        })
                    })?,
                ShardingDimension::Replicated | ShardingDimension::Unconstrained => 1,
            };

            local_dimension_size
                .checked_mul(manual_partition_count)
                .ok_or_else(|| ShardMapTraceError::Overflow {
                    context: format!("computing global output size for output #{output_index} dimension #{dimension}"),
                })
        })
        .collect()
}

fn build_shardings(
    mesh: &LogicalMesh,
    manual_axes: &[String],
    shardings: Vec<Sharding>,
    value_kind: &'static str,
) -> Result<Vec<Sharding>, ShardMapError> {
    let manual_axis_names = manual_axes.iter().map(String::as_str).collect::<HashSet<_>>();
    shardings
        .into_iter()
        .enumerate()
        .map(|(value_index, sharding)| {
            if sharding.mesh != *mesh {
                return Err(ShardMapError::ShardingError(ShardingError::MeshMismatch {
                    expected: mesh.clone(),
                    actual: sharding.mesh.clone(),
                }));
            }
            validate_manual_axis_order(&sharding, &manual_axis_names, value_kind, value_index)?;
            Ok(sharding.without_auto_axes())
        })
        .collect()
}

fn normalize_manual_axes(mesh: &LogicalMesh, manual_axes: Vec<String>) -> Result<Vec<String>, ShardMapError> {
    let selected_manual_axes = if manual_axes.is_empty() {
        None
    } else {
        let mut selected_manual_axes = HashSet::new();
        for axis_name in manual_axes {
            if !mesh.axis_indices.contains_key(axis_name.as_str()) {
                return Err(ShardMapError::ShardingError(ShardingError::UnknownMeshAxisName { name: axis_name }));
            }
            if mesh.axis_type(axis_name.as_str()) != Some(MeshAxisType::Manual) {
                return Err(ShardMapError::ShardingError(ShardingError::ExpectedManualMeshAxis { name: axis_name }));
            }
            selected_manual_axes.insert(axis_name);
        }
        Some(selected_manual_axes)
    };
    let manual_axes = mesh
        .axes
        .iter()
        .filter_map(|axis| {
            (axis.r#type == MeshAxisType::Manual
                && match &selected_manual_axes {
                    None => true,
                    Some(selected_manual_axes) => selected_manual_axes.contains(axis.name.as_str()),
                })
            .then_some(axis.name.clone())
        })
        .collect::<Vec<_>>();
    if manual_axes.is_empty() {
        return Err(ShardMapError::MeshHasNoManualAxes);
    }
    Ok(manual_axes)
}

fn validate_manual_axis_order(
    sharding: &Sharding,
    manual_axes: &HashSet<&str>,
    value_kind: &'static str,
    value_index: usize,
) -> Result<(), ShardMapError> {
    for (dimension, partition_dimension) in sharding.dimensions.iter().enumerate() {
        if let ShardingDimension::Sharded(axis_names) = partition_dimension {
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

fn local_shape_for_sharding(
    sharding: &Sharding,
    manual_axis_names: HashSet<&str>,
    global_shape: &[usize],
    value_kind: &'static str,
    value_index: usize,
) -> Result<Vec<usize>, ShardMapError> {
    if sharding.rank() != global_shape.len() {
        return Err(ShardMapError::RankMismatch {
            value_kind,
            value_index,
            partition_rank: sharding.rank(),
            shape_rank: global_shape.len(),
        });
    }

    let mut local_shape = Vec::with_capacity(global_shape.len());
    for (dimension, (partition_dimension, dimension_size)) in
        sharding.dimensions.iter().zip(global_shape.iter().copied()).enumerate()
    {
        let manual_partition_count = match partition_dimension {
            ShardingDimension::Sharded(axis_names) => axis_names
                .iter()
                .filter(|axis_name| manual_axis_names.contains(axis_name.as_str()))
                .try_fold(1usize, |partition_count, axis_name| -> Result<usize, ShardMapError> {
                    let axis_size = sharding
                        .mesh
                        .axis_size(axis_name)
                        .ok_or_else(|| ShardingError::UnknownMeshAxisName { name: axis_name.clone() })?;
                    Ok(partition_count * axis_size)
                })?,
            ShardingDimension::Replicated | ShardingDimension::Unconstrained => 1,
        };

        if dimension_size % manual_partition_count != 0 {
            return Err(ShardMapError::ManualAxisIntroducesPadding {
                value_kind,
                value_index,
                dimension,
                dimension_size,
                manual_partition_count,
            });
        }

        local_shape.push(dimension_size / manual_partition_count);
    }
    Ok(local_shape)
}

#[cfg(test)]
fn render_shardy_sharding_list(shardings: &[Sharding], manual_axes: &[String]) -> String {
    let mut result = String::from("[");
    for (sharding_index, sharding) in shardings.iter().enumerate() {
        if sharding_index > 0 {
            result.push_str(", ");
        }
        result.push_str(stripped_shardy_tensor_sharding(sharding, manual_axes).as_str());
    }
    result.push(']');
    result
}

fn shardy_tensor_sharding_per_value<'c, 't>(
    shardings: &[Sharding],
    manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> TensorShardingPerValueAttributeRef<'c, 't> {
    let shardings = shardings
        .iter()
        .map(|sharding| manual_computation_tensor_sharding(sharding, manual_axes, context))
        .collect::<Vec<_>>();
    context.shardy_tensor_sharding_per_value(shardings.as_slice())
}

fn manual_computation_tensor_sharding<'c, 't>(
    sharding: &Sharding,
    manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> TensorShardingAttributeRef<'c, 't> {
    let mesh_symbol_ref = context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME);
    let dim_shardings = manual_computation_dimension_shardings(sharding, manual_axes, context);
    let replicated_axis_names = sharding.replicated_axes();
    let replicated_axes = replicated_axis_names
        .iter()
        .map(|axis_name| context.shardy_axis_ref(axis_name, None))
        .collect::<Vec<_>>();
    let unreduced_axes = sharding
        .unreduced_axes
        .iter()
        .map(|axis_name| context.shardy_axis_ref(axis_name, None))
        .collect::<Vec<_>>();
    context.shardy_tensor_sharding(
        mesh_symbol_ref,
        dim_shardings.as_slice(),
        replicated_axes.as_slice(),
        unreduced_axes.as_slice(),
    )
}

fn manual_computation_dimension_shardings<'c, 't>(
    sharding: &Sharding,
    manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> Vec<DimensionShardingAttributeRef<'c, 't>> {
    let manual_axis_names = manual_axes.iter().map(String::as_str).collect::<HashSet<_>>();
    let free_axis_names = sharding
        .mesh
        .axes
        .iter()
        .filter_map(|axis| (!manual_axis_names.contains(axis.name.as_str())).then_some(axis.name.as_str()))
        .collect::<HashSet<_>>();
    let mut used_axes = HashSet::new();
    for partition_dimension in &sharding.dimensions {
        if let ShardingDimension::Sharded(axis_names) = partition_dimension {
            used_axes.extend(axis_names.iter().map(String::as_str));
        }
    }
    used_axes.extend(sharding.unreduced_axes.iter().map(String::as_str));
    used_axes.extend(sharding.reduced_manual_axes.iter().map(String::as_str));
    let has_unused_free_axes = free_axis_names.iter().any(|axis_name| !used_axes.contains(axis_name));

    sharding
        .dimensions
        .iter()
        .map(|partition_dimension| match partition_dimension {
            ShardingDimension::Replicated => context.shardy_dimension_sharding([], !has_unused_free_axes, None),
            ShardingDimension::Sharded(axis_names) => {
                let axes =
                    axis_names.iter().map(|axis_name| context.shardy_axis_ref(axis_name, None)).collect::<Vec<_>>();
                let contains_free_axis =
                    axis_names.iter().any(|axis_name| free_axis_names.contains(axis_name.as_str()));
                context.shardy_dimension_sharding(axes, !(contains_free_axis || has_unused_free_axes), None)
            }
            ShardingDimension::Unconstrained => context.shardy_dimension_sharding([], false, None),
        })
        .collect()
}

#[cfg(test)]
fn stripped_shardy_tensor_sharding(sharding: &Sharding, manual_axes: &[String]) -> String {
    let mut result =
        format!("<@{SHARDY_MESH_SYMBOL_NAME}, {}>", render_manual_computation_dimensions(sharding, manual_axes));

    let replicated_axes = sharding.replicated_axes();
    result.pop();
    if !replicated_axes.is_empty() {
        result.push_str(", replicated={");
        for (axis_index, axis_name) in replicated_axes.iter().enumerate() {
            if axis_index > 0 {
                result.push_str(", ");
            }
            result.push('"');
            result.push_str(escape_shardy_string(axis_name).as_str());
            result.push('"');
        }
        result.push('}');
    }

    if !sharding.unreduced_axes.is_empty() {
        result.push_str(", unreduced={");
        for (axis_index, axis_name) in sharding.unreduced_axes.iter().enumerate() {
            if axis_index > 0 {
                result.push_str(", ");
            }
            result.push('"');
            result.push_str(escape_shardy_string(axis_name).as_str());
            result.push('"');
        }
        result.push('}');
    }

    result.push('>');
    result
}

#[cfg(test)]
fn render_manual_computation_dimensions(sharding: &Sharding, manual_axes: &[String]) -> String {
    let manual_axis_names = manual_axes.iter().map(String::as_str).collect::<HashSet<_>>();
    let free_axis_names = sharding
        .mesh
        .axes
        .iter()
        .filter_map(|axis| (!manual_axis_names.contains(axis.name.as_str())).then_some(axis.name.as_str()))
        .collect::<HashSet<_>>();
    let mut used_axes = HashSet::new();
    for partition_dimension in &sharding.dimensions {
        if let ShardingDimension::Sharded(axis_names) = partition_dimension {
            used_axes.extend(axis_names.iter().map(String::as_str));
        }
    }
    used_axes.extend(sharding.unreduced_axes.iter().map(String::as_str));
    used_axes.extend(sharding.reduced_manual_axes.iter().map(String::as_str));
    let has_unused_free_axes = free_axis_names.iter().any(|axis_name| !used_axes.contains(axis_name));

    let mut result = String::from("[");
    for (dimension_index, partition_dimension) in sharding.dimensions.iter().enumerate() {
        if dimension_index > 0 {
            result.push_str(", ");
        }

        match partition_dimension {
            ShardingDimension::Replicated => {
                if has_unused_free_axes {
                    result.push_str("{?}");
                } else {
                    result.push_str("{}");
                }
            }
            ShardingDimension::Sharded(axis_names) => {
                let contains_free_axis =
                    axis_names.iter().any(|axis_name| free_axis_names.contains(axis_name.as_str()));
                result.push('{');
                for (axis_index, axis_name) in axis_names.iter().enumerate() {
                    if axis_index > 0 {
                        result.push_str(", ");
                    }
                    result.push('"');
                    result.push_str(escape_shardy_string(axis_name).as_str());
                    result.push('"');
                }
                if contains_free_axis || has_unused_free_axes {
                    result.push_str(", ?");
                }
                result.push('}');
            }
            ShardingDimension::Unconstrained => result.push_str("{?}"),
        }
    }
    result.push(']');
    result
}

#[cfg(test)]
fn render_shardy_axes<A: AsRef<str>>(axis_names: &[A]) -> String {
    let mut result = String::from("{");
    for (axis_index, axis_name) in axis_names.iter().enumerate() {
        if axis_index > 0 {
            result.push_str(", ");
        }
        result.push('"');
        result.push_str(escape_shardy_string(axis_name.as_ref()).as_str());
        result.push('"');
    }
    result.push('}');
    result
}

#[cfg(test)]
fn escape_shardy_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, Program, load_cpu_plugin};

    use super::*;
    use crate::sharding::DeviceMesh;
    use crate::sharding::MeshDevice;
    use crate::sharding::ShardingDimension;
    use crate::sharding::{MeshAxis, MeshAxisType, Sharding};
    use crate::tracing_v2::{FloatExt, OneLike, grad, vmap};
    use crate::types::data_types::DataType;
    use crate::xla::arrays::Array;

    fn test_logical_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
    }

    fn test_logical_mesh_data_model() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("data", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("model", 4, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    fn test_logical_mesh_data_model_explicit() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("data", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("model", 4, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap()
    }

    fn test_logical_mesh_without_manual_axes() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    fn empty_axes() -> Vec<&'static str> {
        Vec::new()
    }

    fn static_sharded_array_type(data_type: DataType, global_shape: &[usize], sharding: Sharding) -> ArrayType {
        ArrayType::new(
            data_type,
            Shape::new(global_shape.iter().copied().map(Size::Static).collect()),
            None,
            Some(sharding),
        )
        .unwrap()
    }

    fn test_sharding(mesh: &LogicalMesh, dimensions: Vec<ShardingDimension>, unreduced_axes: Vec<String>) -> Sharding {
        Sharding::with_unreduced_axes(mesh.clone(), dimensions, unreduced_axes).unwrap()
    }

    fn test_sharding_with_varying(
        mesh: &LogicalMesh,
        dimensions: Vec<ShardingDimension>,
        unreduced_axes: Vec<String>,
        reduced_manual_axes: Vec<String>,
        varying_manual_axes: Vec<String>,
    ) -> Sharding {
        Sharding::with_manual_axes(mesh.clone(), dimensions, unreduced_axes, reduced_manual_axes, varying_manual_axes)
            .unwrap()
    }

    fn test_spmd_compilation_options(partition_count: usize) -> CompilationOptions {
        CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: partition_count as i64,
                use_spmd_partitioning: true,
                use_shardy_partitioner: true,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        }
    }

    fn f32_values_to_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * size_of::<f32>());
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn two_f32s_from_bytes(bytes: &[u8]) -> [f32; 2] {
        assert_eq!(bytes.len(), 2 * size_of::<f32>());
        let first = f32::from_ne_bytes(bytes[..size_of::<f32>()].try_into().unwrap());
        let second = f32::from_ne_bytes(bytes[size_of::<f32>()..].try_into().unwrap());
        [first, second]
    }

    fn f32s_from_bytes(bytes: &[u8]) -> Vec<f32> {
        assert_eq!(bytes.len() % size_of::<f32>(), 0);
        bytes
            .chunks_exact(size_of::<f32>())
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>()
    }

    fn assert_two_f32s_approx_eq(actual: [f32; 2], expected: [f32; 2]) {
        let first_delta = (actual[0] - expected[0]).abs();
        let second_delta = (actual[1] - expected[1]).abs();
        assert!(first_delta <= 1e-5, "expected {} ~= {}; absolute error {}", actual[0], expected[0], first_delta);
        assert!(second_delta <= 1e-5, "expected {} ~= {}; absolute error {}", actual[1], expected[1], second_delta);
    }

    #[test]
    fn test_shard_map_uses_manual_axes_from_mesh() {
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![])],
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![])],
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.manual_axes(), vec!["x".to_string(), "y".to_string()].as_slice());
        assert_eq!(shard_map.in_shardings()[0].replicated_axes(), vec!["y"]);
        assert_eq!(shard_map.out_shardings()[0].replicated_axes(), vec!["y"]);
    }

    #[test]
    fn test_shard_map_can_select_manual_axis_subset() {
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x", "y"])], vec![])],
            Vec::new(),
            vec!["x".into()],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.manual_axes(), vec!["x".to_string()].as_slice());
        assert_eq!(shard_map.local_input_shape(0, &[8]).unwrap(), vec![4]);
        assert_eq!(shard_map.to_shardy_in_shardings_attribute(), r#"[<@mesh, [{"x", "y", ?}]>]"#);
        assert_eq!(shard_map.to_shardy_manual_axes_attribute(), r#"{"x"}"#);
    }

    #[test]
    fn test_shard_map_function_rejects_mesh_without_manual_axes() {
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();
        let mesh = test_logical_mesh_without_manual_axes();
        let result: Result<TracedShardMap<ArrayType, ArrayType>, ShardMapTraceError> = shard_map(
            |x| x.clone() + x,
            global_input_type,
            mesh.clone(),
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]),
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]),
        );

        assert!(matches!(result, Err(ShardMapTraceError::MeshHasNoManualAxes)));
    }

    #[test]
    fn test_shard_map_rejects_free_axis_before_manual_axis() {
        let mesh = test_logical_mesh_data_model();
        let result = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["model", "data"])], vec![])],
            Vec::new(),
            vec![],
            true,
        );

        assert_eq!(
            result,
            Err(ShardMapError::ManualAxisMustPrecedeFreeAxis {
                value_kind: "input",
                value_index: 0,
                dimension: 0,
                free_axis_name: "model".to_string(),
                manual_axis_name: "data".to_string(),
            })
        );
    }

    #[test]
    fn test_shard_map_local_input_shape_for_all_manual_axes() {
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x", "y"])], vec![])],
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.local_input_shape(0, &[16]).unwrap(), vec![4]);
    }

    #[test]
    fn test_shard_map_local_input_shape_for_mixed_manual_and_free_axes() {
        let mesh = test_logical_mesh_data_model();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["data", "model"])], vec![])],
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.local_input_shape(0, &[16]).unwrap(), vec![8]);
    }

    #[test]
    fn test_shard_map_local_output_shape() {
        let mesh = test_logical_mesh_data_model();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![test_sharding(
                &mesh,
                vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()],
                vec![],
            )],
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.local_output_shape(0, &[32, 8]).unwrap(), vec![16, 8]);
    }

    #[test]
    fn test_shard_map_local_shape_rejects_padding_from_manual_axes() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Manual).unwrap()]).unwrap();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![])],
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(
            shard_map.local_input_shape(0, &[10]),
            Err(ShardMapError::ManualAxisIntroducesPadding {
                value_kind: "input",
                value_index: 0,
                dimension: 0,
                dimension_size: 10,
                manual_partition_count: 3,
            })
        );
    }

    #[test]
    fn test_shard_map_local_shape_rejects_rank_mismatch() {
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![])],
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(
            shard_map.local_input_shape(0, &[8, 4]),
            Err(ShardMapError::RankMismatch { value_kind: "input", value_index: 0, partition_rank: 1, shape_rank: 2 })
        );
    }

    #[test]
    fn test_shard_map_renders_in_shardings_attribute() {
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![])],
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.to_shardy_in_shardings_attribute(), r#"[<@mesh, [{"x"}], replicated={"y"}>]"#);
    }

    #[test]
    fn test_shard_map_renders_free_axes_as_open_dimension_shardings() {
        let mesh = test_logical_mesh_data_model();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["data", "model"])], vec![])],
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.to_shardy_in_shardings_attribute(), r#"[<@mesh, [{"data", ?}]>]"#);
    }

    #[test]
    fn test_shard_map_renders_explicit_axes_in_traced_shardings() {
        let mesh = test_logical_mesh_data_model_explicit();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["data", "model"])], vec![])],
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.to_shardy_in_shardings_attribute(), r#"[<@mesh, [{"data", "model", ?}]>]"#);
    }

    #[test]
    fn test_shard_map_renders_out_shardings_attribute() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![test_sharding(&mesh, vec![ShardingDimension::replicated()], vec![])],
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.to_shardy_out_shardings_attribute(), r#"[<@mesh, [{?}], replicated={"x"}>]"#);
    }

    #[test]
    fn test_shard_map_renders_manual_axes_attribute() {
        let shard_map = ShardMap::new(
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
                MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            ])
            .unwrap(),
            Vec::new(),
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.manual_axes(), vec!["y".to_string()].as_slice());
        assert_eq!(shard_map.to_shardy_manual_axes_attribute(), r#"{"y"}"#);
    }

    #[test]
    fn test_shard_map_renders_manual_computation_attributes() {
        let mesh = test_logical_mesh_data_model();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["data"])], vec![])],
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["data"])], vec![])],
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(
            shard_map.to_shardy_manual_computation_attributes(),
            r#"in_shardings=[<@mesh, [{"data", ?}]>] out_shardings=[<@mesh, [{"data", ?}]>] manual_axes={"data"}"#
        );
    }

    #[test]
    fn test_derive_local_input_types_adds_varying_axes_from_in_specs() {
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![])],
            Vec::new(),
            vec!["x".into()],
            true,
        )
        .unwrap();
        let global_input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(test_sharding_with_varying(
                &mesh,
                vec![ShardingDimension::replicated()],
                vec![],
                vec![],
                vec!["y".into()],
            )),
        )
        .unwrap();
        let global_input_types = derive_global_input_types(&shard_map, &vec![global_input_type]).unwrap();
        let local_input_types = derive_local_input_types(&shard_map, &global_input_types).unwrap();

        assert_eq!(local_input_types[0].shape, Shape::new(vec![Size::Static(4)]));
        assert_eq!(
            local_input_types[0]
                .sharding
                .as_ref()
                .expect("local shard_map input should keep sharding metadata")
                .varying_manual_axes,
            BTreeSet::from(["x".to_string(), "y".to_string()])
        );
    }

    #[test]
    fn test_derive_global_input_types_rejects_mismatched_reduced_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input_sharding = Sharding::with_manual_axes(
            mesh.clone(),
            vec![ShardingDimension::replicated()],
            empty_axes(),
            ["x"],
            empty_axes(),
        )
        .unwrap();
        let shard_map =
            ShardMap::new(mesh.clone(), vec![input_sharding], Vec::new(), vec!["x".into(), "y".into()], true).unwrap();
        let global_input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::replicated()],
                    empty_axes(),
                    ["y"],
                    empty_axes(),
                )
                .unwrap(),
            ),
        )
        .unwrap();

        assert_eq!(
            derive_global_input_types(&shard_map, &vec![global_input_type]),
            Err(ShardMapTraceError::ShardingStateMismatch {
                value_kind: "input",
                value_index: 0,
                state_kind: "reduced axes",
                expected: vec!["x".to_string()],
                actual: vec!["y".to_string()],
            })
        );
    }

    #[test]
    fn test_derive_local_input_types_preserve_unreduced_and_reduced_axes_from_in_specs() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input_sharding = Sharding::with_manual_axes(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"])],
            ["y"],
            ["z"],
            empty_axes(),
        )
        .unwrap();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![input_sharding.clone()],
            Vec::new(),
            vec!["x".into(), "y".into(), "z".into()],
            true,
        )
        .unwrap();
        let global_input_types = derive_global_input_types(
            &shard_map,
            &vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, Some(input_sharding)).unwrap(),
            ],
        )
        .unwrap();
        let local_input_types = derive_local_input_types(&shard_map, &global_input_types).unwrap();

        assert_eq!(
            local_input_types[0]
                .sharding
                .as_ref()
                .expect("local shard_map input should keep sharding metadata")
                .unreduced_axes,
            BTreeSet::from(["y".to_string()])
        );
        assert_eq!(
            local_input_types[0]
                .sharding
                .as_ref()
                .expect("local shard_map input should keep sharding metadata")
                .reduced_manual_axes,
            BTreeSet::from(["z".to_string()])
        );
    }

    #[test]
    fn test_derive_global_output_types_drops_active_manual_varying_axes_and_preserves_outer_ones() {
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![])],
            vec!["x".into()],
            true,
        )
        .unwrap();
        let local_output_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(test_sharding_with_varying(
                &mesh,
                vec![ShardingDimension::replicated()],
                vec![],
                vec![],
                vec!["x".into(), "y".into()],
            )),
        )
        .unwrap();
        let global_output_types = derive_global_output_types(&shard_map, &vec![local_output_type]).unwrap();

        assert_eq!(global_output_types[0].shape, Shape::new(vec![Size::Static(8)]));
        assert_eq!(
            global_output_types[0]
                .sharding
                .as_ref()
                .expect("global shard_map output should keep sharding metadata")
                .varying_manual_axes,
            BTreeSet::from(["y".to_string()])
        );
    }

    #[test]
    fn test_derive_global_output_types_implicitly_adopts_unreduced_axes_from_out_specs() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let output_sharding =
            Sharding::with_unreduced_axes(mesh.clone(), vec![ShardingDimension::sharded(["x"])], ["y"]).unwrap();
        let shard_map =
            ShardMap::new(mesh.clone(), Vec::new(), vec![output_sharding.clone()], vec!["x".into(), "y".into()], true)
                .unwrap();
        let local_output_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, None).unwrap();
        let global_output_types = derive_global_output_types(&shard_map, &vec![local_output_type]).unwrap();

        assert_eq!(
            global_output_types[0]
                .sharding
                .as_ref()
                .expect("global shard_map output should keep sharding metadata")
                .unreduced_axes,
            BTreeSet::from(["y".to_string()])
        );
    }

    #[test]
    fn test_derive_global_output_types_rejects_extra_local_unreduced_axes_when_check_vma_is_enabled() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap()],
            vec!["x".into(), "y".into()],
            true,
        )
        .unwrap();
        let local_output_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::with_unreduced_axes(mesh, vec![ShardingDimension::replicated()], ["y"]).unwrap()),
        )
        .unwrap();

        assert_eq!(
            derive_global_output_types(&shard_map, &vec![local_output_type]),
            Err(ShardMapTraceError::ShardingStateMismatch {
                value_kind: "output",
                value_index: 0,
                state_kind: "unreduced axes",
                expected: Vec::new(),
                actual: vec!["y".to_string()],
            })
        );
    }

    #[test]
    fn test_derive_global_output_types_rejects_reduced_axis_mismatch_when_check_vma_is_enabled() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let output_sharding = Sharding::with_manual_axes(
            mesh.clone(),
            vec![ShardingDimension::replicated()],
            empty_axes(),
            ["x"],
            empty_axes(),
        )
        .unwrap();
        let shard_map =
            ShardMap::new(mesh.clone(), Vec::new(), vec![output_sharding], vec!["x".into(), "y".into()], true).unwrap();
        let local_output_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, None).unwrap();

        assert_eq!(
            derive_global_output_types(&shard_map, &vec![local_output_type]),
            Err(ShardMapTraceError::ShardingStateMismatch {
                value_kind: "output",
                value_index: 0,
                state_kind: "reduced axes",
                expected: vec!["x".to_string()],
                actual: Vec::new(),
            })
        );
    }

    #[test]
    fn test_derive_global_output_types_rejects_omitted_varying_manual_axis_when_check_vma_is_enabled() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![test_sharding(&mesh, vec![ShardingDimension::replicated()], vec![])],
            vec![],
            true,
        )
        .unwrap();
        let local_output_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(test_sharding_with_varying(
                &mesh,
                vec![ShardingDimension::replicated()],
                vec![],
                vec![],
                vec!["x".into()],
            )),
        )
        .unwrap();

        assert_eq!(
            derive_global_output_types(&shard_map, &vec![local_output_type]),
            Err(ShardMapTraceError::OutputVaryingManualAxisNotInOutSpecs {
                output_index: 0,
                axis_name: "x".to_string(),
            })
        );
    }

    #[test]
    fn test_derive_global_output_types_ignores_omitted_varying_manual_axis_when_check_vma_is_disabled() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![test_sharding(&mesh, vec![ShardingDimension::replicated()], vec![])],
            vec![],
            false,
        )
        .unwrap();
        let local_output_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(test_sharding_with_varying(
                &mesh,
                vec![ShardingDimension::replicated()],
                vec![],
                vec![],
                vec!["x".into()],
            )),
        )
        .unwrap();
        let global_output_types = derive_global_output_types(&shard_map, &vec![local_output_type]).unwrap();

        assert_eq!(
            global_output_types[0]
                .sharding
                .as_ref()
                .expect("global shard_map output should keep sharding metadata")
                .varying_manual_axes,
            BTreeSet::<String>::new()
        );
    }

    #[test]
    fn test_zero_like_and_one_like_clear_varying_manual_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let tensor = ShardMapTensor::new(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Size::Static(4)]),
                None,
                Some(
                    Sharding::with_manual_axes(
                        mesh.clone(),
                        vec![ShardingDimension::replicated()],
                        ["y"],
                        ["z"],
                        ["x"],
                    )
                    .unwrap(),
                ),
            )
            .unwrap(),
        );

        assert_eq!(
            tensor.zero_like().tpe().sharding.expect("zero_like should keep sharding metadata").unreduced_axes,
            BTreeSet::from(["y".to_string()])
        );
        assert_eq!(
            tensor
                .zero_like()
                .tpe()
                .sharding
                .expect("zero_like should keep sharding metadata")
                .reduced_manual_axes,
            BTreeSet::from(["z".to_string()])
        );
        assert_eq!(
            tensor
                .zero_like()
                .tpe()
                .sharding
                .expect("zero_like should keep sharding metadata")
                .varying_manual_axes,
            BTreeSet::<String>::new()
        );
        assert_eq!(
            tensor.one_like().tpe().sharding.expect("one_like should keep sharding metadata").unreduced_axes,
            BTreeSet::from(["y".to_string()])
        );
        assert_eq!(
            tensor
                .one_like()
                .tpe()
                .sharding
                .expect("one_like should keep sharding metadata")
                .reduced_manual_axes,
            BTreeSet::from(["z".to_string()])
        );
        assert_eq!(
            tensor
                .one_like()
                .tpe()
                .sharding
                .expect("one_like should keep sharding metadata")
                .varying_manual_axes,
            BTreeSet::<String>::new()
        );
    }

    #[test]
    fn test_shard_map_trace_derives_types_and_renders_mlir() {
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x| x.clone() + x,
            global_input_type.clone(),
            mesh.clone(),
            input_sharding.clone(),
            input_sharding.clone(),
        )
        .unwrap();
        let expected_global_input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, Some(input_sharding.clone()))
                .unwrap();
        let expected_local_input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(2)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    empty_axes(),
                    empty_axes(),
                    ["x"],
                )
                .unwrap(),
            ),
        )
        .unwrap();

        assert_eq!(traced.global_input_types(), &expected_global_input_type);
        assert_eq!(traced.local_input_types(), &expected_local_input_type);
        assert_eq!(traced.local_output_types(), &expected_local_input_type);
        assert_eq!(traced.global_output_types(), &expected_global_input_type);
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_shard_map_trace_hides_auto_axes_in_type_level_shardings() {
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(16)]), None, None).unwrap();
        let mesh = test_logical_mesh_data_model();
        let input_sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["data", "model"])], vec![]);
        let projected_sharding = input_sharding.without_auto_axes();
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x| x.clone() + x,
            global_input_type.clone(),
            mesh.clone(),
            input_sharding.clone(),
            input_sharding,
        )
        .unwrap();
        let expected_global_input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(16)]), None, Some(projected_sharding.clone()))
                .unwrap();
        let expected_local_input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["data"])],
                    empty_axes(),
                    empty_axes(),
                    ["data"],
                )
                .unwrap(),
            ),
        )
        .unwrap();

        assert_eq!(traced.global_input_types(), &expected_global_input_type);
        assert_eq!(traced.local_input_types(), &expected_local_input_type);
        assert_eq!(traced.global_output_types(), &expected_global_input_type);
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["data"=2, "model"=4]>
                  func.func @main(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"data", ?}]>] out_shardings=[<@mesh, [{"data", ?}]>] manual_axes={"data"} (%arg1: tensor<8xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<8xf32>
                      sdy.return %1 : tensor<8xf32>
                    } : (tensor<16xf32>) -> tensor<16xf32>
                    return %0 : tensor<16xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_shard_map_trace_can_render_nested_shard_maps() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let inner_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let outer_sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let inner_sharding = test_sharding(&inner_mesh, vec![ShardingDimension::sharded(["y"])], vec![]);
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            {
                let inner_mesh = inner_mesh.clone();
                let inner_sharding = inner_sharding.clone();
                move |x: ShardMapTracer| {
                    let nested: ShardMapTracer = shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                        |y: ShardMapTracer| y.clone() + y,
                        x.clone(),
                        inner_mesh.clone(),
                        inner_sharding.clone(),
                        inner_sharding.clone(),
                    )
                    .expect("nested shard_map should trace");
                    nested + x
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap(),
            mesh,
            outer_sharding.clone(),
            outer_sharding,
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", ?}]>] out_shardings=[<@mesh, [{"x", ?}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
                      %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh, [{"y", ?}]>] out_shardings=[<@mesh, [{"y", ?}]>] manual_axes={"y"} (%arg2: tensor<2xf32>) {
                        %3 = stablehlo.add %arg2, %arg2 : tensor<2xf32>
                        sdy.return %3 : tensor<2xf32>
                      } : (tensor<4xf32>) -> tensor<4xf32>
                      %2 = stablehlo.add %1, %arg1 : tensor<4xf32>
                      sdy.return %2 : tensor<4xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_shard_map_trace_rejects_dynamic_input_types() {
        let dynamic_input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Dynamic(None)]), None, None).unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let result: Result<TracedShardMap<ArrayType, ArrayType>, ShardMapTraceError> = shard_map(
            |x| x.clone() + x,
            dynamic_input_type,
            mesh.clone(),
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]),
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]),
        );

        assert!(matches!(
            result,
            Err(ShardMapTraceError::DynamicShapeNotSupported { value_kind: "input", value_index: 0, dimension: 0 })
        ));
    }

    #[test]
    fn test_shard_map_infers_single_input_closure_argument_type() {
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x| x.clone() + x,
            global_input_type,
            mesh.clone(),
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]),
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]),
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_traced_shard_map_executes_end_to_end_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) }))
            .expect("failed to create 4-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);

        let mesh_devices = client_devices
            .iter()
            .map(|device| MeshDevice::new(device.id().unwrap(), device.process_index().unwrap()))
            .collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            mesh_devices,
        )
        .unwrap();

        let sharding =
            Sharding::new(device_mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x| x.clone() + x,
            global_input_type,
            device_mesh.logical_mesh.clone(),
            sharding.clone(),
            sharding.clone(),
        )
        .unwrap();
        let mlir_program = traced.to_mlir_module("main").unwrap();

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        f32_values_to_bytes(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let input_array =
            Array::new(static_sharded_array_type(DataType::F32, &[8], sharding), device_mesh, input_buffers).unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 4);
        let expected_values_by_device = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                (device.id().unwrap(), [device_index as f32 * 4.0 + 2.0, device_index as f32 * 4.0 + 4.0])
            })
            .collect::<HashMap<_, _>>();
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();

        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), 0, None, Some(file!()), None, None)
            .unwrap();

        assert_eq!(outputs.len(), execution_device_ids.len());
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            output.done.r#await().unwrap();
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(
                two_f32s_from_bytes(output_bytes.as_slice()),
                *expected_values_by_device.get(&device_id).unwrap()
            );
        }
    }

    #[test]
    fn test_traced_shard_map_matmul_renders_and_executes_end_to_end_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(8) }))
            .expect("failed to create 8-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 8);

        let mesh_devices = client_devices
            .iter()
            .map(|device| MeshDevice::new(device.id().unwrap(), device.process_index().unwrap()))
            .collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 8, MeshAxisType::Manual).unwrap()]).unwrap(),
            mesh_devices,
        )
        .unwrap();

        let lhs_sharding = Sharding::new(
            device_mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let rhs_sharding = Sharding::replicated(device_mesh.logical_mesh.clone(), 2);
        let output_sharding = Sharding::new(
            device_mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let global_input_types = (
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(4)]), None, None).unwrap(),
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4), Size::Static(2)]), None, None).unwrap(),
        );
        let traced: TracedShardMap<(ArrayType, ArrayType), ArrayType> = shard_map(
            |(lhs, rhs)| lhs.matmul(rhs),
            global_input_types,
            device_mesh.logical_mesh.clone(),
            (lhs_sharding.clone(), rhs_sharding.clone()),
            output_sharding.clone(),
        )
        .unwrap();
        let mlir_program = traced.to_mlir_module("main").unwrap();

        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=8]>
                  func.func @main(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], replicated={"x"}>}) -> (tensor<8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {}], replicated={"x"}>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg2: tensor<1x4xf32>, %arg3: tensor<4x2xf32>) {
                      %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4xf32>, tensor<4x2xf32>) -> tensor<1x2xf32>
                      sdy.return %1 : tensor<1x2xf32>
                    } : (tensor<8x4xf32>, tensor<4x2xf32>) -> tensor<8x2xf32>
                    return %0 : tensor<8x2xf32>
                  }
                }
            "#}
        );

        let lhs_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(row_index, device)| {
                let row = row_index as f32;
                client
                    .buffer(
                        f32_values_to_bytes(&[row, row + 1.0, row + 2.0, row + 3.0]).as_slice(),
                        BufferType::F32,
                        [1u64, 4u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let rhs_values = [1.0f32, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0];
        let rhs_buffers = client_devices
            .iter()
            .map(|device| {
                client
                    .buffer(
                        f32_values_to_bytes(rhs_values.as_slice()).as_slice(),
                        BufferType::F32,
                        [4u64, 2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let lhs_array = Array::new(
            static_sharded_array_type(DataType::F32, &[8, 4], lhs_sharding.clone()),
            device_mesh.clone(),
            lhs_buffers,
        )
        .unwrap();
        let rhs_array =
            Array::new(static_sharded_array_type(DataType::F32, &[4, 2], rhs_sharding), device_mesh, rhs_buffers)
                .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(8)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 8);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let row_start_by_device = execution_device_ids
            .iter()
            .map(|device_id| {
                let row_start = lhs_array.shard_for_device(*device_id).unwrap().slice[0].start;
                (*device_id, row_start)
            })
            .collect::<HashMap<_, _>>();

        let execute_arguments =
            Array::into_execute_arguments(vec![lhs_array, rhs_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), 0, None, Some(file!()), None, None)
            .unwrap();

        assert_eq!(outputs.len(), execution_device_ids.len());
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            output.done.r#await().unwrap();
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values = two_f32s_from_bytes(output_bytes.as_slice());
            let row = *row_start_by_device.get(&device_id).unwrap() as f32;
            assert_eq!(values[0], 4.0 * row + 8.0);
            assert_eq!(values[1], 4.0 * row + 4.0);
        }
    }

    #[test]
    fn test_traced_shard_map_composes_grad_and_vmap_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) }))
            .expect("failed to create 4-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);

        let mesh_devices = client_devices
            .iter()
            .map(|device| MeshDevice::new(device.id().unwrap(), device.process_index().unwrap()))
            .collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            mesh_devices,
        )
        .unwrap();

        let sharding =
            Sharding::new(device_mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x: ShardMapTracer| {
                let gradient: ShardMapTracer =
                    grad(|y: ShardMapTracer| y.sin(), x.clone()).expect("gradient inside shard_map should succeed");
                let lanes: Vec<ShardMapTracer> = vmap(
                    |y: crate::tracing_v2::Batch<ShardMapTracer>| y.clone() + y.one_like(),
                    vec![gradient.clone(), gradient],
                )
                .expect("vmap should succeed");
                lanes[0].clone() + lanes[1].clone()
            },
            global_input_type,
            device_mesh.logical_mesh.clone(),
            sharding.clone(),
            sharding.clone(),
        )
        .unwrap();
        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.cosine %arg1 : tensor<2xf32>
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                      %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
                      %3 = stablehlo.multiply %1, %2 : tensor<2xf32>
                      %4 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
                      %5 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
                      %6 = stablehlo.concatenate %4, %5, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
                      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                      %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2xf32>
                      %8 = stablehlo.broadcast_in_dim %7, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
                      %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<2x2xf32>
                      %10 = stablehlo.add %6, %9 : tensor<2x2xf32>
                      %11 = stablehlo.slice %10 [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
                      %12 = stablehlo.reshape %11 : (tensor<1x2xf32>) -> tensor<2xf32>
                      %13 = stablehlo.slice %10 [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
                      %14 = stablehlo.reshape %13 : (tensor<1x2xf32>) -> tensor<2xf32>
                      %15 = stablehlo.add %12, %14 : tensor<2xf32>
                      sdy.return %15 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        f32_values_to_bytes(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let input_array =
            Array::new(static_sharded_array_type(DataType::F32, &[8], sharding), device_mesh, input_buffers).unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 4);
        let expected_values_by_device = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let first_input = device_index as f32 * 2.0 + 1.0;
                let second_input = device_index as f32 * 2.0 + 2.0;
                (device.id().unwrap(), [2.0 * first_input.cos() + 2.0, 2.0 * second_input.cos() + 2.0])
            })
            .collect::<HashMap<_, _>>();
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();

        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), 0, None, Some(file!()), None, None)
            .unwrap();

        assert_eq!(outputs.len(), execution_device_ids.len());
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            output.done.r#await().unwrap();
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_two_f32s_approx_eq(
                two_f32s_from_bytes(output_bytes.as_slice()),
                *expected_values_by_device.get(&device_id).unwrap(),
            );
        }
    }

    #[test]
    fn test_trace_with_sharding_constraint_renders_mlir() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();

        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    with_sharding_constraint(x.sin(), sharding.clone())
                        .expect("with_sharding_constraint should stage on traced XLA values")
                }
            },
            global_input_type.clone(),
        )
        .unwrap();

        let expected_output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, Some(sharding.clone())).unwrap();
        assert_eq!(traced.global_input_types(), &global_input_type);
        assert_eq!(traced.global_output_types(), &expected_output_type);
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = stablehlo.sine %arg0 : tensor<8xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<8xf32>
                    return %1 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_traced_xla_grad_around_shard_map_renders_and_executes_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) }))
            .expect("failed to create 4-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);

        let mesh_devices = client_devices
            .iter()
            .map(|device| MeshDevice::new(device.id().unwrap(), device.process_index().unwrap()))
            .collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            mesh_devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();

        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh.clone();
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    grad(
                        {
                            let mesh = mesh.clone();
                            let sharding = sharding.clone();
                            move |y: ShardMapTracer| {
                                shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                                    |local_x: ShardMapTracer| local_x.sin(),
                                    y,
                                    mesh.clone(),
                                    sharding.clone(),
                                    sharding.clone(),
                                )
                                .expect("shard_map inside grad should trace")
                            }
                        },
                        x,
                    )
                    .expect("grad around shard_map should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
                    %1 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %3 = stablehlo.cosine %arg1 : tensor<2xf32>
                      sdy.return %3 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    %2 = sdy.manual_computation(%0, %1) in_shardings=[<@mesh, [{"x"}]>, <@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>, %arg2: tensor<2xf32>) {
                      %3 = stablehlo.multiply %arg2, %arg1 : tensor<2xf32>
                      sdy.return %3 : tensor<2xf32>
                    } : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
                    return %2 : tensor<8xf32>
                  }
                }
            "#}
        );

        let input_values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [input_values[device_index * 2], input_values[device_index * 2 + 1]];
                client
                    .buffer(
                        f32_values_to_bytes(shard_values.as_slice()).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array =
            Array::new(static_sharded_array_type(DataType::F32, &[8], sharding), device_mesh, input_buffers).unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 4);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), 0, None, Some(file!()), None, None)
            .unwrap();

        assert_eq!(outputs.len(), execution_device_ids.len());
        let expected_values_by_device = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                (
                    device.id().unwrap(),
                    vec![input_values[device_index * 2].cos(), input_values[device_index * 2 + 1].cos()],
                )
            })
            .collect::<HashMap<_, _>>();
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            output.done.r#await().unwrap();
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let actual_values = f32s_from_bytes(output_bytes.as_slice());
            let expected_values = expected_values_by_device.get(&device_id).unwrap();
            assert_eq!(actual_values.len(), expected_values.len());
            for (actual, expected) in actual_values.into_iter().zip(expected_values.iter().copied()) {
                let delta = (actual - expected).abs();
                assert!(delta <= 1e-5, "expected {actual} ~= {expected}; absolute error {delta}");
            }
        }
    }

    #[test]
    fn test_shard_map_manual_computation_executes_end_to_end_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) }))
            .expect("failed to create 4-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);

        let mesh_devices = client_devices
            .iter()
            .map(|device| MeshDevice::new(device.id().unwrap(), device.process_index().unwrap()))
            .collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            mesh_devices,
        )
        .unwrap();

        let sharding =
            Sharding::new(device_mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let shard_map = ShardMap::new(
            device_mesh.logical_mesh.clone(),
            vec![sharding.clone()],
            vec![sharding.clone()],
            vec![],
            true,
        )
        .unwrap();
        assert_eq!(shard_map.local_input_shape(0, &[8]).unwrap(), vec![2]);
        assert_eq!(shard_map.local_output_shape(0, &[8]).unwrap(), vec![2]);

        let context = MlirContext::new();
        let input_sharding = shard_map.in_shardings()[0].to_shardy(context.unknown_location()).to_string();
        let output_sharding = shard_map.out_shardings()[0].to_shardy(context.unknown_location()).to_string();
        let manual_computation_attributes = shard_map.to_shardy_manual_computation_attributes();
        let mesh_module = context.module(context.unknown_location());
        let mesh_operation = mesh_module
            .body()
            .append_operation(shard_map.mesh().to_shardy(context.unknown_location()))
            .to_string();

        let mlir_program = format!(
            r#"
                module {{
                    {mesh_operation}
                    func.func @main(
                        %arg0: tensor<8xf32> {{sdy.sharding = {input_sharding}}}
                    ) -> (tensor<8xf32> {{sdy.sharding = {output_sharding}}}) {{
                        %0 = sdy.manual_computation(%arg0) {manual_computation_attributes} (%arg1: tensor<2xf32>) {{
                            %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32>
                            sdy.return %1 : tensor<2xf32>
                        }} : (tensor<8xf32>) -> tensor<8xf32>
                        return %0 : tensor<8xf32>
                    }}
                }}
            "#
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        f32_values_to_bytes(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let input_array =
            Array::new(static_sharded_array_type(DataType::F32, &[8], sharding), device_mesh, input_buffers).unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 4);
        let expected_values_by_device = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                (device.id().unwrap(), [device_index as f32 * 4.0 + 2.0, device_index as f32 * 4.0 + 4.0])
            })
            .collect::<HashMap<_, _>>();
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();

        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), 0, None, Some(file!()), None, None)
            .unwrap();

        assert_eq!(outputs.len(), execution_device_ids.len());
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            output.done.r#await().unwrap();
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(
                two_f32s_from_bytes(output_bytes.as_slice()),
                *expected_values_by_device.get(&device_id).unwrap()
            );
        }
    }
}
