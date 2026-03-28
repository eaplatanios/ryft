//! Manual SPMD metadata for Shardy `sdy.manual_computation`.
//!
//! This module provides the tracing-backed `shard_map` surface for `ryft-core::xla`.
//!
//! The default public entry point is [`shard_map`], which stages a Rust closure over shard-local
//! tensor types derived from global [`ArrayType`] metadata, lowers the resulting `tracing_v2`
//! graph to StableHLO/Shardy MLIR, and returns a [`TracedShardMap`] handle for inspection and
//! lowering.
//!
//! Internally, the module still keeps a small metadata model of JAX's `shard_map`: a logical mesh,
//! per-input and per-output shardings, and the set of mesh axes that are handled manually inside
//! the computation body. That internal metadata is responsible for:
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
//! The public [`shard_map`] function accepts structured [`PartitionSpec`] values and internally
//! lowers them to [`NamedSharding`] values.
//!
//! The shard-map implementation derives its manual axes from the mesh itself: every axis whose type is
//! [`Manual`](crate::sharding::MeshAxisType::Manual) is treated as manual for the
//! `sdy.manual_computation` region.
//!
//! This matters because the surrounding mesh still determines which axes are handled manually
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
use std::{collections::HashSet, fmt::Debug};

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
use crate::sharding::{LogicalMesh, MeshAxisType, SHARDY_MESH_SYMBOL_NAME, ShardingError};
use crate::tracing_v2::{
    CompiledFunction, FloatExt, JitTracer, Linearized, MatrixOps, OneLike, TraceError, TraceValue, ZeroLike, jit,
};
use crate::types::{ArrayType, Shape, Size, Typed};

use super::lowering::LoweringError;
use super::sharding::{NamedSharding, PartitionDimension, PartitionSpec};

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

impl ZeroLike for ShardMapTensor {
    fn zero_like(&self) -> Self {
        Self::constant(self.r#type.clone(), ShardMapConstantKind::Zero)
    }
}

impl OneLike for ShardMapTensor {
    fn one_like(&self) -> Self {
        Self::constant(self.r#type.clone(), ShardMapConstantKind::One)
    }
}

impl Add for ShardMapTensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.r#type == rhs.r#type { Self::new(self.r#type) } else { Self::new(self.r#type) }
    }
}

impl Mul for ShardMapTensor {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.r#type == rhs.r#type { Self::new(self.r#type) } else { Self::new(self.r#type) }
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
            ) if left_data_type == right_data_type && left_cols == right_rows => Self::new(ArrayType::new(
                left_data_type,
                Shape::new(vec![Size::Static(*left_rows), Size::Static(*right_cols)]),
                None,
            )),
            _ => Self::new(self.r#type),
        }
    }

    fn transpose_matrix(self) -> Self {
        match self.r#type.shape.dimensions.as_slice() {
            [first, second] => {
                Self::new(ArrayType::new(self.r#type.data_type, Shape::new(vec![*second, *first]), None))
            }
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
    type Return<Input, Output>
    where
        Input: Parameterized<Self, ParameterStructure: Clone>,
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<PartitionSpec>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
        Output::Family: ParameterizedFamily<PartitionSpec>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Linearized<ShardMapTracer>>;

    /// Invokes [`shard_map`] for one specific tracing regime.
    fn invoke<F, Input, Output>(
        function: F,
        inputs: Input,
        mesh: LogicalMesh,
        in_specs: Input::To<PartitionSpec>,
        out_specs: Output::To<PartitionSpec>,
    ) -> Result<Self::Return<Input, Output>, ShardMapTraceError>
    where
        Input: Parameterized<Self, ParameterStructure: Clone>,
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<PartitionSpec>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
        Output::Family: ParameterizedFamily<PartitionSpec>
            + ParameterizedFamily<ShardMapTensor>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Linearized<ShardMapTracer>>,
        F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>;
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
pub fn trace<F, Input, Output>(
    function: F,
    global_input_types: Input,
) -> Result<TracedXlaProgram<Input, Output>, ShardMapTraceError>
where
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
    F: FnOnce(TracedXlaInput<Input>) -> TracedXlaOutput<Output>,
{
    let (global_output_types, compiled) = trace_xla_function(function, &global_input_types)?;
    Ok(TracedXlaProgram { global_input_types, global_output_types, compiled })
}

/// Stages a traced shard-map body over the provided mesh and partition specs.
///
/// This is the ergonomic public entry point for traced XLA shard-map staging. It mirrors the
/// function-first shape of JAX's `shard_map` while adapting it to Rust and `tracing_v2` by
/// requiring explicit `global_input_types`.
///
/// Mesh axes whose type is [`Manual`](crate::sharding::MeshAxisType::Manual) define the manual axes
/// of the computation. Structured `in_specs` and `out_specs` follow the same `Parameterized`
/// layout as the corresponding input and output types. The body closure receives only the traced
/// local inputs, which lets common cases compile cleanly as `|x| ...` or `|(lhs, rhs)| ...`
/// without explicit tracer annotations.
///
/// # Parameters
///
///   - `function`: Body closure to trace over local shard-map values.
///   - `global_input_types`: Global input array types used to derive the local body argument types.
///   - `mesh`: Logical mesh that the manual computation is defined over.
///   - `in_specs`: Structured partition specs for the global inputs.
///   - `out_specs`: Structured partition specs for the global outputs.
#[allow(private_bounds, private_interfaces)]
pub fn shard_map<F, Input, Output, Leaf>(
    function: F,
    inputs: Input,
    mesh: LogicalMesh,
    in_specs: Input::To<PartitionSpec>,
    out_specs: Output::To<PartitionSpec>,
) -> Result<<Leaf as ShardMapInvocationLeaf>::Return<Input, Output>, ShardMapTraceError>
where
    Leaf: ShardMapInvocationLeaf,
    Input: Parameterized<Leaf, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<PartitionSpec>
        + ParameterizedFamily<ShardMapTensor>
        + ParameterizedFamily<ShardMapTracer>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<PartitionSpec>
        + ParameterizedFamily<ShardMapTensor>
        + ParameterizedFamily<ShardMapTracer>
        + ParameterizedFamily<Linearized<ShardMapTracer>>,
    F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>,
{
    Leaf::invoke(function, inputs, mesh, in_specs, out_specs)
}

/// Traced shard-map program backed by a staged `tracing_v2` graph.
#[allow(private_bounds, private_interfaces)]
pub struct TracedShardMap<Input, Output>
where
    Input: Parameterized<ArrayType>,
    Input::Family: ParameterizedFamily<ShardMapTensor>,
    Output: Parameterized<ArrayType>,
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
pub struct TracedXlaProgram<Input, Output>
where
    Input: Parameterized<ArrayType>,
    Input::Family: ParameterizedFamily<ShardMapTensor>,
    Output: Parameterized<ArrayType>,
    Output::Family: ParameterizedFamily<ShardMapTensor>,
{
    global_input_types: Input,
    global_output_types: Output,
    compiled: CompiledFunction<ShardMapTensor, Input::To<ShardMapTensor>, Output::To<ShardMapTensor>>,
}

/// Metadata describing one manual SPMD computation over a mesh.
///
/// A `ShardMap` stores the mesh plus the validated per-input and per-output shardings.
///
/// The manual axes are not stored separately; they are always derived from the mesh axes whose
/// type is [`Manual`](crate::sharding::MeshAxisType::Manual).
///
/// The public constructors accept [`PartitionSpec`] values because that is the natural
/// JAX-facing surface. Internally, those partition specs are converted to [`NamedSharding`]
/// values and then projected into traced/type-level semantics, so `Auto` mesh axes remain
/// hidden while `Manual` axes still drive the manual-computation body.
///
/// Reference: https://docs.jax.dev/en/latest/notebooks/shard_map.html.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ShardMap {
    mesh: LogicalMesh,
    in_shardings: Vec<NamedSharding>,
    out_shardings: Vec<NamedSharding>,
}

impl ShardMap {
    /// Creates a `ShardMap` whose manual axes are derived from the mesh.
    ///
    /// Every mesh axis with type [`Manual`](crate::sharding::MeshAxisType::Manual) is treated as
    /// manual inside the body. The constructor returns [`ShardMapError::MeshHasNoManualAxes`] if
    /// the mesh contains no manual axes.
    ///
    /// # Parameters
    ///
    ///   - `mesh`: Logical mesh that the manual computation is defined over.
    ///   - `in_specs`: Per-input partition specs for the global inputs.
    ///   - `out_specs`: Per-output partition specs for the global outputs.
    pub(crate) fn new(
        mesh: LogicalMesh,
        in_specs: Vec<PartitionSpec>,
        out_specs: Vec<PartitionSpec>,
    ) -> Result<Self, ShardMapError> {
        let in_shardings = build_named_shardings(&mesh, in_specs, "input")?;
        let out_shardings = build_named_shardings(&mesh, out_specs, "output")?;
        Ok(Self { mesh, in_shardings, out_shardings })
    }

    /// Builds a shard map directly from already-validated shardings.
    pub(crate) fn from_shardings(
        mesh: LogicalMesh,
        in_shardings: Vec<NamedSharding>,
        out_shardings: Vec<NamedSharding>,
    ) -> Self {
        Self { mesh, in_shardings, out_shardings }
    }

    /// Returns the logical mesh of this manual computation.
    pub(crate) fn mesh(&self) -> &LogicalMesh {
        &self.mesh
    }

    /// Returns the validated per-input shardings.
    pub(crate) fn in_shardings(&self) -> &[NamedSharding] {
        self.in_shardings.as_slice()
    }

    /// Returns the validated per-output shardings.
    pub(crate) fn out_shardings(&self) -> &[NamedSharding] {
        self.out_shardings.as_slice()
    }

    /// Returns the manual mesh axes in mesh order.
    fn manual_axes(&self) -> Vec<&str> {
        self.mesh
            .axes
            .iter()
            .filter_map(|axis| (axis.r#type == MeshAxisType::Manual).then_some(axis.name.as_str()))
            .collect()
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
        local_shape_for_sharding(&self.in_shardings[input_index], global_shape, "input", input_index)
    }

    /// Returns the local body shape for output `output_index`.
    ///
    /// # Parameters
    ///
    ///   - `output_index`: Index of the output sharding to use.
    ///   - `global_shape`: Global output shape associated with that output.
    #[cfg(test)]
    fn local_output_shape(&self, output_index: usize, global_shape: &[usize]) -> Result<Vec<usize>, ShardMapError> {
        local_shape_for_sharding(&self.out_shardings[output_index], global_shape, "output", output_index)
    }

    /// Renders the Shardy `in_shardings=[...]` attribute payload.
    ///
    /// The returned string is suitable for direct insertion into an `sdy.manual_computation`
    /// operation.
    ///
    #[cfg(test)]
    fn to_shardy_in_shardings_attribute(&self) -> String {
        render_shardy_sharding_list(self.in_shardings.as_slice())
    }

    /// Renders the Shardy `out_shardings=[...]` attribute payload.
    ///
    #[cfg(test)]
    fn to_shardy_out_shardings_attribute(&self) -> String {
        render_shardy_sharding_list(self.out_shardings.as_slice())
    }

    /// Renders the Shardy `manual_axes={...}` attribute payload.
    #[cfg(test)]
    fn to_shardy_manual_axes_attribute(&self) -> String {
        let manual_axes = self.manual_axes();
        render_shardy_axes(manual_axes.as_slice())
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
        shardy_tensor_sharding_per_value(self.in_shardings.as_slice(), context)
    }

    /// Builds the typed Shardy `out_shardings` attribute used by `sdy.manual_computation`.
    pub(crate) fn to_shardy_out_shardings<'c, 't>(
        &self,
        context: &'c MlirContext<'t>,
    ) -> TensorShardingPerValueAttributeRef<'c, 't> {
        shardy_tensor_sharding_per_value(self.out_shardings.as_slice(), context)
    }

    /// Builds the typed Shardy `manual_axes` attribute used by `sdy.manual_computation`.
    pub(crate) fn to_shardy_manual_axes<'c, 't>(&self, context: &'c MlirContext<'t>) -> ManualAxesAttributeRef<'c, 't> {
        let manual_axes = self.manual_axes();
        context.shardy_manual_axes(manual_axes.as_slice())
    }

    /// Traces a shard-map body over local body tensor types using `tracing_v2::jit`.
    ///
    /// # Parameters
    ///
    ///   - `function`: Body closure to trace over local shard-map values.
    ///   - `global_input_types`: Global input array types in the same leaf order as the shard-map
    ///     input shardings.
    pub(crate) fn trace<F, Input, Output>(
        &self,
        function: F,
        global_input_types: Input,
    ) -> Result<TracedShardMap<Input, Output>, ShardMapTraceError>
    where
        Input: Parameterized<ArrayType, ParameterStructure: Clone>,
        Input::Family: ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
        Output::Family: ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
        F: FnOnce(ShardMapLocalTraceInput<Input>) -> ShardMapLocalTraceOutput<Output>,
    {
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
impl<Input, Output> TracedShardMap<Input, Output>
where
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<ShardMapTensor>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
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
impl<Input, Output> TracedXlaProgram<Input, Output>
where
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<ShardMapTensor>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
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
    pub(crate) fn from_traced<Input, Output>(traced: &TracedShardMap<Input, Output>) -> Self
    where
        Input: Parameterized<ArrayType, ParameterStructure: Clone>,
        Input::Family: ParameterizedFamily<ShardMapTensor>,
        Output: Parameterized<ArrayType, ParameterStructure: Clone>,
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

fn derive_local_input_types<Input>(
    shard_map: &ShardMap,
    global_input_types: &Input,
) -> Result<Input, ShardMapTraceError>
where
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
{
    let global_input_type_count = global_input_types.parameter_count();
    if global_input_type_count != shard_map.in_shardings().len() {
        return Err(ShardMapTraceError::InputTypeCountMismatch {
            expected: shard_map.in_shardings().len(),
            actual: global_input_type_count,
        });
    }

    let structure = global_input_types.parameter_structure();
    let local_input_types = global_input_types
        .parameters()
        .cloned()
        .enumerate()
        .map(|(input_index, global_input_type)| {
            ensure_static_array_type(&global_input_type, "input", input_index)?;
            let global_shape = static_shape_values(&global_input_type, "input", input_index)?;
            let local_shape = shard_map.local_input_shape(input_index, &global_shape)?;
            Ok::<ArrayType, ShardMapTraceError>(ArrayType::new(
                global_input_type.data_type,
                Shape::new(local_shape.into_iter().map(Size::Static).collect()),
                global_input_type.layout.clone(),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Input::from_parameters(structure, local_input_types)?)
}

fn traced_input_tensors<Input>(local_input_types: &Input) -> Result<Input::To<ShardMapTensor>, ShardMapTraceError>
where
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<ShardMapTensor>,
{
    let structure = local_input_types.parameter_structure();
    let tensors = local_input_types.parameters().cloned().map(ShardMapTensor::new).collect::<Vec<_>>();
    Ok(Input::To::<ShardMapTensor>::from_parameters(structure, tensors)?)
}

fn trace_xla_function<F, Input, Output>(
    function: F,
    input_types: &Input,
) -> Result<
    (Output, CompiledFunction<ShardMapTensor, Input::To<ShardMapTensor>, Output::To<ShardMapTensor>>),
    ShardMapTraceError,
>
where
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<ShardMapTensor> + ParameterizedFamily<ShardMapTracer>,
    F: FnOnce(TracedXlaInput<Input>) -> TracedXlaOutput<Output>,
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

pub(crate) fn derive_global_output_types<Output>(
    shard_map: &ShardMap,
    local_output_types: &Output,
) -> Result<Output, ShardMapTraceError>
where
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
{
    let local_output_type_count = local_output_types.parameter_count();
    if local_output_type_count != shard_map.out_shardings().len() {
        return Err(ShardMapTraceError::OutputTypeCountMismatch {
            expected: shard_map.out_shardings().len(),
            actual: local_output_type_count,
        });
    }

    let structure = local_output_types.parameter_structure();
    let global_output_types = local_output_types
        .parameters()
        .cloned()
        .enumerate()
        .map(|(output_index, local_output_type)| {
            ensure_static_array_type(&local_output_type, "output", output_index)?;
            let global_shape = global_shape_for_sharding(
                shard_map.out_shardings()[output_index].partition_spec(),
                shard_map.mesh(),
                static_shape_values(&local_output_type, "output", output_index)?,
                output_index,
            )?;
            Ok::<ArrayType, ShardMapTraceError>(ArrayType::new(
                local_output_type.data_type,
                Shape::new(global_shape.into_iter().map(Size::Static).collect()),
                local_output_type.layout.clone(),
            ))
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
    partition_spec: &PartitionSpec,
    mesh: &LogicalMesh,
    local_shape: Vec<usize>,
    output_index: usize,
) -> Result<Vec<usize>, ShardMapTraceError> {
    if partition_spec.rank() != local_shape.len() {
        return Err(ShardMapTraceError::RankMismatch {
            value_kind: "output",
            value_index: output_index,
            partition_rank: partition_spec.rank(),
            shape_rank: local_shape.len(),
        });
    }

    let manual_axis_names = mesh
        .axes
        .iter()
        .filter_map(|axis| (axis.r#type == MeshAxisType::Manual).then_some(axis.name.as_str()))
        .collect::<HashSet<_>>();
    partition_spec
        .dimensions()
        .iter()
        .zip(local_shape)
        .enumerate()
        .map(|(dimension, (partition_dimension, local_dimension_size))| {
            let manual_partition_count = match partition_dimension {
                PartitionDimension::Sharded(axis_names) => axis_names
                    .iter()
                    .filter(|axis_name| manual_axis_names.contains(axis_name.as_str()))
                    .try_fold(1usize, |partition_count, axis_name| {
                        let axis_size = mesh.axis_size(axis_name).ok_or_else(|| {
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
                PartitionDimension::Unsharded | PartitionDimension::Unconstrained => 1,
            };

            local_dimension_size
                .checked_mul(manual_partition_count)
                .ok_or_else(|| ShardMapTraceError::Overflow {
                    context: format!("computing global output size for output #{output_index} dimension #{dimension}"),
                })
        })
        .collect()
}

fn build_named_shardings(
    mesh: &LogicalMesh,
    partition_specs: Vec<PartitionSpec>,
    value_kind: &'static str,
) -> Result<Vec<NamedSharding>, ShardMapError> {
    let manual_axes = manual_axes_from_mesh(mesh)?;
    let manual_axis_names = manual_axes.iter().map(String::as_str).collect::<HashSet<_>>();
    partition_specs
        .into_iter()
        .enumerate()
        .map(|(value_index, partition_spec)| {
            validate_manual_axis_order(&partition_spec, &manual_axis_names, value_kind, value_index)?;
            Ok(NamedSharding::new(mesh.clone(), partition_spec)?.project_for_traced_sharding())
        })
        .collect()
}

fn manual_axes_from_mesh(mesh: &LogicalMesh) -> Result<Vec<String>, ShardMapError> {
    let manual_axes = mesh
        .axes
        .iter()
        .filter_map(|axis| (axis.r#type == MeshAxisType::Manual).then_some(axis.name.clone()))
        .collect::<Vec<_>>();
    if manual_axes.is_empty() {
        return Err(ShardMapError::MeshHasNoManualAxes);
    }
    Ok(manual_axes)
}

fn validate_manual_axis_order(
    partition_spec: &PartitionSpec,
    manual_axes: &HashSet<&str>,
    value_kind: &'static str,
    value_index: usize,
) -> Result<(), ShardMapError> {
    for (dimension, partition_dimension) in partition_spec.dimensions().iter().enumerate() {
        if let PartitionDimension::Sharded(axis_names) = partition_dimension {
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

fn used_axes_in_partition_spec(partition_spec: &PartitionSpec) -> HashSet<&str> {
    let mut used_axes = HashSet::new();
    for partition_dimension in partition_spec.dimensions() {
        if let PartitionDimension::Sharded(axis_names) = partition_dimension {
            for axis_name in axis_names {
                used_axes.insert(axis_name.as_str());
            }
        }
    }
    for axis_name in partition_spec.unreduced_axes() {
        used_axes.insert(axis_name.as_str());
    }
    used_axes
}

fn local_shape_for_sharding(
    sharding: &NamedSharding,
    global_shape: &[usize],
    value_kind: &'static str,
    value_index: usize,
) -> Result<Vec<usize>, ShardMapError> {
    let partition_spec = sharding.partition_spec();
    if partition_spec.rank() != global_shape.len() {
        return Err(ShardMapError::RankMismatch {
            value_kind,
            value_index,
            partition_rank: partition_spec.rank(),
            shape_rank: global_shape.len(),
        });
    }

    let manual_axis_names = sharding
        .mesh()
        .axes
        .iter()
        .filter_map(|axis| (axis.r#type == MeshAxisType::Manual).then_some(axis.name.as_str()))
        .collect::<HashSet<_>>();
    let mut local_shape = Vec::with_capacity(global_shape.len());
    for (dimension, (partition_dimension, dimension_size)) in
        partition_spec.dimensions().iter().zip(global_shape.iter().copied()).enumerate()
    {
        let manual_partition_count = match partition_dimension {
            PartitionDimension::Sharded(axis_names) => axis_names
                .iter()
                .filter(|axis_name| manual_axis_names.contains(axis_name.as_str()))
                .try_fold(1usize, |partition_count, axis_name| -> Result<usize, ShardMapError> {
                    let axis_size = sharding
                        .mesh()
                        .axis_size(axis_name)
                        .ok_or_else(|| ShardingError::UnknownMeshAxisName { name: axis_name.clone() })?;
                    Ok(partition_count * axis_size)
                })?,
            PartitionDimension::Unsharded | PartitionDimension::Unconstrained => 1,
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
fn render_shardy_sharding_list(shardings: &[NamedSharding]) -> String {
    let mut result = String::from("[");
    for (sharding_index, sharding) in shardings.iter().enumerate() {
        if sharding_index > 0 {
            result.push_str(", ");
        }
        result.push_str(stripped_shardy_tensor_sharding(sharding).as_str());
    }
    result.push(']');
    result
}

fn shardy_tensor_sharding_per_value<'c, 't>(
    shardings: &[NamedSharding],
    context: &'c MlirContext<'t>,
) -> TensorShardingPerValueAttributeRef<'c, 't> {
    let shardings = shardings
        .iter()
        .map(|sharding| manual_computation_tensor_sharding(sharding, context))
        .collect::<Vec<_>>();
    context.shardy_tensor_sharding_per_value(shardings.as_slice())
}

fn manual_computation_tensor_sharding<'c, 't>(
    sharding: &NamedSharding,
    context: &'c MlirContext<'t>,
) -> TensorShardingAttributeRef<'c, 't> {
    let mesh_symbol_ref = context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME);
    let dim_shardings = manual_computation_dimension_shardings(sharding.mesh(), sharding.partition_spec(), context);
    let replicated_axis_names = sharding.replicated_axes();
    let replicated_axes = replicated_axis_names
        .iter()
        .map(|axis_name| context.shardy_axis_ref(axis_name, None))
        .collect::<Vec<_>>();
    let unreduced_axes = sharding
        .partition_spec()
        .unreduced_axes()
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
    mesh: &LogicalMesh,
    partition_spec: &PartitionSpec,
    context: &'c MlirContext<'t>,
) -> Vec<DimensionShardingAttributeRef<'c, 't>> {
    let manual_axis_names = mesh
        .axes
        .iter()
        .filter_map(|axis| (axis.r#type == MeshAxisType::Manual).then_some(axis.name.as_str()))
        .collect::<HashSet<_>>();
    let free_axis_names = mesh
        .axes
        .iter()
        .filter_map(|axis| (!manual_axis_names.contains(axis.name.as_str())).then_some(axis.name.as_str()))
        .collect::<HashSet<_>>();
    let used_axes = used_axes_in_partition_spec(partition_spec);
    let has_unused_free_axes = free_axis_names.iter().any(|axis_name| !used_axes.contains(axis_name));

    partition_spec
        .dimensions()
        .iter()
        .map(|partition_dimension| match partition_dimension {
            PartitionDimension::Unsharded => context.shardy_dimension_sharding(&[], !has_unused_free_axes, None),
            PartitionDimension::Sharded(axis_names) => {
                let axes =
                    axis_names.iter().map(|axis_name| context.shardy_axis_ref(axis_name, None)).collect::<Vec<_>>();
                let contains_free_axis =
                    axis_names.iter().any(|axis_name| free_axis_names.contains(axis_name.as_str()));
                context.shardy_dimension_sharding(axes.as_slice(), !(contains_free_axis || has_unused_free_axes), None)
            }
            PartitionDimension::Unconstrained => context.shardy_dimension_sharding(&[], false, None),
        })
        .collect()
}

#[cfg(test)]
fn stripped_shardy_tensor_sharding(sharding: &NamedSharding) -> String {
    let mut result = format!(
        "<@{SHARDY_MESH_SYMBOL_NAME}, {}",
        render_manual_computation_dimensions(sharding.mesh(), sharding.partition_spec())
    );

    let replicated_axes = sharding.replicated_axes();
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

    if !sharding.partition_spec().unreduced_axes().is_empty() {
        result.push_str(", unreduced={");
        for (axis_index, axis_name) in sharding.partition_spec().unreduced_axes().iter().enumerate() {
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
fn render_manual_computation_dimensions(mesh: &LogicalMesh, partition_spec: &PartitionSpec) -> String {
    let manual_axis_names = mesh
        .axes
        .iter()
        .filter_map(|axis| (axis.r#type == MeshAxisType::Manual).then_some(axis.name.as_str()))
        .collect::<HashSet<_>>();
    let free_axis_names = mesh
        .axes
        .iter()
        .filter_map(|axis| (!manual_axis_names.contains(axis.name.as_str())).then_some(axis.name.as_str()))
        .collect::<HashSet<_>>();
    let used_axes = used_axes_in_partition_spec(partition_spec);
    let has_unused_free_axes = free_axis_names.iter().any(|axis_name| !used_axes.contains(axis_name));

    let mut result = String::from("[");
    for (dimension_index, partition_dimension) in partition_spec.dimensions().iter().enumerate() {
        if dimension_index > 0 {
            result.push_str(", ");
        }

        match partition_dimension {
            PartitionDimension::Unsharded => {
                if has_unused_free_axes {
                    result.push_str("{?}");
                } else {
                    result.push_str("{}");
                }
            }
            PartitionDimension::Sharded(axis_names) => {
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
            PartitionDimension::Unconstrained => result.push_str("{?}"),
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
    use std::collections::HashMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, Program, load_cpu_plugin};

    use super::*;
    use crate::sharding::{MeshAxis, MeshAxisType};
    use crate::tracing_v2::{FloatExt, OneLike, grad, vmap};
    use crate::types::data_types::DataType;
    use crate::xla::arrays::Array;
    use crate::xla::sharding::{DeviceMesh, MeshDevice, PartitionDimension, PartitionSpec, ShardingContext};

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
        let shard_map = ShardMap::new(
            test_logical_mesh_2x2(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("x")])],
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("x")])],
        )
        .unwrap();

        assert_eq!(shard_map.manual_axes(), vec!["x", "y"]);
        assert_eq!(shard_map.in_shardings()[0].replicated_axes(), vec!["y"]);
        assert_eq!(shard_map.out_shardings()[0].replicated_axes(), vec!["y"]);
    }

    #[test]
    fn test_shard_map_function_rejects_mesh_without_manual_axes() {
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None);
        let result: Result<TracedShardMap<ArrayType, ArrayType>, ShardMapTraceError> = shard_map(
            |x| x.clone() + x,
            global_input_type,
            test_logical_mesh_without_manual_axes(),
            PartitionSpec::new(vec![PartitionDimension::sharded("x")]),
            PartitionSpec::new(vec![PartitionDimension::sharded("x")]),
        );

        assert!(matches!(result, Err(ShardMapTraceError::MeshHasNoManualAxes)));
    }

    #[test]
    fn test_shard_map_rejects_free_axis_before_manual_axis() {
        let result = ShardMap::new(
            test_logical_mesh_data_model(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded_by(["model", "data"])])],
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
            })
        );
    }

    #[test]
    fn test_shard_map_local_input_shape_for_all_manual_axes() {
        let shard_map = ShardMap::new(
            test_logical_mesh_2x2(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded_by(["x", "y"])])],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(shard_map.local_input_shape(0, &[16]).unwrap(), vec![4]);
    }

    #[test]
    fn test_shard_map_local_input_shape_for_mixed_manual_and_free_axes() {
        let shard_map = ShardMap::new(
            test_logical_mesh_data_model(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded_by(["data", "model"])])],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(shard_map.local_input_shape(0, &[16]).unwrap(), vec![8]);
    }

    #[test]
    fn test_shard_map_local_output_shape() {
        let shard_map = ShardMap::new(
            test_logical_mesh_data_model(),
            Vec::new(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("data"), PartitionDimension::unsharded()])],
        )
        .unwrap();

        assert_eq!(shard_map.local_output_shape(0, &[32, 8]).unwrap(), vec![16, 8]);
    }

    #[test]
    fn test_shard_map_local_shape_rejects_padding_from_manual_axes() {
        let shard_map = ShardMap::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Manual).unwrap()]).unwrap(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("x")])],
            Vec::new(),
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
        let shard_map = ShardMap::new(
            test_logical_mesh_2x2(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("x")])],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(
            shard_map.local_input_shape(0, &[8, 4]),
            Err(ShardMapError::RankMismatch { value_kind: "input", value_index: 0, partition_rank: 1, shape_rank: 2 })
        );
    }

    #[test]
    fn test_shard_map_renders_in_shardings_attribute() {
        let shard_map = ShardMap::new(
            test_logical_mesh_2x2(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("x")])],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(shard_map.to_shardy_in_shardings_attribute(), r#"[<@mesh, [{"x"}], replicated={"y"}>]"#);
    }

    #[test]
    fn test_shard_map_renders_free_axes_as_open_dimension_shardings() {
        let shard_map = ShardMap::new(
            test_logical_mesh_data_model(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded_by(["data", "model"])])],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(shard_map.to_shardy_in_shardings_attribute(), r#"[<@mesh, [{"data", ?}]>]"#);
    }

    #[test]
    fn test_shard_map_renders_explicit_axes_in_traced_shardings() {
        let shard_map = ShardMap::new(
            test_logical_mesh_data_model_explicit(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded_by(["data", "model"])])],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(shard_map.to_shardy_in_shardings_attribute(), r#"[<@mesh, [{"data", "model", ?}]>]"#);
    }

    #[test]
    fn test_shard_map_renders_out_shardings_attribute() {
        let shard_map = ShardMap::new(
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
                MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            ])
            .unwrap(),
            Vec::new(),
            vec![PartitionSpec::new(vec![PartitionDimension::unsharded()])],
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
        )
        .unwrap();

        assert_eq!(shard_map.manual_axes(), vec!["y"]);
        assert_eq!(shard_map.to_shardy_manual_axes_attribute(), r#"{"y"}"#);
    }

    #[test]
    fn test_shard_map_renders_manual_computation_attributes() {
        let shard_map = ShardMap::new(
            test_logical_mesh_data_model(),
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("data")])],
            vec![PartitionSpec::new(vec![PartitionDimension::sharded("data")])],
        )
        .unwrap();

        assert_eq!(
            shard_map.to_shardy_manual_computation_attributes(),
            r#"in_shardings=[<@mesh, [{"data", ?}]>] out_shardings=[<@mesh, [{"data", ?}]>] manual_axes={"data"}"#
        );
    }

    #[test]
    fn test_shard_map_trace_derives_types_and_renders_mlir() {
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None);
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x| x.clone() + x,
            global_input_type.clone(),
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            PartitionSpec::new(vec![PartitionDimension::sharded("x")]),
            PartitionSpec::new(vec![PartitionDimension::sharded("x")]),
        )
        .unwrap();

        assert_eq!(traced.global_input_types(), &global_input_type);
        assert_eq!(traced.local_input_types(), &ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None));
        assert_eq!(
            traced.local_output_types(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None)
        );
        assert_eq!(traced.global_output_types(), &global_input_type);
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
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(16)]), None);
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x| x.clone() + x,
            global_input_type.clone(),
            test_logical_mesh_data_model(),
            PartitionSpec::new(vec![PartitionDimension::sharded_by(["data", "model"])]),
            PartitionSpec::new(vec![PartitionDimension::sharded_by(["data", "model"])]),
        )
        .unwrap();

        assert_eq!(traced.global_input_types(), &global_input_type);
        assert_eq!(traced.local_input_types(), &ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None));
        assert_eq!(traced.global_output_types(), &global_input_type);
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
        let outer_partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded("x")]);
        let inner_partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded("y")]);
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            {
                let inner_mesh = inner_mesh.clone();
                let inner_partition_spec = inner_partition_spec.clone();
                move |x: ShardMapTracer| {
                    let nested: ShardMapTracer = shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                        |y: ShardMapTracer| y.clone() + y,
                        x.clone(),
                        inner_mesh.clone(),
                        inner_partition_spec.clone(),
                        inner_partition_spec.clone(),
                    )
                    .expect("nested shard_map should trace");
                    nested + x
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None),
            mesh,
            outer_partition_spec.clone(),
            outer_partition_spec,
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
        let dynamic_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Dynamic(None)]), None);
        let result: Result<TracedShardMap<ArrayType, ArrayType>, ShardMapTraceError> = shard_map(
            |x| x.clone() + x,
            dynamic_input_type,
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            PartitionSpec::new(vec![PartitionDimension::sharded("x")]),
            PartitionSpec::new(vec![PartitionDimension::sharded("x")]),
        );

        assert!(matches!(
            result,
            Err(ShardMapTraceError::DynamicShapeNotSupported { value_kind: "input", value_index: 0, dimension: 0 })
        ));
    }

    #[test]
    fn test_shard_map_infers_single_input_closure_argument_type() {
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None);
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x| x.clone() + x,
            global_input_type,
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            PartitionSpec::new(vec![PartitionDimension::sharded("x")]),
            PartitionSpec::new(vec![PartitionDimension::sharded("x")]),
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
        let device_mesh =
            DeviceMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()], mesh_devices).unwrap();

        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded("x")]);
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None);
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x| x.clone() + x,
            global_input_type,
            device_mesh.logical_mesh().clone(),
            partition_spec.clone(),
            partition_spec.clone(),
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
            Array::from_sharding(vec![8], DataType::F32, device_mesh, partition_spec, input_buffers).unwrap();
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
        let device_mesh =
            DeviceMesh::new(vec![MeshAxis::new("x", 8, MeshAxisType::Manual).unwrap()], mesh_devices).unwrap();

        let lhs_partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]);
        let rhs_partition_spec = PartitionSpec::replicated(2);
        let output_partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]);
        let global_input_types = (
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(4)]), None),
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4), Size::Static(2)]), None),
        );
        let traced: TracedShardMap<(ArrayType, ArrayType), ArrayType> = shard_map(
            |(lhs, rhs)| lhs.matmul(rhs),
            global_input_types,
            device_mesh.logical_mesh().clone(),
            (lhs_partition_spec.clone(), rhs_partition_spec.clone()),
            output_partition_spec.clone(),
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

        let lhs_array = Array::from_sharding(
            vec![8, 4],
            DataType::F32,
            device_mesh.clone(),
            lhs_partition_spec.clone(),
            lhs_buffers,
        )
        .unwrap();
        let rhs_array =
            Array::from_sharding(vec![4, 2], DataType::F32, device_mesh, rhs_partition_spec, rhs_buffers).unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(8)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 8);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let row_start_by_device = execution_device_ids
            .iter()
            .map(|device_id| {
                let row_start = lhs_array.shard_for_device(*device_id).unwrap().slices()[0].start;
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
        let device_mesh =
            DeviceMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()], mesh_devices).unwrap();

        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded("x")]);
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None);
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
            device_mesh.logical_mesh().clone(),
            partition_spec.clone(),
            partition_spec.clone(),
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
            Array::from_sharding(vec![8], DataType::F32, device_mesh, partition_spec, input_buffers).unwrap();
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
        let device_mesh =
            DeviceMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()], mesh_devices).unwrap();
        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded("x")]);
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None);

        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let partition_spec = partition_spec.clone();
                move |x: ShardMapTracer| {
                    grad(
                        {
                            let mesh = mesh.clone();
                            let partition_spec = partition_spec.clone();
                            move |y: ShardMapTracer| {
                                shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                                    |local_x: ShardMapTracer| local_x.sin(),
                                    y,
                                    mesh.clone(),
                                    partition_spec.clone(),
                                    partition_spec.clone(),
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
            Array::from_sharding(vec![8], DataType::F32, device_mesh, partition_spec, input_buffers).unwrap();
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
        let device_mesh =
            DeviceMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()], mesh_devices).unwrap();

        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded("x")]);
        let shard_map = ShardMap::new(
            device_mesh.logical_mesh().clone(),
            vec![partition_spec.clone()],
            vec![partition_spec.clone()],
        )
        .unwrap();
        assert_eq!(shard_map.local_input_shape(0, &[8]).unwrap(), vec![2]);
        assert_eq!(shard_map.local_output_shape(0, &[8]).unwrap(), vec![2]);

        let input_sharding =
            shard_map.in_shardings()[0].to_shardy_tensor_sharding_attribute(ShardingContext::ExplicitSharding);
        let output_sharding =
            shard_map.out_shardings()[0].to_shardy_tensor_sharding_attribute(ShardingContext::ExplicitSharding);
        let manual_computation_attributes = shard_map.to_shardy_manual_computation_attributes();
        let context = MlirContext::new();
        let mesh_module = context.module(context.unknown_location());
        let mesh_operation = mesh_module
            .body()
            .append_operation(shard_map.mesh().to_shardy_mesh(context.unknown_location()))
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
            Array::from_sharding(vec![8], DataType::F32, device_mesh, partition_spec, input_buffers).unwrap();
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
