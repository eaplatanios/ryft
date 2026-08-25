use std::fmt::Display;
use std::marker::PhantomData;

// TODO(eaplatanios): Review this module.

// TODO(eaplatanios): Why this import?
use crate::arrays::batching::align_array_batch;
use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatching, ArrayIrType, ArrayType, Dimension,
    DimensionType, DimensionValue, DimensionVariable, Layout, RaggedAxis, ShardingDimension, TiledLayout,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    batch_projected_operation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::operations::{CUSTOM_JVP_OPERATION_NAME, CUSTOM_VJP_OPERATION_NAME};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation, impl_reference_free_dischargeable_operation};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::control_flow::scan::ScanOperation;
use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
use crate::operations::manipulation::broadcasting::DynamicBroadcastOperation;
use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
use crate::parameters::Placeholder;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Effect, Effects, Operation, OperationFormatter, OperationProjection, ProgramBuilder, ProgramError, RegionInterface,
    Type, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};

/// Canonical operation name for [`CustomCallOperation`].
pub const CUSTOM_CALL_OPERATION_NAME: &str = "custom_call";

/// Typed configuration attribute value carried by a [`CustomCallOperation`] and forwarded to the foreign kernel.
/// The variants deliberately cover only the encodings every supporting backend must decode: strings, Booleans,
/// 64-bit signed integers, and 64-bit floating-point values. The `From` conversions let attribute values be passed
/// directly to [`CustomCallOperation::with_attribute`] (e.g., `.with_attribute("scale", 2.0)`); `&str` and `String`
/// convert into [`String`](Self::String), `bool` into [`Boolean`](Self::Boolean), `i64` into [`I64`](Self::I64),
/// and `f64` into [`F64`](Self::F64).
#[derive(Clone, Debug, PartialEq)]
pub enum CustomCallAttribute {
    /// UTF-8 string value.
    String(String),

    /// Boolean value.
    Boolean(bool),

    /// 64-bit signed-integer value.
    I64(i64),

    /// 64-bit floating-point value.
    F64(f64),
}

impl Display for CustomCallAttribute {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(string) => formatter.write_str(string),
            Self::Boolean(boolean) => write!(formatter, "{boolean}"),
            Self::I64(integer) => write!(formatter, "{integer}"),
            Self::F64(float) => write!(formatter, "{float:?}"),
        }
    }
}

impl From<&str> for CustomCallAttribute {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<String> for CustomCallAttribute {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<bool> for CustomCallAttribute {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

impl From<i64> for CustomCallAttribute {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<f64> for CustomCallAttribute {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}

/// Declares that one flat array output of a [`CustomCallOperation`] aliases one flat array input.
///
/// Aliasing requires the input and output to describe the same logical array. It allows a backend to reuse the input
/// buffer for the output without changing Ryft's functional SSA semantics. The indices never include mixed trailing
/// dimension operands or backend-internal effect tokens.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomCallInputOutputAlias {
    /// Index of the aliased array input.
    input_index: usize,

    /// Index of the array output that aliases the input.
    output_index: usize,
}

impl CustomCallInputOutputAlias {
    /// Creates an alias from the array input at `input_index` to the array output at `output_index`.
    #[inline]
    pub fn new(input_index: usize, output_index: usize) -> Self {
        Self { input_index, output_index }
    }

    /// Returns the index of the aliased array input.
    #[inline]
    pub fn input_index(&self) -> usize {
        self.input_index
    }

    /// Returns the index of the array output that aliases the input.
    #[inline]
    pub fn output_index(&self) -> usize {
        self.output_index
    }
}

impl Display for CustomCallInputOutputAlias {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}->{}", self.input_index, self.output_index)
    }
}

/// Behavior a [`CustomCallOperation`] requests when the batching transform maps one of its operands. Ryft cannot
/// derive a batching rule for an opaque kernel, so the author of the call declares which of the few universally
/// meaningful strategies applies, using [`with_batching`](CustomCallOperation::with_batching). This mirrors JAX's
/// `vmap_method` selection on
/// [`jax.ffi.ffi_call`](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.ffi_call.html). A call whose operands are
/// all replicated never consults this behavior: it is bound unchanged.
///
/// Two of JAX's selections are deliberately absent:
///
///   - `expand_dims` (broadcast every operand to a size-1 batch axis and call the kernel once) contradicts Ryft's
///     input/output aliasing: a replicated operand would enter carrying a size-1 axis while its aliased output must
///     gain the full batch extent `b`, so the alias would no longer describe one logical array and the buffer could
///     not be reused. [`BroadcastAll`](Self::BroadcastAll) is the aliasing-compatible member of that pair.
///   - `legacy_vectorized` (a mode in which the kernel silently promises to handle arbitrary leading axes) is an
///     XLA-legacy mode that JAX has already removed, so Ryft never introduces it.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CustomCallBatching {
    /// Report a [`BatchingError::UnsupportedOperation`] naming the mapped operand. This is the default because a
    /// foreign kernel's contract is opaque: silently choosing a strategy could execute the kernel on buffers it
    /// never agreed to accept.
    #[default]
    Rejected,

    /// Apply the kernel once per batch item through a `scan` whose body performs exactly one unbatched call.
    /// Mapped operands are realigned to batch axis `0` and sliced one row per iteration, replicated operands become
    /// invariant loop carries, and the per-iteration results are stacked back on batch axis `0`. The kernel therefore
    /// observes exactly the buffers it would have seen without the transform, at the cost of `b` sequential calls;
    /// a side-effecting kernel consequently runs `b` ordered times. The optional `unroll` factor is forwarded to
    /// [`ScanOperation::with_unroll`], and is a lowering-only knob that trades code size for loop overhead.
    Sequential {
        /// Lowering-only number of body copies emitted per loop trip, or [`None`] to keep one call per trip. The
        /// factor must be at least `1` and must evenly divide the batch extent.
        unroll: Option<usize>,
    },

    /// Align every operand to batch axis `0` and call the kernel exactly once on batch-prefixed buffers, declaring
    /// batch-prefixed output types. The kernel must itself understand the leading batch axis. Replicated operands
    /// are materialized across the batch first, so every operand and result carries the same leading extent, aliases
    /// stay type-preserving, and a side-effecting kernel runs exactly once.
    BroadcastAll,
}

impl Display for CustomCallBatching {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected => formatter.write_str("rejected"),
            Self::Sequential { unroll: None } => formatter.write_str("sequential"),
            Self::Sequential { unroll: Some(unroll) } => write!(formatter, "sequential(unroll={unroll})"),
            Self::BroadcastAll => formatter.write_str("broadcast_all"),
        }
    }
}

/// Names one packed custom-call operand axis whose live extent is carried by another, already-declared operand.
/// The bound [`DimensionVariable`] gives the ragged dimension stable identity across transformation replays, while
/// `extent_operand_index` identifies the ordinary integer scalar operand that the unbatched foreign kernel receives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CustomCallRaggedInputBinding {
    /// Name used by output bindings to refer to this input binding.
    name: String,

    /// Index of the packed array operand.
    operand_index: usize,

    /// Axis of the packed array operand whose live prefix is ragged.
    axis: usize,

    /// Index of the existing scalar integer operand carrying the live extent.
    extent_operand_index: usize,

    /// Stable identity and runtime bounds of the ragged dimension.
    dimension: DimensionVariable,
}

impl CustomCallRaggedInputBinding {
    /// Creates a named input binding between one packed operand axis and one existing scalar extent operand.
    pub fn new<N: Into<String>>(
        name: N,
        operand_index: usize,
        axis: usize,
        extent_operand_index: usize,
        dimension: DimensionVariable,
    ) -> Self {
        Self { name: name.into(), operand_index, axis, extent_operand_index, dimension }
    }

    /// Returns the binding name used by output bindings.
    #[inline]
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Returns the index of the packed array operand.
    #[inline]
    pub fn operand_index(&self) -> usize {
        self.operand_index
    }

    /// Returns the bound axis of the packed array operand.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the index of the existing operand carrying the live extent.
    #[inline]
    pub fn extent_operand_index(&self) -> usize {
        self.extent_operand_index
    }

    /// Returns the stable identity and runtime bounds of the ragged dimension.
    #[inline]
    pub fn dimension(&self) -> &DimensionVariable {
        &self.dimension
    }
}

impl Display for CustomCallRaggedInputBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}:operand({})@{}<=operand({}):{}",
            self.name, self.operand_index, self.axis, self.extent_operand_index, self.dimension,
        )
    }
}

/// Declares how one positional custom-call output relates to the operation's ragged input bindings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CustomCallRaggedOutputBinding {
    /// Preserve the named input binding on the output axis `axis`, reusing the exact same extent value and dimension
    /// identity. The axis may be relocated when the output does not alias its input.
    Preserved {
        /// Name of the input binding being preserved.
        input_binding: String,

        /// Packed output axis carrying the preserved ragged dimension.
        axis: usize,
    },

    /// Produce a dense output. Every input binding not preserved by any output is considered deliberately consumed by
    /// the call's declared padding-independent semantics.
    Consumed,

    /// Produce a ragged output whose live extents come from another, ordinary integer scalar output of the same call.
    Fresh {
        /// Packed output axis whose live prefix is described by the fresh extents.
        axis: usize,

        /// Index of the existing integer scalar output carrying the live extent.
        extent_output_index: usize,

        /// Stable identity and runtime bounds of the fresh ragged dimension.
        dimension: DimensionVariable,
    },
}

impl Display for CustomCallRaggedOutputBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preserved { input_binding, axis } => write!(formatter, "preserve({input_binding})@{axis}"),
            Self::Consumed => formatter.write_str("consume"),
            Self::Fresh { axis, extent_output_index, dimension } => {
                write!(formatter, "fresh@{axis}<=output({extent_output_index}):{dimension}")
            }
        }
    }
}

/// Declared calling convention that lets [`CustomCallOperation`] discharge one level of ragged batching without
/// changing the foreign kernel's signature. Input bindings point at extent operands that already exist in the call,
/// and output bindings either preserve one of those bindings, consume raggedness, or point at an existing extent
/// output. Declaring this contract promises that no live output element depends on padded input elements.
///
/// The contract intentionally supports one ragged axis per operand and one ragged batching level. A second ragged
/// batching level is rejected because representing it requires associating each ragged axis with another independent
/// extent value. Ordinary dense `BroadcastAll` batching remains composable: the contract records every accumulated
/// leading dense axis so that its existing extent operands and outputs retain their complete axis mapping. The
/// declaration supplies no differentiation semantics: ragged custom calls continue to follow the ordinary custom-call
/// differentiation contract. Padding remains unspecified downstream; the result [`RaggedAxis`] metadata is what lets
/// extent-aware operations avoid observing it.
/// Runtime extent values must lie within the declared [`DimensionVariable`] bounds and must not exceed their packed
/// physical axis. Eager foreign-kernel implementations are responsible for checking that precondition when decoding
/// their ordinary extent operands and outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CustomCallRaggedContract {
    /// Named packed-input bindings, in declaration order.
    input_bindings: Vec<CustomCallRaggedInputBinding>,

    /// Positional output bindings, with exactly one entry per declared output.
    output_bindings: Vec<CustomCallRaggedOutputBinding>,

    /// Number of leading dense batch axes accumulated as `BroadcastAll` repeatedly batches this contract.
    batch_prefix_count: usize,

    /// Whether an earlier batching pass discharged active ragged input bindings.
    ragged_discharged: bool,
}

impl CustomCallRaggedContract {
    /// Creates a ragged calling convention over existing custom-call operands and outputs.
    ///
    /// # Parameters
    ///
    ///   - `input_bindings`: Named packed-input axes and their existing scalar extent operands.
    ///   - `output_bindings`: One positional relationship for every declared custom-call output.
    pub fn new(
        input_bindings: Vec<CustomCallRaggedInputBinding>,
        output_bindings: Vec<CustomCallRaggedOutputBinding>,
    ) -> Self {
        Self { input_bindings, output_bindings, batch_prefix_count: 0, ragged_discharged: false }
    }

    /// Returns the named packed-input bindings in declaration order.
    #[inline]
    pub fn input_bindings(&self) -> &[CustomCallRaggedInputBinding] {
        self.input_bindings.as_slice()
    }

    /// Returns the positional output bindings.
    #[inline]
    pub fn output_bindings(&self) -> &[CustomCallRaggedOutputBinding] {
        self.output_bindings.as_slice()
    }

    /// Returns a contract describing the `BroadcastAll` call after all operands and outputs gain another leading
    /// dense batch dimension.
    fn batch_prefixed(&self, discharges_ragged: bool) -> Self {
        let mut contract = self.clone();
        for binding in &mut contract.input_bindings {
            binding.axis += 1;
        }
        for binding in &mut contract.output_bindings {
            match binding {
                CustomCallRaggedOutputBinding::Preserved { axis, .. }
                | CustomCallRaggedOutputBinding::Fresh { axis, .. } => *axis += 1,
                CustomCallRaggedOutputBinding::Consumed => {}
            }
        }
        contract.batch_prefix_count += 1;
        contract.ragged_discharged |= discharges_ragged;
        contract
    }

    /// Returns a contract recording that active ragged bindings were discharged without changing operand or output
    /// axes. This transition is used by `Sequential`, whose scan body sees one unbatched slice at a time.
    fn ragged_discharged(&self) -> Self {
        let mut contract = self.clone();
        contract.ragged_discharged = true;
        contract
    }

    /// Returns the unique active dimensions that are absent from every preserved ragged output.
    fn consumed_dimensions<V>(&self, active: &[(String, RaggedAxis<V>)]) -> Vec<DimensionVariable> {
        let mut consumed = Vec::new();
        for (_, axis) in active {
            let dimension = axis.dimension();
            let preserved = active.iter().any(|(name, candidate)| {
                candidate.dimension() == dimension
                    && self.output_bindings.iter().any(|binding| {
                        matches!(
                            binding,
                            CustomCallRaggedOutputBinding::Preserved { input_binding, .. }
                                if input_binding == name
                        )
                    })
            });
            if !preserved && !consumed.contains(dimension) {
                consumed.push(dimension.clone());
            }
        }
        consumed
    }

    /// Returns the complete extent-axis mapping for a new ragged level whose packed data and extent operand carry the
    /// current mapped axis at `data_batch_axis` and `extent_batch_axis`, respectively.
    fn active_extent_axes(&self, data_batch_axis: usize, extent_batch_axis: usize) -> Vec<usize> {
        (0..=self.batch_prefix_count)
            .map(|extent_axis| {
                if extent_axis == extent_batch_axis {
                    data_batch_axis
                } else {
                    let prefix_axis = extent_axis - usize::from(extent_axis > extent_batch_axis);
                    prefix_axis + usize::from(prefix_axis >= data_batch_axis)
                }
            })
            .collect()
    }

    /// Returns this contract after renaming every dimension identity.
    fn renamed(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        let mut contract = self.clone();
        for binding in &mut contract.input_bindings {
            binding.dimension =
                DimensionType::new(binding.dimension.clone()).rename_identities(renaming)?.variable().clone();
        }
        for binding in &mut contract.output_bindings {
            if let CustomCallRaggedOutputBinding::Fresh { dimension, .. } = binding {
                *dimension = DimensionType::new(dimension.clone()).rename_identities(renaming)?.variable().clone();
            }
        }
        Ok(contract)
    }
}

impl Display for CustomCallRaggedContract {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("{inputs=[")?;
        for (index, binding) in self.input_bindings.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            write!(formatter, "{binding}")?;
        }
        formatter.write_str("], outputs=[")?;
        for (index, binding) in self.output_bindings.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            write!(formatter, "{binding}")?;
        }
        formatter.write_str("]")?;
        if self.batch_prefix_count != 0 {
            write!(formatter, ", batch_prefix_count={}", self.batch_prefix_count)?;
        }
        if self.ragged_discharged {
            formatter.write_str(", ragged_discharged=true")?;
        }
        formatter.write_str("}")
    }
}

/// Universe-neutral view of one custom-call array input during ragged contract validation.
struct CustomCallRaggedInput<'a, V> {
    /// Packed value used to verify the declared extent operand.
    value: &'a V,

    /// Normalized position of the mapped batch axis.
    batch_axis: Option<usize>,

    /// Bounded ragged axes carried by the input.
    ragged_axes: &'a [RaggedAxis<V>],

    /// Physical per-item array type expected by the foreign kernel.
    physical_type: ArrayType,
}

/// [`Operation`] that calls a foreign kernel registered with the executing backend under a target name — the
/// analogue of [`jax.ffi.ffi_call`](https://docs.jax.dev/en/latest/ffi.html). The operation is opaque to Ryft:
/// its output types are declared up front instead of inferred, and typed [`CustomCallAttribute`]s are forwarded
/// verbatim to the kernel as its configuration.
///
/// In the array IR, each dynamic axis occurrence in the declared outputs requires one trailing first-class
/// dimension operand, ordered first by output and then by axis. Type inference verifies that each operand defines the
/// exact variable referenced by its corresponding output axis. These logical result extents do not enter the foreign
/// kernel ABI: only the leading array operands are passed to the kernel. Eager execution and backend lowering use the
/// trailing operands to verify or attach the declared logical sizes to the returned buffers.
///
/// The type parameter selects that operand contract without introducing a second semantic operation:
///
///   - `CustomCallOperation<ArrayType>` accepts only the foreign kernel's array operands and is suitable for
///     programs over homogeneous arrays whose result extents are fully described by their array types.
///   - `CustomCallOperation<ArrayIrType>` additionally accepts the trailing first-class dimension operands
///     described above and is suitable for mixed array/dimension programs.
///
/// Both forms carry the same target, output declarations, attributes, effects, rendering, and backend-kernel
/// semantics. Conversion into the mixed form only reparameterizes the operation family; it moves the existing
/// descriptor without copying its owned metadata.
///
/// The XLA backend lowers this operation to a
/// [`stablehlo.custom_call`](https://openxla.org/stablehlo/spec#custom_call) using the typed FFI calling convention
/// (`api_version = 4`), with the attributes carried as the `backend_config` dictionary. Handlers are registered with
/// the executing PJRT client under the same target name (e.g., via `ryft-pjrt`'s `Client::register_ffi_handler`).
/// The reference array backend cannot execute foreign kernels, so eager interpretation on it reports an error.
///
/// Because the kernel is opaque, Ryft cannot derive its transform rules. Differentiating it reports an error
/// directing users to wrap the call with [`custom_jvp`](crate::differentiation::custom_jvp()) or
/// [`custom_vjp`](crate::differentiation::custom_vjp()), which supply the missing derivative. Those wrappers do *not*
/// supply a batching rule: each of them structurally batches its own primal region, so a mapped operand reaches this
/// same operation and meets this same batching contract. Batching a call whose operands are all replicated binds it
/// unchanged, because a region-free foreign kernel cannot observe the transform's named axis. A mapped operand is
/// instead governed by the [`CustomCallBatching`] behavior selected with [`with_batching`](Self::with_batching):
/// the default [`Rejected`](CustomCallBatching::Rejected) reports an error naming that operand,
/// [`Sequential`](CustomCallBatching::Sequential) applies the kernel once per batch item through a `scan`, and
/// [`BroadcastAll`](CustomCallBatching::BroadcastAll) hands the kernel batch-prefixed buffers in a single call.
/// A call that uses explicit packed buffers and ordinary scalar extent operands may additionally declare a
/// [`CustomCallRaggedContract`] with [`with_ragged_contract`](Self::with_ragged_contract). The declaration never adds,
/// removes, hides, or reorders operands or outputs. It only lets batching verify that the exact extent value attached
/// to an input [`RaggedAxis`] is already present at the declared operand index, use the
/// selected [`CustomCallBatching`] strategy unchanged, and attach preserved or fresh ragged metadata to the declared
/// outputs. Calls without this declaration retain the default ragged-input rejection. The contract deliberately
/// supports one ragged axis per operand and one ragged batching level; differentiation remains governed by the same
/// custom JVP/VJP wrappers as dense calls.
/// Marking the call as side-effecting via [`with_side_effect`](Self::with_side_effect)
/// reports [`Effect::OrderedIo`], which keeps the call alive through dead-code elimination and preserves its
/// execution order relative to other ordered effects; the lowered custom call is then also marked
/// `has_side_effect = true` so the XLA compiler never elides or reorders it.
///
/// # Backend Contract
///
/// This operation is backend-independent by design, and this payload is the entire portable contract: a target
/// name resolved in the executing backend's kernel registry, declared output types, typed configuration
/// attributes, and an effect flag. A backend supports the operation by providing (1) a process- or client-level
/// registry that resolves target names to executable kernels at execution time, (2) a calling convention that
/// hands the kernel its input buffers, output buffers matching the declared output types, and the decoded
/// attributes, and (3) an execution engine that honors [`Effect::OrderedIo`] for side-effecting calls. Backends
/// that cannot execute foreign kernels (like the reference array backend) reject interpretation with a clear
/// error instead of guessing.
///
/// Array layouts come from the canonical [`ArrayType`] descriptors of the operands and results. Portable flat-array
/// buffer aliases are declared with [`CustomCallInputOutputAlias`]. Backend-specific vocabulary must never grow on
/// this payload: encodings such as XLA's FFI API version, `backend_config` representation, tuple alias paths, result
/// tiling attributes, or called-computation references belong in the owning backend's lowering (or in a backend-owned
/// operation). If a configuration knob only makes sense for one backend, it does not belong on this operation.
#[derive(Clone, Debug)]
pub struct CustomCallOperation<T: Type> {
    /// Name under which the foreign kernel is registered with the executing backend.
    target_name: String,

    /// Declared output types of the call, returned verbatim by type inference.
    output_types: Vec<ArrayType>,

    /// Typed configuration attributes forwarded to the kernel, in insertion order.
    attributes: Vec<(String, CustomCallAttribute)>,

    /// Flat array input/output buffer aliases, in declaration order.
    input_output_aliases: Vec<CustomCallInputOutputAlias>,

    /// Whether the call has observable side effects beyond its returned outputs.
    has_side_effect: bool,

    /// Behavior requested when the batching transform maps one of this call's operands.
    batching: CustomCallBatching,

    /// Optional declared calling convention for discharging one level of ragged batching.
    ragged_contract: Option<CustomCallRaggedContract>,

    /// Type universe that determines the operation's operand contract.
    marker: PhantomData<fn() -> T>,
}

impl CustomCallOperation<ArrayType> {
    /// Creates a new [`CustomCallOperation`] with the provided target name and declared output types.
    ///
    /// # Parameters
    ///
    ///   - `target_name`: Name under which the foreign kernel is registered with the executing backend.
    ///   - `output_types`: Declared output types of the call, returned verbatim by type inference.
    #[inline]
    pub fn new<N: Into<String>>(target_name: N, output_types: Vec<ArrayType>) -> Self {
        Self {
            target_name: target_name.into(),
            output_types,
            attributes: Vec::new(),
            input_output_aliases: Vec::new(),
            has_side_effect: false,
            batching: CustomCallBatching::default(),
            ragged_contract: None,
            marker: PhantomData,
        }
    }
}

impl<T: Type> CustomCallOperation<T> {
    /// Returns a copy of this [`CustomCallOperation`] with the provided typed configuration attribute appended.
    #[inline]
    pub fn with_attribute<N: Into<String>, V: Into<CustomCallAttribute>>(mut self, name: N, value: V) -> Self {
        self.attributes.push((name.into(), value.into()));
        self
    }

    /// Returns this operation with an alias from array input `input_index` to array output `output_index`.
    ///
    /// Index bounds and type compatibility are validated during type inference, when the input types are available.
    /// Each input and output can participate in at most one alias.
    pub fn with_input_output_alias(mut self, input_index: usize, output_index: usize) -> Result<Self, TypeError> {
        if let Some(alias) = self
            .input_output_aliases
            .iter()
            .find(|alias| alias.input_index == input_index || alias.output_index == output_index)
        {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` cannot add alias {input_index}->{output_index} because alias \
                 `{alias}` already uses the same input or output",
            )));
        }
        self.input_output_aliases.push(CustomCallInputOutputAlias::new(input_index, output_index));
        Ok(self)
    }

    /// Returns a copy of this [`CustomCallOperation`] marked as having observable side effects.
    #[inline]
    pub fn with_side_effect(mut self) -> Self {
        self.has_side_effect = true;
        self
    }

    /// Returns a copy of this [`CustomCallOperation`] requesting the provided [`CustomCallBatching`] behavior when
    /// the batching transform maps one of its operands. Refer to the documentation of [`CustomCallBatching`] for the
    /// available behaviors and for why the default rejects mapped operands.
    #[inline]
    pub fn with_batching(mut self, batching: CustomCallBatching) -> Self {
        self.batching = batching;
        self
    }

    /// Returns a copy of this operation carrying the provided declared ragged calling convention. Structural
    /// validation occurs during type inference, when the operand types are available.
    #[inline]
    pub fn with_ragged_contract(mut self, contract: CustomCallRaggedContract) -> Self {
        self.ragged_contract = Some(contract);
        self
    }

    /// Returns the name under which the foreign kernel is registered with the executing backend.
    #[inline]
    pub fn target_name(&self) -> &str {
        self.target_name.as_str()
    }

    /// Returns the declared output types of the call.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        self.output_types.as_slice()
    }

    /// Returns the typed configuration attributes forwarded to the kernel, in insertion order.
    #[inline]
    pub fn attributes(&self) -> &[(String, CustomCallAttribute)] {
        self.attributes.as_slice()
    }

    /// Returns the flat array input/output buffer aliases in declaration order.
    #[inline]
    pub fn input_output_aliases(&self) -> &[CustomCallInputOutputAlias] {
        self.input_output_aliases.as_slice()
    }

    /// Returns whether the call has observable side effects beyond its returned outputs.
    #[inline]
    pub fn has_side_effect(&self) -> bool {
        self.has_side_effect
    }

    /// Returns the [`CustomCallBatching`] behavior requested when the batching transform maps one of this call's
    /// operands.
    #[inline]
    pub fn batching(&self) -> CustomCallBatching {
        self.batching
    }

    /// Returns the declared ragged calling convention, when present.
    #[inline]
    pub fn ragged_contract(&self) -> Option<&CustomCallRaggedContract> {
        self.ragged_contract.as_ref()
    }

    /// Returns this payload with every declared output identity renamed according to `renaming`.
    fn renamed(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self {
            target_name: self.target_name.clone(),
            output_types: self
                .output_types
                .iter()
                .map(|r#type| r#type.rename_identities(renaming))
                .collect::<Result<Vec<_>, _>>()?,
            attributes: self.attributes.clone(),
            input_output_aliases: self.input_output_aliases.clone(),
            has_side_effect: self.has_side_effect,
            batching: self.batching,
            ragged_contract: self.ragged_contract.as_ref().map(|contract| contract.renamed(renaming)).transpose()?,
            marker: PhantomData,
        })
    }

    /// Returns the number of trailing first-class output-extent operands this call consumes in the mixed universe,
    /// which is one per dynamic axis occurrence across its declared outputs.
    fn dynamic_output_dimension_count(&self) -> usize {
        self.output_types
            .iter()
            .flat_map(|output_type| output_type.shape().dimensions())
            .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
            .count()
    }

    /// Returns this call's declared output types with a leading batch dimension inserted, which is the declaration a
    /// [`CustomCallBatching::BroadcastAll`] call hands to its kernel.
    ///
    /// An output that aliases an input takes the aligned input's packed type verbatim, because an alias asserts that
    /// the two describe one logical array and the alignment already established that array's batched type. Every
    /// other output inserts the batch dimension itself. The inserted axis is the most major dimension, so an explicit
    /// [`TiledLayout`] shifts each of its logical dimension indices by one and gains the new axis as its most major
    /// physical dimension: layouts are part of the foreign kernel's buffer contract and must not be silently dropped.
    /// A [`StridedLayout`](crate::arrays::StridedLayout) declaration is rejected instead, because a correct batch
    /// stride depends on element sizes the layout does not carry.
    ///
    /// # Parameters
    ///
    ///   - `aligned_input_types`: Packed types of this call's array operands after alignment to the batch axis.
    ///   - `batch_dimension`: Mapped-axis [`Dimension`] inserted as each output's new leading axis.
    ///   - `axis_sharding`: Placement assigned to the inserted axis of outputs that carry sharding metadata.
    fn batch_prefixed_output_types(
        &self,
        aligned_input_types: &[ArrayType],
        batch_dimension: Dimension,
        axis_sharding: &ShardingDimension,
    ) -> Result<Vec<ArrayType>, BatchingError> {
        self.output_types
            .iter()
            .enumerate()
            .map(|(output_index, output_type)| {
                if let Some(aligned_type) = self
                    .input_output_aliases
                    .iter()
                    .find(|alias| alias.output_index == output_index)
                    .and_then(|alias| aligned_input_types.get(alias.input_index))
                {
                    return Ok(aligned_type.clone());
                }
                let mut batched_type = output_type.with_inserted_dimension(0, batch_dimension.clone())?;
                if let Some(sharding) = output_type.sharding() {
                    let sharding = sharding
                        .with_inserted_dimension(0, axis_sharding.clone())
                        .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;
                    batched_type = batched_type
                        .with_sharding(sharding)
                        .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;
                }
                Ok(match output_type.layout() {
                    None => batched_type,
                    Some(Layout::Tiled(layout)) => {
                        let minor_to_major = layout
                            .minor_to_major()
                            .iter()
                            .map(|axis| axis + 1)
                            .chain(std::iter::once(0))
                            .collect::<Vec<_>>();
                        batched_type
                            .with_layout(Layout::Tiled(TiledLayout::new(minor_to_major, layout.tiles().to_vec())))
                    }
                    Some(layout @ Layout::Strided(_)) => {
                        return Err(BatchingError::UnsupportedOperation {
                            message: format!(
                                "custom call `{}` cannot batch output {output_index} because its strided layout \
                                 `{layout}` does not determine the byte stride of the inserted batch axis",
                                self.target_name,
                            ),
                        });
                    }
                })
            })
            .collect()
    }

    /// Returns the [`BatchingError`] reported when operand `index` carries the mapped `batch_axis` and this call has
    /// no way to thread that axis through its opaque kernel.
    fn mapped_operand_error(&self, index: usize, batch_axis: BatchAxis) -> BatchingError {
        BatchingError::UnsupportedOperation {
            message: format!(
                "custom call `{}` has no batching rule for operand {index} mapped at batch axis {}; invoke a kernel \
                 that understands the batch axis, or select an explicit batching behavior with \
                 `CustomCallOperation::with_batching`",
                self.target_name,
                batch_axis.axis().map(|axis| axis.to_string()).unwrap_or_else(|| "replicated".to_string()),
            ),
        }
    }

    /// Renders this payload independently of its homogeneous or composite operation contract.
    fn render_operation(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CUSTOM_CALL_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("target", &self.target_name)?;
            for (name, value) in &self.attributes {
                operation.field(name, value)?;
            }
            for alias in &self.input_output_aliases {
                operation.field("input_output_alias", alias)?;
            }
            if self.has_side_effect {
                operation.field("has_side_effect", true)?;
            }
            if self.batching != CustomCallBatching::default() {
                operation.field("batching", self.batching)?;
            }
            if let Some(contract) = &self.ragged_contract {
                operation.field("ragged_contract", contract)?;
            }
            Ok(())
        })
    }
}

impl From<CustomCallOperation<ArrayType>> for CustomCallOperation<ArrayIrType> {
    fn from(operation: CustomCallOperation<ArrayType>) -> Self {
        Self {
            target_name: operation.target_name,
            output_types: operation.output_types,
            attributes: operation.attributes,
            input_output_aliases: operation.input_output_aliases,
            has_side_effect: operation.has_side_effect,
            batching: operation.batching,
            ragged_contract: operation.ragged_contract,
            marker: PhantomData,
        }
    }
}

impl From<CustomCallOperation<ArrayIrType>> for CustomCallOperation<ArrayType> {
    fn from(operation: CustomCallOperation<ArrayIrType>) -> Self {
        Self {
            target_name: operation.target_name,
            output_types: operation.output_types,
            attributes: operation.attributes,
            input_output_aliases: operation.input_output_aliases,
            has_side_effect: operation.has_side_effect,
            batching: operation.batching,
            ragged_contract: operation.ragged_contract,
            marker: PhantomData,
        }
    }
}

impl<T: Type> Display for CustomCallOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render_operation(formatter, 0)
    }
}

impl<T: Type> CustomCallOperation<T> {
    /// Validates flat input/output aliases against the array operands of this operation.
    fn validate_input_output_aliases(&self, input_types: &[&ArrayType]) -> Result<(), TypeError> {
        for alias in &self.input_output_aliases {
            let Some(input_type) = input_types.get(alias.input_index) else {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` alias `{alias}` refers to input {} but the call has {} array \
                     inputs",
                    alias.input_index,
                    input_types.len(),
                )));
            };
            let Some(output_type) = self.output_types.get(alias.output_index) else {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` alias `{alias}` refers to output {} but the call has {} outputs",
                    alias.output_index,
                    self.output_types.len(),
                )));
            };
            if *input_type != output_type {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` alias `{alias}` requires matching input and output types but \
                     input {} has type `{}` and output {} has type `{}`",
                    alias.input_index, input_type, alias.output_index, output_type,
                )));
            }
        }
        Ok(())
    }

    /// Validates one declared packed ragged axis and its finite physical capacity.
    fn validate_ragged_axis(
        &self,
        kind: &str,
        index: usize,
        axis: usize,
        r#type: &ArrayType,
        dimension: &DimensionVariable,
    ) -> Result<(), TypeError> {
        let Some(physical_dimension) = r#type.shape().dimensions().get(axis) else {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` ragged contract {kind} {index} axis {axis} is out of bounds for type \
                 `{type}`",
            )));
        };
        let Dimension::Static(physical_extent) = physical_dimension else {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` ragged contract {kind} {index} axis {axis} must have a finite static \
                 physical bound but has dimension `{physical_dimension}`",
            )));
        };
        if dimension.bounds().upper().is_none_or(|upper| upper.saturating_sub(1) > *physical_extent) {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` ragged dimension `{dimension}` with bounds {} exceeds the physical \
                 extent {physical_extent} of {kind} {index} axis {axis}",
                dimension.bounds(),
            )));
        }
        Ok(())
    }

    /// Validates this operation's optional ragged calling convention against its complete array signature.
    fn validate_ragged_contract(&self, input_types: &[&ArrayType]) -> Result<(), TypeError> {
        let Some(contract) = &self.ragged_contract else {
            return Ok(());
        };
        if contract.output_bindings.len() != self.output_types.len() {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` ragged contract declares {} output bindings but the call has {} \
                 outputs",
                contract.output_bindings.len(),
                self.output_types.len(),
            )));
        }
        let expected_extent_rank = contract.batch_prefix_count;
        let expected_extent_type = match expected_extent_rank {
            0 => "an integer scalar".to_string(),
            1 => "a batch-prefixed integer vector".to_string(),
            rank => format!("a rank-{rank} batch-prefixed integer tensor"),
        };

        for (binding_index, binding) in contract.input_bindings.iter().enumerate() {
            if let Some(existing) =
                contract.input_bindings[..binding_index].iter().find(|existing| existing.name == binding.name)
            {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` ragged input binding `{}` duplicates binding `{}`",
                    binding.name, existing.name,
                )));
            }
            if let Some(existing) = contract.input_bindings[..binding_index]
                .iter()
                .find(|existing| existing.operand_index == binding.operand_index)
            {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` ragged input bindings `{}` and `{}` both bind operand {}",
                    existing.name, binding.name, binding.operand_index,
                )));
            }
            if let Some(existing) = contract.input_bindings[..binding_index].iter().find(|existing| {
                existing.dimension == binding.dimension && existing.extent_operand_index != binding.extent_operand_index
            }) {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` ragged input bindings `{}` and `{}` reuse dimension `{}` with \
                     different extent operands {} and {}",
                    existing.name,
                    binding.name,
                    binding.dimension,
                    existing.extent_operand_index,
                    binding.extent_operand_index,
                )));
            }
            let Some(input_type) = input_types.get(binding.operand_index) else {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` ragged input binding `{}` refers to operand {} but the call has \
                     {} array inputs",
                    binding.name,
                    binding.operand_index,
                    input_types.len(),
                )));
            };
            self.validate_ragged_axis("input", binding.operand_index, binding.axis, input_type, &binding.dimension)?;
            let Some(extent_type) = input_types.get(binding.extent_operand_index) else {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` ragged input binding `{}` refers to extent operand {} but the \
                     call has {} array inputs",
                    binding.name,
                    binding.extent_operand_index,
                    input_types.len(),
                )));
            };
            if extent_type.rank() != expected_extent_rank || !extent_type.data_type().is_integer() {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` ragged input binding `{}` requires extent operand {} to be \
                     {expected_extent_type} but got `{extent_type}`",
                    binding.name, binding.extent_operand_index,
                )));
            }
        }

        for (output_index, binding) in contract.output_bindings.iter().enumerate() {
            match binding {
                CustomCallRaggedOutputBinding::Preserved { input_binding, axis } => {
                    let Some(input_binding) =
                        contract.input_bindings.iter().find(|binding| binding.name == *input_binding)
                    else {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` ragged output {output_index} preserves unknown input \
                             binding `{input_binding}`",
                        )));
                    };
                    self.validate_ragged_axis(
                        "output",
                        output_index,
                        *axis,
                        &self.output_types[output_index],
                        &input_binding.dimension,
                    )?;
                    if let Some(alias) =
                        self.input_output_aliases.iter().find(|alias| alias.output_index == output_index)
                        && (alias.input_index != input_binding.operand_index || *axis != input_binding.axis)
                    {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` alias `{alias}` conflicts with preserved ragged binding \
                             `{}` because aliases require the same packed input, physical axis, dimension identity, \
                             and extent binding",
                            input_binding.name,
                        )));
                    }
                }
                CustomCallRaggedOutputBinding::Consumed => {
                    if let Some(alias) = self.input_output_aliases.iter().find(|alias| {
                        alias.output_index == output_index
                            && contract.input_bindings.iter().any(|binding| binding.operand_index == alias.input_index)
                    }) {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` consumed ragged output {output_index} cannot retain alias \
                             `{alias}`",
                        )));
                    }
                }
                CustomCallRaggedOutputBinding::Fresh { axis, extent_output_index, dimension } => {
                    if let Some(input_binding) =
                        contract.input_bindings.iter().find(|binding| binding.dimension == *dimension)
                    {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` fresh ragged output {output_index} dimension \
                             `{dimension}` is already declared by input binding `{}`",
                            input_binding.name,
                        )));
                    }
                    if let Some((existing_output_index, existing_extent_output_index)) =
                        contract.output_bindings[..output_index].iter().enumerate().find_map(
                            |(existing_output_index, binding)| match binding {
                                CustomCallRaggedOutputBinding::Fresh {
                                    extent_output_index: existing_extent_output_index,
                                    dimension: existing_dimension,
                                    ..
                                } if existing_dimension == dimension
                                    && existing_extent_output_index != extent_output_index =>
                                {
                                    Some((existing_output_index, existing_extent_output_index))
                                }
                                _ => None,
                            },
                        )
                    {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` fresh ragged outputs {existing_output_index} and \
                             {output_index} reuse dimension `{dimension}` with different extent outputs \
                             {existing_extent_output_index} and {extent_output_index}",
                        )));
                    }
                    self.validate_ragged_axis(
                        "output",
                        output_index,
                        *axis,
                        &self.output_types[output_index],
                        dimension,
                    )?;
                    let Some(extent_type) = self.output_types.get(*extent_output_index) else {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` fresh ragged output {output_index} refers to extent \
                             output {extent_output_index} but the call has {} outputs",
                            self.output_types.len(),
                        )));
                    };
                    if extent_type.rank() != expected_extent_rank || !extent_type.data_type().is_integer() {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` fresh ragged output {output_index} requires extent output \
                             {extent_output_index} to be {expected_extent_type} but got `{extent_type}`",
                        )));
                    }
                    if let Some(alias) = self.input_output_aliases.iter().find(|alias| {
                        alias.output_index == output_index
                            && contract.input_bindings.iter().any(|binding| binding.operand_index == alias.input_index)
                    }) {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_CALL_OPERATION_NAME}` fresh ragged output {output_index} cannot retain alias \
                             `{alias}`",
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Validates homogeneous or mixed ragged input carriers against the declaration and returns active bindings.
    fn active_ragged_bindings<V: Clone + PartialEq>(
        &self,
        inputs: &[CustomCallRaggedInput<'_, V>],
    ) -> Result<Vec<(String, RaggedAxis<V>)>, BatchingError> {
        let Some(contract) = &self.ragged_contract else {
            if let Some((index, ragged_axis)) = inputs
                .iter()
                .enumerate()
                .find_map(|(index, input)| input.ragged_axes.first().map(|axis| (index, axis)))
            {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "custom call `{}` does not support bounded ragged dimension `{}` on operand {}",
                        self.target_name,
                        ragged_axis.dimension(),
                        index,
                    ),
                });
            }
            return Ok(Vec::new());
        };
        if contract.ragged_discharged && inputs.iter().any(|input| !input.ragged_axes.is_empty()) {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("custom call `{}` does not support nested ragged batching", self.target_name),
            });
        }

        self.validate_ragged_contract(&inputs.iter().map(|input| &input.physical_type).collect::<Vec<_>>())?;

        let mut active = Vec::new();
        for (operand_index, input) in inputs.iter().enumerate() {
            if input.ragged_axes.len() > 1 {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "custom call `{}` supports at most one ragged axis per operand but operand {operand_index} \
                         carries {}",
                        self.target_name,
                        input.ragged_axes.len(),
                    ),
                });
            }
            let Some(ragged_axis) = input.ragged_axes.first() else {
                continue;
            };
            let Some(batch_axis) = input.batch_axis else {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "custom call `{}` cannot discharge ragged operand {operand_index} without a mapped batch axis",
                        self.target_name,
                    ),
                });
            };
            let logical_axis = ragged_axis.axis() - usize::from(batch_axis < ragged_axis.axis());
            let Some(binding) = contract
                .input_bindings
                .iter()
                .find(|binding| binding.operand_index == operand_index && binding.axis == logical_axis)
            else {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "custom call `{}` ragged contract does not bind operand {operand_index} axis {logical_axis}",
                        self.target_name,
                    ),
                });
            };
            if ragged_axis.dimension() != &binding.dimension {
                return Err(BatchingError::InvalidBatchMetadata {
                    message: format!(
                        "custom call `{}` ragged input binding `{}` expects dimension `{}` but operand \
                         {operand_index} carries `{}`",
                        self.target_name,
                        binding.name,
                        binding.dimension,
                        ragged_axis.dimension(),
                    ),
                });
            }
            let extent_operand = &inputs[binding.extent_operand_index];
            let Some(extent_batch_axis) = extent_operand.batch_axis else {
                return Err(BatchingError::InvalidBatchMetadata {
                    message: format!(
                        "custom call `{}` ragged input binding `{}` requires mapped extent operand {}",
                        self.target_name, binding.name, binding.extent_operand_index,
                    ),
                });
            };
            let expected_extent_axes = contract.active_extent_axes(batch_axis, extent_batch_axis);
            if ragged_axis.extent_axes() != expected_extent_axes {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "custom call `{}` supports one ragged batching level, but operand {operand_index} ragged \
                         dimension `{}` has extent-axis mapping `{:?}` instead of `{:?}`",
                        self.target_name,
                        ragged_axis.dimension(),
                        ragged_axis.extent_axes(),
                        expected_extent_axes,
                    ),
                });
            }
            if extent_operand.value != ragged_axis.extents() {
                return Err(BatchingError::InvalidBatchMetadata {
                    message: format!(
                        "custom call `{}` ragged input binding `{}` requires operand {} to be the exact extent value \
                         carried by operand {operand_index}",
                        self.target_name, binding.name, binding.extent_operand_index,
                    ),
                });
            }
            active.push((binding.name.clone(), ragged_axis.clone()));
        }
        Ok(active)
    }

    /// Wraps unchanged homogeneous outputs as replicated and attaches any fresh ragged output metadata.
    fn replicated_array_outputs<V: Value<Type = ArrayType>>(
        &self,
        values: Vec<V>,
    ) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        let mut outputs = Vec::with_capacity(values.len());
        for (output_index, value) in values.iter().cloned().enumerate() {
            let output = ArrayBatch::replicated(value);
            let ragged_axis =
                self.ragged_contract.as_ref().and_then(|contract| match &contract.output_bindings[output_index] {
                    CustomCallRaggedOutputBinding::Fresh { axis, extent_output_index, dimension } => {
                        Some(RaggedAxis::new(
                            *axis,
                            values[*extent_output_index].clone(),
                            dimension.clone(),
                            (0..contract.batch_prefix_count).collect(),
                        ))
                    }
                    CustomCallRaggedOutputBinding::Preserved { .. } | CustomCallRaggedOutputBinding::Consumed => None,
                });
            outputs.push(match ragged_axis {
                Some(ragged_axis) => output.with_ragged_axes(vec![ragged_axis])?,
                None => output,
            });
        }
        Ok(outputs)
    }

    /// Attaches declared ragged metadata to batch-prefixed homogeneous outputs and records consumed input dimensions.
    fn array_ragged_outputs<C: Context<Type = ArrayType>, P: ArrayBatchingPolicy<C>>(
        &self,
        values: Vec<C::Value>,
        inputs: &[ArrayBatch<C::Value>],
        active: &[(String, RaggedAxis<C::Value>)],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        let contract = self.ragged_contract.as_ref().unwrap();
        let extent_axes = (0..=contract.batch_prefix_count).collect::<Vec<_>>();
        let mut outputs = Vec::with_capacity(values.len());
        for (output_index, value) in values.iter().cloned().enumerate() {
            let ragged_axis = match &contract.output_bindings[output_index] {
                CustomCallRaggedOutputBinding::Preserved { input_binding, axis } => {
                    active.iter().find(|(name, _)| name == input_binding).map(|(_, source)| {
                        let binding =
                            contract.input_bindings.iter().find(|binding| binding.name == *input_binding).unwrap();
                        RaggedAxis::new(
                            *axis + 1,
                            inputs[binding.extent_operand_index].value().clone(),
                            source.dimension().clone(),
                            extent_axes.clone(),
                        )
                    })
                }
                CustomCallRaggedOutputBinding::Consumed => None,
                CustomCallRaggedOutputBinding::Fresh { axis, extent_output_index, dimension } => Some(RaggedAxis::new(
                    *axis + 1,
                    values[*extent_output_index].clone(),
                    dimension.clone(),
                    extent_axes.clone(),
                )),
            };
            let output = ArrayBatch::new(value, BatchAxis::new(0))?;
            outputs.push(match ragged_axis {
                Some(ragged_axis) => output.with_ragged_axes(vec![ragged_axis])?,
                None => output,
            });
        }
        let consumed = contract.consumed_dimensions(active);
        Ok(BatchedOutputs::new(outputs, consumed))
    }

    /// Wraps unchanged mixed-universe outputs as replicated and attaches any fresh ragged output metadata.
    fn replicated_array_ir_outputs<V: Value<Type = ArrayIrType>>(
        &self,
        values: Vec<V>,
    ) -> Result<Vec<ArrayIrBatch<V>>, BatchingError> {
        let mut outputs = Vec::with_capacity(values.len());
        for (output_index, value) in values.iter().cloned().enumerate() {
            let output = ArrayIrBatch::replicated(value);
            let ragged_axis =
                self.ragged_contract.as_ref().and_then(|contract| match &contract.output_bindings[output_index] {
                    CustomCallRaggedOutputBinding::Fresh { axis, extent_output_index, dimension } => {
                        Some(RaggedAxis::new(
                            *axis,
                            values[*extent_output_index].clone(),
                            dimension.clone(),
                            (0..contract.batch_prefix_count).collect(),
                        ))
                    }
                    CustomCallRaggedOutputBinding::Preserved { .. } | CustomCallRaggedOutputBinding::Consumed => None,
                });
            outputs.push(match ragged_axis {
                Some(ragged_axis) => output.with_ragged_axes(vec![ragged_axis])?,
                None => output,
            });
        }
        Ok(outputs)
    }

    /// Attaches declared ragged metadata to batch-prefixed mixed-universe outputs and records consumed dimensions.
    fn array_ir_ragged_outputs<C: Context<Type = ArrayIrType>>(
        &self,
        values: Vec<C::Value>,
        inputs: &[ArrayIrBatch<C::Value>],
        active: &[(String, RaggedAxis<C::Value>)],
    ) -> Result<BatchedOutputs<C, ArrayIrBatching>, BatchingError> {
        let contract = self.ragged_contract.as_ref().unwrap();
        let extent_axes = (0..=contract.batch_prefix_count).collect::<Vec<_>>();
        let mut outputs = Vec::with_capacity(values.len());
        for (output_index, value) in values.iter().cloned().enumerate() {
            let ragged_axis = match &contract.output_bindings[output_index] {
                CustomCallRaggedOutputBinding::Preserved { input_binding, axis } => {
                    active.iter().find(|(name, _)| name == input_binding).map(|(_, source)| {
                        let binding =
                            contract.input_bindings.iter().find(|binding| binding.name == *input_binding).unwrap();
                        RaggedAxis::new(
                            *axis + 1,
                            inputs[binding.extent_operand_index].value().clone(),
                            source.dimension().clone(),
                            extent_axes.clone(),
                        )
                    })
                }
                CustomCallRaggedOutputBinding::Consumed => None,
                CustomCallRaggedOutputBinding::Fresh { axis, extent_output_index, dimension } => Some(RaggedAxis::new(
                    *axis + 1,
                    values[*extent_output_index].clone(),
                    dimension.clone(),
                    extent_axes.clone(),
                )),
            };
            let output = ArrayIrBatch::new(value, BatchAxis::new(0))?;
            outputs.push(match ragged_axis {
                Some(ragged_axis) => output.with_ragged_axes(vec![ragged_axis])?,
                None => output,
            });
        }
        let consumed = contract.consumed_dimensions(active);
        Ok(BatchedOutputs::new(outputs, consumed))
    }
}

impl Operation for CustomCallOperation<ArrayType> {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        CUSTOM_CALL_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        self.validate_input_output_aliases(&input_types.iter().collect::<Vec<_>>())?;
        self.validate_ragged_contract(&input_types.iter().collect::<Vec<_>>())?;

        // The homogeneous universe has no way to ground a dynamic result extent: only the mixed form accepts the
        // trailing first-class dimension operands that define one.
        for output_type in &self.output_types {
            if output_type.static_shape().is_none() {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` requires explicit result-extent operands for dynamic output type \
                     {output_type}",
                )));
            }
        }
        Ok(self.output_types.clone())
    }

    #[inline]
    fn effects(&self) -> Effects {
        if self.has_side_effect { Effects::single(Effect::OrderedIo) } else { Effects::PURE }
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as crate::Type>::Identity>,
    ) -> Result<Self, TypeError> {
        self.renamed(renaming)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render_operation(formatter, indentation)
    }
}

impl Operation for CustomCallOperation<ArrayIrType> {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        CUSTOM_CALL_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        let dynamic_output_dimensions = self
            .output_types
            .iter()
            .flat_map(|output_type| output_type.shape().dimensions())
            .filter_map(Dimension::variable)
            .collect::<Vec<_>>();
        let Some(array_input_count) = input_types.len().checked_sub(dynamic_output_dimensions.len()) else {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_CALL_OPERATION_NAME}` expects {} trailing output-extent dimensions but only {} inputs were \
                 provided",
                dynamic_output_dimensions.len(),
                input_types.len(),
            )));
        };
        let array_input_types =
            input_types[..array_input_count].iter().map(<&ArrayType>::try_from).collect::<Result<Vec<_>, _>>()?;
        self.validate_input_output_aliases(array_input_types.as_slice())?;
        self.validate_ragged_contract(array_input_types.as_slice())?;
        for (input_type, expected_variable) in input_types[array_input_count..].iter().zip(dynamic_output_dimensions) {
            let actual_variable = <&crate::arrays::DimensionType>::try_from(input_type)?.variable();
            if actual_variable != expected_variable {
                return Err(TypeError::invalid(format!(
                    "`{CUSTOM_CALL_OPERATION_NAME}` output-extent operand defines dimension variable \
                     `{actual_variable}`, but the corresponding declared output axis refers to \
                     `{expected_variable}`",
                )));
            }
        }
        Ok(self.output_types.iter().cloned().map(Into::into).collect())
    }

    #[inline]
    fn effects(&self) -> Effects {
        if self.has_side_effect { Effects::single(Effect::OrderedIo) } else { Effects::PURE }
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        self.renamed(renaming)
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render_operation(formatter, indentation)
    }
}

impl<C: Domain<Type = ArrayType, Value: CustomCall>> InterpretableOperation<C> for CustomCallOperation<ArrayType> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        C::Value::custom_call(self, inputs)
    }
}

// Partial evaluation defers to the default fold-or-residualize behavior of `Program::partially_evaluate`. An all-known
// custom call folds into the known side (executing there only if the known-side context can run foreign kernels), and a
// side-effecting residual call survives dead-code elimination because `Operation::effects` is not `Effects::PURE`.
impl<T: Type, C: Context<Type = T, Operation: From<CustomCallOperation<T>>>> PartiallyEvaluatableOperation<C>
    for CustomCallOperation<T>
where
    CustomCallOperation<T>: Operation<Type = T>,
{
}

impl_reference_free_dischargeable_operation!(<T> CustomCallOperation<T> where T: Type);

impl_differentiable_operation! {
    <T> CustomCallOperation<T>,
    jvp<C>
    where
        T: Type,
    {
        |operation, _context, _driver, _inputs| {
            // Foreign kernels are opaque, so there is no derivative to derive: differentiation reports an error
            // directing users to wrap the call with `custom_jvp` or `custom_vjp`, which is also how JAX handles
            // `ffi_call` differentiation.
            Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "custom call `{}` has no differentiation rule; wrap it with `{}` or `{}` to provide one",
                    operation.target_name,
                    CUSTOM_JVP_OPERATION_NAME,
                    CUSTOM_VJP_OPERATION_NAME,
                ),
            }
            .into())
        }
    },
    transpose = @nonlinear,
}

/// Homogeneous-array batching rule for [`CustomCallOperation`]. A foreign kernel is opaque, so Ryft cannot derive how
/// a batch axis threads through it. A call whose operands are *all replicated* is nevertheless bound unchanged through
/// the parent context and reports replicated outputs, matching JAX, which only invokes a batching rule once some
/// operand is actually mapped.
///
/// That all-replicated shortcut is sound *for this operation specifically* because a custom call is region-free by
/// construction: [`Operation::infer_output_types`] rejects every attached region, so the kernel is a leaf whose only
/// observable inputs are its operands. A foreign kernel therefore cannot observe the transform's named axis, and
/// running it unchanged over replicated operands computes exactly what each batch item would have computed on its
/// own. The shortcut must never be generalized to region-carrying operations, because a region can contain a
/// named-axis operation whose value differs per batch item even when every operand of the enclosing instruction is
/// replicated. `.tasks/plan_custom_derivative_batching_axis_parity.md` records the JAX fixture pinning that
/// counterexample (`vmap` with `in_axes=None`, an explicit extent, and a named-axis index still produces
/// `[0, 1, 2]`), which is why the custom-derivative wrappers always batch their regions structurally.
///
/// A mapped operand is instead governed by the call's own [`CustomCallBatching`] behavior:
/// [`Rejected`](CustomCallBatching::Rejected) reports a [`BatchingError::UnsupportedOperation`] naming that operand
/// and its mapped axis, [`Sequential`](CustomCallBatching::Sequential) stages one [`ScanOperation`] whose body
/// performs a single unbatched call (mapped operands realigned to batch axis `0` and sliced per iteration, replicated
/// operands threaded as invariant carries), and [`BroadcastAll`](CustomCallBatching::BroadcastAll) aligns every
/// operand to batch axis `0` and rebinds one call whose declared outputs gain the same leading batch dimension. Both
/// mapped behaviors keep the staged program's size independent of the batch extent and compose with nested batching,
/// because the rewritten instruction is bound through the parent context and carries the same behavior selection.
///
/// [`Sequential`](CustomCallBatching::Sequential) requires a statically known mapped extent: the scan trip count is a
/// host `usize` in this universe. The mixed [`ArrayIrType`] rule below owns the dynamic-extent case, where the trip
/// count is a first-class dimension operand.
impl<C: Context<Type = ArrayType>, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for CustomCallOperation<ArrayType>
where
    C::Value: PartialEq,
    C::Operation: From<CustomCallOperation<ArrayType>> + From<ScanOperation<C::Constant>>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        let ragged_inputs = inputs
            .iter()
            .map(|input| -> Result<_, BatchingError> {
                Ok(CustomCallRaggedInput {
                    value: input.value(),
                    batch_axis: input.batch_axis_position(),
                    ragged_axes: input.ragged_axes(),
                    physical_type: input.r#type().unbatched_type(input.batch_axis())?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let active_ragged_bindings = self.active_ragged_bindings(ragged_inputs.as_slice())?;
        let Some((index, mapped)) = inputs.iter().enumerate().find(|(_, input)| !input.batch_axis().is_replicated())
        else {
            let values = inputs.iter().map(ArrayBatch::value).cloned().collect::<Vec<_>>();
            let outputs = context.parent().bind(self.clone(), Vec::new(), values.as_slice())?;
            return Ok(self.replicated_array_outputs(outputs)?.into());
        };

        match self.batching {
            CustomCallBatching::Rejected => Err(self.mapped_operand_error(index, mapped.batch_axis())),
            CustomCallBatching::Sequential { unroll } => {
                // Realign every mapped operand to batch axis 0 so the scan consumes one per-item row per iteration,
                // and keep replicated operands as invariant loop carries.
                let mut carry_indices = Vec::new();
                let mut stacked_indices = Vec::new();
                let mut aligned = Vec::with_capacity(inputs.len());
                for (index, input) in inputs.iter().enumerate() {
                    if input.batch_axis().is_replicated() {
                        carry_indices.push(index);
                        aligned.push(input.clone());
                    } else {
                        stacked_indices.push(index);
                        aligned.push(P::match_axis(context, input, Axis::from(0))?);
                    }
                }

                // Build the scan body: one unbatched application of this same call over `[carries..., slices...]`,
                // returning the unchanged carries followed by that item's outputs.
                let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
                let mut operands = vec![None; inputs.len()];
                let carry_inputs = carry_indices
                    .iter()
                    .map(|&index| {
                        let input_type = aligned[index].r#type().unbatched_type(aligned[index].batch_axis())?;
                        let input = builder.add_input(input_type);
                        operands[index] = Some(input);
                        Ok(input)
                    })
                    .collect::<Result<Vec<_>, BatchingError>>()?;
                for &index in &stacked_indices {
                    let input_type = aligned[index].r#type().unbatched_type(aligned[index].batch_axis())?;
                    operands[index] = Some(builder.add_input(input_type));
                }
                let operands = operands.into_iter().map(Option::unwrap).collect::<Vec<_>>();
                let ragged_contract = self.ragged_contract.as_ref().map(|contract| {
                    if active_ragged_bindings.is_empty() { contract.clone() } else { contract.ragged_discharged() }
                });
                let operation = Self { ragged_contract, ..self.clone() };
                let outputs = builder.add_instruction(operation, Vec::new(), operands, None)?.to_vec();
                let body_outputs = carry_inputs.iter().copied().chain(outputs).collect::<Vec<_>>();
                let body = builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
                    body_outputs,
                    vec![Placeholder; inputs.len()],
                    vec![Placeholder; carry_inputs.len() + self.output_types.len()],
                )?;

                let mut scan = ScanOperation::<C::Constant>::new(carry_inputs.len(), P::axis_size(context)?);
                if let Some(unroll) = unroll {
                    scan = scan.with_unroll(unroll)?;
                }
                let packed = carry_indices
                    .iter()
                    .chain(stacked_indices.iter())
                    .map(|&index| aligned[index].value().clone())
                    .collect::<Vec<_>>();
                let mut outputs = context.parent().bind(scan, vec![body], packed.as_slice())?;
                check_count!("output", outputs, carry_inputs.len() + self.output_types.len(), ProgramError);
                outputs.drain(..carry_inputs.len());
                if self.ragged_contract.is_none() {
                    Ok(outputs
                        .into_iter()
                        .map(|value| ArrayBatch::new(value, Some(0)))
                        .collect::<Result<Vec<_>, _>>()?
                        .into())
                } else {
                    self.array_ragged_outputs::<C, P>(outputs, aligned.as_slice(), active_ragged_bindings.as_slice())
                }
            }
            CustomCallBatching::BroadcastAll => {
                let aligned = inputs
                    .iter()
                    .map(|input| P::match_axis(context, input, Axis::from(0)))
                    .collect::<Result<Vec<_>, _>>()?;
                let aligned_types = aligned.iter().map(|batch| batch.r#type().into_owned()).collect::<Vec<_>>();
                let output_types = self.batch_prefixed_output_types(
                    aligned_types.as_slice(),
                    P::axis_dimension(context)?,
                    context.axis_sharding(),
                )?;
                let values = aligned.iter().map(ArrayBatch::value).cloned().collect::<Vec<_>>();
                let ragged_contract = self
                    .ragged_contract
                    .as_ref()
                    .map(|contract| contract.batch_prefixed(!active_ragged_bindings.is_empty()));
                let operation = Self { output_types, ragged_contract, ..self.clone() };
                let outputs = context.parent().bind(operation, Vec::new(), values.as_slice())?;
                if self.ragged_contract.is_none() {
                    Ok(outputs
                        .into_iter()
                        .map(|value| ArrayBatch::new(value, Some(0)))
                        .collect::<Result<Vec<_>, _>>()?
                        .into())
                } else {
                    self.array_ragged_outputs::<C, P>(outputs, aligned.as_slice(), active_ragged_bindings.as_slice())
                }
            }
        }
    }
}

/// Mixed array/dimension batching rule for [`CustomCallOperation`]. It applies the same all-replicated shortcut and
/// the same [`CustomCallBatching`] behaviors as the homogeneous rule above, with two composite-universe additions.
///
/// Every trailing first-class output-extent operand must be replicated: a per-batch-item extent would make the call's
/// results ragged, which the array IR cannot represent. An extent-free call whose mapped extent is statically known is
/// exactly the homogeneous contract, so it delegates to the projected homogeneous rule through
/// [`batch_projected_operation`]. A dynamic mapped extent stays here, because a first-class dimension is not an array
/// value and therefore cannot cross the projected array boundary as a scan trip count or a broadcast extent.
///
/// [`Sequential`](CustomCallBatching::Sequential) threads the replicated extents as leading invariant scan carries and
/// consumes the mapped rows one per iteration, so the body's call sees exactly the per-item extents it declared.
/// [`BroadcastAll`](CustomCallBatching::BroadcastAll) instead rebinds one call whose declared outputs gain the mapped
/// batch dimension, prepending the transform's extent value to each output's trailing extent group when that batch
/// dimension is itself dynamic.
impl<C: Context<Type = ArrayIrType>> BatchableOperation<C, ArrayIrBatching> for CustomCallOperation<ArrayIrType>
where
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: PartialEq + ValueProjection<ArrayType, Projected: PartialEq + Transpose + Value<Type = ArrayType>>,
    C::Operation: From<CustomCallOperation<ArrayIrType>>
        + From<DynamicBroadcastOperation>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + From<ScanOperation<C::Constant>>
        + OperationProjection<ArrayType>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<CustomCallOperation<ArrayType>>
        + From<ScanOperation<<C::Constant as ValueProjection<ArrayType>>::Projected>>
        + From<TransposeOperation>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatching>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatching>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatching>, BatchingError> {
        let extent_count = self.dynamic_output_dimension_count();
        let Some(array_input_count) = inputs.len().checked_sub(extent_count) else {
            return Err(ProgramError::InvalidInputCount { expected: extent_count, actual: inputs.len() }.into());
        };
        let (arrays, extents) = inputs.split_at(array_input_count);
        let ragged_inputs = arrays
            .iter()
            .map(|input| -> Result<_, BatchingError> {
                let value_type = input.value().r#type();
                Ok(CustomCallRaggedInput {
                    value: input.value(),
                    batch_axis: input.batch_axis_position(),
                    ragged_axes: input.ragged_axes(),
                    physical_type: <&ArrayType>::try_from(value_type.as_ref())?.unbatched_type(input.batch_axis())?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let active_ragged_bindings = self.active_ragged_bindings(ragged_inputs.as_slice())?;
        for extent in extents {
            extent.validate_replicated_dimension()?;
        }
        let batch_dimension = <&DimensionType>::try_from(context.axis_extent().r#type().as_ref())?.to_dimension();
        if extents.is_empty() && batch_dimension.value().is_some() {
            return batch_projected_operation(context, &CustomCallOperation::<ArrayType>::from(self.clone()), inputs);
        }

        let Some((index, mapped)) = arrays.iter().enumerate().find(|(_, input)| !input.batch_axis().is_replicated())
        else {
            let values = inputs.iter().map(ArrayIrBatch::value).cloned().collect::<Vec<_>>();
            let outputs = context.parent().bind(self.clone(), Vec::new(), values.as_slice())?;
            return Ok(self.replicated_array_ir_outputs(outputs)?.into());
        };

        match self.batching {
            CustomCallBatching::Rejected => Err(self.mapped_operand_error(index, mapped.batch_axis())),
            CustomCallBatching::Sequential { unroll } => {
                let mut carry_indices = Vec::new();
                let mut stacked_indices = Vec::new();
                let mut aligned = Vec::with_capacity(arrays.len());
                for (index, input) in arrays.iter().enumerate() {
                    if input.batch_axis().is_replicated() {
                        carry_indices.push(index);
                        aligned.push(input.clone());
                    } else {
                        stacked_indices.push(index);
                        aligned.push(align_array_batch(context, input.clone(), Axis::from(0))?);
                    }
                }

                // The replicated extents lead the carries so the body's call can reuse them verbatim as its own
                // trailing extent operands, exactly as the unbatched call declared them.
                let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
                let mut carry_inputs =
                    extents.iter().map(|extent| builder.add_input(extent.unbatched_type().clone())).collect::<Vec<_>>();
                let mut operands = vec![None; arrays.len()];
                for &index in &carry_indices {
                    let value_type = aligned[index].value().r#type();
                    let input_type =
                        <&ArrayType>::try_from(value_type.as_ref())?.unbatched_type(aligned[index].batch_axis())?;
                    let input = builder.add_input(input_type.into());
                    carry_inputs.push(input);
                    operands[index] = Some(input);
                }
                for &index in &stacked_indices {
                    let value_type = aligned[index].value().r#type();
                    let input_type =
                        <&ArrayType>::try_from(value_type.as_ref())?.unbatched_type(aligned[index].batch_axis())?;
                    operands[index] = Some(builder.add_input(input_type.into()));
                }
                let operands = operands
                    .into_iter()
                    .map(Option::unwrap)
                    .chain(carry_inputs[..extents.len()].iter().copied())
                    .collect::<Vec<_>>();
                let ragged_contract = self.ragged_contract.as_ref().map(|contract| {
                    if active_ragged_bindings.is_empty() { contract.clone() } else { contract.ragged_discharged() }
                });
                let operation = Self { ragged_contract, ..self.clone() };
                let outputs = builder.add_instruction(operation, Vec::new(), operands, None)?.to_vec();
                let body_outputs = carry_inputs.iter().copied().chain(outputs).collect::<Vec<_>>();
                let body = builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
                    body_outputs,
                    vec![Placeholder; carry_inputs.len() + stacked_indices.len()],
                    vec![Placeholder; carry_inputs.len() + self.output_types.len()],
                )?;

                let mut scan = ScanOperation::<C::Constant>::new(carry_inputs.len(), batch_dimension.clone());
                if let Some(unroll) = unroll {
                    scan = scan.with_unroll(unroll)?;
                }
                let mut packed = extents.iter().map(|extent| extent.value().clone()).collect::<Vec<_>>();
                packed.extend(
                    carry_indices.iter().chain(stacked_indices.iter()).map(|&index| aligned[index].value().clone()),
                );
                if batch_dimension.variable().is_some() {
                    packed.push(context.axis_extent().clone());
                }
                let mut outputs = context.parent().bind(scan, vec![body], packed.as_slice())?;
                check_count!("output", outputs, carry_inputs.len() + self.output_types.len(), ProgramError);
                outputs.drain(..carry_inputs.len());
                if self.ragged_contract.is_none() {
                    Ok(outputs
                        .into_iter()
                        .map(|value| ArrayIrBatch::new(value, BatchAxis::new(0)))
                        .collect::<Result<Vec<_>, _>>()?
                        .into())
                } else {
                    self.array_ir_ragged_outputs::<C>(outputs, aligned.as_slice(), active_ragged_bindings.as_slice())
                }
            }
            CustomCallBatching::BroadcastAll => {
                let aligned = arrays
                    .iter()
                    .map(|input| align_array_batch(context, input.clone(), Axis::from(0)))
                    .collect::<Result<Vec<_>, _>>()?;
                let aligned_types = aligned
                    .iter()
                    .map(|batch| Ok(<&ArrayType>::try_from(batch.value().r#type().as_ref())?.clone()))
                    .collect::<Result<Vec<_>, TypeError>>()?;
                let output_types = self.batch_prefixed_output_types(
                    aligned_types.as_slice(),
                    batch_dimension.clone(),
                    context.axis_sharding(),
                )?;
                let ragged_contract = self
                    .ragged_contract
                    .as_ref()
                    .map(|contract| contract.batch_prefixed(!active_ragged_bindings.is_empty()));
                let operation = Self { output_types, ragged_contract, ..self.clone() };

                // Regroup the trailing extents: each output's inserted batch axis is its new leading dynamic axis,
                // followed by that output's originally declared extents in axis order.
                let mut values = aligned.iter().map(|batch| batch.value().clone()).collect::<Vec<_>>();
                let mut declared_extents = extents.iter();
                for output_type in &self.output_types {
                    if batch_dimension.variable().is_some() {
                        values.push(context.axis_extent().clone());
                    }
                    for dimension in output_type.shape().dimensions() {
                        if matches!(dimension, Dimension::Dynamic(_)) {
                            values.push(declared_extents.next().unwrap().value().clone());
                        }
                    }
                }
                let outputs = context.parent().bind(operation, Vec::new(), values.as_slice())?;
                if self.ragged_contract.is_none() {
                    Ok(outputs
                        .into_iter()
                        .map(|value| ArrayIrBatch::new(value, BatchAxis::new(0)))
                        .collect::<Result<Vec<_>, _>>()?
                        .into())
                } else {
                    self.array_ir_ragged_outputs::<C>(outputs, aligned.as_slice(), active_ragged_bindings.as_slice())
                }
            }
        }
    }
}

/// Represents the ability to call foreign kernels registered with the executing backend. [`CustomCall`] stages or
/// executes a [`CustomCallOperation`]; refer to its documentation for the calling convention and the transform
/// rules. The capability method dispatches through the first input's context, so it needs at least one input
/// (zero-input custom calls can still be staged directly through a program builder).
pub trait CustomCall: Sized {
    /// Calls the foreign kernel described by `operation` with the provided inputs, returning one value per
    /// declared output type, and a [`ProgramError`] if something goes wrong.
    fn custom_call<'a, I: IntoIterator<Item = &'a Self>>(
        operation: &CustomCallOperation<ArrayType>,
        inputs: I,
    ) -> Result<Vec<Self>, ProgramError>
    where
        Self: 'a;
}

/// Any context-carrying value calls foreign kernels by binding a [`CustomCallOperation<ArrayType>`] through its own
/// context. The conversion bound makes this disjoint from the eager reference value types (whose context operation is
/// [`ConstantOperation`]), so it covers the transform tracers and
/// backend-owned values without conflicting with concrete implementations.
impl<V: Value<Type = ArrayType>> CustomCall for V
where
    V::DispatchDomain: Context<Operation: From<CustomCallOperation<ArrayType>>>,
{
    fn custom_call<'a, I: IntoIterator<Item = &'a Self>>(
        operation: &CustomCallOperation<ArrayType>,
        inputs: I,
    ) -> Result<Vec<Self>, ProgramError>
    where
        Self: 'a,
    {
        let inputs = inputs.into_iter().cloned().collect::<Vec<_>>();
        let Some(first) = inputs.first() else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "the custom-call capability dispatches through its first input's context, so calling '{}' with \
                     no inputs requires staging the operation through a program builder instead",
                    operation.target_name(),
                ),
            });
        };
        first.dispatch_domain().bind(operation.clone(), Vec::new(), inputs.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayBatching, ArrayIrBatch, ArrayIrBatching, ArrayIrOperation, ArrayIrValue,
        ArrayOperation, DataType, Dimension, DimensionBounds, DimensionType, DimensionValue, DimensionVariable,
        RaggedAxis, Shape, ShardingDimension, StridedLayout,
    };
    use crate::batching::{
        BatchAxis, BatchableOperation, BatchedProgram, BatchingContext, BatchingError, BatchingTracer,
        ProgramBatchingOutputAxesPolicy,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{DifferentiationError, TransposableOperation};
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};
    use crate::tracing::{DomainTracer, Trace, TracingContext};

    use super::*;

    /// Returns the `f32[2]` array type used throughout these tests.
    fn vector_type() -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]))
    }

    // Returns a one-input preserved-output ragged contract over a packed `f32[4]` value and scalar extent operand.
    fn preserved_ragged_contract(dimension: DimensionVariable) -> CustomCallRaggedContract {
        CustomCallRaggedContract::new(
            vec![CustomCallRaggedInputBinding::new("data", 0, 0, 1, dimension)],
            vec![CustomCallRaggedOutputBinding::Preserved { input_binding: "data".to_string(), axis: 0 }],
        )
    }

    #[test]
    fn test_custom_call_operation_contract() {
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()])
            .with_attribute("scale", 2.0)
            .with_attribute("count", 4i64)
            .with_attribute("verbose", true)
            .with_attribute("label", "x")
            .with_input_output_alias(0, 0)
            .unwrap()
            .with_side_effect();

        assert_eq!(operation.target_name(), "ryft.test.add_one");
        assert_eq!(operation.output_types(), &[vector_type()]);
        assert_eq!(
            operation.attributes(),
            &[
                ("scale".to_string(), CustomCallAttribute::F64(2.0)),
                ("count".to_string(), CustomCallAttribute::I64(4)),
                ("verbose".to_string(), CustomCallAttribute::Boolean(true)),
                ("label".to_string(), CustomCallAttribute::String("x".to_string())),
            ],
        );
        assert_eq!(operation.input_output_aliases(), &[CustomCallInputOutputAlias::new(0, 0)]);
        assert!(operation.has_side_effect());
        assert_eq!(operation.name(), CUSTOM_CALL_OPERATION_NAME);
        assert_eq!(operation.effects(), Effects::single(Effect::OrderedIo));
        assert_eq!(operation.infer_output_types(&[vector_type()], &[]), Ok(vec![vector_type()]),);
        // Long attribute lists wrap onto one line per field.
        assert_eq!(
            operation.to_string(),
            indoc! {"
                custom_call [
                    target=ryft.test.add_one,
                    scale=2.0,
                    count=4,
                    verbose=true,
                    label=x,
                    input_output_alias=0->0,
                    has_side_effect=true,
                ]
            "}
            .trim_end(),
        );

        let pure = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);
        assert_eq!(pure.effects(), Effects::PURE);
        assert_eq!(pure.to_string(), "custom_call [target=ryft.test.add_one]");
        let roundtrip =
            CustomCallOperation::<ArrayType>::from(CustomCallOperation::<ArrayIrType>::from(operation.clone()));
        assert_eq!(roundtrip.input_output_aliases(), operation.input_output_aliases());
        assert!(matches!(
            operation.clone().with_input_output_alias(0, 1),
            Err(TypeError::Invalid { message })
                if message == "`custom_call` cannot add alias 0->1 because alias `0->0` already uses the same input \
                               or output",
        ));
        assert!(matches!(
            operation.clone().with_input_output_alias(1, 0),
            Err(TypeError::Invalid { message })
                if message == "`custom_call` cannot add alias 1->0 because alias `0->0` already uses the same input \
                               or output",
        ));
        assert_eq!(
            CustomCallOperation::new("ryft.test.add_one", vec![vector_type()])
                .with_input_output_alias(1, 0)
                .unwrap()
                .infer_output_types(&[vector_type()], &[]),
            Err(TypeError::invalid("`custom_call` alias `1->0` refers to input 1 but the call has 1 array inputs",)),
        );
        assert_eq!(
            CustomCallOperation::new("ryft.test.add_one", vec![vector_type()])
                .with_input_output_alias(0, 1)
                .unwrap()
                .infer_output_types(&[vector_type()], &[]),
            Err(TypeError::invalid("`custom_call` alias `0->1` refers to output 1 but the call has 1 outputs",)),
        );
        assert_eq!(
            CustomCallOperation::new("ryft.test.add_one", vec![ArrayType::scalar(DataType::F32)])
                .with_input_output_alias(0, 0)
                .unwrap()
                .infer_output_types(&[vector_type()], &[]),
            Err(TypeError::invalid(
                "`custom_call` alias `0->0` requires matching input and output types but input 0 has type `f32[2]` \
                 and output 0 has type `f32[]`",
            )),
        );

        // Composite output extents are positional SSA operands: output-major and then axis-major.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::new(2, Some(17)).unwrap());
        let dynamic_output_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Dynamic(rows.clone()),
                Dimension::Static(3),
                Dimension::Dynamic(columns.clone()),
            ]),
        );
        let dynamic_operation = CustomCallOperation::<ArrayIrType>::from(CustomCallOperation::new(
            "ryft.test.dynamic",
            vec![dynamic_output_type.clone()],
        ));
        let input_types = vec![
            vector_type().into(),
            DimensionType::new(rows.clone()).into(),
            DimensionType::new(columns.clone()).into(),
        ];
        let aliased_dynamic_operation = CustomCallOperation::<ArrayIrType>::from(
            CustomCallOperation::new("ryft.test.dynamic", vec![dynamic_output_type.clone()])
                .with_input_output_alias(0, 0)
                .unwrap(),
        );
        assert_eq!(
            aliased_dynamic_operation.infer_output_types(
                &[
                    dynamic_output_type.clone().into(),
                    DimensionType::new(rows.clone()).into(),
                    DimensionType::new(columns.clone()).into(),
                ],
                &[],
            ),
            Ok(vec![dynamic_output_type.clone().into()]),
        );
        assert_eq!(dynamic_operation.infer_output_types(&input_types, &[]), Ok(vec![dynamic_output_type.into()]));
        assert_eq!(
            dynamic_operation.infer_output_types(
                &[vector_type().into(), DimensionType::new(columns).into(), DimensionType::new(rows).into()],
                &[],
            ),
            Err(TypeError::invalid(
                "`custom_call` output-extent operand defines dimension variable `columns`, but the corresponding \
                 declared output axis refers to `rows`",
            )),
        );
        assert_eq!(
            dynamic_operation.infer_output_types(
                &[DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap(),))
                    .into()],
                &[],
            ),
            Err(TypeError::invalid(
                "`custom_call` expects 2 trailing output-extent dimensions but only 1 inputs were provided",
            )),
        );
    }

    #[test]
    fn test_custom_call_ragged_contract_validation_and_propagation() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let packed_type = ArrayType::new_static(DataType::F32, [4]);
        let extent_type = ArrayType::scalar(DataType::I32);
        let contract = preserved_ragged_contract(length.clone());
        let input_binding = &contract.input_bindings()[0];
        assert_eq!(input_binding.to_string(), "data:operand(0)@0<=operand(1):length");
        assert_eq!(
            format!("{input_binding:?}"),
            "CustomCallRaggedInputBinding { name: \"data\", operand_index: 0, axis: 0, extent_operand_index: 1, \
             dimension: DimensionVariable { name: \"length\", bounds: DimensionBounds { lower: 0, upper: Some(5) }, \
             .. } }",
        );
        let preserved = &contract.output_bindings()[0];
        assert_eq!(preserved.to_string(), "preserve(data)@0");
        assert_eq!(format!("{preserved:?}"), "Preserved { input_binding: \"data\", axis: 0 }");
        let consumed = CustomCallRaggedOutputBinding::Consumed;
        assert_eq!(consumed.to_string(), "consume");
        assert_eq!(format!("{consumed:?}"), "Consumed");
        let fresh = CustomCallRaggedOutputBinding::Fresh { axis: 1, extent_output_index: 2, dimension: length.clone() };
        assert_eq!(fresh.to_string(), "fresh@1<=output(2):length");
        assert_eq!(
            format!("{fresh:?}"),
            "Fresh { axis: 1, extent_output_index: 2, dimension: DimensionVariable { name: \"length\", \
             bounds: DimensionBounds { lower: 0, upper: Some(5) }, .. } }",
        );
        let rendered_contract = "{inputs=[data:operand(0)@0<=operand(1):length], outputs=[preserve(data)@0]}";
        assert_eq!(contract.to_string(), rendered_contract);
        assert_eq!(
            format!("{contract:?}"),
            "CustomCallRaggedContract { input_bindings: [CustomCallRaggedInputBinding { name: \"data\", \
             operand_index: 0, axis: 0, extent_operand_index: 1, dimension: DimensionVariable { name: \"length\", \
             bounds: DimensionBounds { lower: 0, upper: Some(5) }, .. } }], output_bindings: [Preserved { \
             input_binding: \"data\", axis: 0 }], batch_prefix_count: 0, \
             ragged_discharged: false }",
        );
        let operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
            .with_batching(CustomCallBatching::BroadcastAll)
            .with_ragged_contract(contract.clone());
        assert_eq!(operation.ragged_contract(), Some(&contract));
        assert_eq!(
            operation.infer_output_types(&[packed_type.clone(), extent_type.clone()], &[]),
            Ok(vec![packed_type.clone()])
        );
        assert_eq!(
            operation
                .clone()
                .with_input_output_alias(0, 0)
                .unwrap()
                .infer_output_types(&[packed_type.clone(), extent_type.clone()], &[]),
            Ok(vec![packed_type.clone()]),
        );
        assert_eq!(
            operation.to_string(),
            indoc! {"
                custom_call [
                    target=ryft.test.ragged,
                    batching=broadcast_all,
                    ragged_contract={inputs=[data:operand(0)@0<=operand(1):length], \
                 outputs=[preserve(data)@0]},
                ]
            "}
            .trim_end(),
        );

        // The declaration survives both universe conversions and identity renaming.
        let mixed = CustomCallOperation::<ArrayIrType>::from(operation.clone());
        assert_eq!(mixed.ragged_contract(), Some(&contract));
        assert_eq!(CustomCallOperation::<ArrayType>::from(mixed).ragged_contract(), Some(&contract));
        let renamed_length = DimensionVariable::new("renamed_length", DimensionBounds::new(0, Some(5)).unwrap());
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(length, renamed_length.clone()).unwrap();
        let renamed = operation.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.ragged_contract().unwrap().input_bindings()[0].dimension(), &renamed_length);

        // Fresh output identities are renamed independently from input bindings.
        let output_length = DimensionVariable::new("output_length", DimensionBounds::new(0, Some(5)).unwrap());
        let renamed_output_length =
            DimensionVariable::new("renamed_output_length", DimensionBounds::new(0, Some(5)).unwrap());
        let fresh_operation = CustomCallOperation::new(
            "ryft.test.fresh_ragged",
            vec![packed_type.clone(), ArrayType::scalar(DataType::I32)],
        )
        .with_ragged_contract(CustomCallRaggedContract::new(
            vec![CustomCallRaggedInputBinding::new("data", 0, 0, 1, renamed_length.clone())],
            vec![
                CustomCallRaggedOutputBinding::Fresh {
                    axis: 0,
                    extent_output_index: 1,
                    dimension: output_length.clone(),
                },
                CustomCallRaggedOutputBinding::Consumed,
            ],
        ));
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(output_length, renamed_output_length.clone()).unwrap();
        let renamed = fresh_operation.rename_type_identities(&renaming).unwrap();
        assert!(matches!(
            &renamed.ragged_contract().unwrap().output_bindings()[0],
            CustomCallRaggedOutputBinding::Fresh { dimension, .. } if dimension == &renamed_output_length,
        ));

        assert_eq!(
            CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new(
                        "data",
                        0,
                        0,
                        1,
                        DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap()),
                    )],
                    Vec::new(),
                ))
                .infer_output_types(&[packed_type.clone(), extent_type.clone()], &[]),
            Err(TypeError::invalid(
                "`custom_call` ragged contract declares 0 output bindings but the call has 1 outputs",
            )),
        );
        assert_eq!(
            CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![
                        CustomCallRaggedInputBinding::new(
                            "left",
                            0,
                            0,
                            1,
                            DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap()),
                        ),
                        CustomCallRaggedInputBinding::new(
                            "right",
                            0,
                            0,
                            1,
                            DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap()),
                        ),
                    ],
                    vec![CustomCallRaggedOutputBinding::Preserved { input_binding: "left".to_string(), axis: 0 }],
                ))
                .infer_output_types(&[packed_type.clone(), extent_type.clone()], &[]),
            Err(TypeError::invalid("`custom_call` ragged input bindings `left` and `right` both bind operand 0",)),
        );
        assert_eq!(
            CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new(
                        "data",
                        0,
                        0,
                        1,
                        DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap()),
                    )],
                    vec![CustomCallRaggedOutputBinding::Preserved { input_binding: "missing".to_string(), axis: 0 }],
                ))
                .infer_output_types(&[packed_type.clone(), extent_type.clone()], &[]),
            Err(TypeError::invalid("`custom_call` ragged output 0 preserves unknown input binding `missing`",)),
        );
        assert_eq!(
            CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                .with_ragged_contract(preserved_ragged_contract(DimensionVariable::new(
                    "length",
                    DimensionBounds::new(0, Some(6)).unwrap(),
                )))
                .infer_output_types(&[packed_type.clone(), extent_type.clone()], &[]),
            Err(TypeError::invalid(
                "`custom_call` ragged dimension `length` with bounds [0, 6) exceeds the physical extent 4 of input \
                 0 axis 0",
            )),
        );
        assert_eq!(
            CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                .with_ragged_contract(preserved_ragged_contract(DimensionVariable::new(
                    "length",
                    DimensionBounds::new(0, Some(5)).unwrap(),
                )))
                .infer_output_types(&[packed_type.clone(), ArrayType::scalar(DataType::F32)], &[]),
            Err(TypeError::invalid(
                "`custom_call` ragged input binding `data` requires extent operand 1 to be an integer scalar but got \
                 `f32[]`",
            )),
        );

        // Alias preservation requires the same bound input and physical axis. Consumed and fresh outputs cannot alias
        // a ragged-bound input, but an alias to an unrelated dense input remains valid.
        let matrix_type = ArrayType::new_static(DataType::F32, [4, 4]);
        let alias_conflict = CustomCallOperation::new("ryft.test.ragged", vec![matrix_type.clone()])
            .with_input_output_alias(0, 0)
            .unwrap()
            .with_ragged_contract(CustomCallRaggedContract::new(
                vec![CustomCallRaggedInputBinding::new(
                    "data",
                    0,
                    0,
                    1,
                    DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap()),
                )],
                vec![CustomCallRaggedOutputBinding::Preserved { input_binding: "data".to_string(), axis: 1 }],
            ));
        assert!(matches!(
            alias_conflict.infer_output_types(&[matrix_type, extent_type.clone()], &[]),
            Err(TypeError::Invalid { message })
                if message == "`custom_call` alias `0->0` conflicts with preserved ragged binding `data` because \
                    aliases require the same packed input, physical axis, dimension identity, and extent binding",
        ));
        assert!(matches!(
            CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                .with_input_output_alias(0, 0)
                .unwrap()
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new(
                        "data",
                        0,
                        0,
                        1,
                        DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap()),
                    )],
                    vec![CustomCallRaggedOutputBinding::Consumed],
                ))
                .infer_output_types(&[packed_type.clone(), extent_type.clone()], &[]),
            Err(TypeError::Invalid { message })
                if message == "`custom_call` consumed ragged output 0 cannot retain alias `0->0`",
        ));
        assert!(matches!(
            CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone(), ArrayType::scalar(DataType::I32)],)
                .with_input_output_alias(0, 0)
                .unwrap()
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new(
                        "data",
                        0,
                        0,
                        1,
                        DimensionVariable::new("input_length", DimensionBounds::new(0, Some(5)).unwrap()),
                    )],
                    vec![
                        CustomCallRaggedOutputBinding::Fresh {
                            axis: 0,
                            extent_output_index: 1,
                            dimension: DimensionVariable::new(
                                "output_length",
                                DimensionBounds::new(0, Some(5)).unwrap(),
                            ),
                        },
                        CustomCallRaggedOutputBinding::Consumed,
                    ],
                ))
                .infer_output_types(&[packed_type.clone(), extent_type.clone()], &[]),
            Err(TypeError::Invalid { message })
                if message == "`custom_call` fresh ragged output 0 cannot retain alias `0->0`",
        ));
        assert_eq!(
            CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                .with_input_output_alias(1, 0)
                .unwrap()
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new(
                        "data",
                        0,
                        0,
                        2,
                        DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap()),
                    )],
                    vec![CustomCallRaggedOutputBinding::Consumed],
                ))
                .infer_output_types(&[packed_type.clone(), packed_type.clone(), extent_type.clone()], &[]),
            Ok(vec![packed_type.clone()]),
        );
        assert_eq!(
            CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone(), ArrayType::scalar(DataType::I32)],)
                .with_input_output_alias(1, 0)
                .unwrap()
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new(
                        "data",
                        0,
                        0,
                        2,
                        DimensionVariable::new("input_length", DimensionBounds::new(0, Some(5)).unwrap()),
                    )],
                    vec![
                        CustomCallRaggedOutputBinding::Fresh {
                            axis: 0,
                            extent_output_index: 1,
                            dimension: DimensionVariable::new(
                                "output_length",
                                DimensionBounds::new(0, Some(5)).unwrap(),
                            ),
                        },
                        CustomCallRaggedOutputBinding::Consumed,
                    ],
                ))
                .infer_output_types(&[packed_type.clone(), packed_type.clone(), extent_type.clone()], &[]),
            Ok(vec![packed_type.clone(), ArrayType::scalar(DataType::I32)]),
        );

        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        assert!(matches!(
            CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone(), ArrayType::scalar(DataType::I32)],)
                .with_ragged_contract(CustomCallRaggedContract::new(
                    vec![CustomCallRaggedInputBinding::new("data", 0, 0, 1, length.clone())],
                    vec![
                        CustomCallRaggedOutputBinding::Fresh { axis: 0, extent_output_index: 1, dimension: length },
                        CustomCallRaggedOutputBinding::Consumed,
                    ],
                ))
                .infer_output_types(&[packed_type, extent_type], &[]),
            Err(TypeError::Invalid { message })
                if message == "`custom_call` fresh ragged output 0 dimension `length` is already declared by input \
                    binding `data`",
        ));
    }

    #[test]
    fn test_custom_call_ragged_contract_validates_complete_dense_prefix_rank() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3, 4]);
        let twice_prefixed = preserved_ragged_contract(length.clone()).batch_prefixed(false).batch_prefixed(false);
        assert!(matches!(
            CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                .with_ragged_contract(twice_prefixed)
                .infer_output_types(&[packed_type.clone(), ArrayType::new_static(DataType::I32, [2])], &[]),
            Err(TypeError::Invalid { message })
                if message == "`custom_call` ragged input binding `data` requires extent operand 1 to be a rank-2 \
                    batch-prefixed integer tensor but got `i32[2]`",
        ));

        let fresh_contract = CustomCallRaggedContract::new(
            Vec::new(),
            vec![
                CustomCallRaggedOutputBinding::Fresh { axis: 0, extent_output_index: 1, dimension: length },
                CustomCallRaggedOutputBinding::Consumed,
            ],
        )
        .batch_prefixed(false)
        .batch_prefixed(false);
        assert!(matches!(
            CustomCallOperation::new(
                "ryft.test.fresh_ragged",
                vec![packed_type.clone(), ArrayType::new_static(DataType::I32, [2])],
            )
            .with_ragged_contract(fresh_contract)
            .infer_output_types(std::slice::from_ref(&packed_type), &[]),
            Err(TypeError::Invalid { message })
                if message == "`custom_call` fresh ragged output 0 requires extent output 1 to be a rank-2 \
                    batch-prefixed integer tensor but got `i32[2]`",
        ));
    }

    #[test]
    fn test_custom_call_ragged_contract_reuses_extent_sources_for_shared_dimensions() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let packed_type = ArrayType::new_static(DataType::F32, [4]);
        let extent_type = ArrayType::scalar(DataType::I32);
        let input_types = [packed_type.clone(), packed_type.clone(), extent_type.clone(), extent_type.clone()];
        let shared_input_contract = |second_extent_operand_index| {
            CustomCallRaggedContract::new(
                vec![
                    CustomCallRaggedInputBinding::new("lhs", 0, 0, 2, length.clone()),
                    CustomCallRaggedInputBinding::new("rhs", 1, 0, second_extent_operand_index, length.clone()),
                ],
                vec![CustomCallRaggedOutputBinding::Consumed],
            )
        };
        assert_eq!(
            CustomCallOperation::new("ryft.test.shared_input_dimension", vec![packed_type.clone()])
                .with_ragged_contract(shared_input_contract(2))
                .infer_output_types(&input_types, &[]),
            Ok(vec![packed_type.clone()]),
        );
        assert!(matches!(
            CustomCallOperation::new("ryft.test.shared_input_dimension", vec![packed_type.clone()])
                .with_ragged_contract(shared_input_contract(3))
                .infer_output_types(&input_types, &[]),
            Err(TypeError::Invalid { message })
                if message == "`custom_call` ragged input bindings `lhs` and `rhs` reuse dimension `length` with \
                    different extent operands 2 and 3",
        ));

        let output_types =
            [packed_type.clone(), extent_type.clone(), packed_type.clone(), extent_type.clone()].to_vec();
        let shared_output_contract = |second_extent_output_index| {
            CustomCallRaggedContract::new(
                Vec::new(),
                vec![
                    CustomCallRaggedOutputBinding::Fresh { axis: 0, extent_output_index: 1, dimension: length.clone() },
                    CustomCallRaggedOutputBinding::Consumed,
                    CustomCallRaggedOutputBinding::Fresh {
                        axis: 0,
                        extent_output_index: second_extent_output_index,
                        dimension: length.clone(),
                    },
                    CustomCallRaggedOutputBinding::Consumed,
                ],
            )
        };
        assert_eq!(
            CustomCallOperation::new("ryft.test.shared_output_dimension", output_types.clone())
                .with_ragged_contract(shared_output_contract(1))
                .infer_output_types(&[], &[]),
            Ok(output_types.clone()),
        );
        assert!(matches!(
            CustomCallOperation::new("ryft.test.shared_output_dimension", output_types)
                .with_ragged_contract(shared_output_contract(3))
                .infer_output_types(&[], &[]),
            Err(TypeError::Invalid { message })
                if message == "`custom_call` fresh ragged outputs 0 and 2 reuse dimension `length` with different \
                    extent outputs 1 and 3",
        ));
    }

    #[test]
    fn test_custom_call_stages_through_the_tracer_capability() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);
                Ok(CustomCall::custom_call(&operation, std::slice::from_ref(&x))?.remove(0))
            },
            vector_type(),
        )
        .unwrap();
        let program = program.to_flat_program();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = custom_call [target=ryft.test.add_one] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    /// The homogeneous form has no way to ground a dynamic result extent, because only the mixed form accepts the
    /// trailing first-class dimension operands that define one. Type inference rejects such a declaration instead of
    /// returning an ungrounded output type.
    #[test]
    fn test_custom_call_rejects_dynamic_output_types_without_extent_operands() {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let dynamic_output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(3)]));
        let operation = CustomCallOperation::new("ryft.test.dynamic", vec![vector_type(), dynamic_output_type]);
        assert_eq!(
            operation.infer_output_types(&[vector_type()], &[]),
            Err(TypeError::invalid(
                "`custom_call` requires explicit result-extent operands for dynamic output type f32[rows, 3]",
            )),
        );
    }

    #[test]
    fn test_custom_call_is_rejected_by_the_reference_backend() {
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);
        let result = InterpretableOperation::<EagerContext<Array>>::interpret(
            &operation,
            &EagerContext::new(),
            &EmptyRegionDriver,
            &[Array::vector(vec![1.0, 2.0])],
        );
        assert!(matches!(
            result,
            Err(ProgramError::UnsupportedOperation { message })
                if message == "the reference array backend cannot execute the foreign kernel `ryft.test.add_one`",
        ));
    }

    #[test]
    fn test_custom_call_rejects_differentiation_and_batching() {
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(vector_type());
        let output = builder.add_instruction(operation.clone(), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        assert!(matches!(
            program.jvp(),
            Err(error)
                if error.to_string().contains("custom call `ryft.test.add_one` has no differentiation rule"),
        ));
        assert!(matches!(
            program.batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(error)
                if error.to_string()
                    == "custom call `ryft.test.add_one` has no batching rule for operand 0 mapped at batch axis 0; \
                        invoke a kernel that understands the batch axis, or select an explicit batching behavior \
                        with `CustomCallOperation::with_batching`",
        ));
    }

    /// A custom call whose operands are all replicated is bound unchanged and reports replicated outputs. This is the
    /// JAX-parity behavior: a batching rule is consulted only once an operand is actually mapped, and this shortcut is
    /// sound because the region-free foreign kernel cannot observe the transform's axis.
    #[test]
    fn test_custom_call_batches_all_replicated_operands_unchanged() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);
                Ok(vec![CustomCall::custom_call(&operation, inputs.iter())?.remove(0)])
            },
            vec![vector_type(), vector_type()],
        )
        .unwrap();

        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::replicated()]);
        let batched = batched.to_flat_program();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:f32[2] = custom_call [target=ryft.test.add_one] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_batch_rejects_ragged_operands_before_binding() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let input = ArrayBatch::new(Array::matrix(2, 3, vec![1.0_f32; 6]), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1_i32, 3]), variable, vec![0])])
            .unwrap();
        let operation = CustomCallOperation::new("ryft.test.side_effect", vec![vector_type()])
            .with_batching(CustomCallBatching::BroadcastAll)
            .with_side_effect();
        let context = BatchingContext::<_, ArrayBatching>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        assert!(matches!(
            operation.batch(&context, &EmptyRegionDriver, &[input]),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "custom call `ryft.test.side_effect` does not support bounded ragged dimension `length` \
                    on operand 0",
        ));
    }

    #[test]
    fn test_custom_call_ragged_contract_discharges_sequential_and_broadcast_batching() -> Result<(), ProgramError> {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let packed_type = ArrayType::new_static(DataType::F32, [4]);
        let extent_type = ArrayType::scalar(DataType::I32);

        for behavior in [CustomCallBatching::Sequential { unroll: None }, CustomCallBatching::BroadcastAll] {
            let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
            let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 4]));
            let extents = trace.input(ArrayType::new_static(DataType::I32, [2]));
            let context = BatchingContext::<_, ArrayBatching>::new(trace.clone(), 2);
            let data = ArrayBatch::new(packed, BatchAxis::new(0))?.with_ragged_axes(vec![RaggedAxis::new(
                1,
                extents.clone(),
                length.clone(),
                vec![0],
            )])?;
            let extent_operand = ArrayBatch::new(extents.clone(), BatchAxis::new(0))?;
            let operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                .with_batching(behavior)
                .with_ragged_contract(preserved_ragged_contract(length.clone()));
            assert_eq!(
                operation.infer_output_types(&[packed_type.clone(), extent_type.clone()], &[]),
                Ok(vec![packed_type.clone()])
            );

            let (outputs, evidence) =
                operation.batch(&context, &EmptyRegionDriver, &[data, extent_operand])?.into_parts();
            assert!(evidence.is_empty());
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].ragged_axes().len(), 1);
            assert_eq!(outputs[0].ragged_axes()[0].axis(), 1);
            assert_eq!(outputs[0].ragged_axes()[0].extent_axes(), &[0]);
            assert_eq!(outputs[0].ragged_axes()[0].dimension(), &length);
            assert_eq!(outputs[0].ragged_axes()[0].extents(), &extents);

            let output_id = outputs[0].value().atom_id().unwrap();
            let program = trace.builder().borrow().clone().build::<Vec<Array>, Vec<Array>>(
                vec![output_id],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )?;
            let rendered = program.to_string();
            if matches!(behavior, CustomCallBatching::Sequential { .. }) {
                assert_eq!(
                    rendered,
                    indoc! {"
                        lambda %0:f32[2, 4], %1:i32[2] .
                        let %2:f32[2, 4] = scan [carry_count=0, length=2, reverse=false] %0 %1 [
                            body={
                                lambda %0:f32[4], %1:i32[] .
                                let %2:f32[4] = custom_call [
                                    target=ryft.test.ragged,
                                    batching=sequential,
                                    ragged_contract={inputs=[data:operand(0)@0<=operand(1):length], \
                         outputs=[preserve(data)@0], ragged_discharged=true},
                                ] %0 %1
                                in (%2)
                            },
                        ]
                        in (%2)
                    "}
                    .trim_end(),
                );
            } else {
                assert_eq!(
                    rendered,
                    indoc! {"
                        lambda %0:f32[2, 4], %1:i32[2] .
                        let %2:f32[2, 4] = custom_call [
                            target=ryft.test.ragged,
                            batching=broadcast_all,
                            ragged_contract={inputs=[data:operand(0)@1<=operand(1):length], \
                         outputs=[preserve(data)@1], batch_prefix_count=1, \
                         ragged_discharged=true},
                        ] %0 %1
                        in (%2)
                    "}
                    .trim_end(),
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_custom_call_ragged_contract_rejects_extent_identity_mismatch_and_nested_batching() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 4]));
        let extents = trace.input(ArrayType::new_static(DataType::I32, [2]));
        let other_extents = trace.input(ArrayType::new_static(DataType::I32, [2]));
        let context = BatchingContext::<_, ArrayBatching>::new(trace, 2);
        let data = ArrayBatch::new(packed, BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents, length.clone(), vec![0])])
            .unwrap();
        let extent_operand = ArrayBatch::new(other_extents, BatchAxis::new(0)).unwrap();
        let operation = CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [4])])
            .with_batching(CustomCallBatching::BroadcastAll)
            .with_ragged_contract(preserved_ragged_contract(length));
        assert!(matches!(
            operation.batch(&context, &EmptyRegionDriver, &[data.clone(), extent_operand.clone()]),
            Err(BatchingError::InvalidBatchMetadata { message })
                if message == "custom call `ryft.test.ragged` ragged input binding `data` requires operand 1 to be \
                               the exact extent value carried by operand 0",
        ));

        let nested = CustomCallOperation {
            ragged_contract: operation.ragged_contract.as_ref().map(CustomCallRaggedContract::ragged_discharged),
            ..operation
        };
        assert!(matches!(
            nested.batch(&context, &EmptyRegionDriver, &[data, extent_operand]),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "custom call `ryft.test.ragged` does not support nested ragged batching",
        ));
    }

    #[test]
    fn test_array_ir_sequential_custom_call_records_ragged_discharge_in_the_scan_body() -> Result<(), ProgramError> {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let batch_size = DimensionVariable::new("batch_size", DimensionBounds::new(1, Some(5)).unwrap());
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 4]).into());
        let extents = trace.input(ArrayType::new_static(DataType::I32, [2]).into());
        let axis_extent = trace.input(DimensionType::new(batch_size).into());
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), axis_extent);
        let data = ArrayIrBatch::new(packed, BatchAxis::new(0))?.with_ragged_axes(vec![RaggedAxis::new(
            1,
            extents.clone(),
            length.clone(),
            vec![0],
        )])?;
        let extent_operand = ArrayIrBatch::new(extents, BatchAxis::new(0))?;
        let operation = CustomCallOperation::<ArrayIrType>::from(
            CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [4])])
                .with_batching(CustomCallBatching::Sequential { unroll: None })
                .with_ragged_contract(preserved_ragged_contract(length)),
        );
        let output = operation.batch(&context, &EmptyRegionDriver, &[data, extent_operand])?.into_parts().0.remove(0);
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output.value().atom_id().unwrap()],
            vec![Placeholder; 3],
            vec![Placeholder],
        )?;
        let scan = program
            .entry_region()
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayIrOperation::Scan(_)))
            .unwrap();
        let body = program.region(scan.regions()[0])?;
        let contract = body
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                ArrayIrOperation::CustomCall(operation) => operation.ragged_contract(),
                _ => None,
            })
            .unwrap();
        assert!(contract.ragged_discharged);
        Ok(())
    }

    #[test]
    fn test_custom_call_consumes_shared_dimensions_only_when_no_binding_is_preserved() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let contract = CustomCallRaggedContract::new(
            vec![
                CustomCallRaggedInputBinding::new("lhs", 0, 0, 2, length.clone()),
                CustomCallRaggedInputBinding::new("rhs", 1, 0, 2, length.clone()),
            ],
            vec![CustomCallRaggedOutputBinding::Preserved { input_binding: "rhs".to_string(), axis: 0 }],
        );
        let active = vec![
            ("lhs".to_string(), RaggedAxis::new(0, Array::scalar(2_i32), length.clone(), Vec::new())),
            ("rhs".to_string(), RaggedAxis::new(0, Array::scalar(2_i32), length.clone(), Vec::new())),
        ];
        assert!(contract.consumed_dimensions(active.as_slice()).is_empty());

        let consumed_contract = CustomCallRaggedContract::new(
            contract.input_bindings.clone(),
            vec![CustomCallRaggedOutputBinding::Consumed],
        );
        assert_eq!(consumed_contract.consumed_dimensions(active.as_slice()), vec![length]);
    }

    #[test]
    fn test_custom_call_ragged_contract_accepts_first_ragged_discharge_after_a_dense_prefix() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 3, 4]));
        let extents = trace.input(ArrayType::new_static(DataType::I32, [3, 2]));
        let context = BatchingContext::<_, ArrayBatching>::new(trace.clone(), 2);
        let data = ArrayBatch::new(packed, BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(2, extents.clone(), length.clone(), vec![1, 0])])
            .unwrap();
        let extent_operand = ArrayBatch::new(extents, BatchAxis::new(1)).unwrap();
        let operation =
            CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [3, 4])])
                .with_batching(CustomCallBatching::BroadcastAll)
                .with_ragged_contract(preserved_ragged_contract(length.clone()).batch_prefixed(false));

        let outputs = operation.batch(&context, &EmptyRegionDriver, &[data, extent_operand]).unwrap().into_parts().0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].ragged_axes().len(), 1);
        assert_eq!(outputs[0].ragged_axes()[0].axis(), 2);
        assert_eq!(outputs[0].ragged_axes()[0].extent_axes(), &[0, 1]);
        assert_eq!(outputs[0].ragged_axes()[0].dimension(), &length);
        assert_eq!(
            outputs[0].ragged_axes()[0].extents().r#type().into_owned(),
            ArrayType::new_static(DataType::I32, [2, 3]),
        );

        let builder = trace.builder();
        let builder = builder.borrow();
        let staged_contract = builder
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                ArrayOperation::CustomCall(operation) => operation.ragged_contract(),
                _ => None,
            })
            .unwrap();
        assert_eq!(staged_contract.batch_prefix_count, 2);
        assert!(staged_contract.ragged_discharged);
        drop(builder);

        // The mixed-universe path performs the same alignment and must attach the aligned extent value as well.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 3, 4]).into());
        let extents = trace.input(ArrayType::new_static(DataType::I32, [3, 2]).into());
        let axis_extent = trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace, axis_extent);
        let data = ArrayIrBatch::new(packed, BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(2, extents.clone(), length.clone(), vec![1, 0])])
            .unwrap();
        let extent_operand = ArrayIrBatch::new(extents, BatchAxis::new(1)).unwrap();
        let operation = CustomCallOperation::<ArrayIrType>::from(
            CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [3, 4])])
                .with_batching(CustomCallBatching::BroadcastAll)
                .with_ragged_contract(preserved_ragged_contract(length).batch_prefixed(false)),
        );
        let outputs = operation.batch(&context, &EmptyRegionDriver, &[data, extent_operand]).unwrap().into_parts().0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].ragged_axes()[0].axis(), 2);
        assert_eq!(outputs[0].ragged_axes()[0].extent_axes(), &[0, 1]);
        let extent_type = outputs[0].ragged_axes()[0].extents().r#type();
        assert_eq!(
            <&ArrayType>::try_from(extent_type.as_ref()).unwrap(),
            &ArrayType::new_static(DataType::I32, [2, 3]),
        );
    }

    #[test]
    fn test_custom_call_ragged_contract_attaches_fresh_output_extents() -> Result<(), ProgramError> {
        let input_length = DimensionVariable::new("input_length", DimensionBounds::new(0, Some(5)).unwrap());
        let output_length = DimensionVariable::new("output_length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 4]));
        let extents = trace.input(ArrayType::new_static(DataType::I32, [2]));
        let context = BatchingContext::<_, ArrayBatching>::new(trace, 2);
        let data = ArrayBatch::new(packed, BatchAxis::new(0))?.with_ragged_axes(vec![RaggedAxis::new(
            1,
            extents.clone(),
            input_length.clone(),
            vec![0],
        )])?;
        let extent_operand = ArrayBatch::new(extents, BatchAxis::new(0))?;
        let operation = CustomCallOperation::new(
            "ryft.test.fresh_ragged",
            vec![ArrayType::new_static(DataType::F32, [4]), ArrayType::scalar(DataType::I32)],
        )
        .with_batching(CustomCallBatching::BroadcastAll)
        .with_ragged_contract(CustomCallRaggedContract::new(
            vec![CustomCallRaggedInputBinding::new("data", 0, 0, 1, input_length.clone())],
            vec![
                CustomCallRaggedOutputBinding::Fresh {
                    axis: 0,
                    extent_output_index: 1,
                    dimension: output_length.clone(),
                },
                CustomCallRaggedOutputBinding::Consumed,
            ],
        ));
        let (outputs, evidence) = operation.batch(&context, &EmptyRegionDriver, &[data, extent_operand])?.into_parts();
        assert_eq!(evidence, vec![input_length]);
        assert_eq!(outputs[0].ragged_axes().len(), 1);
        assert_eq!(outputs[0].ragged_axes()[0].dimension(), &output_length);
        assert_eq!(outputs[0].ragged_axes()[0].extents(), outputs[1].value());
        assert!(outputs[1].ragged_axes().is_empty());
        Ok(())
    }

    #[test]
    fn test_custom_call_ragged_contract_attaches_fresh_extents_without_ragged_inputs() -> Result<(), ProgramError> {
        let output_length = DimensionVariable::new("output_length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = trace.input(ArrayType::new_static(DataType::F32, [2, 4]));
        let replicated_input = trace.input(ArrayType::new_static(DataType::F32, [4]));
        let context = BatchingContext::<_, ArrayBatching>::new(trace, 2);
        let operation = CustomCallOperation::new(
            "ryft.test.fresh_ragged",
            vec![ArrayType::new_static(DataType::F32, [4]), ArrayType::scalar(DataType::I32)],
        )
        .with_batching(CustomCallBatching::BroadcastAll)
        .with_ragged_contract(CustomCallRaggedContract::new(
            Vec::new(),
            vec![
                CustomCallRaggedOutputBinding::Fresh {
                    axis: 0,
                    extent_output_index: 1,
                    dimension: output_length.clone(),
                },
                CustomCallRaggedOutputBinding::Consumed,
            ],
        ));
        let (outputs, evidence) = operation
            .batch(&context, &EmptyRegionDriver, &[ArrayBatch::new(input, BatchAxis::new(0))?])?
            .into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs[0].ragged_axes().len(), 1);
        assert_eq!(outputs[0].ragged_axes()[0].dimension(), &output_length);
        assert_eq!(outputs[0].ragged_axes()[0].extents(), outputs[1].value());
        assert!(outputs[1].ragged_axes().is_empty());

        let (outputs, evidence) = operation
            .batch(&context, &EmptyRegionDriver, &[ArrayBatch::replicated(replicated_input)])?
            .into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].ragged_axes()[0].axis(), 0);
        assert!(outputs[0].ragged_axes()[0].extent_axes().is_empty());
        assert_eq!(outputs[0].ragged_axes()[0].extents(), outputs[1].value());
        Ok(())
    }

    #[test]
    fn test_array_ir_custom_call_ragged_contract_survives_projection() -> Result<(), ProgramError> {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let packed = trace.input(ArrayType::new_static(DataType::F32, [2, 4]).into());
        let extents = trace.input(ArrayType::new_static(DataType::I32, [2]).into());
        let axis_extent = trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace, axis_extent);
        let data = ArrayIrBatch::new(packed, BatchAxis::new(0))?.with_ragged_axes(vec![RaggedAxis::new(
            1,
            extents.clone(),
            length.clone(),
            vec![0],
        )])?;
        let extent_operand = ArrayIrBatch::new(extents.clone(), BatchAxis::new(0))?;
        let operation = CustomCallOperation::<ArrayIrType>::from(
            CustomCallOperation::new("ryft.test.ragged", vec![ArrayType::new_static(DataType::F32, [4])])
                .with_batching(CustomCallBatching::BroadcastAll)
                .with_ragged_contract(preserved_ragged_contract(length.clone())),
        );
        let (outputs, evidence) = operation.batch(&context, &EmptyRegionDriver, &[data, extent_operand])?.into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].ragged_axes().len(), 1);
        assert_eq!(outputs[0].ragged_axes()[0].axis(), 1);
        assert_eq!(outputs[0].ragged_axes()[0].dimension(), &length);
        assert_eq!(outputs[0].ragged_axes()[0].extents(), &extents);
        Ok(())
    }

    #[test]
    fn test_array_ir_custom_call_attaches_fresh_ragged_output_to_replicated_call() -> Result<(), ProgramError> {
        let output_length = DimensionVariable::new("output_length", DimensionBounds::new(0, Some(5)).unwrap());
        let batch_size = DimensionVariable::new("batch_size", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = CustomCallOperation::<ArrayIrType>::from(
            CustomCallOperation::new(
                "ryft.test.fresh_ragged",
                vec![ArrayType::new_static(DataType::F32, [4]), ArrayType::scalar(DataType::I32)],
            )
            .with_ragged_contract(CustomCallRaggedContract::new(
                Vec::new(),
                vec![
                    CustomCallRaggedOutputBinding::Fresh {
                        axis: 0,
                        extent_output_index: 1,
                        dimension: output_length.clone(),
                    },
                    CustomCallRaggedOutputBinding::Consumed,
                ],
            )),
        );
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = trace.input(ArrayType::new_static(DataType::F32, [4]).into());
        let axis_extent = trace.input(DimensionType::new(batch_size).into());
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace, axis_extent);
        let (outputs, evidence) =
            operation.batch(&context, &EmptyRegionDriver, &[ArrayIrBatch::replicated(input)])?.into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].ragged_axes()[0].axis(), 0);
        assert_eq!(outputs[0].ragged_axes()[0].dimension(), &output_length);
        assert!(outputs[0].ragged_axes()[0].extent_axes().is_empty());
        assert_eq!(outputs[0].ragged_axes()[0].extents(), outputs[1].value());
        Ok(())
    }

    #[test]
    fn test_array_ir_custom_call_fresh_output_composes_under_nested_dense_batching() -> Result<(), ProgramError> {
        let output_length = DimensionVariable::new("output_length", DimensionBounds::new(0, Some(5)).unwrap());
        let operation = CustomCallOperation::<ArrayIrType>::from(
            CustomCallOperation::new(
                "ryft.test.fresh_ragged",
                vec![ArrayType::new_static(DataType::F32, [4]), ArrayType::scalar(DataType::I32)],
            )
            .with_batching(CustomCallBatching::BroadcastAll)
            .with_ragged_contract(CustomCallRaggedContract::new(
                Vec::new(),
                vec![
                    CustomCallRaggedOutputBinding::Fresh {
                        axis: 0,
                        extent_output_index: 1,
                        dimension: output_length.clone(),
                    },
                    CustomCallRaggedOutputBinding::Consumed,
                ],
            )),
        );
        let output_types = operation
            .output_types
            .iter()
            .map(|output_type| output_type.with_inserted_dimension(0, Dimension::Static(3)))
            .collect::<Result<Vec<_>, _>>()?;
        let operation = CustomCallOperation {
            output_types,
            ragged_contract: operation.ragged_contract.as_ref().map(|contract| contract.batch_prefixed(false)),
            ..operation
        };

        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let outer = DimensionVariable::new("outer", DimensionBounds::new(1, Some(5)).unwrap());
        let axis_extent = trace.input(DimensionType::new(outer.clone()).into());
        let input = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(outer), Dimension::Static(3), Dimension::Static(4)]),
            )
            .into(),
        );
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), axis_extent);
        let (outputs, evidence) = operation
            .batch(&context, &EmptyRegionDriver, &[ArrayIrBatch::new(input, BatchAxis::new(0))?])?
            .into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].ragged_axes().len(), 1);
        assert_eq!(outputs[0].ragged_axes()[0].axis(), 2);
        assert_eq!(outputs[0].ragged_axes()[0].extent_axes(), &[0, 1]);
        assert_eq!(outputs[0].ragged_axes()[0].dimension(), &output_length);
        assert_eq!(outputs[0].ragged_axes()[0].extents(), outputs[1].value());

        let output_ids = outputs.iter().map(|output| output.value().atom_id()).collect::<Result<Vec<_>, _>>()?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            output_ids,
            vec![Placeholder, Placeholder],
            vec![Placeholder, Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<outer ∈ [1, 5)>, %1:f32[outer, 3, 4] .
                let %2:f32[outer, 3, 4], %3:i32[outer, 3] = custom_call [
                    target=ryft.test.fresh_ragged,
                    batching=broadcast_all,
                    ragged_contract={inputs=[], outputs=[fresh@2<=output(1):output_length, consume], \
                 batch_prefix_count=2},
                ] %1 %0 %0
                in (%2, %3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    /// The mixed universe applies the same all-replicated shortcut, including the trailing first-class output-extent
    /// operand, which stays an ordinary replicated operand of the unchanged call.
    #[test]
    fn test_array_ir_custom_call_batches_all_replicated_operands_unchanged() -> Result<(), ProgramError> {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let operation = CustomCallOperation::<ArrayIrType>::from(CustomCallOperation::new(
            "ryft.test.dynamic",
            vec![output_type.clone()],
        ));

        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = trace.input(vector_type().into());
        let extent = trace.input(DimensionType::new(rows).into());
        let context = BatchingContext::<_, ArrayIrBatching>::new(
            trace.clone(),
            trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())),
        );
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(input)),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent)),
        ];
        let [output] = context.bind(operation, Vec::new(), &inputs)?.try_into().unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(output_type));

        let output_id = output.into_batch().into_value().atom_id().unwrap();
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output_id],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:dimension<rows ∈ [1, 9)> .
                let %2:dimension<2> = const 2
                    %3:f32[rows] = custom_call [target=ryft.test.dynamic] %0 %1
                in (%3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_array_ir_custom_call_rejects_transforms() {
        let operation = CustomCallOperation::<ArrayIrType>::from(CustomCallOperation::new(
            "ryft.test.add_one",
            vec![vector_type()],
        ));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(vector_type().into());
        let output = builder.add_instruction(operation.clone(), Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            program.jvp(),
            Err(error)
                if error.to_string()
                    == "custom call `ryft.test.add_one` has no differentiation rule; wrap it with `custom_jvp` or \
                        `custom_vjp` to provide one",
        ));

        let batching_context = BatchingContext::<_, ArrayIrBatching>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let mapped = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])),
            BatchAxis::new(0),
        )
        .unwrap();
        assert!(matches!(
            operation.batch(&batching_context, &EmptyRegionDriver, &[mapped]),
            Err(BatchingError::UnsupportedOperation { message })
                if message
                    == "custom call `ryft.test.add_one` has no batching rule for operand 0 mapped at batch axis 0; \
                        invoke a kernel that understands the batch axis, or select an explicit batching behavior \
                        with `CustomCallOperation::with_batching`",
        ));

        let mut transposition_context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert!(matches!(
            operation.transpose(&mut transposition_context, &EmptyRegionDriver, &[], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `custom_call` is not transposable",
        ));
    }

    /// The batching selection is rendered as an ordinary operation field, but only when it differs from the default
    /// `Rejected` behavior, so every existing rendering stays byte-for-byte unchanged. The selection survives both
    /// directions of the homogeneous/mixed conversion and identity renaming.
    #[test]
    fn test_custom_call_batching_selection_renders_only_when_non_default() {
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);
        assert_eq!(operation.batching(), CustomCallBatching::Rejected);
        assert_eq!(operation.to_string(), "custom_call [target=ryft.test.add_one]");

        let sequential = operation.clone().with_batching(CustomCallBatching::Sequential { unroll: None });
        assert_eq!(sequential.batching(), CustomCallBatching::Sequential { unroll: None });
        assert_eq!(sequential.to_string(), "custom_call [target=ryft.test.add_one, batching=sequential]");

        let unrolled = operation.clone().with_batching(CustomCallBatching::Sequential { unroll: Some(2) });
        assert_eq!(unrolled.to_string(), "custom_call [target=ryft.test.add_one, batching=sequential(unroll=2)]");

        let broadcast = operation.with_batching(CustomCallBatching::BroadcastAll).with_side_effect();
        assert_eq!(
            broadcast.to_string(),
            "custom_call [target=ryft.test.add_one, has_side_effect=true, batching=broadcast_all]",
        );

        // Both conversions into and out of the mixed family, and identity renaming, preserve the selection.
        let mixed = CustomCallOperation::<ArrayIrType>::from(broadcast.clone());
        assert_eq!(mixed.batching(), CustomCallBatching::BroadcastAll);
        assert_eq!(CustomCallOperation::<ArrayType>::from(mixed).batching(), CustomCallBatching::BroadcastAll);
        assert_eq!(
            broadcast.rename_type_identities(&TypeIdentityRenaming::default()).unwrap().batching(),
            CustomCallBatching::BroadcastAll,
        );
    }

    /// `Sequential` stages one carry-free-per-operand `scan` whose body performs exactly one unbatched call: the
    /// mapped operand is sliced one row per iteration while the replicated operand rides along as an invariant carry.
    /// A side-effecting kernel therefore runs once per batch item, in iteration order.
    #[test]
    fn test_custom_call_batches_sequentially_through_a_scan() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()])
                    .with_side_effect()
                    .with_batching(CustomCallBatching::Sequential { unroll: None });
                Ok(vec![CustomCall::custom_call(&operation, inputs.iter())?.remove(0)])
            },
            vec![vector_type(), vector_type()],
        )
        .unwrap();

        let (batched, output_axes) = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        // The staged program keeps exactly one call, inside a three-trip scan body: the side effect therefore occurs
        // three times in iteration order rather than once over a batch-prefixed buffer.
        assert_eq!(batched.effects(), Effects::single(Effect::OrderedIo));
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3, 2], %1:f32[2] .
                let %2:f32[2], %3:f32[3, 2] = scan [carry_count=1, length=3, reverse=false] %1 %0 [
                    body={
                        lambda %0:f32[2], %1:f32[2] .
                        let %2:f32[2] = custom_call [target=ryft.test.add_one, has_side_effect=true, \
                 batching=sequential] %1 %0
                        in (%0, %2)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
    }

    /// The lowering-only unroll factor is wired through to the staged `scan`.
    #[test]
    fn test_custom_call_sequential_batching_forwards_the_unroll_factor() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()])
                    .with_batching(CustomCallBatching::Sequential { unroll: Some(2) });
                Ok(vec![CustomCall::custom_call(&operation, inputs.iter())?.remove(0)])
            },
            vec![vector_type()],
        )
        .unwrap();

        let (batched, _) = program
            .batched(4, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[4, 2] .
                let %1:f32[4, 2] = scan [carry_count=0, length=4, reverse=false, unroll=2] %0 [
                    body={
                        lambda %0:f32[2] .
                        let %1:f32[2] = custom_call [target=ryft.test.add_one, batching=sequential(unroll=2)] %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );

        // An unroll factor that does not divide the batch extent is rejected by the scan contract.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()])
                    .with_batching(CustomCallBatching::Sequential { unroll: Some(3) });
                Ok(vec![CustomCall::custom_call(&operation, inputs.iter())?.remove(0)])
            },
            vec![vector_type()],
        )
        .unwrap();
        assert!(
            program
                .batched(
                    4,
                    ShardingDimension::Replicated,
                    &[BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural
                )
                .is_err(),
        );
    }

    /// `BroadcastAll` materializes every operand on the batch axis and rebinds exactly one call whose declared
    /// outputs gain the same leading extent. An aliased output takes its aligned input's packed type verbatim, so the
    /// alias keeps describing one logical array, and an explicit tiled layout shifts to keep the batch axis most
    /// major.
    #[test]
    fn test_custom_call_batches_by_broadcasting_all_operands() {
        let column_major = vector_type()
            .with_inserted_dimension(1, Dimension::Static(3))
            .unwrap()
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new())));
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let column_major = vector_type()
                    .with_inserted_dimension(1, Dimension::Static(3))
                    .unwrap()
                    .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new())));
                let operation =
                    CustomCallOperation::new("ryft.test.scaled_add", vec![vector_type(), column_major.clone()])
                        .with_input_output_alias(0, 0)
                        .unwrap()
                        .with_batching(CustomCallBatching::BroadcastAll);
                let outputs = CustomCall::custom_call(&operation, inputs.iter())?;
                Ok(outputs)
            },
            vec![vector_type(), vector_type()],
        )
        .unwrap();
        assert_eq!(program.output_types()[1], column_major);

        let (batched, output_axes) = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::new(0)]);
        // The replicated operand is materialized across the batch and the kernel is invoked exactly once. Output 0
        // aliases input 0 and therefore takes that operand's packed type, while output 1 keeps its declared tiled
        // layout with every logical index shifted by one and the inserted batch axis as its most major dimension.
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3, 2], %1:f32[2] .
                let %2:f32[3, 2] = broadcast [output_type=f32[3, 2], output_axes=[1]] %1
                    %3:f32[3, 2], %4:f32[3, 2, 3][layout=tiled{1,2,0}] = custom_call [target=ryft.test.scaled_add, \
                 input_output_alias=0->0, batching=broadcast_all] %0 %2
                in (%3, %4)
            "}
            .trim_end(),
        );
    }

    /// A strided output layout cannot be shifted onto a batched declaration, because the byte stride of the inserted
    /// axis is not derivable from the layout alone, so `BroadcastAll` reports that explicitly.
    #[test]
    fn test_custom_call_broadcast_batching_rejects_strided_output_layouts() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let strided = vector_type().with_layout(Layout::Strided(StridedLayout::new(vec![4])));
                let operation = CustomCallOperation::new("ryft.test.add_one", vec![strided])
                    .with_batching(CustomCallBatching::BroadcastAll);
                Ok(vec![CustomCall::custom_call(&operation, inputs.iter())?.remove(0)])
            },
            vec![vector_type()],
        )
        .unwrap();
        assert!(matches!(
            program.batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message
                    == "custom call `ryft.test.add_one` cannot batch output 0 because its strided layout \
                        `strided{4}` does not determine the byte stride of the inserted batch axis",
        ));
    }

    /// Nested batching composes: the inner rule stages its rewritten instruction through the parent context, which is
    /// itself a batching context, so the outer level batches that instruction structurally.
    #[test]
    fn test_custom_call_batching_composes_under_nested_batching() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()])
                    .with_batching(CustomCallBatching::BroadcastAll);
                Ok(vec![CustomCall::custom_call(&operation, inputs.iter())?.remove(0)])
            },
            vec![vector_type()],
        )
        .unwrap();
        let (inner, _) = program
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        let (outer, output_axes) = inner
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            outer.to_string(),
            indoc! {"
                lambda %0:f32[2, 3, 2] .
                let %1:f32[2, 3, 2] = custom_call [target=ryft.test.add_one, batching=broadcast_all] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_ragged_contract_composes_under_nested_dense_batching() {
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let packed_type = ArrayType::new_static(DataType::F32, [4]);
        let extent_type = ArrayType::scalar(DataType::I32);
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                let operation = CustomCallOperation::new("ryft.test.ragged", vec![packed_type.clone()])
                    .with_batching(CustomCallBatching::BroadcastAll)
                    .with_ragged_contract(preserved_ragged_contract(length.clone()));
                Ok(vec![CustomCall::custom_call(&operation, inputs.iter())?.remove(0)])
            },
            vec![packed_type.clone(), extent_type],
        )
        .unwrap();
        let (inner, _) = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        let (outer, output_axes) = inner
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            outer.to_string(),
            indoc! {"
                lambda %0:f32[2, 3, 4], %1:i32[2, 3] .
                let %2:f32[2, 3, 4] = custom_call [
                    target=ryft.test.ragged,
                    batching=broadcast_all,
                    ragged_contract={inputs=[data:operand(0)@2<=operand(1):length], \
                 outputs=[preserve(data)@2], batch_prefix_count=2},
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    /// The mixed rule threads replicated first-class output extents as leading invariant scan carries, so the body's
    /// call declares exactly the per-item extents it was given, and it supports a dynamic mapped extent by consuming
    /// it as the scan's trailing trip-count operand.
    #[test]
    fn test_array_ir_custom_call_batches_sequentially_with_extent_carries() -> Result<(), ProgramError> {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let operation = CustomCallOperation::<ArrayIrType>::from(
            CustomCallOperation::new("ryft.test.dynamic", vec![output_type])
                .with_batching(CustomCallBatching::Sequential { unroll: None }),
        );

        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let mapped = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2)])).into(),
        );
        let extent = trace.input(DimensionType::new(rows).into());
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), batch_extent);
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent)),
        ];
        let [output] = context.bind(operation, Vec::new(), &inputs)?.try_into().unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));

        let output_id = output.into_batch().into_value().atom_id().unwrap();
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output_id],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 2], %2:dimension<rows ∈ [1, 9)> .
                let %3:dimension<rows ∈ [1, 9)>, %4:f32[batch, rows] = scan [carry_count=1, length=batch, \
                 reverse=false] %2 %1 %0 [
                    body={
                        lambda %0:dimension<rows ∈ [1, 9)>, %1:f32[2] .
                        let %2:f32[rows] = custom_call [target=ryft.test.dynamic, batching=sequential] %1 %0
                        in (%0, %2)
                    },
                ]
                in (%4)
            "}
            .trim_end(),
        );
        Ok(())
    }

    /// The mixed `BroadcastAll` rule rebinds one call whose declared outputs gain the mapped batch dimension, and
    /// regroups the trailing extents so each output's new leading dynamic axis is grounded by the transform's extent.
    #[test]
    fn test_array_ir_custom_call_broadcasts_all_operands_with_regrouped_extents() -> Result<(), ProgramError> {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let operation = CustomCallOperation::<ArrayIrType>::from(
            CustomCallOperation::new("ryft.test.dynamic", vec![output_type])
                .with_batching(CustomCallBatching::BroadcastAll),
        );

        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let mapped = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2)])).into(),
        );
        let extent = trace.input(DimensionType::new(rows).into());
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), batch_extent);
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent)),
        ];
        let [output] = context.bind(operation, Vec::new(), &inputs)?.try_into().unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));

        let output_id = output.into_batch().into_value().atom_id().unwrap();
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output_id],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 2], %2:dimension<rows ∈ [1, 9)> .
                let %3:f32[batch, rows] = custom_call [target=ryft.test.dynamic, batching=broadcast_all] %1 %0 %2
                in (%3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    /// Side-effect occurrence counts differ between the two behaviors, which is exactly why the selection is
    /// explicit. Both stage a single call instruction and both keep the call's [`Effect::OrderedIo`], so neither is
    /// eliminated or reordered, but `Sequential` executes it once per batch item through the scan's ordered trips
    /// while `BroadcastAll` executes it exactly once over batch-prefixed buffers.
    #[test]
    fn test_custom_call_batching_side_effect_occurrence_counts() {
        for behavior in [CustomCallBatching::Sequential { unroll: None }, CustomCallBatching::BroadcastAll] {
            let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
                |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                    let operation = CustomCallOperation::new("ryft.test.record", vec![vector_type()])
                        .with_side_effect()
                        .with_batching(behavior);
                    Ok(vec![CustomCall::custom_call(&operation, inputs.iter())?.remove(0)])
                },
                vec![vector_type()],
            )
            .unwrap();
            let (batched, _) = program
                .batched(
                    3,
                    ShardingDimension::Replicated,
                    &[BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts();

            let rendered = batched.to_string();
            assert_eq!(rendered.matches(CUSTOM_CALL_OPERATION_NAME).count(), 1, "{behavior}: {rendered}");
            assert_eq!(batched.effects(), Effects::single(Effect::OrderedIo), "{behavior}: {rendered}");
            if matches!(behavior, CustomCallBatching::Sequential { .. }) {
                assert!(rendered.contains("scan [carry_count=0, length=3, reverse=false]"), "{rendered}");
            } else {
                assert!(!rendered.contains("scan"), "{rendered}");
            }
        }
    }

    /// Input/output aliasing survives batching under both explicit behaviors, which is what makes them the only two
    /// alias-compatible strategies. `Sequential` preserves the alias per iteration, where the body's call sees exactly
    /// the unbatched types the alias was declared against, and `BroadcastAll` preserves it because the aliased output
    /// takes its aligned input's packed type verbatim.
    #[test]
    fn test_custom_call_batching_preserves_input_output_aliases() {
        for behavior in [CustomCallBatching::Sequential { unroll: None }, CustomCallBatching::BroadcastAll] {
            let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
                |inputs: Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>| {
                    let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()])
                        .with_input_output_alias(0, 0)?
                        .with_batching(behavior);
                    Ok(vec![CustomCall::custom_call(&operation, inputs.iter())?.remove(0)])
                },
                vec![vector_type()],
            )
            .unwrap();
            let (batched, output_axes) = program
                .batched(
                    3,
                    ShardingDimension::Replicated,
                    &[BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts();

            let rendered = batched.to_string();
            assert_eq!(output_axes, vec![BatchAxis::new(0)], "{behavior}: {rendered}");
            assert!(rendered.contains("input_output_alias=0->0"), "{behavior}: {rendered}");
            assert_eq!(
                batched.output_types(),
                vec![ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]))],
                "{behavior}: {rendered}",
            );
        }
    }
}
