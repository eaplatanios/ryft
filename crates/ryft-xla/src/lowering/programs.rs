use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, Atom, AtomId, BroadcastOperation, CallOperation, DataType, DeviceMesh, Dimension, EagerContext,
    EffectClass, ExternalReferenceBinding, FlatProgram, Instruction, LogicalMesh, Memory, MeshAxisType, Parameterized,
    PlaceOperation, PlacementRequest, Program, ProgramError, RegionId, RegionPlacementRequests, RegionRef, RegionRole,
    RequestedPlacement, Shape, Sharding, ShardingDimension, StopGradientOperation, TagOperation,
    TransferToMemoryOperation, Type as CoreType, Typed, TypedProgram, Value,
};
use ryft_mlir::dialects::{func, shardy, stable_hlo};
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, DetachedBlock, Error as MlirError, Location, LocationRef,
    Operation as MlirOperation, OperationPrintingFlags, Region, SymbolVisibility, Type as MlirType, TypeAndAttributes,
    Value as MlirValue, ValueRef, WalkOrder, WalkResult,
};

use crate::arrays::host::ReductionChange;
use crate::devices::MemoryExtension;
use crate::lowering::contexts::ProgramLoweringContext;
use crate::lowering::sharding::{
    LogicalMeshExtension, OpenDimensions, SHARDY_MESH_SYMBOL_NAME, ShardingExtension,
    validate_declared_pending_reduction, validate_partial_completion, validate_pending_reduction_change,
};
use crate::lowering::types::{
    ArrayTypeLoweringExtension, TypeLoweringExtension, lower_runtime_dimension_size_i64,
    lower_unplaced_constant_output, validate_boundary_memory,
};
use crate::lowering::{
    EffectTokens, Lowerer, LoweringError, ProgramLoweringOperation, RaggedDotLoweringStrategy,
    lower_instruction_operation,
};
use crate::operations::call::{CallFunctionMap, collect_call_functions};
use crate::{
    CompilationError, Constant, Error, ExecutionError, RetainedArray, RuntimeRequirement, Type,
    normalize_runtime_requirements,
};

/// StableHLO/Shardy MLIR module that a staged [`Program`] lowers to, paired with what executing it requires.
pub struct MlirProgram {
    /// Textual StableHLO/Shardy module of this [`MlirProgram`].
    stable_hlo: String,

    /// Mapping from the logical inputs and outputs of the entry function to executable slots.
    boundary_mapping: MlirProgramBoundaryMapping,

    /// Canonical runtime contracts required by emitted calls and synthesized checks.
    runtime_requirements: Arc<[RuntimeRequirement]>,

    /// Module mesh that every sharding and `shard_map` boundary of the module refers to, or `None` when the module
    /// declares no mesh. Compilation checks it against the topology of the device mesh that it targets.
    mesh: Option<LogicalMesh>,
}

impl MlirProgram {
    /// Pairs the rendered `stable_hlo` module with the entry function's `boundary_mapping`, its canonical
    /// `runtime_requirements`, and its module `mesh`.
    #[inline]
    fn new(
        stable_hlo: String,
        boundary_mapping: MlirProgramBoundaryMapping,
        runtime_requirements: Arc<[RuntimeRequirement]>,
        mesh: Option<LogicalMesh>,
    ) -> Self {
        Self { stable_hlo, boundary_mapping, runtime_requirements, mesh }
    }

    /// Returns the rendered StableHLO module of this [`MlirProgram`].
    #[inline]
    pub fn stable_hlo(&self) -> &str {
        &self.stable_hlo
    }

    /// Returns the mapping from the logical inputs and outputs of the entry function to executable slots.
    #[inline]
    pub fn boundary_mapping(&self) -> &MlirProgramBoundaryMapping {
        &self.boundary_mapping
    }

    /// Returns the module mesh that every sharding and `shard_map` boundary of the module refers to, or `None` when the
    /// module declares no mesh. Compilation checks it against the topology of the device mesh that it targets.
    #[inline]
    pub fn mesh(&self) -> Option<&LogicalMesh> {
        self.mesh.as_ref()
    }

    /// Returns the rendered StableHLO module of this lowering, consuming it.
    #[inline]
    pub fn into_stable_hlo(self) -> String {
        self.stable_hlo
    }

    /// Consumes this [`MlirProgram`] and returns its textual module, its entry boundary mapping, and its runtime
    /// requirements. The mapping records whether execution requires the host runtime assertion handler.
    #[inline]
    pub fn into_parts(self) -> (String, MlirProgramBoundaryMapping, Arc<[RuntimeRequirement]>) {
        (self.stable_hlo, self.boundary_mapping, self.runtime_requirements)
    }
}

/// Mapping from logical inputs and outputs to the corresponding slots of an XLA executable. Ryft retains every program
/// input and output in its staged and compiled function metadata. Statically shaped [`DataType::Zero`] leaves carry no
/// runtime payload and are omitted from the StableHLO/PJRT signature, and each mapping stores an executable index for
/// each materialized leaf and `None` for each erased leaf. Dynamically shaped [`DataType::Zero`] leaves deliberately
/// retain private `i1` carriers because the current executable ABI has no independent shape transport. Current XLA
/// compilation may impose stricter limits on dynamic tensors than this projection layer.
///
/// The constructor uses the logical types to compute slot indices and hidden extents. Callers supply the types and
/// sharding annotations separately when projecting them into executable order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlirProgramBoundaryMapping {
    /// Executable slot index for each flattened logical input, or `None` for an erased input.
    input_mapping: Arc<[Option<usize>]>,

    /// Executable slot index for each flattened logical output, or `None` for an erased output.
    output_mapping: Arc<[Option<usize>]>,

    /// Whether each logical flattened input is erased because the executable never reads it, or empty when every input
    /// is read.
    unread_inputs: Arc<[bool]>,

    /// Hidden scalar inputs that carry the runtime extents of tensors padded to their bounds.
    input_dimensions: Arc<[MlirProgramBoundaryMappingInputDimension]>,

    /// Hidden scalar outputs that report bounded dynamic logical output extents.
    output_dimensions: Arc<[MlirProgramBoundaryMappingOutputDimension]>,

    /// Whether the executable carries the ordered I/O boundary token (i.e., whether the program contains
    /// [`OrderedIo`](EffectClass::OrderedIo) or [`DeviceOrderedIo`](EffectClass::DeviceOrderedIo) effects).
    ordered_io: bool,

    /// Whether every ordered I/O occurrence is [`DeviceOrderedIo`](EffectClass::DeviceOrderedIo), so the boundary token
    /// only needs per-device ordering and the executable may run on several devices.
    device_ordered_io: bool,

    /// Whether the executable contains unordered I/O.
    unordered_io: bool,

    /// Whether lowering emitted a runtime assertion, including synthesized checks.
    requires_assertion_handler: bool,
}

impl MlirProgramBoundaryMapping {
    /// Derives the executable boundary mapping for the provided logical input and output types.
    pub(crate) fn new(input_types: &[ArrayType], output_types: &[ArrayType]) -> Self {
        /// Maps each flattened logical type to its executable array slot. Statically shaped [`DataType::Zero`]
        /// entries receive [`None`]. All other entries get consecutive slots starting at zero, in signature order.
        /// This includes dynamically shaped zeros, whose slots carry their runtime shape.
        fn array_slot_mapping(types: &[ArrayType]) -> Arc<[Option<usize>]> {
            let mut executable_index = 0;
            types
                .iter()
                .map(|r#type| {
                    if r#type.data_type().is_zero() && r#type.static_shape().is_some() {
                        None
                    } else {
                        let index = executable_index;
                        executable_index += 1;
                        Some(index)
                    }
                })
                .collect()
        }

        let input_mapping = array_slot_mapping(input_types);
        let mut executable_input_index = input_mapping.iter().flatten().count();
        let mut input_dimensions = Vec::new();
        for (logical_input_index, input_type) in input_types.iter().enumerate() {
            for (axis, dimension) in input_type.shape().dimensions().iter().enumerate() {
                if dimension.value().is_none() && dimension.upper_bound().is_some() {
                    input_dimensions.push(MlirProgramBoundaryMappingInputDimension {
                        axis,
                        logical_input_index,
                        executable_input_index,
                    });
                    executable_input_index += 1;
                }
            }
        }

        let output_mapping = array_slot_mapping(output_types);
        let mut executable_output_index = output_mapping.iter().flatten().count();
        let mut output_dimensions = Vec::new();
        for (logical_output_index, output_type) in output_types.iter().enumerate() {
            for (axis, dimension) in output_type.shape().dimensions().iter().enumerate() {
                if dimension.value().is_none() && dimension.upper_bound().is_some() {
                    output_dimensions.push(MlirProgramBoundaryMappingOutputDimension {
                        axis,
                        logical_output_index,
                        executable_output_index,
                    });
                    executable_output_index += 1;
                }
            }
        }

        Self {
            input_mapping,
            output_mapping,
            unread_inputs: Arc::from([]),
            input_dimensions: input_dimensions.into(),
            output_dimensions: output_dimensions.into(),
            ordered_io: false,
            device_ordered_io: false,
            unordered_io: false,
            requires_assertion_handler: false,
        }
    }

    /// Erases the logical inputs that `unread_inputs` flags (i.e., one flag per logical input), which the executable
    /// never reads, and records them, so that executions and persisted copies of the executable read which inputs are
    /// unread rather than infer it from the mapping, which cannot tell an unread input from a read one that has no
    /// buffer (e.g., of the zero data type). Every flagged input must have a static shape. Flags that state no unread
    /// input leave the mapping unchanged.
    pub(crate) fn with_unread_inputs(mut self, unread_inputs: &[bool]) -> Self {
        if !unread_inputs.contains(&true) {
            return self;
        }

        debug_assert_eq!(unread_inputs.len(), self.input_mapping.len());

        let mut executable_index = 0;
        self.input_mapping = self
            .input_mapping
            .iter()
            .zip(unread_inputs)
            .map(|(mapping, is_unread)| {
                if mapping.is_none() || *is_unread {
                    return None;
                }
                executable_index += 1;
                Some(executable_index - 1)
            })
            .collect();

        let input_dimensions = self.input_dimensions.iter().enumerate().map(|(offset, dimension)| {
            MlirProgramBoundaryMappingInputDimension { executable_input_index: executable_index + offset, ..*dimension }
        });

        self.input_dimensions = input_dimensions.collect();
        self.unread_inputs = unread_inputs.into();
        self
    }

    /// Records that lowering emitted a runtime assertion, so that executions need the runtime assertion handler.
    #[inline]
    pub(crate) fn require_assertion_handler(&mut self) {
        self.requires_assertion_handler = true;
    }

    /// Attaches the effect metadata known at compilation time, also used when restoring persistent executable metadata.
    /// `device_ordered_io` is only meaningful together with `ordered_io` and is cleared otherwise.
    #[inline]
    pub(crate) fn with_effects(
        mut self,
        ordered_io: bool,
        unordered_io: bool,
        requires_assertion_handler: bool,
        device_ordered_io: bool,
    ) -> Self {
        self.ordered_io = ordered_io;
        self.device_ordered_io = ordered_io && device_ordered_io;
        self.unordered_io = unordered_io;
        self.requires_assertion_handler = requires_assertion_handler;
        self
    }

    /// Returns the executable slot index of each logical input, or [`None`] for an erased input.
    #[inline]
    pub fn input_mapping(&self) -> &[Option<usize>] {
        &self.input_mapping
    }

    /// Returns the executable slot index of each logical output, or [`None`] for an erased output.
    #[inline]
    pub fn output_mapping(&self) -> &[Option<usize>] {
        &self.output_mapping
    }

    /// Returns whether each logical input is erased because the executable never reads it, or an empty slice when every
    /// input is read.
    #[inline]
    pub fn unread_inputs(&self) -> &[bool] {
        &self.unread_inputs
    }

    /// Returns whether the logical input at `index` is erased because the executable never reads it.
    #[inline]
    pub fn is_unread_input(&self, index: usize) -> bool {
        self.unread_inputs.get(index).copied().unwrap_or(false)
    }

    /// Returns the hidden scalar inputs carrying bounded dynamic input extents, in executable order.
    #[inline]
    pub fn input_dimensions(&self) -> &[MlirProgramBoundaryMappingInputDimension] {
        &self.input_dimensions
    }

    /// Returns the hidden scalar outputs carrying bounded dynamic output extents, in executable order.
    #[inline]
    pub fn output_dimensions(&self) -> &[MlirProgramBoundaryMappingOutputDimension] {
        &self.output_dimensions
    }

    /// Returns whether this executable carries an ordered I/O boundary token.
    #[inline]
    pub fn has_ordered_io(&self) -> bool {
        self.ordered_io
    }

    /// Returns whether every ordered I/O occurrence is `DeviceOrderedIo`, permitting execution on several devices with
    /// per-device ordering.
    #[inline]
    pub fn has_device_ordered_io(&self) -> bool {
        self.device_ordered_io
    }

    /// Returns whether this executable contains unordered I/O.
    #[inline]
    pub fn has_unordered_io(&self) -> bool {
        self.unordered_io
    }

    /// Returns whether lowering emitted a runtime assertion, including synthesized checks.
    #[inline]
    pub fn requires_assertion_handler(&self) -> bool {
        self.requires_assertion_handler
    }

    /// Returns whether execution participates in the effect barrier, which is the case for executables with ordered or
    /// unordered I/O and for executables that require the runtime assertion handler.
    #[inline]
    pub fn has_effects(&self) -> bool {
        self.ordered_io || self.unordered_io || self.requires_assertion_handler
    }

    /// Returns the ordinal of the hidden ordered I/O token input, which follows every array and extent input.
    #[inline]
    pub fn ordered_io_input_index(&self) -> Option<usize> {
        self.ordered_io.then(|| self.input_mapping.iter().flatten().count() + self.input_dimensions.len())
    }

    /// Returns the ordinal of the hidden ordered I/O token output, which follows every array and extent output.
    #[inline]
    pub fn ordered_io_output_index(&self) -> Option<usize> {
        self.ordered_io.then(|| self.output_mapping.iter().flatten().count() + self.output_dimensions.len())
    }

    /// Returns the number of array inputs, including hidden extent scalars but excluding tokens.
    #[inline]
    pub fn input_array_count(&self) -> usize {
        self.input_mapping.iter().flatten().count() + self.input_dimensions.len()
    }

    /// Returns the number of array outputs, including hidden extent scalars but excluding tokens.
    #[inline]
    pub fn output_array_count(&self) -> usize {
        self.output_mapping.iter().flatten().count() + self.output_dimensions.len()
    }

    /// Returns the total number of executable inputs, including hidden extent scalars and the ordered I/O token
    /// when present.
    #[inline]
    pub fn executable_input_count(&self) -> usize {
        self.input_mapping.iter().flatten().count() + self.input_dimensions.len() + usize::from(self.ordered_io)
    }

    /// Returns the total number of executable outputs, including hidden extent scalars and the ordered I/O token
    /// when present.
    #[inline]
    pub fn executable_output_count(&self) -> usize {
        self.output_array_count() + usize::from(self.ordered_io)
    }

    /// Returns the executable's array input types for the types of the logical `inputs`, one per flattened logical
    /// input. Bounded dynamic dimensions use their maximum extents, and hidden extent scalars follow the data inputs.
    /// When present, the ordered I/O token occupies a separate slot.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidInputCount`] when `inputs` does not have one entry per logical input.
    pub fn executable_input_types(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, Error> {
        check_count!("input", inputs, self.input_mapping.len(), ProgramError);
        let mut types = Self::select_mapped_entries(&self.input_mapping, inputs);
        let mut converted = vec![false; types.len()];
        for input_dimension in self.input_dimensions.iter() {
            // The conversion rewrites every axis of the owning tensor at once, so an input with several dynamic axes
            // is converted only for its first hidden extent entry.
            let executable_index = self.input_mapping[input_dimension.logical_input_index].unwrap();
            if std::mem::replace(&mut converted[executable_index], true) {
                continue;
            }
            let dimensions = types[executable_index]
                .shape()
                .dimensions()
                .iter()
                .map(|dimension| {
                    // Dimension bounds validate an exclusive upper bound above a non-negative lower bound, so a bounded
                    // dynamic axis always yields a static bound here. Only an unbounded axis stays dynamic, and the
                    // materialization of bounded inputs rejects it as having no static upper shape.
                    dimension
                        .upper_bound()
                        .and_then(|upper| upper.checked_sub(1))
                        .map_or_else(|| dimension.clone(), Dimension::Static)
                })
                .collect::<Vec<_>>();
            types[executable_index] = types[executable_index].clone().with_shape(Shape::new(dimensions));
        }
        types.extend(self.input_dimensions.iter().map(|_| ArrayType::scalar(DataType::I32)));
        Ok(types)
    }

    /// Returns the executable's array output types for the types of the logical `outputs`, one per flattened logical
    /// output, with hidden extent scalars after the data outputs. When present, the ordered I/O token occupies a
    /// separate slot.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidOutputCount`] when `outputs` does not have one entry per logical output.
    pub fn executable_output_types(&self, outputs: &[ArrayType]) -> Result<Vec<ArrayType>, Error> {
        check_count!("output", outputs, self.output_mapping.len(), ProgramError);
        let mut types = Self::select_mapped_entries(&self.output_mapping, outputs);
        types.extend(self.output_dimensions.iter().map(|_| ArrayType::scalar(DataType::I64)));
        Ok(types)
    }

    /// Returns the input `shardings`, one per flattened logical input, in executable order, with a replicated scalar
    /// sharding for each hidden extent on the mesh of the input that owns it.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidInputCount`] when `shardings` does not have one entry per logical input.
    pub fn executable_input_shardings(&self, shardings: &[Sharding]) -> Result<Vec<Sharding>, Error> {
        check_count!("input", shardings, self.input_mapping.len(), ProgramError);
        let mut executable_shardings = Self::select_mapped_entries(&self.input_mapping, shardings);
        executable_shardings.extend(
            self.input_dimensions
                .iter()
                .map(|dimension| Sharding::replicated(shardings[dimension.logical_input_index].mesh().clone(), 0)),
        );
        Ok(executable_shardings)
    }

    /// Returns input shardings like [`Self::executable_input_shardings`], leaving a hidden extent input unannotated
    /// when the input that owns it is unannotated.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidInputCount`] when `shardings` does not have one entry per logical input.
    fn executable_optional_input_shardings(
        &self,
        shardings: &[Option<Sharding>],
    ) -> Result<Vec<Option<Sharding>>, Error> {
        check_count!("input", shardings, self.input_mapping.len(), ProgramError);
        let mut executable_shardings = Self::select_mapped_entries(&self.input_mapping, shardings);
        executable_shardings.extend(self.input_dimensions.iter().map(|dimension| {
            shardings[dimension.logical_input_index]
                .as_ref()
                .map(|sharding| Sharding::replicated(sharding.mesh().clone(), 0))
        }));
        Ok(executable_shardings)
    }

    /// Returns the output `shardings`, one per flattened logical output, in executable order, including the sharding
    /// of each hidden extent output, which is replicated on the mesh of the output that owns the extent, or absent
    /// when that output is unannotated.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidOutputCount`] when `shardings` does not have one entry per logical output.
    pub fn executable_optional_output_shardings(
        &self,
        shardings: &[Option<Sharding>],
    ) -> Result<Vec<Option<Sharding>>, Error> {
        check_count!("output", shardings, self.output_mapping.len(), ProgramError);
        let mut executable_shardings = Self::select_mapped_entries(&self.output_mapping, shardings);
        executable_shardings.extend(self.output_dimensions.iter().map(|dimension| {
            shardings[dimension.logical_output_index]
                .as_ref()
                .map(|sharding| Sharding::replicated(sharding.mesh().clone(), 0))
        }));
        Ok(executable_shardings)
    }

    /// Clones the entries in `inputs` for logical inputs that have an executable array slot. Supply one entry per
    /// flattened logical input, including captures. For example, the input mapping `[Some(0), None, Some(1)]` maps
    /// `[a, b, c]` to `[a, c]`. Hidden extent scalars and effect tokens have separate slots.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidInputCount`] when `inputs` does not have one entry per logical input.
    #[inline]
    pub fn select_input_entries<T: Clone>(&self, inputs: &[T]) -> Result<Vec<T>, Error> {
        check_count!("input", inputs, self.input_mapping.len(), ProgramError);
        Ok(Self::select_mapped_entries(&self.input_mapping, inputs))
    }

    /// Clones the entries in `outputs` for logical outputs that have an executable array slot. Supply one entry per
    /// flattened logical output. For example, the output mapping `[None, Some(0)]` maps `[a, b]` to `[b]`. Hidden
    /// extent scalars and effect tokens have separate slots.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidOutputCount`] when `outputs` does not have one entry per logical output.
    #[inline]
    pub fn select_output_entries<T: Clone>(&self, outputs: &[T]) -> Result<Vec<T>, Error> {
        check_count!("output", outputs, self.output_mapping.len(), ProgramError);
        Ok(Self::select_mapped_entries(&self.output_mapping, outputs))
    }

    /// Clones entries from `values` whose corresponding entries in `mapping` are `Some(_)`. Executable indices are
    /// consecutive in logical order, so keeping these entries in sequence gives executable order. `mapping` and
    /// `values` must have the same length, which every caller validates before it selects.
    fn select_mapped_entries<T: Clone>(mapping: &[Option<usize>], values: &[T]) -> Vec<T> {
        debug_assert_eq!(mapping.len(), values.len());
        mapping
            .iter()
            .zip(values)
            .filter_map(|(executable_index, value)| executable_index.map(|_| value.clone()))
            .collect()
    }
}

/// A bounded dynamic input axis whose extent is carried by a hidden scalar executable input.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MlirProgramBoundaryMappingInputDimension {
    /// Axis within the logical input.
    axis: usize,

    /// Logical flattened input containing the dynamic axis.
    logical_input_index: usize,

    /// Executable input slot that carries the runtime extent.
    executable_input_index: usize,
}

impl MlirProgramBoundaryMappingInputDimension {
    /// Returns the dynamic axis within the logical input.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the logical flattened input containing this dynamic axis.
    #[inline]
    pub fn logical_input_index(&self) -> usize {
        self.logical_input_index
    }

    /// Returns the index of the executable input slot that carries the runtime extent.
    #[inline]
    pub fn executable_input_index(&self) -> usize {
        self.executable_input_index
    }
}

/// A bounded dynamic output axis whose extent is carried by a hidden scalar executable output.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MlirProgramBoundaryMappingOutputDimension {
    /// Axis within the logical output.
    axis: usize,

    /// Logical flattened output containing the dynamic axis.
    logical_output_index: usize,

    /// Executable output slot that carries the runtime extent.
    executable_output_index: usize,
}

impl MlirProgramBoundaryMappingOutputDimension {
    /// Returns the dynamic axis within the logical output.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the logical flattened output containing this dynamic axis.
    #[inline]
    pub fn logical_output_index(&self) -> usize {
        self.logical_output_index
    }

    /// Returns the index of the executable output slot that carries the runtime extent.
    #[inline]
    pub fn executable_output_index(&self) -> usize {
        self.executable_output_index
    }
}

/// Lowering of staged [`Program`]s of the XLA backend, over any [`ProgramLoweringOperation`], to StableHLO/Shardy MLIR
/// modules. References and mutable state must have been discharged from a program before it is lowered.
pub trait ProgramLoweringExtension {
    /// Lowers this program according to `options` and returns its [`MlirProgram`], which pairs the textual module
    /// with what executing it requires.
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name to use for the outer `func.func` operation.
    ///   - `options`: Options that determine the entry signature and how the program is prepared.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Core`] when an input or output of the program whose type the options do not provide is not an
    /// array, when pruning or simplifying the program fails, or when the program imports an array whose type names a
    /// mesh of another topology than the meshes of the signature, because a plain trace keeps the type of an array that
    /// it imports (a function compiled through a [`Session`](crate::Session), whose captures follow its call, receives
    /// it where it runs instead). Lowering failures use the corresponding [`Error`] variants (e.g.,
    /// [`LoweringError::UnresolvedReference`] or [`LoweringError::UnresolvedState`] for undischarged references or
    /// state, [`LoweringError::InvalidFunctionName`] for an invalid function name, mismatched signature shardings,
    /// or operations that the backend cannot lower).
    fn to_mlir<S: AsRef<str>>(&self, function_name: S, options: &ProgramLoweringOptions) -> Result<MlirProgram, Error>;

    /// Renders a full StableHLO/Shardy MLIR module for this program according to `options`.
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name to use for the outer `func.func` operation.
    ///   - `options`: Options that determine the entry signature and how the program is prepared.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::to_mlir`].
    #[inline]
    fn to_rendered_mlir<S: AsRef<str>>(
        &self,
        function_name: S,
        options: &ProgramLoweringOptions,
    ) -> Result<String, Error> {
        Ok(self.to_mlir(function_name, options)?.into_stable_hlo())
    }
}

impl<O, Input, Output> ProgramLoweringExtension for Program<Constant, O, Input, Output>
where
    O: ProgramLoweringOperation,
    Input: Parameterized<Constant>,
    Output: Parameterized<Constant>,
{
    fn to_mlir<S: AsRef<str>>(&self, function_name: S, options: &ProgramLoweringOptions) -> Result<MlirProgram, Error> {
        let target_platform = options.target_platform();
        let pruned;
        let program = if options.prune {
            pruned = self.pruned_for_lowering(target_platform)?;
            &pruned
        } else {
            self
        };

        // Only arrays cross the boundary of an executable, so the types that the program provides must be arrays.
        let as_array_types = |types: Vec<Type>| {
            types
                .iter()
                .map(|r#type| <&ArrayType>::try_from(r#type).cloned().map_err(ProgramError::from))
                .collect::<Result<Vec<_>, _>>()
        };
        let input_types = match &options.input_types {
            Some(input_types) => input_types.clone(),
            None => as_array_types(self.entry_region_ref().input_types())?,
        };
        let output_types = match &options.output_types {
            Some(output_types) => output_types.clone(),
            None => as_array_types(self.entry_region_ref().output_types())?,
        };

        // Arrays that session constants refer to enter the module as hidden inputs that precede the public inputs,
        // which are annotated as their runtime arrays are placed, as compilation for a `Session` passes them.
        let regions = program.entry_region_ref().emitted_regions(target_platform, CallTraversal::Enter);
        let session_constants = ProgramSessionConstants::from_regions(regions, target_platform);
        let capture_types = options.capture_types.clone().unwrap_or_else(|| {
            session_constants.inputs.iter().map(|constant| constant.r#type().into_owned()).collect()
        });
        let exact_input_shardings = match &options.exact_input_shardings {
            Some(EntryShardingOverride::Values(shardings)) => Some(shardings.clone()),
            Some(EntryShardingOverride::Unannotated) => Some(vec![None; capture_types.len() + input_types.len()]),
            None => None,
        };
        let input_shardings: Vec<Option<Sharding>> = match (&exact_input_shardings, &options.input_shardings) {
            (Some(shardings), _) => shardings.iter().skip(capture_types.len()).cloned().collect(),
            (None, Some(shardings)) => shardings.iter().cloned().map(Some).collect(),
            (None, None) => input_types.iter().map(|r#type| r#type.sharding().cloned()).collect(),
        };

        // A plain trace preserves the types of the arrays that enter it (i.e., `CapturePlacement::Preserve`), so an
        // array that it imports keeps its own type, which names the mesh of the array, and a signature over a mesh of
        // another topology cannot hold it. This is reported with its cause and the way forward, rather than as the
        // disagreement of the meshes that lowering would find.
        if options.capture_types.is_none() {
            let mesh = input_shardings
                .iter()
                .flatten()
                .chain(options.output_shardings.iter().flatten())
                .chain(output_types.iter().filter_map(ArrayType::sharding))
                .map(Sharding::mesh)
                .next();
            if let Some(mesh) = mesh {
                for constant in &session_constants.inputs {
                    let declared_type = constant.r#type();
                    if let Some(sharding) = declared_type.sharding()
                        && !sharding.mesh().has_same_topology(mesh)
                    {
                        return Err(ProgramError::InvalidArgument {
                            message: format!(
                                "the traced program imports an array of type `{declared_type}`, whose mesh does not \
                                 have the topology of the mesh `{mesh}` of the signature of the module, because a \
                                 plain `trace` keeps the type of an array that it imports; stage or compile the \
                                 function for a device mesh over `{mesh}` through a `Session` instead (e.g., with \
                                 `stage` or `compile`), which receives the array on that mesh",
                            ),
                        }
                        .into());
                    }
                }
            }
        }

        let logical_input_shardings = match exact_input_shardings {
            Some(shardings) => shardings,
            None => {
                let capture_shardings: Vec<Option<Sharding>> = if options.capture_types.is_some() {
                    capture_types.iter().map(|r#type| r#type.sharding().cloned()).collect()
                } else {
                    session_constants.inputs.iter().map(|constant| Some(constant.sharding().clone())).collect()
                };
                capture_shardings
                    .iter()
                    .chain(&input_shardings)
                    .map(|sharding| sharding.as_ref().map(Sharding::with_unconstrained_dimensions_replicated))
                    .collect::<Vec<_>>()
            }
        };

        let (output_shardings, open_outputs) = match &options.exact_output_shardings {
            Some(EntryShardingOverride::Values(shardings)) => {
                (shardings.clone(), options.open_outputs.clone().unwrap_or_default())
            }
            Some(EntryShardingOverride::Unannotated) => {
                (vec![None; output_types.len()], options.open_outputs.clone().unwrap_or_default())
            }
            None => {
                let explicit = options.output_shardings.as_deref();
                let sources =
                    program.output_sharding_sources(output_types.len(), |constant| constant.sharding().clone());
                let shardings = entry_output_shardings(&output_types, explicit, &sources, &input_shardings)?;
                let open_outputs = match &options.open_outputs {
                    Some(open_outputs) => open_outputs.clone(),
                    None => entry_outputs_with_compiler_chosen_placement(&output_types, explicit, &sources, &shardings),
                };
                (shardings, open_outputs)
            }
        };

        let function_name = function_name.as_ref();
        let (capture_types, input_types, output_types) =
            (capture_types.as_slice(), input_types.as_slice(), output_types.as_slice());
        let (logical_input_shardings, output_shardings, open_outputs) =
            (logical_input_shardings.as_slice(), output_shardings.as_slice(), open_outputs.as_slice());
        let supports_shardy = options.supports_shardy();
        let ragged_dot_lowering_strategy = options.ragged_dot_lowering_strategy();
        let boundary = program.entry_boundary(
            function_name,
            session_constants,
            capture_types,
            input_types,
            output_types,
            logical_input_shardings,
            output_shardings,
            open_outputs,
            options,
        )?;

        let context = MlirContext::new();
        let location = context.unknown_location();
        let mut module = context.module(location)?;

        // Module-scoped lowering context, shared between the entry function body and private functions below so channel
        // ids stay unique and target, representation policy, named compositions, and provenance state reach nested
        // bodies. The arrays that the program only passes on to calls that never read them follow the session
        // constants in the table, and the entry function binds them to values that require no buffer.
        let lowering_context = ProgramLoweringContext::new()
            .with_target_platform(target_platform)
            .with_ragged_dot_lowering_strategy(ragged_dot_lowering_strategy)
            .with_session_constants(
                boundary.session_constants.inputs.iter().chain(&boundary.session_constants.forwarded),
            );

        // The module declares the mesh that the lowered shardings refer to, which is recorded as they are lowered
        // (refer to `ProgramLoweringContext::require_mesh`). The signature shardings are recorded first, so that the
        // module mesh keeps their axis types.
        for sharding in logical_input_shardings.iter().chain(output_shardings).flatten() {
            lowering_context.require_mesh(sharding.mesh())?;
        }

        // Deduplicate `call` callees that occur more than once into shared private `func.func`s, so repeated nested
        // programs (e.g., identical transformer blocks or the per-block primal and pullback programs of `grad`) lower
        // to one function plus N `func.call`s instead of N inlined copies. The map is empty for modules without
        // repeated calls, in which case every `call` inlines exactly as before.
        let nested_functions = Rc::new(collect_call_functions(program, target_platform)?);

        // Each shared callee is emitted as a private function ahead of the entry function. Its own repeated `call`s
        // lower to calls of the other shared functions, which are referred to by symbol, so the order of emission
        // does not matter.
        let mut module_block = module.body()?;
        let location_ref = location.as_ref();
        for function in nested_functions.functions() {
            let (symbol, program) = (function.symbol(), function.program());
            let functions = Some(&nested_functions);
            let function =
                lower_private_function(symbol, program, functions, &lowering_context, &context, location_ref)?;
            module_block.append_operation(function)?;
        }

        let (function_arguments, function_results) = boundary.lower_attributes(
            logical_input_shardings,
            output_shardings,
            open_outputs,
            supports_shardy,
            target_platform,
            &context,
            location.as_ref(),
        )?;

        module_block.append_operation({
            let argument_types =
                function_arguments.iter().map(|argument| (argument.r#type, location)).collect::<Vec<_>>();
            let function_block = context.block(argument_types.as_slice());
            boundary.lower_body(
                program,
                capture_types.len(),
                logical_input_shardings,
                supports_shardy,
                &function_block,
                &nested_functions,
                &lowering_context,
                &context,
                location.as_ref(),
            )?;
            let mut function_region = context.region();
            function_region.append_block(function_block)?;
            func::func(
                boundary.function_name.as_str(),
                func::FuncAttributes { arguments: function_arguments, results: function_results, ..Default::default() },
                function_region,
                location,
            )?
        })?;

        // The decompositions of the named compositions that lowering emitted are recorded as it requests them (refer
        // to `ProgramLoweringContext::require_named_composition`). Each one is emitted as a private function ahead of
        // the other functions of the module, in the order of the requests. A decomposition is traced independently of
        // the lowered program, so it contains no shared `call`, and lowering it can request further decompositions,
        // which is why they are read by position.
        let mut index = 0;
        while let Some(function) = lowering_context.named_composition(index) {
            let (symbol, program) = (function.symbol(), function.program());
            let function = lower_private_function(symbol, program, None, &lowering_context, &context, location_ref)?;
            module_block.insert_operation(function, index)?;
            index += 1;
        }

        let mut boundary_mapping = boundary.boundary_mapping;

        // Everything that lowering emitted over a mesh refers to the recorded one, which is declared ahead of the
        // functions that refer to it.
        let mesh = lowering_context.mesh();
        if let Some(mesh) = &mesh {
            module_block.insert_operation(mesh.lower(location)?, 0)?;
        }

        // On a single device each sharding holds the whole value, so a target without Shardy can drop its mesh and
        // sharding constraints. If another Shardy operation prevents removal, name it in the lowering diagnostic.
        if !supports_shardy && let Err(error) = module.remove_shardy_operations() {
            let mut unsupported = None;
            module.as_operation()?.walk(WalkOrder::PreOrder, |operation| match operation.name().as_str() {
                Ok(name) if name.starts_with("sdy.") && name != "sdy.sharding_constraint" && name != "sdy.mesh" => {
                    unsupported = Some(name.to_string());
                    WalkResult::Interrupt
                }
                _ => WalkResult::Advance,
            });
            return Err(match unsupported {
                Some(operation) => Error::Lowering(LoweringError::UnsupportedOperation {
                    operation,
                    reason: "the target plugin cannot compile with Shardy".to_string(),
                }),
                None => error.into(),
            });
        }

        module.remove_unused_stable_hlo_constants()?;

        if !module.verify()? {
            return Err(Error::Lowering(LoweringError::VerificationFailure));
        }

        if lowering_context.requires_assertion_handler() {
            boundary_mapping.require_assertion_handler();
        }

        let mut runtime_requirements = lowering_context.runtime_requirements();
        if boundary_mapping.requires_assertion_handler() {
            runtime_requirements.push(RuntimeRequirement::builtin("ryft.assert"));
        }

        if boundary_mapping.has_ordered_io() {
            runtime_requirements.push(RuntimeRequirement::builtin("ryft.ordered-io"));
        }

        let runtime_requirements = normalize_runtime_requirements(runtime_requirements)?.into();

        // The StableHLO text is part of the compilation cache key. Use plain text for modules without provenance.
        // Otherwise, include parsable locations and complete constants so the cached artifact retains both.
        // Programs that differ only in provenance can therefore have distinct cache entries.
        let stable_hlo = if lowering_context.has_provenance() {
            module
                .as_operation()?
                .to_string_with_flags(OperationPrintingFlags {
                    elements_attribute_size_threshold: None,
                    resource_string_size_threshold: None,
                    enable_debug_information: true,
                    pretty_print_debug_information: false,
                    ..OperationPrintingFlags::default()
                })
                .map_err(|error| MlirError::internal(format!("serialized module is not valid UTF-8: {error}")))?
        } else {
            module.to_string()
        };

        Ok(MlirProgram::new(stable_hlo, boundary_mapping, runtime_requirements, mesh))
    }
}

/// Options for lowering a staged [`Program`] with [`ProgramLoweringExtension`]. By default, the entry signature is
/// read from the program. The global types of its inputs and outputs are those of the program, and the arrays that the
/// program imports (i.e., its session constants) enter the entry function as hidden inputs that precede the public
/// inputs. The inputs carry the sharding of each of their global types that has one, with the hidden inputs placed
/// as their arrays are. The outputs carry the placement that the program gives them (i.e., the target of a
/// [`place`](PlaceOperation) that produces them, including over auto mesh axes, or the sharding of the input that they
/// return, also through the callee of a `call`) or otherwise the sharding of their global types. Inputs and outputs
/// replicate the unconstrained dimensions of their shardings, which their runtime placements read as replicated, and
/// outputs whose placement over auto mesh axes nothing fixes are left open over those axes, so that the compiler
/// chooses it. A value without a sharding is left unannotated, so that Shardy propagates its placement from the
/// program.
///
/// The program is pruned before it is lowered, which drops the dead outputs of instructions that carry regions together
/// with the body computation that only they use (e.g., the dead primal outputs of a forward mode `shard_map` whose
/// residuals feed a gradient), and is then simplified. A `shard_map` whose ordinary outputs are all dead but whose body
/// has observable effects (e.g., prints with the [`DeviceOrderedIo`](EffectClass::DeviceOrderedIo) effect) is kept with
/// no ordinary outputs and lowers to a manual computation that only threads its effect token.
#[derive(Clone, Debug)]
pub struct ProgramLoweringOptions {
    /// Types of the hidden capture inputs, or [`None`] for the types of the session constants of the program.
    capture_types: Option<Vec<ArrayType>>,

    /// Global types of the inputs, or [`None`] for those of the program.
    input_types: Option<Vec<ArrayType>>,

    /// Global types of the outputs, or [`None`] for those of the program.
    output_types: Option<Vec<ArrayType>>,

    /// Shardings of the inputs that replace those of their global types.
    input_shardings: Option<Vec<Sharding>>,

    /// Shardings of the outputs that replace those that the program and their global types give them.
    output_shardings: Option<Vec<Sharding>>,

    /// Shardings supplied directly for the program's entry inputs, including captures, in place of derived ones.
    exact_input_shardings: Option<EntryShardingOverride>,

    /// Shardings supplied directly for the program's entry outputs, in place of derived ones.
    exact_output_shardings: Option<EntryShardingOverride>,

    /// Whether the program is pruned and simplified before it is lowered.
    prune: bool,

    /// Platform that the program is lowered for.
    target_platform: Option<String>,

    /// Whether the target supports the Shardy dialect.
    supports_shardy: bool,

    /// External reference state recipes whose aliases the entry boundary records.
    reference_bindings: Vec<ExternalReferenceBinding>,

    /// Public inputs whose buffers the caller gives up to every execution.
    donated_inputs: Vec<bool>,

    /// Public inputs that the program never reads.
    unread_inputs: Vec<bool>,

    /// Whether the entry function exports the runtime ordered I/O token chain as an extra input and output.
    export_effects: bool,

    /// Representation of ragged dots.
    ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,

    /// Outputs that are lowered open over auto mesh axes, or [`None`] to use the output sharding defaults.
    open_outputs: Option<Vec<bool>>,
}

impl ProgramLoweringOptions {
    /// Creates the default [`ProgramLoweringOptions`], which the documentation of this type describes.
    #[inline]
    pub fn new() -> Self {
        Self {
            capture_types: None,
            input_types: None,
            output_types: None,
            input_shardings: None,
            output_shardings: None,
            exact_input_shardings: None,
            exact_output_shardings: None,
            prune: true,
            target_platform: None,
            supports_shardy: true,
            reference_bindings: Vec::new(),
            donated_inputs: Vec::new(),
            unread_inputs: Vec::new(),
            export_effects: false,
            ragged_dot_lowering_strategy: RaggedDotLoweringStrategy::default(),
            open_outputs: None,
        }
    }

    /// Returns these options with `capture_types` as the types of the hidden capture inputs that precede the public
    /// inputs, in the order of the capture table, in place of the types of the session constants of the program. The
    /// types must end with the types of those session constants.
    #[inline]
    pub fn with_capture_types(mut self, capture_types: Vec<ArrayType>) -> Self {
        self.capture_types = Some(capture_types);
        self
    }

    /// Returns these options with `input_types` as the global types of the inputs, in place of those of the program.
    #[inline]
    pub fn with_input_types<Input: Parameterized<ArrayType>>(mut self, input_types: &Input) -> Self {
        self.input_types = Some(input_types.parameters().cloned().collect());
        self
    }

    /// Returns these options with `output_types` as the global types of the outputs, in place of those of the program.
    #[inline]
    pub fn with_output_types<Output: Parameterized<ArrayType>>(mut self, output_types: &Output) -> Self {
        self.output_types = Some(output_types.parameters().cloned().collect());
        self
    }

    /// Returns these options with `input_shardings` as the shardings of the inputs, one per input, in place of those
    /// of their global types. This is what the XLA Single Program Multiple Data (SPMD) partitioner reads to drive
    /// boundary slicing of the buffers of each device, including for shapes whose dimensions are not divisible by
    /// the partition count (e.g., shape `[5]` on 2 partitions producing `[3]` + `[2]`).
    #[inline]
    pub fn with_input_shardings(mut self, input_shardings: Vec<Sharding>) -> Self {
        self.input_shardings = Some(input_shardings);
        self
    }

    /// Returns these options with `output_shardings` as the shardings of the outputs, one per output, in place of
    /// those that the program and their global types give them.
    #[inline]
    pub fn with_output_shardings(mut self, output_shardings: Vec<Sharding>) -> Self {
        self.output_shardings = Some(output_shardings);
        self
    }

    /// Returns these options with `input_shardings` attached directly to the program's entry inputs, in place of the
    /// shardings that the default options derive. Supply one optional sharding per input, with the hidden capture
    /// inputs first. An input without a sharding is left unannotated.
    #[inline]
    pub fn with_exact_input_shardings(mut self, input_shardings: Vec<Option<Sharding>>) -> Self {
        self.exact_input_shardings = Some(EntryShardingOverride::Values(input_shardings));
        self
    }

    /// Returns these options with `output_shardings` attached directly to the program's entry outputs, in place of the
    /// shardings that the default options derive. Supply one optional sharding per output. An output without a sharding
    /// is left unannotated. Outputs are closed over auto mesh axes by default. Use [`Self::with_open_outputs`] to
    /// choose which annotated outputs to leave open.
    #[inline]
    pub fn with_exact_output_shardings(mut self, output_shardings: Vec<Option<Sharding>>) -> Self {
        self.exact_output_shardings = Some(EntryShardingOverride::Values(output_shardings));
        self
    }

    /// Returns these options with no sharding attached to any input or output of the entry function, so that Shardy
    /// propagates every placement from the program.
    #[inline]
    pub fn without_boundary_shardings(mut self) -> Self {
        self.exact_input_shardings = Some(EntryShardingOverride::Unannotated);
        self.exact_output_shardings = Some(EntryShardingOverride::Unannotated);
        self
    }

    /// Returns these options with the program lowered as it is, without pruning or simplifying it first.
    #[inline]
    pub fn without_pruning(mut self) -> Self {
        self.prune = false;
        self
    }

    /// Returns these options with the module lowered for `target_platform`.
    ///
    /// Use the case-sensitive PJRT name returned by [`ryft_pjrt::Client::platform_name`] (e.g., `"cpu"`, `"cuda"`, or
    /// `"rocm"`). [`None`] uses the default lowering rules without selecting a platform. The name selects
    /// platform-specific lowering rules. For lowering that also takes a client's detected capabilities into account,
    /// use a [`Session`](crate::Session).
    #[inline]
    pub fn with_target_platform(mut self, target_platform: Option<&str>) -> Self {
        self.target_platform = target_platform.map(str::to_string);
        self
    }

    /// Returns these options with `supports_shardy` as whether the target supports the Shardy dialect.
    /// Lowering for a target without it removes the Shardy operations from the module.
    #[inline]
    pub(crate) fn with_shardy_support(mut self, supports_shardy: bool) -> Self {
        self.supports_shardy = supports_shardy;
        self
    }

    /// Returns these options with the external reference state aliases of a discharged program recorded on its
    /// entry boundary, and with the entry function exporting its ordered I/O token chain.
    ///
    /// `reference_bindings` are the external state recipes of the discharge artifact, and each must name a distinct
    /// logical state input. Each state input must survive the executable boundary. A mutated state must be a static
    /// device array and name a hidden output of the identical type and sharding. Mutated pairs are recorded as
    /// `tf.aliasing_output` aliases between inputs and outputs, which are non-semantic hints that the buffers may alias
    /// and that merely permit backend buffer reuse. The runtime never donates reference state inputs, so an alias never
    /// authorizes in-place mutation of the state buffer of the caller. Read-only state may be finite and bounded
    /// dynamic because it has no hidden alias. Lowering validates the executable buffer alias contract, and the XLA
    /// session checks the supported memory, sharding, and dynamic shape classes beforehand.
    ///
    /// `donated_inputs` flags the public inputs whose buffers the caller gives up to every execution (i.e., one flag
    /// per public input, or none at all). Each donated input is paired with an output that it can share its buffers
    /// with and the pair is recorded as the same kind of alias, as JAX does for the donated arguments of a `jit`
    /// function. An output can share the buffers of an input when both are static device arrays of the identical type
    /// and sharding, or, for an input without such an output, when both have shardings under which every device holds
    /// shards of the same type (e.g., a donated input with a pending sum and an output that completes it, as in the
    /// compiled reshard of a donated array), and in both cases only when the output is not open on a mesh of more than
    /// one device, because the compiler could then place it differently than the input. Without such an alias the
    /// outputs of every execution are written to new buffers, which is correct but forces backends that cache launch
    /// state per buffer address (e.g., the command buffers of the XLA GPU backend) to rebuild that state for every
    /// execution of a program that updates its inputs in a loop.
    ///
    /// `unread_inputs` flags the public inputs that the program never reads (i.e., one flag per public input, or none
    /// at all), which the caller states because it cannot be read from `program` (e.g., an input of the program of an
    /// eager call that the call passes as a capture which its callee never reads). Such an input with a static shape
    /// has no executable input slot, like a capture that no emitted region reads.
    #[inline]
    pub(crate) fn with_reference_bindings(
        mut self,
        reference_bindings: Vec<ExternalReferenceBinding>,
        donated_inputs: Vec<bool>,
        unread_inputs: Vec<bool>,
    ) -> Self {
        self.reference_bindings = reference_bindings;
        self.donated_inputs = donated_inputs;
        self.unread_inputs = unread_inputs;
        self.export_effects = true;
        self
    }

    /// Returns these options with `strategy` as the representation of ragged dots.
    #[inline]
    pub fn with_ragged_dot_lowering_strategy(mut self, strategy: RaggedDotLoweringStrategy) -> Self {
        self.ragged_dot_lowering_strategy = strategy;
        self
    }

    /// Returns these options with the annotated outputs flagged in `open_outputs` lowered open over auto mesh axes,
    /// so that the compiler chooses their placement over those axes. Supply one flag per logical output in output
    /// order, or an empty vector to close every annotated output. Lowering returns
    /// [`CompilationError::InvalidOptions`] for any other length.
    #[inline]
    pub fn with_open_outputs(mut self, open_outputs: Vec<bool>) -> Self {
        self.open_outputs = Some(open_outputs);
        self
    }

    /// Returns the hidden capture input type overrides, or [`None`] to use the types of the session constants.
    #[inline]
    pub fn capture_types(&self) -> Option<&[ArrayType]> {
        self.capture_types.as_deref()
    }

    /// Returns the global input type overrides, or [`None`] to use the program's input types.
    #[inline]
    pub fn input_types(&self) -> Option<&[ArrayType]> {
        self.input_types.as_deref()
    }

    /// Returns the global output type overrides, or [`None`] to use the program's output types.
    #[inline]
    pub fn output_types(&self) -> Option<&[ArrayType]> {
        self.output_types.as_deref()
    }

    /// Returns the public input sharding overrides used to derive entry input annotations, if supplied.
    #[inline]
    pub fn input_shardings(&self) -> Option<&[Sharding]> {
        self.input_shardings.as_deref()
    }

    /// Returns the output sharding overrides used to derive entry output annotations, if supplied.
    #[inline]
    pub fn output_shardings(&self) -> Option<&[Sharding]> {
        self.output_shardings.as_deref()
    }

    /// Returns the exact entry input sharding override, if supplied, including any hidden captures first.
    #[inline]
    pub fn exact_input_shardings(&self) -> Option<&EntryShardingOverride> {
        self.exact_input_shardings.as_ref()
    }

    /// Returns the exact entry output sharding override, if supplied.
    #[inline]
    pub fn exact_output_shardings(&self) -> Option<&EntryShardingOverride> {
        self.exact_output_shardings.as_ref()
    }

    /// Returns whether lowering prunes and simplifies the program first.
    #[inline]
    pub fn prunes(&self) -> bool {
        self.prune
    }

    /// Returns the case-sensitive PJRT platform name used for lowering, if supplied.
    #[inline]
    pub fn target_platform(&self) -> Option<&str> {
        self.target_platform.as_deref()
    }

    /// Returns whether the target supports the Shardy dialect.
    #[inline]
    pub(crate) fn supports_shardy(&self) -> bool {
        self.supports_shardy
    }

    /// Returns the external reference state recipes whose aliases the entry boundary records.
    #[inline]
    pub(crate) fn reference_bindings(&self) -> &[ExternalReferenceBinding] {
        &self.reference_bindings
    }

    /// Returns the flags of the public inputs whose buffers the caller gives up to every execution.
    #[inline]
    pub(crate) fn donated_inputs(&self) -> &[bool] {
        &self.donated_inputs
    }

    /// Returns the flags of the public inputs that the program never reads.
    #[inline]
    pub(crate) fn unread_inputs(&self) -> &[bool] {
        &self.unread_inputs
    }

    /// Returns whether the entry function exports the runtime ordered I/O token chain as an extra input and output.
    #[inline]
    pub(crate) fn exports_effects(&self) -> bool {
        self.export_effects
    }

    /// Returns the representation of ragged dots.
    #[inline]
    pub fn ragged_dot_lowering_strategy(&self) -> RaggedDotLoweringStrategy {
        self.ragged_dot_lowering_strategy
    }

    /// Returns the output flags for openness over auto mesh axes, or [`None`] to use the output sharding defaults.
    /// An empty slice closes every annotated output.
    #[inline]
    pub fn open_outputs(&self) -> Option<&[bool]> {
        self.open_outputs.as_deref()
    }
}

impl Default for ProgramLoweringOptions {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Exact entry boundary sharding overrides reported by [`ProgramLoweringOptions`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntryShardingOverride {
    /// Leave every value unannotated so that Shardy propagates placements from the program.
    Unannotated,

    /// Attach the supplied shardings in signature order and leave values with [`None`] unannotated.
    Values(Vec<Option<Sharding>>),
}

/// Boundary of the entry function of a lowered module, which [`ProgramLoweringAnalysisExtension::entry_boundary`]
/// derives from a program before anything is emitted.
pub(crate) struct ProgramBoundary {
    /// Name of the entry function.
    function_name: String,

    /// Mapping from the program's logical inputs and outputs to executable slots.
    boundary_mapping: MlirProgramBoundaryMapping,

    /// Session constants bound by the entry function.
    session_constants: ProgramSessionConstants,

    /// Types of the logical inputs (i.e., of the captures followed by the public inputs).
    logical_input_types: Vec<ArrayType>,

    /// Types of the executable's array arguments, including hidden extent scalars.
    executable_input_types: Vec<ArrayType>,

    /// Types of the executable's array results, including hidden extent scalars.
    executable_output_types: Vec<ArrayType>,

    /// Maps each executable input index to the output index with which it may share buffers.
    aliases: HashMap<usize, usize>,
}

impl ProgramBoundary {
    /// Returns a map from executable input indices to the output indices with which they may share buffers. The aliases
    /// of `reference_states` are validated and paired first. Each donated input then gets an output that no earlier
    /// input is paired with. The type and sharding slices are in logical boundary order, with the first
    /// `capture_count` input entries describing captures.
    ///
    /// # Errors
    ///
    /// Returns [`LoweringError::InvalidReferenceStateAbi`] for a reference state whose input and output cannot alias,
    /// and [`ExecutionError::DonationFlagCountMismatch`] when `donated_inputs` does not have one flag for each public
    /// input.
    #[allow(clippy::too_many_arguments)]
    fn aliases(
        reference_states: &[ExternalReferenceBinding],
        donated_inputs: &[bool],
        open_outputs: &[bool],
        capture_count: usize,
        boundary_mapping: &MlirProgramBoundaryMapping,
        input_types: &[ArrayType],
        output_types: &[ArrayType],
        input_shardings: &[Option<Sharding>],
        output_shardings: &[Option<Sharding>],
    ) -> Result<HashMap<usize, usize>, Error> {
        let mut aliases = HashMap::with_capacity(reference_states.len());
        let mut state_inputs = HashSet::with_capacity(reference_states.len());
        let mut aliased_outputs = HashSet::with_capacity(reference_states.len());
        for state in reference_states {
            let logical_input_index = state.source().flat_input_index(capture_count)?;
            if !state_inputs.insert(logical_input_index) {
                return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: format!("logical state input {logical_input_index} appears more than once"),
                }));
            }

            let input_type = input_types.get(logical_input_index).ok_or_else(|| {
                Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: format!("logical input index {logical_input_index} is out of range"),
                })
            })?;

            let executable_input_index = boundary_mapping.input_mapping()[logical_input_index].ok_or_else(|| {
                Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                    message: format!(
                        "logical state input {logical_input_index} is erased from the executable boundary"
                    ),
                })
            })?;

            if let Some(logical_output_index) = state.output_index() {
                let output_type = output_types.get(logical_output_index).ok_or_else(|| {
                    Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                        message: format!("logical output index {logical_output_index} is out of range"),
                    })
                })?;

                let executable_output_index =
                    boundary_mapping.output_mapping()[logical_output_index].ok_or_else(|| {
                        Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                            message: format!(
                                "logical state output {logical_output_index} is erased from the executable boundary",
                            ),
                        })
                    })?;

                if input_type.static_shape().is_none() || output_type.static_shape().is_none() {
                    return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                        message: format!(
                            "state input {logical_input_index} and output {logical_output_index} must be static \
                             because bounded-dynamic mutation alias compatibility is unsupported",
                        ),
                    }));
                }

                if input_type.memory() != Memory::Device || output_type.memory() != Memory::Device {
                    return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                        message: format!(
                            "state input {logical_input_index} and output {logical_output_index} must use device \
                             memory",
                        ),
                    }));
                }

                if input_type != output_type {
                    return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                        message: format!(
                            "state input {logical_input_index} type `{input_type}` is incompatible with output \
                             {logical_output_index} type `{output_type}`",
                        ),
                    }));
                }

                match (&input_shardings[logical_input_index], &output_shardings[logical_output_index]) {
                    (Some(input_sharding), Some(output_sharding)) if input_sharding != output_sharding => {
                        return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                            message: format!(
                                "state input {logical_input_index} and output {logical_output_index} \
                                 must use the same sharding",
                            ),
                        }));
                    }
                    (Some(_), None) | (None, Some(_)) => {
                        return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                            message:
                                "reference-state aliases require both input and output sharding metadata or neither"
                                    .to_string(),
                        }));
                    }
                    _ => {}
                }

                if !aliased_outputs.insert(executable_output_index) {
                    return Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi {
                        message: "reference-state aliases must be injective".to_string(),
                    }));
                }

                // Distinct logical inputs, which `state_inputs` enforces above, map to distinct executable slots,
                // so this insertion can never displace an earlier alias.
                aliases.insert(executable_input_index, executable_output_index);
            }
        }

        if !donated_inputs.is_empty() && donated_inputs.len() != input_types.len() - capture_count {
            return Err(Error::Execution(ExecutionError::DonationFlagCountMismatch {
                expected_count: input_types.len() - capture_count,
                actual_count: donated_inputs.len(),
            }));
        }

        // Each donated input is paired with the first output that can share its buffers and that no earlier input was
        // paired with. The pairs are the same hints as the reference state aliases above, so an input that finds no
        // output simply stays unpaired. Every donated input is first paired with an output of its own type and
        // sharding, and only an input that finds none is then paired with an output whose shards have the types
        // of its shards, so that the second pass only adds pairs to inputs that the first one leaves unpaired.
        for same_sharding in [true, false] {
            for (input_index, _) in donated_inputs.iter().enumerate().filter(|(_, donated)| **donated) {
                let logical_input_index = capture_count + input_index;
                let input_type = &input_types[logical_input_index];
                let Some(executable_input_index) = boundary_mapping.input_mapping()[logical_input_index] else {
                    continue;
                };

                if state_inputs.contains(&logical_input_index)
                    || aliases.contains_key(&executable_input_index)
                    || input_type.static_shape().is_none()
                    || input_type.memory() != Memory::Device
                {
                    continue;
                }

                let input_sharding = &input_shardings[logical_input_index];

                // The compiler places an open output itself, and only on a single device is that placement certain to
                // be the placement of the input.
                let has_single_device = input_sharding.as_ref().is_none_or(|sharding| {
                    sharding.mesh().axes().iter().map(|axis| axis.size()).product::<usize>() == 1
                });

                let executable_output_index =
                    output_types.iter().enumerate().find_map(|(logical_output_index, output_type)| {
                        let executable_output_index = boundary_mapping.output_mapping()[logical_output_index]?;
                        let is_open = open_outputs.get(logical_output_index).copied().unwrap_or(false);
                        let output_sharding = &output_shardings[logical_output_index];
                        let shares_buffers = match same_sharding {
                            true => output_type == input_type && output_sharding == input_sharding,
                            false => {
                                // Every device must hold shards of the same type of the input and of the output,
                                // although the two shardings may differ (e.g., when the output completes a pending sum
                                // of the input along a mesh axis, or splits a dimension along another axis of the same
                                // size). The output can then take the buffers of the donated input on every device.
                                // That is the case when both shardings are on the same mesh, the types are identical
                                // apart from their shardings (i.e., in data type, shape, layout, and memory), and both
                                // shardings split each dimension into shards of the same static extent.
                                let input_sharding = input_sharding.as_ref()?;
                                let output_sharding = output_sharding.as_ref()?;
                                let shape = input_type.static_shape()?;

                                // The extent of the shards of each dimension, or `None` when a dimension is
                                // unconstrained, names an axis that the mesh does not have, or is not split evenly.
                                let shard_extents = |sharding: &Sharding| {
                                    if sharding.rank() != shape.rank() {
                                        return None;
                                    }

                                    shape
                                        .dimensions()
                                        .iter()
                                        .zip(sharding.dimensions())
                                        .map(|(&extent, dimension)| match dimension {
                                            ShardingDimension::Replicated => Some(extent),
                                            ShardingDimension::Sharded(axes) => {
                                                let count = axes
                                                    .iter()
                                                    .map(|axis| sharding.mesh().axis_size(axis))
                                                    .product::<Option<usize>>()?;
                                                (count != 0 && extent % count == 0).then(|| extent / count)
                                            }
                                            ShardingDimension::Unconstrained => None,
                                        })
                                        .collect::<Option<Vec<_>>>()
                                };

                                let without_sharding = |r#type: &ArrayType| r#type.clone().with_sharding(None).ok();
                                input_sharding.mesh() == output_sharding.mesh()
                                    && without_sharding(input_type) == without_sharding(output_type)
                                    && shard_extents(input_sharding)
                                        .is_some_and(|extents| Some(extents) == shard_extents(output_sharding))
                            }
                        };

                        (shares_buffers
                            && (has_single_device || !is_open)
                            && !aliased_outputs.contains(&executable_output_index))
                        .then_some(executable_output_index)
                    });

                if let Some(executable_output_index) = executable_output_index {
                    aliased_outputs.insert(executable_output_index);
                    aliases.insert(executable_input_index, executable_output_index);
                }
            }
        }

        Ok(aliases)
    }

    /// Returns the MLIR types and attributes of the entry function arguments and results in executable order.
    /// `input_shardings` and `output_shardings` follow logical boundary order, with captures first on the input side.
    /// The attributes declare the sharding, memory kind, layout, and aliased output of each value.
    ///
    /// # Errors
    ///
    /// Returns [`LoweringError::UnsupportedBoundaryMemory`] for a boundary memory that `target_platform` cannot use.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn lower_attributes<'c, 't>(
        &self,
        input_shardings: &[Option<Sharding>],
        output_shardings: &[Option<Sharding>],
        open_outputs: &[bool],
        supports_shardy: bool,
        target_platform: Option<&str>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Result<(Vec<TypeAndAttributes<'c, 't, 'static>>, Vec<TypeAndAttributes<'c, 't, 'static>>), Error> {
        let Self { boundary_mapping, executable_input_types, executable_output_types, aliases, .. } = self;
        let mut argument_types = executable_input_types
            .iter()
            .map(|array_type| array_type.lower(context, location).map(|r#type| r#type.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;

        if boundary_mapping.has_ordered_io() {
            argument_types.push(context.stable_hlo_token_type()?.as_ref());
        }

        let mut result_types = executable_output_types
            .iter()
            .map(|array_type| array_type.lower(context, location).map(|r#type| r#type.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;

        if boundary_mapping.has_ordered_io() {
            result_types.push(context.stable_hlo_token_type()?.as_ref());
        }

        let mut argument_sharding_attributes = boundary_mapping
            .executable_optional_input_shardings(input_shardings)?
            .iter()
            .map(|sharding| {
                sharding.as_ref().map(|sharding| sharding.lower(OpenDimensions::None, location)).transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Results whose placement over auto mesh axes the compiler chooses are open over those axes.
        let executable_open_outputs =
            if open_outputs.is_empty() { Vec::new() } else { boundary_mapping.select_output_entries(open_outputs)? };
        let mut result_sharding_attributes = boundary_mapping
            .executable_optional_output_shardings(output_shardings)?
            .iter()
            .enumerate()
            .map(|(index, sharding)| {
                let open = executable_open_outputs.get(index).copied().unwrap_or(false);
                let open_dimensions = if open { OpenDimensions::All } else { OpenDimensions::None };
                sharding.as_ref().map(|sharding| sharding.lower(open_dimensions, location)).transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;

        // The effect token of a signature side is annotated when any value of that side is, as each device threads its
        // own token.
        if boundary_mapping.has_ordered_io() {
            let token_sharding = context.shardy_tensor_sharding(
                context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME),
                &[],
                &[],
                &[],
                shardy::ReductionOperation::Sum,
            )?;

            for shardings in [&mut argument_sharding_attributes, &mut result_sharding_attributes] {
                if shardings.iter().any(Option::is_some) {
                    shardings.push(Some(token_sharding));
                }
            }
        }

        // Internal layout constraints do not constrain the executable ABI. Declare layouts at the executable boundary,
        // whose types already omit static zeros and include hidden extent scalars, while tokens have no array type.
        // Declare placements for array slots, including captures. The appended effect token carries only ordering.
        // Default device placement needs no attribute, but host identities still need attributes even without
        // transfers.
        let function_arguments = argument_types
            .iter()
            .enumerate()
            .map(|(index, tensor_type)| {
                let mut attributes = HashMap::new();
                if supports_shardy
                    && let Some(sharding) = argument_sharding_attributes.get(index).and_then(Option::as_ref)
                {
                    attributes.insert("sdy.sharding".into(), sharding.as_ref());
                }

                if let Some(r#type) =
                    executable_input_types.get(index).filter(|r#type| r#type.memory() != Memory::Device)
                {
                    validate_boundary_memory(r#type.memory(), target_platform)?;
                    attributes.insert(
                        "mhlo.memory_kind".into(),
                        context.string_attribute(r#type.memory().placement_kind()).as_ref(),
                    );
                }

                if let Some(output_index) = aliases.get(&index) {
                    attributes.insert(
                        "tf.aliasing_output".into(),
                        context.integer_attribute(context.signless_integer_type(64), *output_index as i64).as_ref(),
                    );
                }

                if let Some(array_type) = executable_input_types.get(index)
                    && let Some(layout) = array_type.lower_boundary_layout("input", context)?
                {
                    attributes.insert("mhlo.layout_mode".into(), layout);
                }

                let attributes = (!attributes.is_empty()).then_some(attributes);
                Ok::<_, Error>(TypeAndAttributes { r#type: tensor_type.as_ref(), attributes })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let function_results = result_types
            .iter()
            .enumerate()
            .map(|(index, tensor_type)| {
                let mut attributes = HashMap::new();
                if supports_shardy
                    && let Some(sharding) = result_sharding_attributes.get(index).and_then(Option::as_ref)
                {
                    attributes.insert("sdy.sharding".into(), sharding.as_ref());
                }

                if let Some(r#type) =
                    executable_output_types.get(index).filter(|r#type| r#type.memory() != Memory::Device)
                {
                    validate_boundary_memory(r#type.memory(), target_platform)?;
                    attributes.insert(
                        "mhlo.memory_kind".into(),
                        context.string_attribute(r#type.memory().placement_kind()).as_ref(),
                    );
                }

                if let Some(array_type) = executable_output_types.get(index)
                    && let Some(layout) = array_type.lower_boundary_layout("output", context)?
                {
                    attributes.insert("mhlo.layout_mode".into(), layout);
                }

                let attributes = (!attributes.is_empty()).then_some(attributes);
                Ok::<_, Error>(TypeAndAttributes { r#type: tensor_type.as_ref(), attributes })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok((function_arguments, function_results))
    }

    /// Lowers the entry function body into `function_block`. It reconstructs the logical inputs from the executable
    /// arguments, completes pending input reductions, lowers `program`, and emits `func.return` with the executable
    /// results. `input_shardings` follows logical input order, with the `capture_count` captures first.
    #[allow(clippy::too_many_arguments)]
    fn lower_body<'b, 'c: 'b, 't: 'c, ProgramInput, ProgramOutput, O: ProgramLoweringOperation>(
        &self,
        program: &Program<Constant, O, ProgramInput, ProgramOutput>,
        capture_count: usize,
        input_shardings: &[Option<Sharding>],
        supports_shardy: bool,
        function_block: &'b DetachedBlock<'c, 't>,
        nested_functions: &Rc<CallFunctionMap<O>>,
        lowering_context: &ProgramLoweringContext,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Result<(), Error>
    where
        ProgramInput: Parameterized<Constant>,
        ProgramOutput: Parameterized<Constant>,
    {
        let Self { boundary_mapping, session_constants, logical_input_types, executable_input_types, .. } = self;

        let session_count = session_constants.inputs.len();
        let mut function_block_ref = function_block.as_ref();
        let mut logical_input_values = Vec::with_capacity(logical_input_types.len());
        for (logical_index, array_type) in logical_input_types.iter().enumerate() {
            let value = match boundary_mapping.input_mapping()[logical_index] {
                Some(executable_index) => {
                    let mut value = function_block.argument(executable_index).unwrap().as_ref();
                    let mut refined_type = executable_input_types[executable_index].clone();
                    for input_dimension in boundary_mapping
                        .input_dimensions()
                        .iter()
                        .filter(|dimension| dimension.logical_input_index() == logical_index)
                    {
                        let mut dimensions = refined_type.shape().dimensions().to_vec();
                        dimensions[input_dimension.axis()] =
                            array_type.shape().dimensions()[input_dimension.axis()].clone();
                        refined_type = refined_type.with_shape(Shape::new(dimensions));
                        let size = function_block.argument(input_dimension.executable_input_index()).unwrap().as_ref();
                        let operation = function_block_ref.append_operation(stable_hlo::set_dimension_size(
                            value,
                            size,
                            refined_type.lower(context, location)?,
                            input_dimension.axis(),
                            location,
                        )?)?;
                        value = operation.result(0).unwrap().as_ref();
                    }
                    value
                }
                None => {
                    let value = lower_unplaced_constant_output(
                        std::slice::from_ref(array_type),
                        0,
                        &mut function_block_ref,
                        context,
                        location.as_ref(),
                    )?
                    .into_iter()
                    .next()
                    .unwrap();

                    // An unread input (refer to `unread_logical_inputs` above) is a placeholder that no emitted region
                    // reads and that XLA removes, so its sharding is not applied either.
                    if boundary_mapping.is_unread_input(logical_index) {
                        logical_input_values.push(value);
                        continue;
                    }
                    value
                }
            };

            debug_assert_eq!(value.r#type()?, array_type.lower(context, location)?.as_ref());

            // An input that arrives with a pending reduction that its type does not record (i.e., over auto mesh axes,
            // which an input sharding override states and types never record) is reduced where it enters, because no
            // value of the program can carry that reduction. The sharding constraint is the transition that Shardy
            // accepts, and it completes the reduction (e.g., in the reshard that `Array::to_placement` compiles from
            // such a source onto a sharding without it). An input whose type records a pending reduction and that
            // arrives split along auto mesh axes, which its type cannot record, is constrained to its sharding where
            // it enters as well, so that the constraints that carry its pending reduction through the program keep
            // that split.
            let value = match &input_shardings[logical_index] {
                Some(sharding)
                    if supports_shardy
                        && (sharding
                            .unreduced_axes()
                            .iter()
                            .any(|axis| !array_type.unreduced_axes().contains(axis))
                            || !array_type.unreduced_axes().is_empty()
                                && array_type.sharding().is_some_and(|stated| {
                                    stated.dimensions_with_unconstrained_replicated()
                                        != sharding.dimensions_with_unconstrained_replicated()
                                })) =>
                {
                    // The constraint completes the pending reduction along the axes that the type does not record, with
                    // a collective that must make that completion for the data type of the input. The whole transition
                    // is checked as well, because the axes that the type records stay pending.
                    let unrecorded = |axis: &&String| !array_type.unreduced_axes().contains(*axis);
                    let (completed, kept): (Vec<&String>, Vec<&String>) =
                        sharding.unreduced_axes().iter().partition(unrecorded);
                    let kind = sharding.unreduced_kind();
                    let change = ReductionChange::Complete(kind);
                    let data_type = array_type.data_type();
                    let axes = completed.iter().copied();

                    validate_pending_reduction_change(data_type, change, sharding, axes, "input_shardings")?;
                    validate_partial_completion(data_type, kind, sharding, completed, kept, "input_shardings")?;

                    let reduced = sharding.clone().with_unreduced_axes(array_type.unreduced_axes().iter().cloned())?;
                    let block = &mut function_block_ref;

                    lowering_context.lower_pending_reduction_constraint(&reduced, value, block, location.as_ref())?
                }
                _ => value,
            };

            logical_input_values.push(value);
        }

        let (capture_values, public_input_values) = logical_input_values.split_at(capture_count);

        // Ordinary module callers lower an unlifted program whose entry inputs are only the public inputs. The
        // compilation domain instead supplies the program with lifted captures, so that core reference discharge and
        // lowering observe the same arena. In that form the complete logical input list is also the entry input list,
        // while the leading prefix still serves attached `CaptureReference` constants. No third boundary form exists,
        // so the entry input count must match one of those two shapes exactly. Session constants are not program
        // inputs, so the lifted form omits the trailing session constants of the capture prefix from the entry inputs,
        // while `capture_values` keeps them as its suffix for `ProgramLoweringContext::lower_session_constant`.
        debug_assert!(
            program.input_count() + session_count == logical_input_values.len()
                || program.input_count() == public_input_values.len(),
        );

        let lifted_input_values: Vec<_>;
        let input_values = if program.input_count() + session_count == logical_input_values.len() {
            lifted_input_values = capture_values[..capture_values.len() - session_count]
                .iter()
                .chain(public_input_values)
                .copied()
                .collect();
            lifted_input_values.as_slice()
        } else {
            public_input_values
        };

        // The arrays that the program only passes on to calls that never read them are bound like unread inputs,
        // after the session constants.
        let mut captured_values = Cow::Borrowed(capture_values);
        for constant in &session_constants.forwarded {
            let r#type = constant.r#type();
            let values = lower_unplaced_constant_output(
                std::slice::from_ref(r#type.as_ref()),
                0,
                &mut function_block_ref,
                context,
                location.as_ref(),
            )?;
            captured_values.to_mut().extend(values);
        }

        let mut effect_tokens = EffectTokens::default();
        if let Some(index) = boundary_mapping.ordered_io_input_index() {
            effect_tokens.set(EffectClass::OrderedIo, function_block.argument(index).unwrap().as_ref());
        }

        let logical_outputs = program.lower_inline(
            &captured_values,
            input_values,
            Some(nested_functions),
            lowering_context,
            &mut effect_tokens,
            function_block_ref,
            location.as_ref(),
        )?;

        let mut executable_outputs = boundary_mapping.select_output_entries(logical_outputs.as_slice())?;

        // XLA writes the runtime-size trailer of a bounded dynamic result only in the `SliceToDynamic` that
        // `DynamicPadder` inserts before each output leaf. When several such leaves carry values that XLA proves
        // equal (e.g., the same SSA value or equal computations merged by CSE), one `SliceToDynamic` survives and copy
        // insertion fills the other result with a plain copy of the bounded payload, which leaves its trailer unset
        // (i.e., the buffer reports zero-sized dynamic dimensions). Routing these results through one optimization
        // barrier makes every leaf a distinct barrier result. XLA expands barriers only after copy insertion, so each
        // leaf keeps its own `SliceToDynamic`. A single dynamic result cannot alias a static one, so the barrier is
        // only needed with two or more of them.
        let mut dynamic_output_slots = boundary_mapping
            .output_dimensions()
            .iter()
            .map(|output_dimension| boundary_mapping.output_mapping()[output_dimension.logical_output_index()].unwrap())
            .collect::<Vec<_>>();
        dynamic_output_slots.dedup();
        if dynamic_output_slots.len() > 1 {
            let barrier_inputs =
                dynamic_output_slots.iter().map(|index| executable_outputs[*index]).collect::<Vec<_>>();
            let barrier = function_block_ref
                .append_operation(stable_hlo::optimization_barrier(barrier_inputs.as_slice(), location)?)?;
            for (result_index, executable_index) in dynamic_output_slots.into_iter().enumerate() {
                executable_outputs[executable_index] = barrier.result(result_index).unwrap().as_ref();
            }
        }

        for output_dimension in boundary_mapping.output_dimensions() {
            executable_outputs.push(lower_runtime_dimension_size_i64(
                logical_outputs[output_dimension.logical_output_index()],
                output_dimension.axis(),
                &mut function_block_ref,
                context,
                location.as_ref(),
            )?);
        }

        if boundary_mapping.has_ordered_io() {
            executable_outputs.push(effect_tokens.get(EffectClass::OrderedIo).unwrap());
        }

        function_block_ref.append_operation(func::r#return(executable_outputs.as_slice(), location)?)?;

        Ok(())
    }
}

/// Distinct session constants bound by a lowered program's entry function. Each group preserves the order in which
/// the constants first appear among the emitted region atoms, so every process assigns the same input slots.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ProgramSessionConstants {
    /// Arrays supplied as hidden entry inputs after the lifted captures.
    inputs: Vec<RetainedArray>,

    /// Arrays passed only to unread callee captures. Lowering binds them to values that require no buffer.
    forwarded: Vec<RetainedArray>,
}

impl ProgramSessionConstants {
    /// Classifies the session constants of the supplied regions on `target_platform`. Callers select the regions with
    /// [`RegionLoweringExtension::emitted_regions`], entering callees for the whole program or skipping them for eager
    /// placement within a single capture scope. A live read anywhere makes a constant an input. A constant with no
    /// reads also remains an input when the caller lowers an unpruned program.
    pub(crate) fn from_regions<'r, O: 'r + ProgramLoweringOperation>(
        regions: impl IntoIterator<Item = RegionRef<'r, Constant, O>>,
        target_platform: Option<&str>,
    ) -> Self {
        /// Strongest use observed across all occurrences of a retained array.
        enum Usage {
            Unused,
            Forwarded,
            Read,
        }

        let mut indices = HashMap::new();
        let mut constants = Vec::new();
        for region in regions {
            let mut has_session_constants = false;
            for atom in region.atoms() {
                if let Atom::Constant(Constant::Session(constant)) = atom {
                    has_session_constants = true;
                    indices.entry(constant.identity_key()).or_insert_with(|| {
                        let index = constants.len();
                        constants.push((constant, Usage::Unused));
                        index
                    });
                }
            }

            if has_session_constants {
                region.visit_atom_reads(target_platform, |id, is_live| {
                    if let Some(Atom::Constant(Constant::Session(constant))) = region.atoms().get(id.index()) {
                        let usage = &mut constants[indices[&constant.identity_key()]].1;
                        if is_live {
                            *usage = Usage::Read;
                        } else if matches!(usage, Usage::Unused) {
                            *usage = Usage::Forwarded;
                        }
                    }
                });
            }
        }

        let mut inputs = Vec::new();
        let mut forwarded = Vec::new();
        for (constant, usage) in constants {
            match usage {
                Usage::Unused | Usage::Read => inputs.push(constant.clone()),
                Usage::Forwarded => forwarded.push(constant.clone()),
            }
        }

        Self { inputs, forwarded }
    }

    /// Returns the arrays to supply as hidden inputs, in first-occurrence order.
    #[inline]
    pub(crate) fn into_inputs(self) -> Vec<RetainedArray> {
        self.inputs
    }
}

/// Prepares XLA programs for lowering and analyzes their input use, effects, references, and placements.
pub(crate) trait ProgramLoweringAnalysisExtension: Sized {
    /// Operation family of the program.
    type Operation: ProgramLoweringOperation;

    /// Returns this program as lowering emits it, without the unused outputs of its instructions that carry regions,
    /// together with the computations that only those outputs use (i.e., like [`Program::into_pruned`]), and without
    /// every other instruction that neither an output nor an observable effect needs (i.e., like
    /// [`Program::into_simplified`]). Every entry point that lowers a program (i.e., the compilation of a
    /// [`Session`](crate::Session) and the StableHLO export of traced programs) lowers this program, and
    /// [`RegionPlacementRequests::requests`] analyzes it, so that the validation of placements, the lowering itself,
    /// commitment, and the devices that a program names all see the same instructions. Whether a call still names
    /// devices is read for `target_platform`, so a callee that names devices only inside a body that lowering for
    /// that platform replaces names none.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Program::into_pruned`] and [`Program::into_simplified`].
    fn pruned_for_lowering(&self, target_platform: Option<&str>) -> Result<Self, ProgramError>;

    /// Returns whether an instruction in a region that lowering emits declares ordered state. XLA cannot represent
    /// state in any such region, including state omitted from the effect summary of the enclosing operation. This
    /// checks the regions returned by [`RegionLoweringExtension::emitted_regions`] without a target platform, skipping
    /// dormant rules and bodies of scans that admit no iteration.
    fn contains_unresolved_state(&self) -> bool;

    /// Returns whether a region that lowering emits contains a reference-typed atom or an instruction
    /// that declares a [`ReferenceEffect`](ryft_core::ReferenceEffect). This checks the same regions as
    /// [`Self::contains_unresolved_state`]. Passing through a reference or forwarding a reference capture
    /// can leave reference values in those regions even when no instruction declares ordered state.
    fn contains_unresolved_references(&self) -> bool;

    /// Returns whether this program requests the placement of a value explicitly in any of the regions that lowering
    /// emits, through a `place` with a target, a `constrain_sharding`, or a `call` whose callee names devices (i.e.,
    /// carries a sharding override of its boundary or requests a placement itself), whether or not that value reaches
    /// an output. Only a request of the program that lowering emits (i.e., [`Self::pruned_for_lowering`]) counts, so a
    /// wrapper that drops the only output of a callee that places a value requests nothing. Such calls count because
    /// their callees cannot show their overrides (e.g., a replicated override over auto mesh axes stages nothing).
    /// The callees of calls are among the regions that lowering emits, so a request inside a callee counts too.
    /// A `place` without a target leaves its input where it is, so it requests nothing. A `place` with a device
    /// target (i.e., a [`PlacementRequest::Device`]) is counted, as an explicit device commits an array eagerly.
    /// A `place` with an in-place target (i.e., a [`PlacementRequest::InPlace`]) is counted too. A `reshard` is
    /// not counted.
    ///
    /// With `includes_memory_requests` unset, an in-place target is not counted, because it names no devices. This is
    /// how a compiled function decides whether it names the devices of its mesh, and commits its outputs there.
    ///
    /// The program is analyzed for `target_platform`, so a request inside a body that lowering for that platform
    /// replaces is not counted, and with [`None`] no region is replaced.
    fn has_placement_requests(&self, target_platform: Option<&str>, includes_memory_requests: bool) -> bool
    where
        Self::Operation: 'static;

    /// Returns where the sharding of each of the first `output_count` outputs of this program comes from (refer to
    /// [`OutputShardingSource`]), or [`None`] for an output whose sharding the program does not determine. An output
    /// that a `place` in the entry region produces has the placement of its target, which states the placement along
    /// every mesh axis, including the auto mesh axes that the output type drops. A bufferless output of `condition`,
    /// `scan`, or `while` uses its declared sharding or an entry output override, because the compiler has no output
    /// buffer from which to report a placement along auto axes. An output that is an entry input is placed where that
    /// input enters the program. An output that returns a [`RetainedArray`] is placed as the hidden input that supplies
    /// it, whose sharding `constant_sharding` returns.
    ///
    /// Operations that return the value of an input together with its placement are followed back to that input, so
    /// that an output keeps the placement of a `place` or of an entry input that precedes them. These are a `place`
    /// without a target, which leaves its input where it is, a `place` with a device target, which keeps the sharding
    /// of its input on the only device of the program mesh, `transfer_to_memory`, which only changes the memory of its
    /// input as a `place` with an in-place target does, `stop_gradient` and `tag`, which only mark values for program
    /// transforms, and identity `broadcast` operations that forget a compatible global sharding on an explicit mesh.
    /// A device target that completes a pending reduction over mesh axes of size one gives its output another sharding
    /// than its input, so such an output is placed as its type states instead. A `call` always runs its callee,
    /// whose outputs are the outputs of the call, so an output of a call has the placement that the callee gives the
    /// corresponding callee output, and a callee output that returns a callee input is followed back to the
    /// corresponding input of the call. The boundary of the callee takes precedence, because the callee body cannot
    /// always show it, so an output of the call that a sharding override of a callee output covers is placed as that
    /// override, and one that returns a callee input with an effective boundary sharding (i.e., its override or
    /// otherwise where the executable of the callee places it, refer to [`CallOperation::boundary_input_sharding`])
    /// is placed as that sharding. Lowering constrains that output or input of the call to that sharding as well,
    /// so that the call places it alike whether a program returns it or consumes it (refer to the lowering of
    /// [`CallOperation`]). Other operations that return a value equal to an input are not followed, because they
    /// change its placement (e.g., `reshard` and `constrain_sharding`) or because the compiler is free to place their
    /// outputs differently (e.g., the `optimization_barrier` of rematerialization).
    fn output_sharding_sources<F: Fn(&RetainedArray) -> Sharding>(
        &self,
        output_count: usize,
        constant_sharding: F,
    ) -> Vec<Option<OutputShardingSource>>;

    /// Returns one flag per logical input, set when the executable can omit that input because it is unread. The first
    /// `capture_count` inputs are captures, ending with `session_constant_count` session constants. For the preceding
    /// captures, [`RegionLoweringExtension::used_captures`] determines which the emitted regions read. This mask treats
    /// session constants as read. The caller supplies `unread_public_inputs` for the public inputs that follow the
    /// captures. Only statically shaped inputs outside `reference_state_inputs` can be omitted this way. Compilation
    /// uses these flags for input admission and placement, and the executable records them in
    /// [`MlirProgramBoundaryMapping::unread_inputs`].
    fn unread_logical_inputs(
        &self,
        logical_input_types: &[ArrayType],
        capture_count: usize,
        session_constant_count: usize,
        unread_public_inputs: &[bool],
        reference_state_inputs: &[usize],
        target_platform: Option<&str>,
    ) -> Vec<bool>;

    /// Validates the entry boundary of this program and returns its [`ProgramBoundary`], without emitting anything.
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name of the entry function.
    ///   - `session_constants`: Constants of the emitted regions, in the order used to infer hidden inputs.
    ///   - `capture_types`: Types of the hidden capture inputs that precede the public inputs, which end with the
    ///     types of the session constants of this program.
    ///   - `input_types`: Global types of the inputs.
    ///   - `output_types`: Global types of the outputs.
    ///   - `logical_input_shardings`: Optional sharding of each logical input (i.e., captures first).
    ///   - `output_shardings`: Optional sharding of each output.
    ///   - `open_outputs`: Whether each output is lowered open over auto mesh axes, so that the compiler chooses its
    ///     placement over those axes (i.e., one flag per output, or none at all).
    ///   - `options`: Options that provide the target platform, the reference states, the donated and unread inputs,
    ///     and whether the entry function exports its ordered I/O token chain.
    ///
    /// # Errors
    ///
    /// Returns [`LoweringError::UnresolvedReference`] or [`LoweringError::UnresolvedState`] for a program whose
    /// references or state have not been discharged, [`LoweringError::InvalidFunctionName`] for an invalid function
    /// name, [`ProgramError::MalformedProgram`] when the trailing capture types are not the types of the session
    /// constants, and a compilation error when the shardings or open output flags do not match the boundary or when
    /// they describe a boundary that the executable cannot have.
    #[allow(clippy::too_many_arguments)]
    fn entry_boundary(
        &self,
        function_name: &str,
        session_constants: ProgramSessionConstants,
        capture_types: &[ArrayType],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
        logical_input_shardings: &[Option<Sharding>],
        output_shardings: &[Option<Sharding>],
        open_outputs: &[bool],
        options: &ProgramLoweringOptions,
    ) -> Result<ProgramBoundary, Error>;
}

impl<O, Input, Output> ProgramLoweringAnalysisExtension for Program<Constant, O, Input, Output>
where
    O: ProgramLoweringOperation,
    Input: Parameterized<Constant>,
    Output: Parameterized<Constant>,
{
    type Operation = O;

    #[inline]
    fn pruned_for_lowering(&self, target_platform: Option<&str>) -> Result<Self, ProgramError> {
        O::without_stale_call_meshes(self.clone().into_pruned()?.into_simplified()?, target_platform)
    }

    #[inline]
    fn contains_unresolved_state(&self) -> bool {
        self.entry_region_ref().emitted_regions(None, CallTraversal::Enter).any(|region| {
            region
                .instructions()
                .iter()
                .any(|instruction| instruction.operation().effects().classes().contains(EffectClass::OrderedState))
        })
    }

    #[inline]
    fn contains_unresolved_references(&self) -> bool {
        // Checking declarations as well as atom types catches an incomplete discharge that retyped every atom but
        // left a reference operation behind.
        self.entry_region_ref().emitted_regions(None, CallTraversal::Enter).any(|region| {
            region.atoms().iter().any(|atom| atom.r#type().is_reference())
                || region
                    .instructions()
                    .iter()
                    .any(|instruction| instruction.operation().effects().has_reference_effects())
        })
    }

    #[inline]
    fn has_placement_requests(&self, target_platform: Option<&str>, includes_memory_requests: bool) -> bool
    where
        Self::Operation: 'static,
    {
        RegionPlacementRequests::for_lowering(self.entry_region_ref(), target_platform)
            .has_requests(includes_memory_requests)
    }

    #[inline]
    fn output_sharding_sources<F: Fn(&RetainedArray) -> Sharding>(
        &self,
        output_count: usize,
        constant_sharding: F,
    ) -> Vec<Option<OutputShardingSource>> {
        /// Returns the source of the sharding of each output of `region`, which
        /// [`ProgramLoweringAnalysisExtension::output_sharding_sources`] describes, with
        /// [`OutputShardingSource::Input`] naming an input of `region`. An output that returns a [`RetainedArray`] has
        /// the placement of that array, which enters the executable as a hidden input with that placement, and which
        /// has no buffer whose placement the compiler could report when it has the zero data type. The placements of
        /// the callee regions of `call`s are computed once per callee and memoized in `callee_sources`, because
        /// repeated calls share their callee region. The inputs of `region` and the producer of each of its values,
        /// together with the position of that value among the outputs of its producer, are indexed once, so that
        /// following an output back to an input costs one lookup per step rather than a scan of the inputs or of the
        /// outputs of an instruction.
        fn region_output_sharding_sources<O: ProgramLoweringOperation, F: Fn(&RetainedArray) -> Sharding>(
            region: RegionRef<'_, Constant, O>,
            constant_sharding: &F,
            callee_sources: &mut HashMap<RegionId, Rc<[Option<OutputShardingSource>]>>,
        ) -> Rc<[Option<OutputShardingSource>]> {
            let mut producers = vec![None; region.atoms().len()];
            for (instruction_index, instruction) in region.instructions().iter().enumerate() {
                for (output_index, output) in instruction.outputs().iter().enumerate() {
                    producers[output.index()] = Some((instruction_index, output_index));
                }
            }

            // A value that is several inputs is placed as the first of them, so later positions are visited first.
            let mut input_positions = vec![None; region.atoms().len()];
            for (position, input) in region.input_ids().iter().enumerate().rev() {
                input_positions[input.index()] = Some(position);
            }

            let mut output_source = |output_id: AtomId| {
                let mut value_id = output_id;
                loop {
                    let Some((instruction_index, output_index)) = producers[value_id.index()] else {
                        if let Some(Constant::Session(constant)) = region.atoms()[value_id.index()].as_constant() {
                            return Some(OutputShardingSource::Determined(constant_sharding(constant)));
                        }

                        return input_positions[value_id.index()].map(OutputShardingSource::Input);
                    };

                    let instruction = &region.instructions()[instruction_index];
                    let operation = instruction.operation();
                    let input_index = if let Some(operation) = operation.projected_payload::<PlaceOperation>() {
                        match operation.target() {
                            Some(PlacementRequest::Resolved(target)) => {
                                return Some(OutputShardingSource::Determined(target.sharding().clone()));
                            }
                            Some(PlacementRequest::Device { .. }) => {
                                // A device target completes a pending reduction over mesh axes of size one, so its
                                // output can have another sharding than its input, which is then the placement of the
                                // output.
                                let sharding = |atom: AtomId| match region.atoms()[atom.index()].r#type().as_ref() {
                                    Type::Array(array_type) => array_type.sharding().cloned(),
                                    _ => None,
                                };

                                let output_sharding = sharding(instruction.outputs()[output_index]);
                                if output_sharding != sharding(*instruction.inputs().first()?) {
                                    return output_sharding.map(OutputShardingSource::Determined);
                                }

                                0
                            }
                            Some(PlacementRequest::InPlace { .. }) | None => {
                                // An in-place target leaves its input on the devices where it lives with its sharding,
                                // so the output is placed as its input is.
                                0
                            }
                        }
                    } else if operation.projected_payload::<StopGradientOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<TagOperation<ArrayType>>().is_some()
                        || operation.projected_payload::<TransferToMemoryOperation>().is_some()
                    {
                        // These operations return each of their inputs as the output at the same position, and a memory
                        // transfer keeps the devices and the sharding of its input like an in-place `place` target
                        // does.
                        output_index
                    } else if let Some(broadcast) = operation.projected_payload::<BroadcastOperation>() {
                        let source = region.atoms()[instruction.inputs().first()?.index()].r#type();
                        let Type::Array(source) = source.as_ref() else { return None };
                        let target = broadcast.output_type();

                        // A transpose can forget tracked sharding to match an untracked cotangent type. Keep the split
                        // chosen by an explicit placement. An auto mesh still leaves this choice to the partitioner.
                        let explicit_mesh = source.sharding().is_some_and(|sharding| {
                            sharding.mesh().axes().iter().all(|axis| axis.r#type() == MeshAxisType::Explicit)
                        });
                        if !explicit_mesh
                            || target.sharding().is_some()
                            || source.shape() != target.shape()
                            || !broadcast.output_axes().iter().copied().eq(0..source.rank())
                            || !target.is_refined_by(source)
                        {
                            return None;
                        }

                        0
                    } else {
                        let call = operation.projected_payload::<CallOperation<Type>>()?;

                        // The inputs of the callee receive the inputs of the call positionally, and the outputs of the
                        // call are the outputs of the callee. A sharding override of the boundary of the callee places
                        // the output or the forwarded input that it covers, as an execution of the callee does, which
                        // its body cannot always show (e.g., a replicated override over auto mesh axes stages nothing).
                        if let Some(sharding) = call.output_sharding(output_index) {
                            return Some(OutputShardingSource::Determined(sharding.clone()));
                        }

                        let callee = *instruction.regions().first()?;
                        let sources = match callee_sources.get(&callee) {
                            Some(sources) => Rc::clone(sources),
                            None => {
                                let sources = region_output_sharding_sources(
                                    region.with_id(callee).ok()?,
                                    constant_sharding,
                                    callee_sources,
                                );
                                callee_sources.insert(callee, Rc::clone(&sources));
                                sources
                            }
                        };

                        match sources.get(output_index)?.as_ref()? {
                            OutputShardingSource::Determined(sharding) => {
                                return Some(OutputShardingSource::Determined(sharding.clone()));
                            }
                            OutputShardingSource::Input(index) => {
                                let actual = region.atoms()[instruction.inputs().get(*index)?.index()].r#type();
                                let callee = region.with_id(callee).ok()?;
                                let declared = callee.input_types();
                                match (actual.as_ref(), declared.get(*index)?) {
                                    (Type::Array(actual), declared) => {
                                        match call.boundary_input_sharding(*index, actual, declared) {
                                            Some(sharding) => {
                                                return Some(OutputShardingSource::Determined(sharding.into_owned()));
                                            }
                                            None => *index,
                                        }
                                    }
                                    _ => *index,
                                }
                            }
                        }
                    };

                    value_id = *instruction.inputs().get(input_index)?;
                }
            };

            region.output_ids().iter().map(|&output_id| output_source(output_id)).collect()
        }

        let mut callee_sources = HashMap::new();
        region_output_sharding_sources(self.entry_region_ref(), &constant_sharding, &mut callee_sources)[..output_count]
            .to_vec()
    }

    fn unread_logical_inputs(
        &self,
        logical_input_types: &[ArrayType],
        capture_count: usize,
        session_constant_count: usize,
        unread_public_inputs: &[bool],
        reference_state_inputs: &[usize],
        target_platform: Option<&str>,
    ) -> Vec<bool> {
        let lifted = self.input_count() + session_constant_count == logical_input_types.len();
        let source_capture_count = capture_count.saturating_sub(session_constant_count);
        let used_captures = self.entry_region_ref().used_captures(source_capture_count, lifted, target_platform);
        logical_input_types
            .iter()
            .enumerate()
            .map(|(index, r#type)| {
                let is_read = match index.checked_sub(capture_count) {
                    Some(public_index) => !unread_public_inputs.get(public_index).copied().unwrap_or(false),
                    None => used_captures.get(index).copied().unwrap_or(true),
                };
                !is_read && r#type.static_shape().is_some() && !reference_state_inputs.contains(&index)
            })
            .collect()
    }

    fn entry_boundary(
        &self,
        function_name: &str,
        session_constants: ProgramSessionConstants,
        capture_types: &[ArrayType],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
        logical_input_shardings: &[Option<Sharding>],
        output_shardings: &[Option<Sharding>],
        open_outputs: &[bool],
        options: &ProgramLoweringOptions,
    ) -> Result<ProgramBoundary, Error> {
        let target_platform = options.target_platform();
        let reference_states = options.reference_bindings();
        let (donated_inputs, unread_inputs) = (options.donated_inputs(), options.unread_inputs());
        let export_effects = options.exports_effects();

        // XLA has no references or ordered state, so both must have been discharged from the program. References are
        // checked first, because a program with both gets the more precise diagnostic that way.
        if self.contains_unresolved_references() {
            return Err(Error::Lowering(LoweringError::UnresolvedReference { construct: "program".to_string() }));
        }

        if self.contains_unresolved_state() {
            return Err(Error::Lowering(LoweringError::UnresolvedState { construct: "program".to_string() }));
        }

        // Accept surrounding whitespace and one optional MLIR symbol prefix.
        let function_name = function_name.trim();
        let function_name = function_name.strip_prefix('@').unwrap_or(function_name);
        if function_name.is_empty() || function_name.chars().any(char::is_whitespace) {
            return Err(Error::Lowering(LoweringError::InvalidFunctionName {
                function_name: function_name.to_string(),
            }));
        }

        // Session constants are carried by hidden inputs that form the suffix of the capture inputs. The caller
        // declares them through the trailing `capture_types`, in order of first occurrence in the emitted regions.
        let session_count = session_constants.inputs.len();
        if capture_types.len() < session_count
            || !capture_types[capture_types.len() - session_count..]
                .iter()
                .zip(&session_constants.inputs)
                .all(|(capture_type, constant)| capture_type == constant.r#type().as_ref())
        {
            return Err(ProgramError::MalformedProgram(format!(
                "the trailing capture types of the entry function must be the types of its {session_count} session \
                 constant(s)",
            ))
            .into());
        }

        // The logical inputs are the captures followed by the public inputs. The boundary mapping assigns them and the
        // logical outputs to executable slots.
        let logical_input_types = capture_types.iter().cloned().chain(input_types.iter().cloned()).collect::<Vec<_>>();
        let effects = self.effects().classes();
        let ordered_io = effects.contains(EffectClass::OrderedIo) || effects.contains(EffectClass::DeviceOrderedIo);

        // An input that the executable never reads has no executable slot, so that executions neither admit, require,
        // nor wait for its buffers, and the boundary mapping records it as unread.
        let reference_state_inputs = reference_states
            .iter()
            .filter_map(|state| state.source().flat_input_index(capture_types.len()).ok())
            .collect::<Vec<_>>();
        let unread_logical_inputs = self.unread_logical_inputs(
            &logical_input_types,
            capture_types.len(),
            session_count,
            unread_inputs,
            &reference_state_inputs,
            target_platform,
        );

        // The ordered I/O token chain crosses the boundary only when the options export it. Its ordering is per device
        // unless some effect of the program is ordered across all devices.
        let boundary_mapping = MlirProgramBoundaryMapping::new(logical_input_types.as_slice(), output_types)
            .with_unread_inputs(&unread_logical_inputs)
            .with_effects(
                export_effects && ordered_io,
                effects.contains(EffectClass::UnorderedIo),
                effects.contains(EffectClass::OrderedAssertion),
                !effects.contains(EffectClass::OrderedIo),
            );

        // The shardings and open output flags must describe exactly this boundary. Each is given per logical input
        // or output, and the open output flags may also be left out entirely.
        if logical_input_shardings.len() != logical_input_types.len() {
            return Err(Error::Compilation(CompilationError::ShardingCountMismatch {
                kind: "input",
                expected: logical_input_types.len(),
                actual: logical_input_shardings.len(),
            }));
        }
        if output_shardings.len() != output_types.len() {
            return Err(Error::Compilation(CompilationError::ShardingCountMismatch {
                kind: "output",
                expected: output_types.len(),
                actual: output_shardings.len(),
            }));
        }
        if !open_outputs.is_empty() && open_outputs.len() != output_types.len() {
            return Err(Error::Compilation(CompilationError::InvalidOptions {
                reason: format!(
                    "expected {} open output flags or an empty list, but got {}",
                    output_types.len(),
                    open_outputs.len(),
                ),
            }));
        }

        // A pending reduction that an output sharding declares must be one that lowering can establish for the type of
        // that output.
        for (sharding, output_type) in output_shardings.iter().zip(output_types) {
            if let Some(sharding) = sharding {
                let data_type = output_type.data_type();
                validate_declared_pending_reduction(
                    sharding,
                    output_type.sharding(),
                    data_type,
                    &[],
                    "output_shardings",
                )?;
            }
        }

        // Reference states and donated inputs let the executable reuse the buffers of an input for an output. The
        // pairs are resolved here, before anything is emitted, so that an invalid pairing fails without a module.
        let aliases = ProgramBoundary::aliases(
            reference_states,
            donated_inputs,
            open_outputs,
            capture_types.len(),
            &boundary_mapping,
            &logical_input_types,
            output_types,
            logical_input_shardings,
            output_shardings,
        )?;

        let executable_input_types = boundary_mapping.executable_input_types(logical_input_types.as_slice())?;
        let executable_output_types = boundary_mapping.executable_output_types(output_types)?;

        Ok(ProgramBoundary {
            function_name: function_name.to_string(),
            boundary_mapping,
            session_constants,
            logical_input_types,
            executable_input_types,
            executable_output_types,
            aliases,
        })
    }
}

/// Lowering of a program body into an existing MLIR block.
pub(crate) trait ProgramInlineLoweringExtension {
    /// Operation family of the program.
    type Operation: ProgramLoweringOperation;

    /// Appends the program body to `block` and returns its flat outputs. Captured constants resolve against
    /// `captured_values` and the program's input atoms bind to `input_values`. Each instruction inherits its
    /// location from `location` and its provenance. Calls may use private functions registered in `nested_functions`.
    ///
    /// Ordered effects consume and update the caller's `effect_tokens` in program order. The caller continues from
    /// the final token of each chain after the body returns. An instruction that uses the assertion token marks the
    /// module as requiring the runtime assertion handler.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidInputCount`] before emitting any operations when the input count differs from
    /// the program's input count. Propagates errors from lowering constants and instructions.
    #[allow(clippy::too_many_arguments)]
    fn lower_inline<'b, 'c: 'b, 't: 'c>(
        &self,
        captured_values: &[ValueRef<'b, 'c, 't>],
        input_values: &[ValueRef<'b, 'c, 't>],
        nested_functions: Option<&Rc<CallFunctionMap<Self::Operation>>>,
        lowering_context: &ProgramLoweringContext,
        effect_tokens: &mut EffectTokens<'b, 'c, 't>,
        block: BlockRef<'b, 'c, 't>,
        location: LocationRef<'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error>;
}

impl<O, Input, Output> ProgramInlineLoweringExtension for Program<Constant, O, Input, Output>
where
    O: ProgramLoweringOperation,
    Input: Parameterized<Constant>,
    Output: Parameterized<Constant>,
{
    type Operation = O;

    fn lower_inline<'b, 'c: 'b, 't: 'c>(
        &self,
        captured_values: &[ValueRef<'b, 'c, 't>],
        input_values: &[ValueRef<'b, 'c, 't>],
        nested_functions: Option<&Rc<CallFunctionMap<O>>>,
        lowering_context: &ProgramLoweringContext,
        effect_tokens: &mut EffectTokens<'b, 'c, 't>,
        block: BlockRef<'b, 'c, 't>,
        location: LocationRef<'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let region = self.entry_region_ref();
        let context = block.context();
        let target_platform = lowering_context.target_platform();
        let mut constant_block = block;
        region.interpret_with(
            input_values.to_vec(),
            |_, value| value.lower(captured_values, lowering_context, &mut constant_block, context, location),
            |instruction, inputs| {
                let (input_types, output_types) = region.instruction_boundary_types(instruction);

                // All operations emitted for this instruction inherit its provenance on top of the ambient location.
                let location = lowering_context.lower_provenance(context, instruction.provenance(), location);
                let operation = instruction.operation();

                // A named composite with a backend implementation or a shared function needs no copy of its body.
                let regions = if operation.composite_key().is_some_and(|key| {
                    O::has_composite_implementation(key, &input_types, &output_types, target_platform)
                        || nested_functions
                            .is_some_and(|functions| functions.named(key, &input_types, &output_types).is_some())
                }) {
                    Vec::new()
                } else {
                    instruction
                        .regions()
                        .iter()
                        .map(|attached| RegionRef::new(region.arena(), *attached).map(RegionRef::to_program))
                        .collect::<Result<Vec<_>, _>>()?
                };

                let mut lowerer = Lowerer::new(block, context, location, lowering_context.clone())
                    .with_regions(regions)
                    .with_nested_functions(nested_functions.cloned())
                    .with_captured_values(captured_values)
                    .with_effect_tokens(*effect_tokens);

                let outputs =
                    lower_instruction_operation(operation, &mut lowerer, inputs, &input_types, &output_types)?;

                if lowerer.effect_tokens.ordered_assertion.is_some() {
                    lowering_context.require_assertion_handler();
                }

                *effect_tokens = lowerer.effect_tokens;

                Ok(outputs)
            },
        )
    }
}

/// Analyzes instruction boundaries, emitted computations, and capture use in an XLA program region.
pub(crate) trait RegionLoweringExtension<O: ProgramLoweringOperation>: Sized {
    /// Returns the input and output types of an instruction in this region, in operand and result order.
    fn instruction_boundary_types(self, instruction: &Instruction<O>) -> (Vec<Type>, Vec<Type>);

    /// Visits this region and its emitted computation descendants once each, in the
    /// order of [`RegionRef::computation_regions`]. Dormant rules and bodies omitted by
    /// [`ProgramLoweringOperation::omits_regions`] are skipped. Use [`CallTraversal::Enter`] for
    /// whole-program analyses and [`CallTraversal::Skip`] when resolving captures in this region's scope.
    fn emitted_regions(self, target_platform: Option<&str>, calls: CallTraversal) -> impl Iterator<Item = Self>;

    /// Returns one flag per capture, set when lowering reads that capture in this region or an emitted descendant in
    /// the same capture scope. A captured constant names its position through [`Value::capture_index`]. When `lifted`
    /// is set, the first `capture_count` entry inputs also name the captures in order. Callees have their own capture
    /// indices. At each call, [`Self::visit_atom_reads`] checks the forwarded operands against the callee's unread
    /// captures. Compilation and lowering use these flags so that a capture read only inside an omitted body, such
    /// as a zero-length scan, can be skipped at the executable boundary.
    fn used_captures(self, capture_count: usize, lifted: bool, target_platform: Option<&str>) -> Vec<bool>;

    /// Visits each instruction input and each output of this region, passing its atom ID and whether the read
    /// is live. Outputs are always live. When a call forwards a lifted capture that the operation family marks
    /// [`unread`](super::LowerableOperation::unread_captures) on `target_platform`, its operand is dead and lowering
    /// supplies a bufferless value. Capture analysis and session constant partitioning use this same decision.
    fn visit_atom_reads<F: FnMut(AtomId, bool)>(self, target_platform: Option<&str>, visit: F);
}

impl<O: ProgramLoweringOperation> RegionLoweringExtension<O> for RegionRef<'_, Constant, O> {
    fn instruction_boundary_types(self, instruction: &Instruction<O>) -> (Vec<Type>, Vec<Type>) {
        let types =
            |atoms: &[AtomId]| atoms.iter().map(|atom| self.atoms()[atom.index()].r#type().into_owned()).collect();
        (types(instruction.inputs()), types(instruction.outputs()))
    }

    fn emitted_regions(self, target_platform: Option<&str>, calls: CallTraversal) -> impl Iterator<Item = Self> {
        let mut visited = vec![false; self.arena().len()];
        let mut pending = vec![self.id()];
        std::iter::from_fn(move || {
            while let Some(id) = pending.pop() {
                if std::mem::replace(&mut visited[id.index()], true) {
                    continue;
                }

                // Every pending identifier is this region or an attachment in the same arena.
                let region = self.with_id(id).unwrap();
                for instruction in region.instructions() {
                    let operation = instruction.operation();
                    if operation.omits_regions(target_platform, || region.instruction_boundary_types(instruction))
                        || (calls == CallTraversal::Skip
                            && operation.projected_payload::<CallOperation<Type>>().is_some())
                    {
                        continue;
                    }

                    pending.extend(instruction.regions().iter().copied().enumerate().filter_map(|(index, id)| {
                        (operation.region_role(index) == Some(RegionRole::Computation)).then_some(id)
                    }));
                }

                return Some(region);
            }

            None
        })
    }

    fn used_captures(self, capture_count: usize, lifted: bool, target_platform: Option<&str>) -> Vec<bool> {
        let mut is_read = vec![false; capture_count];
        for region in self.emitted_regions(target_platform, CallTraversal::Skip) {
            let mut lifted_captures = HashMap::new();
            if lifted && region.id() == self.id() {
                let inputs = region.input_ids().iter().take(capture_count);
                lifted_captures.extend(inputs.enumerate().map(|(index, id)| (*id, index)));
            }

            region.visit_atom_reads(target_platform, |id, is_live| {
                if !is_live {
                    return;
                }

                let index = lifted_captures.get(&id).copied().or_else(|| {
                    region.atoms().get(id.index()).and_then(|atom| atom.as_constant()).and_then(Value::capture_index)
                });

                if let Some(index) = index.filter(|index| *index < capture_count) {
                    is_read[index] = true;
                }
            });
        }

        is_read
    }

    fn visit_atom_reads<F: FnMut(AtomId, bool)>(self, target_platform: Option<&str>, mut visit: F) {
        for instruction in self.instructions() {
            let unread = instruction
                .operation()
                .projected_payload::<CallOperation<Type>>()
                .filter(|call| call.capture_count() != 0)
                .and_then(|call| {
                    let callee = self.with_id(*instruction.regions().first()?).ok()?;
                    O::unread_captures(callee, call.capture_count(), target_platform)
                });

            for (index, id) in instruction.inputs().iter().enumerate() {
                visit(*id, !unread.as_deref().is_some_and(|unread| unread.get(index).copied().unwrap_or(false)));
            }
        }

        for id in self.output_ids() {
            visit(*id, true);
        }
    }
}

/// Derives the [`RegionPlacementRequests`] of the program that lowering emits for a region, and the devices that its
/// placement requests name.
pub(crate) trait RegionPlacementRequestsExtension: Sized {
    /// Returns the [`RegionPlacementRequests`] of the program that lowering emits for `region` and `target_platform`,
    /// every output of `region` being used (refer to [`ProgramLoweringAnalysisExtension::pruned_for_lowering`]). For
    /// a target platform, the regions of the instructions whose lowering for that platform omits them are not analyzed
    /// (refer to [`RegionPlacementRequests::for_target`]), and [`None`] analyzes the program for no target (refer to
    /// [`RegionPlacementRequests::new`]). The summary is derived once for each region and platform and retained with
    /// the region.
    fn for_lowering<O: 'static + ProgramLoweringOperation>(
        region: RegionRef<'_, Constant, O>,
        target_platform: Option<&str>,
    ) -> Self;

    /// Returns the devices that the placement requests of the program name (refer to
    /// [`RegionPlacementRequests::requests`]), as the mesh of a request that names devices together with the name
    /// of the operation that makes it, or [`None`] when no request names devices. A device target is resolved as
    /// [`RequestedPlacement`] documents, and a request over a named mesh of the same devices takes precedence over the
    /// canonical mesh (refer to [`merge_requested_placement`]), so that the result does not depend on the order of the
    /// requests. The caller merges the requests of other regions with [`merge_requested_placement`].
    ///
    /// # Errors
    ///
    /// Returns [`LoweringError::UnsupportedOperation`] when two requests name different devices, which no single mesh
    /// runs.
    fn requested_mesh(&self) -> Result<Option<(DeviceMesh, &'static str)>, Error>;
}

impl RegionPlacementRequestsExtension for RegionPlacementRequests {
    #[inline]
    fn for_lowering<O: 'static + ProgramLoweringOperation>(
        region: RegionRef<'_, Constant, O>,
        target_platform: Option<&str>,
    ) -> Self {
        match target_platform {
            None => Self::new(region),
            Some(target_platform) => Self::for_target(region, target_platform, |region, instruction| {
                instruction
                    .operation()
                    .omits_regions(Some(target_platform), || region.instruction_boundary_types(instruction))
            }),
        }
    }

    #[inline]
    fn requested_mesh(&self) -> Result<Option<(DeviceMesh, &'static str)>, Error> {
        let mut requested = None;
        for (request, operation) in self.requests() {
            if let RequestedPlacement::Devices(mesh) = request {
                merge_requested_placement(&mut requested, Cow::Borrowed(mesh), operation)?;
            }
        }
        Ok(requested.map(|(mesh, operation)| (mesh.into_owned(), operation)))
    }
}

/// Whether a traversal follows a [`CallOperation`] into its callee.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum CallTraversal {
    /// Visit the computations that execute across call boundaries.
    Enter,

    /// Stay in the current capture scope. A callee's capture indices refer to its own inputs.
    Skip,
}

/// Source of the sharding of one output of a program, as [`ProgramLoweringAnalysisExtension::output_sharding_sources`]
/// returns it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OutputShardingSource {
    /// The program determines the sharding of the output.
    Determined(Sharding),

    /// The output inherits the sharding of the region input at this index.
    Input(usize),
}

/// Merges `mesh`, the devices that a placement request of the operation named `operation` names, into `requested`, the
/// request that names devices among those seen before it, if any, together with the name of its operation. A request
/// over the canonical mesh of one device (i.e., [`LogicalMesh::single_device`], over which a device target whose input
/// tracks no mesh is resolved, refer to [`RegionPlacementRequestsExtension::requested_mesh`]) gives way to a request
/// over another logical mesh of the same device, whichever comes first, so that a bare device and a named mesh of that
/// device select the named mesh in either order. Requests over two named meshes of the same devices keep the first.
///
/// # Errors
///
/// Returns [`LoweringError::UnsupportedOperation`] when `mesh` names other devices than `requested`, which no single
/// mesh runs.
pub(crate) fn merge_requested_placement<'r>(
    requested: &mut Option<(Cow<'r, DeviceMesh>, &'static str)>,
    mesh: Cow<'r, DeviceMesh>,
    operation: &'static str,
) -> Result<(), Error> {
    match requested {
        None => *requested = Some((mesh, operation)),
        Some((first, _)) if first.devices() == mesh.devices() => {
            let canonical = LogicalMesh::single_device();
            if first.logical_mesh() == &canonical && mesh.logical_mesh() != &canonical {
                *requested = Some((mesh, operation));
            }
        }
        Some((first, _)) => {
            return Err(Error::Lowering(LoweringError::UnsupportedOperation {
                operation: operation.to_string(),
                reason: format!(
                    "it requests a placement onto the devices [{}], while another placement request of the same \
                     program names the devices [{}], and the two do not form one mesh",
                    mesh.devices().iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                    first.devices().iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                ),
            }));
        }
    }
    Ok(())
}

/// Returns the sharding with which the entry function of a program annotates each of its outputs, whose types are
/// `output_types`, and which describes the buffers that the outputs are returned in. These are the `output_shardings`
/// overrides when they are provided, and otherwise the placement that the program gives each output (i.e., the
/// corresponding element of `sharding_sources`, as [`ProgramLoweringAnalysisExtension::output_sharding_sources`]
/// returns them, where an entry input is placed as the corresponding element of `logical_input_shardings`), or the
/// sharding of the output type when the program gives it none (e.g., when that input has no sharding). An output
/// without either has no sharding. Compilation for a [`Session`](crate::Session) and the StableHLO export of a
/// [`TracedProgram`](crate::TracedProgram) both choose the shardings of outputs with this function, so that an
/// exported module places its outputs as the executable that a session compiles for it does.
///
/// An override, a `place` target, an input placement, or an output type may leave dimensions unconstrained, but the
/// runtime placement of an array reads such a dimension as replicated, and the output type records it exactly, so it
/// cannot record the placement that the compiler would choose for it. These dimensions are therefore replicated here
/// (refer to [`Sharding::with_unconstrained_dimensions_replicated`]), whatever operation produces the output (e.g.,
/// arithmetic whose output type inherits an unconstrained dimension from the target of a `place` that precedes it),
/// so that the executable returns the buffers that the runtime placement of the output describes, while the dimensions
/// that they constrain keep their placement. Provided overrides are not checked against `output_types`, so that a count
/// mismatch is still reported when the module is lowered.
///
/// A placement that the program gives an output never keeps reduction state that the output type does not record (i.e.,
/// a pending reduction or a completed-reduction marker over auto mesh axes). No value of a program carries such state:
/// an input whose input sharding override states a pending reduction is reduced where it enters the program (refer to
/// the lowering of entry inputs, which constrains it to its placement with only the unreduced axes that its type
/// records), and a marker states nothing about the buffers, so an output that returns that input, directly, through
/// operations that keep its placement, or through a call whose boundary states the state, returns the completed value.
/// Annotating the output with the placement of the input as it arrives would instead make the executable introduce the
/// pending reduction again, and return an array whose runtime sharding states a reduction that its type does not record
/// and that every consumer of the array rejects, or whose marker makes every consumer move it again.
pub(crate) fn entry_output_shardings(
    output_types: &[ArrayType],
    output_shardings: Option<&[Sharding]>,
    sharding_sources: &[Option<OutputShardingSource>],
    logical_input_shardings: &[Option<Sharding>],
) -> Result<Vec<Option<Sharding>>, Error> {
    match output_shardings {
        Some(shardings) => shardings
            .iter()
            .map(|sharding| Ok(Some(sharding.with_unconstrained_dimensions_replicated())))
            .collect(),
        None => output_types
            .iter()
            .zip(sharding_sources)
            .map(|(output_type, source)| {
                let stated = match source {
                    Some(OutputShardingSource::Determined(sharding)) => Some(sharding),
                    Some(OutputShardingSource::Input(index)) => {
                        logical_input_shardings.get(*index).and_then(Option::as_ref)
                    }
                    None => None,
                };
                let (unreduced_axes, reduced_axes) = (output_type.unreduced_axes(), output_type.reduced_axes());
                match stated {
                    Some(sharding)
                        if sharding.unreduced_axes().iter().any(|axis| !unreduced_axes.contains(axis))
                            || sharding.reduced_axes().iter().any(|axis| !reduced_axes.contains(axis)) =>
                    {
                        let unreduced = sharding.clone().with_unreduced_axes(unreduced_axes.iter().cloned())?;
                        let reduced = unreduced.with_reduced_axes(reduced_axes.iter().cloned())?;
                        Ok(Some(reduced.with_unconstrained_dimensions_replicated()))
                    }
                    stated => {
                        Ok(stated.or(output_type.sharding()).map(Sharding::with_unconstrained_dimensions_replicated))
                    }
                }
            })
            .collect(),
    }
}

/// Returns whether the entry function of a program leaves each of its outputs, whose types are `output_types`
/// and whose annotations are `output_annotations` (i.e., the shardings that [`entry_output_shardings`] returns for
/// `output_shardings` and `sharding_sources`), open over the auto mesh axes of its annotation, so that the compiler
/// chooses the placement of the output along those axes, like JAX lowers the outputs of a `jit` function without
/// `out_shardings`. Compilation for a [`Session`](crate::Session) and the StableHLO export of a
/// [`TracedProgram`](crate::TracedProgram) both decide this with this function, so that an exported module
/// leaves open the outputs whose placement the compiler chooses for the executable that a session compiles for it.
///
/// An output is open when it has an annotation over a mesh with an auto axis larger than one (i.e., one along which the
/// compiler has a choice), and when nothing else fixes its placement. An output sharding override, a placement that the
/// program gives the output (i.e., that a `place` produces it or that it returns an input), reduction state or
/// manual variation in the annotation, which the compiler must not change, and unconstrained dimensions in the output
/// type, which the runtime placement of the output must record as its type does and read as replicated, all fix it.
/// Outputs without an annotation are never open, because they have no annotation that could be.
///
/// Export lowers a program without a device mesh, so it can only decide what the program and its signature state.
/// It cannot know whether compilation runs the program partitioned, because a program with bounded dynamic shapes
/// runs as fully replicated replicas whose outputs are never open, nor which placements the compiler then chooses
/// for the open outputs, which only the compiled executable reports. It also cannot know the runtime placements of
/// inputs without signature shardings, whose types do not record auto mesh axes, so that their annotations and
/// those of the outputs that return them are left to Shardy.
pub(crate) fn entry_outputs_with_compiler_chosen_placement(
    output_types: &[ArrayType],
    output_shardings: Option<&[Sharding]>,
    sharding_sources: &[Option<OutputShardingSource>],
    output_annotations: &[Option<Sharding>],
) -> Vec<bool> {
    output_types
        .iter()
        .zip(sharding_sources)
        .zip(output_annotations)
        .map(|((output_type, source), annotation)| {
            output_shardings.is_none()
                && source.is_none()
                && annotation.as_ref().is_some_and(|sharding| {
                    sharding.mesh().has_untracked_auto_axes()
                        && sharding.unreduced_axes().is_empty()
                        && sharding.reduced_axes().is_empty()
                        && sharding.varying_manual_axes().is_empty()
                })
                && !output_type
                    .sharding()
                    .is_some_and(|sharding| sharding.dimensions().contains(&ShardingDimension::Unconstrained))
        })
        .collect()
}

/// Returns the private `func.func` named `symbol`, whose body is the lowering of `program`, for a module to hold.
/// The function takes the inputs of `program` and returns its outputs, without attributes. The repeated `call`s of
/// `program` lower to calls of the shared functions in `nested_functions`, if any, and inline otherwise. The body is
/// lowered outside any manual region and with the `lowering_context` of the module, so that it shares the channel
/// allocator, the mesh, and the target platform of the entry function. `program` must have no effects when it is
/// lowered (e.g., `collect_call_functions` leaves out callees whose lowering has effects, including synthesized
/// entry guards), because the function threads no effect token. It must also read no session constant, because
/// only the entry function receives the hidden inputs that carry them.
fn lower_private_function<'c, 't: 'c, O: ProgramLoweringOperation>(
    symbol: &str,
    program: &FlatProgram<EagerContext<Constant, O>>,
    nested_functions: Option<&Rc<CallFunctionMap<O>>>,
    lowering_context: &ProgramLoweringContext,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<func::DetachedFuncOperation<'c, 't>, Error> {
    let input_tensor_types = program
        .input_types()
        .iter()
        .map(|r#type| r#type.lower(context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let output_tensor_types = program
        .output_types()
        .iter()
        .map(|r#type| r#type.lower(context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let argument_types = input_tensor_types.iter().map(|r#type| (*r#type, location)).collect::<Vec<_>>();
    let function_block = context.block(argument_types.as_slice());
    let input_values = (0..input_tensor_types.len())
        .map(|index| function_block.argument(index).unwrap().as_ref())
        .collect::<Vec<_>>();

    let mut function_block_ref = function_block.as_ref();
    let outputs = program.lower_inline(
        &[],
        input_values.as_slice(),
        nested_functions,
        lowering_context,
        &mut EffectTokens::default(),
        function_block_ref,
        location,
    )?;

    function_block_ref.append_operation(func::r#return(outputs.as_slice(), location)?)?;
    let mut function_region = context.region();
    function_region.append_block(function_block)?;
    Ok(func::func(
        symbol,
        func::FuncAttributes {
            arguments: input_tensor_types
                .iter()
                .map(|r#type| TypeAndAttributes { r#type: r#type.as_ref(), attributes: None })
                .collect(),
            results: output_tensor_types
                .iter()
                .map(|r#type| TypeAndAttributes { r#type: r#type.as_ref(), attributes: None })
                .collect(),
            visibility: SymbolVisibility::Private,
            ..Default::default()
        },
        function_region,
        location,
    )?)
}

// TODO(eaplatanios): Can you make a pass over the tests in @here making sure that all the test code complies with our
//  conventions around test naming, placement, coverage, non-redundancy, code style, code comments, documentation, etc.?

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_core::{
        AddOperation, ArrayOperation, ArrayType, CALL_OPERATION_NAME, CaptureReference, CastOperation, CompositeKey,
        CompositeOperation, ConditionOperation, ConstrainSharding, Context, Cos, CustomRuleDefinition,
        CustomRuleRegistration, DataType, Device, DeviceMesh, Dimension, DimensionBounds, DimensionToScalarOperation,
        DimensionType, DimensionValue, DimensionVariable, DomainTracingContext, Dot, DotDimensionNumbers, EffectClass,
        EffectClasses, Error as CoreError, FlatProgram, ImportingContext, Layout, LogicalMesh, Memory, MeshAxis,
        MeshAxisType, NamedComposite, OneLike, PLACE_OPERATION_NAME, Place, PlaceOperation, Placeholder, Placement,
        PlacementRequest, Print, PrintOperation, ProgramBuilder, Provenance, ProvenanceScope, ReferenceFreezeOperation,
        ReferenceNewOperation, ReferenceReadOperation, ReferenceSource, ReferenceType, ReferenceWriteOperation,
        ReshapeOperation, Reshard, ScaledDotOperation, ScanOperation, Shape, ShardMapOptions, Sharding,
        ShardingDimension, Sin, StopGradient, Tag, TiledLayout, TransferToMemory, Transpose, TypeError, Typed,
        ValueProjection, ZeroLike, differentiate_in_parent_at, shard_map,
    };
    use ryft_kernels::{KernelBoundaryContract, KernelParameterAccess, validate_kernel_body};
    use ryft_mlir::Size as MlirSize;
    use ryft_pjrt::{
        BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram,
        load_cpu_plugin,
    };

    use crate::lowering::tests::{
        add_call, dynamic_dimension, linear_call_program, neg_branch, scalar_linear_region, shard_map_module,
        test_manual_mesh, test_matrix_type, test_sharding, test_vector_type, unproject_plain_program,
    };
    use crate::lowering::token_threaded_effects;
    use crate::sessions::tests::{domain_mesh, f32_scalar, two_device_mesh};
    use crate::tests::{
        cpu_client, peer_devices, replicated_array, session_with_remote_devices, test_compilation_options,
        values_from_bytes, values_to_bytes,
    };
    use crate::{
        Array, ArrayConstant, ArrayTracer, CompilationError, CompileTracer, CompiledFunction, Constant, FromPjrt,
        Operation, Options, PlacementOptions, Session, StagedFunction, ThroughSession, TracedProgram, Tracer, Type,
        Value, compile, ensure_print_handler_registered, stage, trace, trace_with_context, with_captured_prints,
    };

    use super::*;

    /// Computes `f(x, y) = x * y + sin(x)` for the pair `inputs` of `(x, y)`.
    fn scalar_bilinear_sin<T>(inputs: (T, T)) -> T
    where
        T: Clone + Sin + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        inputs.0.clone() * inputs.1 + inputs.0.sin().unwrap()
    }

    /// Extends a [`scalar_linear_region`] with a reference that is allocated with its `value`, overwritten with its
    /// `residual`, and frozen into its output.
    fn add_reference_state(
        builder: &mut ProgramBuilder<Constant, Operation>,
        residual: AtomId,
        value: AtomId,
    ) -> AtomId {
        let reference = builder
            .add_instruction(Operation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![value], None)
            .unwrap()[0];
        builder
            .add_instruction(
                Operation::ReferenceWrite(ReferenceWriteOperation::new()),
                Vec::new(),
                vec![reference, residual],
                None,
            )
            .unwrap();
        builder
            .add_instruction(
                Operation::ReferenceFreeze(ReferenceFreezeOperation::new()),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0]
    }

    /// Builds a program whose scan of `length` iterations carries a scalar through a body that extends it with
    /// reference state (refer to [`add_reference_state`]).
    fn stateful_scan_program(length: usize) -> FlatProgram<Session> {
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F32));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        builder.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = builder.add_input(scalar_type.clone());
        let output = add_reference_state(&mut builder, carry, carry);
        let body = Arc::new(builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder]).unwrap());
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(scalar_type);
        let regions = vec![builder.intern_callee(&body, None).unwrap()];
        let scan = ScanOperation::<Type>::new(1, length);
        let output = builder.add_instruction(scan, regions, vec![input], None).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Traces a program over a sharded `f64[4]` input that invokes a `shard_map` over the manual mesh of two devices,
    /// whose body prints its local shard with the [`EffectClass::DeviceOrderedIo`] effect and whose only output is
    /// dead, and that returns its input unchanged.
    fn traced_effectful_dead_shard_map() -> (TracedProgram<ArrayType, ArrayType>, Sharding) {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced = trace(
            {
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    let _printed = shard_map(
                        |local_x: ArrayTracer| local_x.print_with_effect_class("body", EffectClass::DeviceOrderedIo),
                        x.clone(),
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap();
                    x
                }
            },
            ArrayType::new_static(DataType::F64, [4]),
        )
        .unwrap();
        (traced, sharding)
    }

    #[test]
    fn test_mlir_program_boundary_mapping_new() {
        let static_zero = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let dynamic_zero = ArrayType::new(DataType::Zero, Shape::new(vec![dynamic_dimension("dynamic_zero", Some(3))]));
        let boolean = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(3)]));
        let boundary_mapping = MlirProgramBoundaryMapping::new(
            &[static_zero.clone(), boolean.clone(), dynamic_zero.clone()],
            &[dynamic_zero, static_zero],
        );

        assert_eq!(boundary_mapping.input_mapping(), &[None, Some(0), Some(1)]);
        assert_eq!(boundary_mapping.output_mapping(), &[Some(0), None]);
        assert_eq!(boundary_mapping.executable_input_count(), 3);
        assert_eq!(boundary_mapping.input_dimensions().len(), 1);
        assert_eq!(boundary_mapping.input_dimensions()[0].logical_input_index(), 2);
        assert_eq!(boundary_mapping.input_dimensions()[0].axis(), 0);
        assert_eq!(boundary_mapping.input_dimensions()[0].executable_input_index(), 2);
        assert_eq!(boundary_mapping.output_dimensions().len(), 1);
        assert_eq!(boundary_mapping.output_dimensions()[0].logical_output_index(), 0);
        assert_eq!(boundary_mapping.output_dimensions()[0].axis(), 0);
        assert_eq!(boundary_mapping.output_dimensions()[0].executable_output_index(), 1);
    }

    #[test]
    fn test_mlir_program_boundary_mapping_with_unread_inputs() {
        // An unread input is erased whatever its data type and recorded as unread, while an input of the zero data type
        // that is read is erased without being recorded, and the hidden extent inputs follow the remaining arrays.
        let static_zero = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let vector = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let dynamic = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", Some(3))]));
        let input_types = [vector.clone(), static_zero.clone(), static_zero, vector, dynamic];
        let boundary_mapping = MlirProgramBoundaryMapping::new(&input_types, &[]);
        assert_eq!(boundary_mapping.clone().with_unread_inputs(&[false; 5]), boundary_mapping);
        let boundary_mapping = boundary_mapping.with_unread_inputs(&[true, true, false, false, false]);
        assert_eq!(boundary_mapping.input_mapping(), &[None, None, None, Some(0), Some(1)]);
        assert_eq!(boundary_mapping.unread_inputs(), &[true, true, false, false, false]);
        assert_eq!([0, 1, 2].map(|index| boundary_mapping.is_unread_input(index)), [true, true, false]);
        assert_eq!(boundary_mapping.input_dimensions()[0].executable_input_index(), 2);
        assert_eq!(boundary_mapping.executable_input_count(), 3);
    }

    #[test]
    fn test_mlir_program_boundary_mapping_with_effects() {
        let scalar = ArrayType::scalar(DataType::F32);
        let dynamic = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("count", Some(3))]));
        let boundary_mapping = MlirProgramBoundaryMapping::new(&[scalar.clone(), dynamic.clone()], &[dynamic, scalar]);

        // Without effects, the boundary has no token and does not participate in the effect barrier.
        assert!(!boundary_mapping.has_ordered_io());
        assert!(!boundary_mapping.has_device_ordered_io());
        assert!(!boundary_mapping.has_unordered_io());
        assert!(!boundary_mapping.requires_assertion_handler());
        assert!(!boundary_mapping.has_effects());
        assert_eq!(boundary_mapping.ordered_io_input_index(), None);
        assert_eq!(boundary_mapping.ordered_io_output_index(), None);
        assert_eq!(boundary_mapping.input_array_count(), 3);
        assert_eq!(boundary_mapping.output_array_count(), 3);
        assert_eq!(boundary_mapping.executable_input_count(), 3);
        assert_eq!(boundary_mapping.executable_output_count(), 3);

        // Ordered I/O appends one token after every array and hidden extent input and output.
        let ordered = boundary_mapping.clone().with_effects(true, false, false, true);
        assert!(ordered.has_ordered_io());
        assert!(ordered.has_device_ordered_io());
        assert!(!ordered.has_unordered_io());
        assert!(ordered.has_effects());
        assert_eq!(ordered.ordered_io_input_index(), Some(3));
        assert_eq!(ordered.ordered_io_output_index(), Some(3));
        assert_eq!(ordered.input_array_count(), 3);
        assert_eq!(ordered.output_array_count(), 3);
        assert_eq!(ordered.executable_input_count(), 4);
        assert_eq!(ordered.executable_output_count(), 4);

        // Device ordering only applies together with ordered I/O, while unordered I/O and runtime assertions still
        // participate in the effect barrier.
        let unordered = boundary_mapping.clone().with_effects(false, true, false, true);
        assert!(!unordered.has_device_ordered_io());
        assert!(unordered.has_unordered_io());
        assert!(unordered.has_effects());
        assert_eq!(unordered.executable_input_count(), 3);
        let asserting = boundary_mapping.with_effects(false, false, true, false);
        assert!(asserting.requires_assertion_handler());
        assert!(asserting.has_effects());
    }

    #[test]
    fn test_mlir_program_boundary_mapping_executable_input_types() {
        // Every bounded dynamic axis of a materialized input uses its maximum extent and adds a hidden `i32`
        // extent input, while statically shaped structural zeros are erased.
        let static_zero = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let dynamic = ArrayType::new(
            DataType::F32,
            Shape::new(vec![dynamic_dimension("rows", Some(3)), dynamic_dimension("columns", Some(5))]),
        );
        let boundary_mapping = MlirProgramBoundaryMapping::new(&[static_zero.clone(), dynamic.clone()], &[]);
        assert_eq!(
            boundary_mapping.executable_input_types(&[static_zero.clone(), dynamic]).unwrap(),
            vec![
                ArrayType::new_static(DataType::F32, [2, 4]),
                ArrayType::scalar(DataType::I32),
                ArrayType::scalar(DataType::I32),
            ],
        );
        assert_eq!(
            boundary_mapping.executable_input_types(&[static_zero]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }.into()),
        );
    }

    #[test]
    fn test_mlir_program_boundary_mapping_executable_output_types() {
        // Dynamic outputs keep their logical types and each bounded dynamic axis appends a hidden `i64` extent output.
        let static_zero = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let dynamic = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", Some(3))]));
        let boundary_mapping = MlirProgramBoundaryMapping::new(&[], &[dynamic.clone(), static_zero.clone()]);
        assert_eq!(
            boundary_mapping.executable_output_types(&[dynamic.clone(), static_zero]).unwrap(),
            vec![dynamic.clone(), ArrayType::scalar(DataType::I64)],
        );
        assert_eq!(
            boundary_mapping.executable_output_types(&[dynamic]),
            Err(ProgramError::InvalidOutputCount { expected: 2, actual: 1 }.into()),
        );
    }

    #[test]
    fn test_mlir_program_boundary_mapping_executable_input_shardings() {
        // Each hidden extent input is a replicated scalar on the mesh of the input that owns the extent, and it
        // stays unannotated when that input is unannotated.
        let mesh = test_manual_mesh("x", 2);
        let static_zero = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let dynamic = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", Some(3))]));
        let boundary_mapping = MlirProgramBoundaryMapping::new(&[static_zero, dynamic.clone(), dynamic], &[]);
        let vector_sharding = Sharding::replicated(mesh.clone(), 1);
        let scalar_sharding = Sharding::replicated(mesh, 0);
        assert_eq!(
            boundary_mapping
                .executable_input_shardings(&[
                    vector_sharding.clone(),
                    vector_sharding.clone(),
                    vector_sharding.clone(),
                ])
                .unwrap(),
            vec![vector_sharding.clone(), vector_sharding.clone(), scalar_sharding.clone(), scalar_sharding.clone()],
        );
        assert_eq!(
            boundary_mapping.executable_input_shardings(std::slice::from_ref(&vector_sharding)),
            Err(ProgramError::InvalidInputCount { expected: 3, actual: 1 }.into()),
        );
        assert_eq!(
            boundary_mapping
                .executable_optional_input_shardings(&[None, Some(vector_sharding.clone()), None])
                .unwrap(),
            vec![Some(vector_sharding), None, Some(scalar_sharding), None],
        );
        assert_eq!(
            boundary_mapping.executable_optional_input_shardings(&[None]),
            Err(ProgramError::InvalidInputCount { expected: 3, actual: 1 }.into()),
        );
    }

    #[test]
    fn test_mlir_program_boundary_mapping_executable_optional_output_shardings() {
        let mesh = test_manual_mesh("x", 2);
        let dynamic = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", Some(3))]));
        let boundary_mapping = MlirProgramBoundaryMapping::new(&[], &[dynamic.clone(), dynamic]);
        let vector_sharding = Sharding::replicated(mesh.clone(), 1);
        assert_eq!(
            boundary_mapping
                .executable_optional_output_shardings(&[Some(vector_sharding.clone()), None])
                .unwrap(),
            vec![Some(vector_sharding), None, Some(Sharding::replicated(mesh, 0)), None],
        );
        assert_eq!(
            boundary_mapping.executable_optional_output_shardings(&[None]),
            Err(ProgramError::InvalidOutputCount { expected: 2, actual: 1 }.into()),
        );
    }

    #[test]
    fn test_mlir_program_boundary_mapping_select_input_entries() {
        // The erased static zero has no executable slot, and the hidden extent input of the dynamic zero is not an
        // entry of the selection.
        let static_zero = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let dynamic_zero = ArrayType::new(DataType::Zero, Shape::new(vec![dynamic_dimension("rows", Some(3))]));
        let boolean = ArrayType::new_static(DataType::Boolean, [3]);
        let boundary_mapping = MlirProgramBoundaryMapping::new(&[static_zero, boolean, dynamic_zero], &[]);
        assert_eq!(boundary_mapping.select_input_entries(&[0, 1, 2]).unwrap(), vec![1, 2]);
        assert_eq!(
            boundary_mapping.select_input_entries(&[0, 1]),
            Err(ProgramError::InvalidInputCount { expected: 3, actual: 2 }.into()),
        );
    }

    #[test]
    fn test_mlir_program_boundary_mapping_select_output_entries() {
        let static_zero = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let dynamic_zero = ArrayType::new(DataType::Zero, Shape::new(vec![dynamic_dimension("rows", Some(3))]));
        let boundary_mapping = MlirProgramBoundaryMapping::new(&[], &[dynamic_zero, static_zero]);
        assert_eq!(boundary_mapping.select_output_entries(&[0, 1]).unwrap(), vec![0]);
        assert_eq!(
            boundary_mapping.select_output_entries(&[0, 1, 2]),
            Err(ProgramError::InvalidOutputCount { expected: 2, actual: 3 }.into()),
        );
    }

    #[test]
    fn test_mlir_program_boundary_mapping_require_assertion_handler() {
        let mut boundary_mapping = MlirProgramBoundaryMapping::new(&[], &[]);
        assert!(!boundary_mapping.requires_assertion_handler());
        boundary_mapping.require_assertion_handler();
        assert!(boundary_mapping.requires_assertion_handler());
        assert!(boundary_mapping.has_effects());
    }

    #[test]
    fn test_program_lowering_options_exact_sharding_accessors() {
        let options = ProgramLoweringOptions::new();
        assert_eq!(options.exact_input_shardings(), None);
        assert_eq!(options.exact_output_shardings(), None);

        let options = options.without_boundary_shardings();
        assert_eq!(options.exact_input_shardings(), Some(&EntryShardingOverride::Unannotated));
        assert_eq!(options.exact_output_shardings(), Some(&EntryShardingOverride::Unannotated));

        let shardings = vec![Some(Sharding::replicated(LogicalMesh::single_device(), 0)), None];
        let options = options.with_exact_input_shardings(shardings.clone()).with_exact_output_shardings(vec![None]);
        assert_eq!(options.exact_input_shardings(), Some(&EntryShardingOverride::Values(shardings)));
        assert_eq!(options.exact_output_shardings(), Some(&EntryShardingOverride::Values(vec![None])));
    }

    #[test]
    fn test_program_to_mlir_omits_shardy_for_targets_without_the_dialect() {
        // A device target that completes a pending reduction over a mesh axis of size one constrains its output, and
        // the signature shardings annotate the entry function. A target whose plugin lacks the Shardy dialect gets
        // neither, nor the mesh that both refer to.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let input_type = test_vector_type(4).with_sharding(unreduced.clone()).unwrap();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(PlaceOperation::new(Device::new(0, 0), None), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let output_types = program
            .output_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned().unwrap())
            .collect::<Vec<_>>();
        let lower = |supports_shardy| {
            program
                .to_mlir(
                    "main",
                    &ProgramLoweringOptions::new()
                        .without_pruning()
                        .with_input_types(&vec![input_type.clone()])
                        .with_output_types(&output_types)
                        .without_boundary_shardings()
                        .with_exact_input_shardings(vec![Some(unreduced.clone())])
                        .with_exact_output_shardings(vec![Some(replicated.clone())])
                        .with_target_platform(Some("cpu"))
                        .with_shardy_support(supports_shardy)
                        .with_reference_bindings(Vec::new(), Vec::new(), Vec::new()),
                )
                .unwrap()
        };
        let lowered = lower(true);
        assert_eq!(lowered.mesh(), Some(&mesh));
        assert_eq!(
            lowered.into_stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced={"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={"x"}>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
        let lowered = lower(false);
        assert_eq!(lowered.mesh(), Some(&mesh));
        assert_eq!(
            lowered.into_stable_hlo(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    return %arg0 : tensor<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_mlir_rejects_manual_computations_for_targets_without_shardy() {
        // The ordered print keeps the manual computation live even though its output is unused.
        let (program, sharding) = traced_effectful_dead_shard_map();
        assert!(matches!(
            program.to_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .with_input_shardings(vec![sharding.clone()])
                    .with_output_shardings(vec![sharding])
                    .with_shardy_support(false),
            ),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                if operation == "sdy.manual_computation" && reason == "the target plugin cannot compile with Shardy",
        ));
    }

    #[test]
    fn test_program_to_mlir_rejects_unresolved_reference_state() {
        assert_eq!(token_threaded_effects(EffectClasses::single(EffectClass::OrderedState)).next(), None);

        // The reference scan over the whole artifact catches the reference atoms and intrinsic semantics before token
        // threading, so two representative reference operations are enough to pin the diagnostic.
        let array_type = ArrayType::scalar(DataType::F32);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(Type::Array(array_type.clone()));
        let reference = builder
            .add_instruction(Operation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        builder
            .add_instruction(
                Operation::ReferenceWrite(ReferenceWriteOperation::new()),
                Vec::new(),
                vec![reference, input],
                None,
            )
            .unwrap();
        let output = builder
            .add_instruction(
                Operation::ReferenceFreeze(ReferenceFreezeOperation::new()),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The reference scan runs before the generic state scan at the module entry, so a program whose state comes
        // from reference operations receives the dedicated reference diagnostic. The state error remains behind it for
        // future state that does not come from references, and `token_threaded_effects` above pins that `OrderedState`
        // has no token slot either way.
        assert!(matches!(
            program.to_mlir("main", &ProgramLoweringOptions::new()),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "program",
        ));
    }

    #[test]
    fn test_program_to_mlir_rejects_preserved_reference_entries() {
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference = builder.add_input(Type::Reference(ReferenceType::new(scalar_type.clone())));
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let contract = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadOnly)]);
        validate_kernel_body(program.entry_region_ref(), &contract).unwrap();
        assert!(matches!(
            program.to_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .with_input_types(&vec![scalar_type.clone()])
                    .with_output_types(&vec![scalar_type]),
            ),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "program",
        ));
    }

    #[test]
    fn test_program_to_mlir_with_reference_states() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let state_type = ArrayType::new_static(DataType::F32, [4]);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let _zero = builder.add_input(Type::Array(zero_type.clone()));
        let mutated = builder.add_input(Type::Array(state_type.clone()));
        let _read_only = builder.add_input(Type::Array(state_type.clone()));
        let program: FlatProgram<Session> =
            builder.build(vec![mutated], vec![Placeholder; 3], vec![Placeholder]).unwrap();
        let input_types = vec![zero_type.clone(), state_type.clone(), state_type.clone()];
        let output_types = vec![state_type.clone()];
        let input_shardings = vec![
            Sharding::replicated(mesh.clone(), 1),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        ];
        let output_shardings = vec![Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()];
        let states = [
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 1 }, Some(0)),
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 2 }, None),
        ];

        let lowered = program
            .to_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings()
                    .with_exact_input_shardings(input_shardings.iter().cloned().map(Some).collect())
                    .with_exact_output_shardings(output_shardings.iter().cloned().map(Some).collect())
                    .with_reference_bindings(states.to_vec(), Vec::new(), Vec::new()),
            )
            .unwrap();
        assert_eq!(lowered.boundary_mapping().input_mapping(), &[None, Some(0), Some(1)]);
        assert_eq!(lowered.boundary_mapping().output_mapping(), &[Some(0)]);
        let expected_signature = concat!(
            "  func.func @main(",
            "%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>, ",
            "tf.aliasing_output = 0 : i64}, ",
            "%arg1: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> ",
            "(tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {",
        );
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                @SIGNATURE@
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                    return %arg0 : tensor<4xf32>
                  }
                }
            "#}
            .replace("@SIGNATURE@", expected_signature),
        );

        let invalid_state = ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None);
        assert!(matches!(
            program.to_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings()
                    .with_exact_input_shardings(input_shardings.iter().cloned().map(Some).collect())
                    .with_exact_output_shardings(output_shardings.iter().cloned().map(Some).collect())
                    .with_reference_bindings((std::slice::from_ref(&invalid_state)).to_vec(), Vec::new(), Vec::new()),
            ),
            Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                if message == "logical state input 0 is erased from the executable boundary",
        ));

        let asymmetric_sharding_message =
            "reference-state aliases require both input and output sharding metadata or neither";
        let options = ProgramLoweringOptions::new()
            .without_pruning()
            .with_input_types(&input_types)
            .with_output_types(&output_types)
            .without_boundary_shardings()
            .with_reference_bindings(states.to_vec(), Vec::new(), Vec::new());
        for options in [
            options.clone().with_exact_input_shardings(input_shardings.iter().cloned().map(Some).collect()),
            options.with_exact_output_shardings(output_shardings.iter().cloned().map(Some).collect()),
        ] {
            assert!(matches!(
                program.to_mlir("main", &options),
                Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                    if message == asymmetric_sharding_message,
            ));
        }

        let mismatched_output_shardings = vec![Sharding::replicated(mesh, 1)];
        assert!(matches!(
            program.to_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings()
                    .with_exact_input_shardings(input_shardings.iter().cloned().map(Some).collect())
                    .with_exact_output_shardings(mismatched_output_shardings.iter().cloned().map(Some).collect())
                    .with_reference_bindings(states.to_vec(), Vec::new(), Vec::new()),
            ),
            Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                if message == "state input 1 and output 0 must use the same sharding",
        ));
    }

    #[test]
    fn test_program_to_mlir_with_reference_states_rejects_bounded_dynamic_aliases() {
        let extent = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let state_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let state = builder.add_input(Type::Array(state_type.clone()));
        let program: FlatProgram<Session> = builder.build(vec![state], vec![Placeholder], vec![Placeholder]).unwrap();
        let reference_state = ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(0));

        assert!(matches!(
            program.to_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&vec![state_type.clone()])
                    .with_output_types(&vec![state_type])
                    .without_boundary_shardings()
                    .with_exact_input_shardings(vec![Some(sharding.clone())])
                    .with_exact_output_shardings(vec![Some(sharding.clone())])
                    .with_reference_bindings((std::slice::from_ref(&reference_state)).to_vec(), Vec::new(), Vec::new()),
            ),
            Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                if message == "state input 0 and output 0 must be static because bounded-dynamic mutation alias \
                               compatibility is unsupported",
        ));
    }

    #[test]
    fn test_program_to_mlir_aliases_donated_inputs() {
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let pair_type = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let first = builder.add_input(Type::Array(vector_type.clone()));
        let second = builder.add_input(Type::Array(vector_type.clone()));
        let third = builder.add_input(Type::Array(pair_type.clone()));
        let program: FlatProgram<Session> =
            builder.build(vec![second, first, third], vec![Placeholder; 3], vec![Placeholder; 3]).unwrap();
        let input_types = vec![vector_type.clone(), vector_type.clone(), pair_type.clone()];
        let output_types = vec![vector_type.clone(), vector_type.clone(), pair_type.clone()];
        let lower_with =
            |shardings: Option<(&[Sharding], &[Sharding])>, donated_inputs: &[bool], open_outputs: &[bool]| {
                let options = ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .with_reference_bindings(Vec::new(), donated_inputs.to_vec(), Vec::new())
                    .with_open_outputs(open_outputs.to_vec());
                let options = match shardings {
                    Some((inputs, outputs)) => options
                        .with_exact_input_shardings(inputs.iter().cloned().map(Some).collect())
                        .with_exact_output_shardings(outputs.iter().cloned().map(Some).collect()),
                    None => options.without_boundary_shardings(),
                };
                program.to_mlir("main", &options)
            };
        let lower = |shardings: Option<&[Sharding]>, donated_inputs: &[bool], open_outputs: &[bool]| {
            lower_with(shardings.map(|shardings| (shardings, shardings)), donated_inputs, open_outputs)
        };
        let signature = |lowered: MlirProgram| {
            let line = lowered.stable_hlo().lines().find(|line| line.contains("func.func @main")).unwrap();
            line.trim().to_string()
        };
        // Pairs of the index of each aliased input and the index of its output.
        let aliases = |lowered: MlirProgram| {
            signature(lowered)
                .split("%arg")
                .skip(1)
                .enumerate()
                .filter_map(|(index, argument)| {
                    let (_, alias) = argument.split_once("tf.aliasing_output = ")?;
                    Some((index, alias.split(' ').next().unwrap().parse::<usize>().unwrap()))
                })
                .collect::<Vec<_>>()
        };

        // Each donated input is paired with the first unpaired output of its type, and other inputs with none.
        assert_eq!(
            signature(lower(None, &[true, true, false], &[]).unwrap()),
            concat!(
                "func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 0 : i64}, ",
                "%arg1: tensor<4xf32> {tf.aliasing_output = 1 : i64}, %arg2: tensor<2xf32>) -> ",
                "(tensor<4xf32>, tensor<4xf32>, tensor<2xf32>) {",
            ),
        );
        assert_eq!(
            signature(lower(None, &[false, true, true], &[]).unwrap()),
            concat!(
                "func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32> {tf.aliasing_output = 0 : i64}, ",
                "%arg2: tensor<2xf32> {tf.aliasing_output = 2 : i64}) -> ",
                "(tensor<4xf32>, tensor<4xf32>, tensor<2xf32>) {",
            ),
        );
        assert!(!signature(lower(None, &[], &[]).unwrap()).contains("tf.aliasing_output"));

        // An open output is only paired on a single device, where the compiler cannot place it differently.
        let shardings = |device_count: usize| {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", device_count, MeshAxisType::Auto).unwrap()]).unwrap();
            vec![Sharding::replicated(mesh, 1); 3]
        };
        let open_outputs = [true, false, false];
        let single_device = signature(lower(Some(&shardings(1)), &[true, false, false], &open_outputs).unwrap());
        assert!(single_device.contains("tf.aliasing_output = 0 : i64"));
        let two_devices = signature(lower(Some(&shardings(2)), &[true, false, false], &open_outputs).unwrap());
        assert!(two_devices.contains("tf.aliasing_output = 1 : i64"));

        // An input that finds no output of its own sharding is paired with an output under which every device holds
        // shards of the same type, such as one that completes its pending sum or splits it along another axis of the
        // same size, and only after every donated input was paired with an output of its own sharding where one is
        // free. Before, only an output of the identical sharding was paired, so the first two cases had no alias.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let split = |axes: &[&str]| {
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(axes.iter().copied())]).unwrap()
        };
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let pending = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let cases = [
            ("completed sum", [pending, replicated.clone()], [replicated.clone(), replicated.clone()], [true, false]),
            ("other axis", [split(&["x"]), replicated.clone()], [replicated.clone(), split(&["y"])], [true, false]),
            (
                "other extents",
                [split(&["x"]), replicated.clone()],
                [replicated.clone(), split(&["x", "y"])],
                [true, false],
            ),
            ("own sharding first", [split(&["x"]), split(&["x"])], [split(&["y"]), split(&["x"])], [true, true]),
        ];
        let outcomes = cases
            .into_iter()
            .map(|(name, [first_input, second_input], [first_output, second_output], [first, second])| {
                let input_shardings = [first_input, second_input, replicated.clone()];
                let output_shardings = [first_output, second_output, replicated.clone()];
                let donated_inputs = [first, second, false];
                let lowered = lower_with(Some((&input_shardings, &output_shardings)), &donated_inputs, &[]);
                (name, aliases(lowered.unwrap()))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            outcomes,
            [
                ("completed sum", vec![(0, 0)]),
                ("other axis", vec![(0, 1)]),
                ("other extents", vec![]),
                ("own sharding first", vec![(0, 1), (1, 0)]),
            ],
        );

        assert!(matches!(
            lower(None, &[true], &[]),
            Err(Error::Execution(ExecutionError::DonationFlagCountMismatch { expected_count: 3, actual_count: 1 })),
        ));
    }

    #[test]
    fn test_program_to_mlir_declares_the_mesh_of_lowered_shardings() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 2);
        let input_type = test_vector_type(4);
        let mut builder = ProgramBuilder::<ArrayConstant, ArrayOperation<ArrayConstant>>::new();
        let input = builder.add_input(input_type.clone());
        builder
            .add_instruction(
                ReshapeOperation::new(vec![4], vec![2, 2]).with_output_sharding(sharding),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![Placeholder], vec![])
            .unwrap()
            .into_unprojected::<Constant, Operation>()
            .unwrap();

        // The mesh exists only on an internal value. The module still declares the symbol that the constraint of the
        // reshape refers to, even though neither the signature nor a sharding operation supplies it.
        let lowered = program.to_mlir("main", &ProgramLoweringOptions::new().without_pruning()).unwrap();
        assert_eq!(lowered.mesh(), Some(&mesh));
        assert_eq!(
            lowered.into_stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) {
                    %0 = stablehlo.reshape %arg0 : (tensor<4xf32>) -> tensor<2x2xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}, {}], replicated={"x"}> : tensor<2x2xf32>
                    return
                  }
                }
            "#},
        );

        // The sharding of a type alone refers to no mesh, so a module in which no sharding is lowered declares none.
        let mut builder = ProgramBuilder::<ArrayConstant, ArrayOperation<ArrayConstant>>::new();
        builder.add_input(input_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap());
        let metadata_only = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![Placeholder], vec![])
            .unwrap()
            .into_unprojected::<Constant, Operation>()
            .unwrap();
        let lowered =
            metadata_only.to_mlir("main", &ProgramLoweringOptions::new().without_boundary_shardings()).unwrap();
        assert_eq!(lowered.mesh(), None);
        assert_eq!(
            lowered.into_stable_hlo(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>) {
                    return
                  }
                }
            "},
        );

        // The mesh of a cast that changes the explicit placement of its input can be the only mesh that the module
        // refers to, including when the output of the cast is internal.
        let output_type = input_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        for operation in [
            CastOperation::<Type>::checked(output_type.clone().into(), vec![input_type.clone().into()])
                .unwrap()
                .into(),
            Operation::Array(ArrayOperation::Cast(
                CastOperation::checked(output_type, vec![input_type.clone()]).unwrap(),
            )),
        ] {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(input_type.clone().into());
            builder.add_instruction(operation, vec![], vec![input], None).unwrap();
            let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![], vec![Placeholder], vec![]).unwrap();
            let lowered = program.to_mlir("main", &ProgramLoweringOptions::new()).unwrap();
            assert_eq!(lowered.mesh(), Some(&mesh));
            assert_eq!(
                lowered.into_stable_hlo(),
                indoc! {r#"
                    module {
                      sdy.mesh @mesh = <["x"=2]>
                      func.func @main(%arg0: tensor<4xf32>) {
                        %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                        return
                      }
                    }
                "#},
            );
        }
    }

    #[test]
    fn test_program_to_mlir_requires_one_mesh_topology() {
        let mesh_2x2 = |names: [&str; 2], r#type: MeshAxisType| {
            LogicalMesh::new(names.map(|name| MeshAxis::new(name, 2, r#type).unwrap()).to_vec()).unwrap()
        };
        let mesh = mesh_2x2(["x", "y"], MeshAxisType::Explicit);
        let input_type = test_vector_type(4);
        let mut builder = ProgramBuilder::<ArrayConstant, ArrayOperation<ArrayConstant>>::new();
        let input = builder.add_input(input_type.clone());
        builder
            .add_instruction(
                ReshapeOperation::new(vec![4], vec![2, 2]).with_output_sharding(Sharding::replicated(mesh.clone(), 2)),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![], vec![Placeholder], vec![])
            .unwrap()
            .into_unprojected::<Constant, Operation>()
            .unwrap();
        // The signature places the input over `signature_mesh`, while the constraint of the reshape refers to `mesh`.
        let lower = |signature_mesh: &LogicalMesh| {
            program.to_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&vec![input_type.clone()])
                    .with_output_types(&Vec::<ArrayType>::new())
                    .without_boundary_shardings()
                    .with_exact_input_shardings(vec![Some(Sharding::replicated(signature_mesh.clone(), 1))]),
            )
        };

        // Device ids follow the order of the axes of the module mesh, so a reordered mesh is another topology.
        let reordered = mesh_2x2(["y", "x"], MeshAxisType::Explicit);
        assert!(matches!(
            lower(&reordered),
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == reordered && actual == mesh,
        ));
        assert_eq!(
            LoweringError::MismatchedMeshTopology { expected: reordered, actual: mesh.clone() }.to_string(),
            "mesh `['x'=2:explicit, 'y'=2:explicit]` does not have the topology of the module mesh \
             `['y'=2:explicit, 'x'=2:explicit]`",
        );

        // Meshes with other axes are not merged into a union.
        let other_axes = mesh_2x2(["x", "z"], MeshAxisType::Explicit);
        assert!(matches!(
            lower(&other_axes),
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == other_axes && actual == mesh,
        ));

        // A mesh with other axis sizes is another topology as well.
        let other_sizes = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        assert!(matches!(
            lower(&other_sizes),
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == other_sizes && actual == mesh,
        ));

        // Meshes that differ only in axis types share the module mesh, which keeps the axis types of the first mesh
        // that lowering refers to, which is that of the signature.
        let manual = mesh_2x2(["x", "y"], MeshAxisType::Manual);
        let lowered = lower(&manual).unwrap();
        assert_eq!(lowered.mesh(), Some(&manual));
        assert_eq!(
            lowered.into_stable_hlo(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2, \"y\"=2]>
                  func.func @main(%arg0: tensor<4xf32> \
                      {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={\"x\", \"y\"}>}) {
                    %0 = stablehlo.reshape %arg0 : (tensor<4xf32>) -> tensor<2x2xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}, {}], replicated={\"x\", \"y\"}> : tensor<2x2xf32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_mlir_declares_no_mesh_for_bodies_that_the_target_replaces() {
        // A named composite call whose body places its input on another mesh refers to that mesh where lowering emits
        // the body, and to none on a target whose backend implementation replaces the whole call
        // (`ryft.tests.xla_named_square` version 1 on CUDA, refer to `omits_regions`), which never lowers the body,
        // as the validation of placements skips it.
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let other = LogicalMesh::new(vec![MeshAxis::new("other", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_devices = DeviceMesh::new(other.clone(), vec![Device::new(2, 0), Device::new(3, 0)]).unwrap();
        let placement = Placement::new(other_devices, Sharding::replicated(other.clone(), 1)).unwrap();
        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let operation = PlaceOperation::new(PlacementRequest::from(placement), None);
            let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
        };
        let key = CompositeKey::new("ryft.tests.xla_named_square".to_string(), 1);
        let named = NamedComposite::new(key, |types: &[Type]| Ok(types.to_vec()), |inputs| Ok(vec![inputs[0].clone()]));
        let rules = CustomRuleRegistration::new(CustomRuleDefinition::new("placing").with_named_composite(named));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(vector_type.into());
        let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
        let regions = vec![builder.intern_callee(&body, None).unwrap()];
        let output = builder.add_instruction(operation, regions, vec![input], None).unwrap()[0];
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder]);
        let program = program.unwrap();
        // The signature is lowered without shardings, so that only the body can refer to a mesh.
        let lower = |target_platform| {
            let options = ProgramLoweringOptions::new().with_target_platform(Some(target_platform));
            program.to_mlir("main", &options.without_boundary_shardings()).unwrap()
        };
        let lowered = lower("cuda");
        assert_eq!(lowered.mesh(), None);
        assert_eq!(
            lowered.into_stable_hlo(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );
        let lowered = lower("cpu");
        assert_eq!(lowered.mesh(), Some(&other));
        assert_eq!(
            lowered.into_stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["other"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"other"}> : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_mlir_declares_the_mesh_of_call_boundaries() {
        // A call constrains its output to the override of the boundary of its callee, which no type records because the
        // override is over an auto mesh axis, so the module declares the mesh of the override even when it is lowered
        // without signature shardings, both for a static program and for one with a bounded dynamic value. The call
        // also constrains its input replicated over that mesh, as an execution of its callee, whose input type has no
        // sharding, places its input (refer to `CallOperation::boundary_input_sharding`).
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let lower = |array_type: ArrayType| {
            let callee = {
                let mut builder = ProgramBuilder::<Constant, Operation>::new();
                let input = builder.add_input(array_type.clone().into());
                Arc::new(
                    builder
                        .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
                        .unwrap(),
                )
            };
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(array_type.clone().into());
            let callee_region = builder.intern_callee(&callee, None).unwrap();
            let call =
                CallOperation::new(0).with_boundary(Some(&device_mesh), Vec::new(), vec![Some(replicated.clone())]);
            let output =
                builder.add_instruction(Operation::Call(call), vec![callee_region], vec![input], None).unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            let types = vec![array_type];
            let lowered = program
                .to_mlir(
                    "main",
                    &ProgramLoweringOptions::new()
                        .with_input_types(&types)
                        .with_output_types(&types)
                        .without_boundary_shardings(),
                )
                .unwrap();
            assert_eq!(lowered.mesh(), Some(&mesh));
            lowered.into_stable_hlo()
        };
        assert_eq!(
            lower(test_vector_type(4)),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}]> : tensor<4xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}]> : tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#},
        );
        let length = DimensionVariable::new("length", DimensionBounds::new(1, Some(4)).unwrap());
        assert_eq!(
            lower(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(length)]))),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<i32>) -> (tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<3xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<3>>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}]> : tensor<?xf32, #stablehlo.bounds<3>>
                    %2 = sdy.sharding_constraint %1 <@mesh, [{}]> : tensor<?xf32, #stablehlo.bounds<3>>
                    %3 = stablehlo.get_dimension_size %2, dim = 0 : (tensor<?xf32, #stablehlo.bounds<3>>) -> tensor<i32>
                    %4 = stablehlo.convert %3 : (tensor<i32>) -> tensor<i64>
                    return %2, %4 : tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_mlir_emits_the_decompositions_that_lowered_scaled_dots_request() {
        // A scaled dot in the body of a scan requests its decomposition when the length of the scan admits an
        // iteration. When its length is zero, lowering emits nothing of the body (refer to `omits_regions`), so
        // nothing requests the decomposition and the module holds no private function.
        let element_type = ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![2.into(), 16.into()]));
        let scale_type = ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![2.into(), 1.into()]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![2.into(), 2.into()]));
        let carry_types = [output_type, element_type.clone(), element_type, scale_type.clone(), scale_type];
        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carries = carry_types.iter().map(|r#type| builder.add_input(r#type.clone().into())).collect::<Vec<_>>();
            let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
            let operation = ScaledDotOperation::new(dimensions, DataType::F32, true, true);
            let output = builder.add_instruction(operation, Vec::new(), carries[1..].to_vec(), None).unwrap()[0];
            let outputs = std::iter::once(output).chain(carries[1..].iter().copied()).collect();
            Arc::new(builder.build(outputs, vec![Placeholder; 6], vec![Placeholder; 5]).unwrap())
        };
        let private_functions = |length: usize| {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let inputs = carry_types.iter().map(|r#type| builder.add_input(r#type.clone().into())).collect();
            let regions = vec![builder.intern_callee(&body, None).unwrap()];
            let scan = ScanOperation::<Type>::new(5, length);
            let outputs = builder.add_instruction(scan, regions, inputs, None).unwrap().to_vec();
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 5], vec![Placeholder; 5])
                .unwrap();
            let lowered = program.to_mlir("main", &ProgramLoweringOptions::new()).unwrap().into_stable_hlo();
            lowered
                .lines()
                .filter_map(|line| line.trim_start().strip_prefix("func.func private @"))
                .map(|line| line.split('(').next().unwrap().to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(private_functions(0), Vec::<String>::new());
        assert_eq!(private_functions(1), vec!["xla.scaled_dot".to_string()]);
    }

    #[test]
    fn test_program_to_rendered_mlir() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(move |x: ArrayTracer| x.sin().unwrap().reshard(&sharding).unwrap(), test_vector_type(8)).unwrap();
        assert_eq!(
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> \
                        (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = stablehlo.sine %arg0 : tensor<8xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{\"x\"}]> : tensor<8xf32>
                    return %1 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_normalizes_function_names() {
        let program = trace(|input: ArrayTracer| input, ArrayType::scalar(DataType::F32)).unwrap();
        let options = ProgramLoweringOptions::new();
        let expected = program.to_rendered_mlir("main", &options).unwrap();
        for function_name in [" main ", " @main "] {
            assert_eq!(program.to_rendered_mlir(function_name, &options).unwrap(), expected);
        }
        for (function_name, normalized_name) in [(" @ ", ""), ("main function", "main function")] {
            assert_eq!(
                program.to_rendered_mlir(function_name, &options),
                Err(Error::Lowering(LoweringError::InvalidFunctionName { function_name: normalized_name.to_string() })),
            );
        }
    }

    #[test]
    fn test_program_to_rendered_mlir_lowers_manual_computations() {
        let global_input_type = test_vector_type(8);
        let mesh = test_manual_mesh("x", 4);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x| Ok(x.clone() + x),
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap(),
            ShardMapOptions::new(),
        )
        .unwrap();

        assert_eq!(
            module,
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
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_lowers_scalar_function() {
        let scalar = ArrayType::scalar(DataType::F64);
        let program: TracedProgram<(ArrayType, ArrayType), ArrayType> =
            trace(scalar_bilinear_sin, (scalar.clone(), scalar)).unwrap();
        let stablehlo = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f64>
                    %1 = stablehlo.sine %arg0 : tensor<f64>
                    %2 = stablehlo.add %0, %1 : tensor<f64>
                    return %2 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_lowers_gradients() {
        let program: TracedProgram<ArrayType, ArrayType> = trace(
            |x: ArrayTracer| {
                differentiate_in_parent_at(x)
                    .gradient(|x| Ok(x.clone() * x.clone() * x.clone() * x.clone() + x.sin()?))
                    .unwrap()
            },
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        let stablehlo = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();

        // The gradient computes one cosine for the derivative of `sin(x)`, expands the `x^4` chain rule into
        // multiplications, accumulates the cotangent contributions with additions, and contains no sine.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.cosine %arg0 : tensor<f64>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %1 = stablehlo.multiply %0, %cst : tensor<f64>
                    %2 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
                    %3 = stablehlo.multiply %2, %arg0 : tensor<f64>
                    %4 = stablehlo.multiply %3, %cst : tensor<f64>
                    %5 = stablehlo.add %1, %4 : tensor<f64>
                    %6 = stablehlo.multiply %arg0, %cst : tensor<f64>
                    %7 = stablehlo.multiply %2, %6 : tensor<f64>
                    %8 = stablehlo.add %5, %7 : tensor<f64>
                    %9 = stablehlo.multiply %arg0, %6 : tensor<f64>
                    %10 = stablehlo.multiply %arg0, %9 : tensor<f64>
                    %11 = stablehlo.add %8, %10 : tensor<f64>
                    %12 = stablehlo.multiply %arg0, %9 : tensor<f64>
                    %13 = stablehlo.add %11, %12 : tensor<f64>
                    return %13 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_lowers_standalone_pullbacks() {
        // Linearize at concrete values and lower the standalone transposed program. Its public inputs are the
        // output cotangent followed by the residuals saved at that point (i.e., the factors cos(x), x, and y).
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let scalar = |value: f64| replicated_array(&session, &mesh, DataType::F64, &[], &value.to_ne_bytes());
        let (_, pullback) = differentiate_in_parent_at((scalar(2.0), scalar(3.0)))
            .vjp(|inputs| Ok(scalar_bilinear_sin(inputs)))
            .unwrap();
        let pullback = pullback.inner().transposed_program(&[]).unwrap().into_program();
        let stablehlo = pullback
            .to_rendered_mlir("main", &ProgramLoweringOptions::new().without_boundary_shardings())
            .unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
                    %0 = stablehlo.multiply %arg3, %arg0 : tensor<f64>
                    %1 = stablehlo.multiply %arg1, %arg0 : tensor<f64>
                    %2 = stablehlo.add %0, %1 : tensor<f64>
                    %3 = stablehlo.multiply %arg2, %arg0 : tensor<f64>
                    return %2, %3 : tensor<f64>, tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_lowers_staged_gradients() {
        let scalar = ArrayType::scalar(DataType::F64);
        let program: TracedProgram<(ArrayType, ArrayType), (ArrayType, ArrayType)> = trace(
            |inputs: (ArrayTracer, ArrayTracer)| {
                differentiate_in_parent_at(inputs).gradient(|inputs| Ok(scalar_bilinear_sin(inputs))).unwrap()
            },
            (scalar.clone(), scalar),
        )
        .unwrap();
        let stablehlo = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();

        // `cos(x)` is computed symbolically from `%arg0` rather than baked in as a constant, both inputs are
        // referenced, and the gradient contains no sine, because the derivative of sine is cosine.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
                    %0 = stablehlo.cosine %arg0 : tensor<f64>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %1 = stablehlo.multiply %0, %cst : tensor<f64>
                    %2 = stablehlo.multiply %arg1, %cst : tensor<f64>
                    %3 = stablehlo.add %1, %2 : tensor<f64>
                    %4 = stablehlo.multiply %arg0, %cst : tensor<f64>
                    return %3, %4 : tensor<f64>, tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_renders_provenance_locations() {
        let global_input_type = test_vector_type(8);
        let mesh = test_manual_mesh("x", 4);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x| {
                let context = x.value().context().clone();

                // One instruction staged under nested scopes, one under a fused origin, and one with unknown
                // provenance, covering every provenance shape in one lowered module.
                let scoped = context.invoke_with_provenance_scope(ProvenanceScope::new("outer"), || {
                    context.invoke_with_provenance_scope(ProvenanceScope::new("inner"), || x.clone() + x.clone())
                });
                let fused = Provenance::fused([
                    Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown()),
                    Provenance::scope(ProvenanceScope::new("b"), Provenance::unknown()),
                ]);
                let product = context.invoke_with_provenance_origin(fused, || scoped.clone() * scoped);
                Ok(product + x)
            },
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap(),
            ShardMapOptions::new(),
        )
        .unwrap();

        // The scoped addition carries the nested chain of name locations, the fused multiplication carries a fused
        // location without metadata, the addition with unknown provenance keeps the unknown base location, and the
        // module and function scaffolding keeps the base location. Locations render because modules that carry
        // provenance serialize with debug information enabled, in the parsable form that is not pretty printed.
        assert_eq!(
            module,
            indoc! {r#"
                #loc = loc(unknown)
                module {
                  sdy.mesh @mesh = <["x"=4]> loc(#loc)
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>} loc(unknown)) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32> loc(unknown)) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32> loc(#loc4)
                      %2 = stablehlo.multiply %1, %1 : tensor<2xf32> loc(#loc5)
                      %3 = stablehlo.add %2, %arg1 : tensor<2xf32> loc(#loc)
                      sdy.return %3 : tensor<2xf32> loc(#loc)
                    } : (tensor<8xf32>) -> tensor<8xf32> loc(#loc)
                    return %0 : tensor<8xf32> loc(#loc)
                  } loc(#loc)
                } loc(#loc)
                #loc1 = loc("inner")
                #loc2 = loc("a")
                #loc3 = loc("b")
                #loc4 = loc("outer"(#loc1))
                #loc5 = loc(fused[#loc2, #loc3])
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_renders_constants_and_supported_ops() {
        let global_input_type = test_matrix_type(4, 4);
        let mesh = test_manual_mesh("x", 2);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x| {
                let product = x.transpose(vec![1, 0])?.dot(&x, &DotDimensionNumbers::matmul())?;
                let waveform = (-product).cos()?.sin()?;
                Ok((waveform.clone() * waveform.one_like()?) + waveform.zero_like()?)
            },
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            ShardMapOptions::new(),
        )
        .unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2x4xf32>) {
                      %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
                      %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
                      %3 = stablehlo.negate %2 : tensor<4x4xf32>
                      %4 = stablehlo.cosine %3 : tensor<4x4xf32>
                      %5 = stablehlo.sine %4 : tensor<4x4xf32>
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                      %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                      %7 = sdy.sharding_constraint %6 <@mesh, [{}, {}]> : tensor<4x4xf32>
                      %8 = stablehlo.multiply %5, %7 : tensor<4x4xf32>
                      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                      %10 = sdy.sharding_constraint %9 <@mesh, [{}, {}]> : tensor<4x4xf32>
                      %11 = stablehlo.add %8, %10 : tensor<4x4xf32>
                      sdy.return %11 : tensor<4x4xf32>
                    } : (tensor<4x4xf32>) -> tensor<8x4xf32>
                    return %0 : tensor<8x4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_preserves_boundary_layouts() {
        let array_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        let zero_type = ArrayType::new_static(DataType::Zero, [2, 3]);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(array_type.clone().into());
        let zero = builder.add_input(zero_type.clone().into());
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![zero, input], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x3xf32> {mhlo.layout_mode = \"{0,1}\"}) \
                      -> (tensor<2x3xf32> {mhlo.layout_mode = \"{0,1}\"}) {
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<2x3xi1>
                    return %arg0 : tensor<2x3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_lowers_captures_as_hidden_inputs() {
        let array_type = test_vector_type(4);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(array_type.clone().into());
        let capture = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, capture], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let capture_types = vec![array_type.clone()];
        let input_types = vec![array_type.clone()];
        let output_types = vec![array_type];
        let stablehlo = program
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_capture_types(capture_types.to_vec())
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings(),
            )
            .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg1, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_lowers_immediate_dimension_constants_in_place() {
        // An immediate dimension constant carries its own extent from the host, so unlike a capture reference, it
        // materializes as a scalar `i64` `stablehlo.constant` in the function body instead of consuming a hidden
        // capture input. That is exactly what makes first-class extents usable where no capture table is reachable,
        // such as inside a `shard_map` manual region.
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let extent = builder.add_constant(Constant::Dimension(DimensionValue::new(extent_type, 4).unwrap()));
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![extent], None).unwrap()[0];
        let program =
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
        let stablehlo = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main() -> tensor<i64> {
                    %c = stablehlo.constant dense<4> : tensor<i64>
                    return %c : tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_lowers_capture_output_as_a_hidden_input() {
        let array_type = test_vector_type(4);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let output = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
        let program =
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
        let input_types: [ArrayType; 0] = [];
        let output_types = vec![array_type.clone()];
        let stablehlo = program
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_capture_types((std::slice::from_ref(&array_type)).to_vec())
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings(),
            )
            .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    return %arg0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_forwards_capture_into_nested_regions() {
        let array_type = test_vector_type(4);
        let branch = || {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let output = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap()
        };
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let true_region = builder.import_program(branch());
        let false_region = builder.import_program(branch());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let output = builder
            .add_instruction(
                Operation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let output_types = vec![array_type.clone()];
        let stablehlo = program
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_capture_types((std::slice::from_ref(&array_type)).to_vec())
                    .with_input_types(&[ArrayType::scalar(DataType::Boolean)])
                    .with_output_types(&output_types)
                    .without_boundary_shardings(),
            )
            .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<i1>) -> tensor<4xf32> {
                    %0 = "stablehlo.if"(%arg1) ({
                      stablehlo.return %arg0 : tensor<4xf32>
                    }, {
                      stablehlo.return %arg0 : tensor<4xf32>
                    }) : (tensor<i1>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_erases_static_zero_space_boundary() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(zero_type.clone().into());
        let program: FlatProgram<Session> = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let stablehlo = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main() {
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                    return
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_preserves_effects_that_produce_an_erased_output() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(zero_type.clone().into());
        let output = builder
            .add_instruction(PrintOperation::<ArrayType>::new("zero"), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program: FlatProgram<Session> = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main() {
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.print(%0, %1) {api_version = 4 : i32, \
                        backend_config = {label = \"zero\", logical_type = \"zero\"}, has_side_effect = true, \
                        operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], \
                        result_layouts = [dense<> : tensor<0xindex>]} : (tensor<3xi1>, !stablehlo.token) \
                        -> !stablehlo.token
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_preserves_dynamic_zero_space_shape_carrier() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![dynamic_dimension("zero", Some(3))]));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(zero_type.clone().into());
        let program: FlatProgram<Session> = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xi1>, %arg1: tensor<i32>) -> (tensor<?xi1, #stablehlo.bounds<2>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<2xi1>, tensor<i32>) -> tensor<?xi1, #stablehlo.bounds<2>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xi1, #stablehlo.bounds<2>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    return %0, %2 : tensor<?xi1, #stablehlo.bounds<2>>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_isolates_bounded_dynamic_outputs_behind_one_barrier() {
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("count", Some(4))]));
        let static_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let dynamic = builder.add_input(dynamic_type.clone().into());
        let r#static = builder.add_input(static_type.clone().into());
        let program: FlatProgram<Session> =
            builder.build(vec![dynamic, r#static, dynamic], vec![Placeholder; 2], vec![Placeholder; 3]).unwrap();
        let stablehlo = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();

        // Both dynamic outputs pass through the barrier, while the static output and the hidden extents bypass it.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<i32>) -> (tensor<?xf32, #stablehlo.bounds<3>>, tensor<2xf32>, tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<3xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<3>>
                    %1:2 = stablehlo.optimization_barrier %0, %0 : tensor<?xf32, #stablehlo.bounds<3>>, tensor<?xf32, #stablehlo.bounds<3>>
                    %2 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<3>>) -> tensor<i32>
                    %3 = stablehlo.convert %2 : (tensor<i32>) -> tensor<i64>
                    %4 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<3>>) -> tensor<i32>
                    %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i64>
                    return %1#0, %arg1, %1#1, %3, %5 : tensor<?xf32, #stablehlo.bounds<3>>, tensor<2xf32>, tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_declares_the_mesh_of_completed_device_reductions() {
        // A device target that completes a pending reduction over a mesh axis of size one constrains its output over
        // the mesh of its type, which neither the signature nor a payload names, so the module declares the mesh
        // because the lowered constraint refers to it.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let input_type = test_vector_type(4).with_sharding(unreduced).unwrap();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(PlaceOperation::new(Device::new(0, 0), None), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let output_types = program
            .output_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            program
                .to_rendered_mlir(
                    "main",
                    &ProgramLoweringOptions::new()
                        .with_input_types(&vec![input_type])
                        .with_output_types(&output_types)
                        .without_boundary_shardings()
                )
                .unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_merges_tracked_placement_into_sharding_constraint() {
        // The input is tracked as sharded over the explicit axis and the constraint places it over the auto axis, so
        // the emitted constraint must carry both placements rather than the constraint alone, which would contradict
        // the tracked type by marking the explicit axis replicated. The output is not the output of a `place`, so its
        // placement along the auto axis is left to the compiler, as compilation leaves it.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let tracked = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let constraint = test_sharding(&mesh, vec![ShardingDimension::sharded(["a"])], vec![]);
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            move |x: ArrayTracer| x.constrain_sharding(&constraint).unwrap(),
            test_vector_type(8).with_sharding(tracked).unwrap(),
        )
        .unwrap();
        assert_eq!(
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2, \"a\"=2]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                        (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\", ?}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{\"x\", \"a\"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_annotates_placed_outputs() {
        // An output that a `place` produces, also through an operation that returns its input unchanged, is annotated
        // with the target of that `place`, including its split along the auto axis that the output type drops, as
        // compilation places it.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let target = Placement::new(mesh, sharding).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(move |x: ArrayTracer| x.place(&target, None).unwrap().stop_gradient().unwrap(), test_vector_type(8))
                .unwrap();
        assert_eq!(
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<8xf32>) -> \
                        (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{\"x\"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_passes_imported_arrays_as_inputs() {
        // An array that the traced function imports becomes a session constant, which enters the module as a hidden
        // input that precedes the public inputs and is annotated as the array is placed, as compilation passes it.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 4f32.to_ne_bytes())
            .unwrap();
        let device = array.mesh().devices()[0];
        let traced = trace_with_context::<_, (), ArrayTracer>(
            |context, ()| {
                let imported =
                    ImportingContext::<Session, ThroughSession>::import(context, Value::Array(array.clone()))?;
                ValueProjection::<ArrayType>::into_projected(imported)?.place(&device, None)
            },
            (),
        )
        .unwrap();
        assert_eq!(
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"device\"=1]>
                  func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> \
                        (tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) {
                    return %arg0 : tensor<f32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_with_options() {
        // Explicit shardings are attached to the inputs and the outputs of the function. The global types have no
        // sharding here, so without explicit shardings the signature is not annotated, as `to_rendered_mlir` renders
        // it.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(|x: ArrayTracer| x.sin().unwrap(), test_vector_type(8)).unwrap();
        let shardings = vec![sharding.clone()];
        assert_eq!(
            traced
                .to_rendered_mlir(
                    "main",
                    &ProgramLoweringOptions::new()
                        .with_input_shardings(shardings.clone())
                        .with_output_shardings(shardings.clone())
                )
                .unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                        (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = stablehlo.sine %arg0 : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
        assert_eq!(
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = stablehlo.sine %arg0 : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
        assert_eq!(
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap()
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_rejects_wrong_open_output_flag_count() {
        let traced = trace(|input: ArrayTracer| (input.clone(), input), test_vector_type(4)).unwrap();
        for (flags, reason) in [
            (vec![true], "expected 2 open output flags or an empty list, but got 1"),
            (vec![true, false, true], "expected 2 open output flags or an empty list, but got 3"),
        ] {
            assert_eq!(
                traced.to_rendered_mlir("main", &ProgramLoweringOptions::new().with_open_outputs(flags)),
                Err(Error::Compilation(CompilationError::InvalidOptions { reason: reason.to_string() })),
            );
        }
    }

    #[test]
    fn test_program_to_rendered_mlir_rejects_boundaries_that_are_not_arrays() {
        // The export reads array types at the boundary of the program, so a program whose input is a dimension is
        // rejected before anything is lowered.
        let dimension_type = Type::Dimension(DimensionValue::constant(3).unwrap().r#type().into_owned());
        let (_, program) =
            DomainTracingContext::<Session>::trace(|inputs: Vec<Tracer>| Ok(inputs), vec![dimension_type]).unwrap();
        assert_eq!(
            program.to_rendered_mlir("main", &ProgramLoweringOptions::new()),
            Err(Error::from(ProgramError::from(TypeError::invalid("expected array type but got dimension type")))),
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_rejects_imports_over_other_topologies() {
        // A plain trace keeps the type of an array that it imports, so a signature over a mesh of another topology
        // cannot hold an array over two devices, and the error names the cause and the way forward.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1);
        let array_type = test_vector_type(4);
        let array = Array::from_host_buffer(
            &session,
            array_type.clone().with_sharding(sharding.clone()).unwrap(),
            None,
            Some(sharding),
            values_to_bytes(&[0f32, 1., 2., 3.]),
        )
        .unwrap();
        let traced = trace_with_context::<_, ArrayType, ArrayTracer>(
            |context, _| {
                let imported =
                    ImportingContext::<Session, ThroughSession>::import(context, Value::Array(array.clone()))?;
                Ok(ValueProjection::<ArrayType>::into_projected(imported)?)
            },
            array_type,
        )
        .unwrap();
        let signature = Sharding::replicated(LogicalMesh::single_device(), 1);
        let signatures = vec![signature.clone()];
        assert_eq!(
            traced.to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .with_input_shardings(signatures.clone())
                    .with_output_shardings(signatures.clone())
            ),
            Err(Error::from(ProgramError::InvalidArgument {
                message: "the traced program imports an array of type \
                          `f32[4][sharding={mesh<['x'=2:auto]>, [{}]}]`, whose mesh does not have the topology of the \
                          mesh `['device'=1:auto]` of the signature of the module, because a plain `trace` keeps the \
                          type of an array that it imports; stage or compile the function for a device mesh over \
                          `['device'=1:auto]` through a `Session` instead (e.g., with `stage` or `compile`), which \
                          receives the array on that mesh"
                    .to_string(),
            })),
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_closes_unconstrained_output_placements() {
        // The runtime placement of an output reads an unconstrained dimension of the target of the `place` that
        // produces it, or of an output sharding override, as replicated, so the output is annotated with that dimension
        // closed, as compilation annotates it, while the constraint inside the function keeps it open.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
                MeshAxis::new("r", 2, MeshAxisType::Auto).unwrap(),
            ])
            .unwrap(),
            (0..4).map(|id| Device::new(id, 0)).collect(),
        )
        .unwrap();
        let input_sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["r"])],
        )
        .unwrap();
        let open_sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::unconstrained()],
        )
        .unwrap();
        let target = Placement::new(mesh, open_sharding.clone()).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(move |x: ArrayTracer| x.place(&target, None).unwrap(), ArrayType::new_static(DataType::I32, [4, 4]))
                .unwrap();
        let expected = indoc! {"
            module {
              sdy.mesh @mesh = <[\"x\"=2, \"r\"=2]>
              func.func @main(%arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {\"r\"}]>}) -> \
                    (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>}) {
                %0 = sdy.sharding_constraint %arg0 <@mesh, [{\"x\"}, {?}]> : tensor<4x4xi32>
                return %0 : tensor<4x4xi32>
              }
            }
        "};
        assert_eq!(
            traced
                .to_rendered_mlir(
                    "main",
                    &ProgramLoweringOptions::new()
                        .with_input_shardings((std::slice::from_ref(&input_sharding)).to_vec())
                )
                .unwrap(),
            expected,
        );
        assert_eq!(
            traced
                .to_rendered_mlir(
                    "main",
                    &ProgramLoweringOptions::new()
                        .with_input_shardings((std::slice::from_ref(&input_sharding)).to_vec())
                        .with_output_shardings((std::slice::from_ref(&open_sharding)).to_vec())
                )
                .unwrap(),
            expected,
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_closes_unconstrained_output_types() {
        // An output that is computed from the output of a `place` inherits the unconstrained dimension of its target in
        // its type, which its runtime placement reads as replicated as well, so it is annotated with that dimension
        // closed, as compilation annotates it.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
                MeshAxis::new("r", 2, MeshAxisType::Auto).unwrap(),
            ])
            .unwrap(),
            (0..4).map(|id| Device::new(id, 0)).collect(),
        )
        .unwrap();
        let input_sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let open_sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::unconstrained()],
        )
        .unwrap();
        let target = Placement::new(mesh, open_sharding).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            move |x: ArrayTracer| {
                let placed = x.place(&target, None).unwrap();
                placed.clone() + placed
            },
            ArrayType::new_static(DataType::I32, [4, 4]),
        )
        .unwrap();
        assert_eq!(
            traced
                .to_rendered_mlir(
                    "main",
                    &ProgramLoweringOptions::new()
                        .with_input_shardings((std::slice::from_ref(&input_sharding)).to_vec())
                )
                .unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2, \"r\"=2]>
                  func.func @main(%arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>}) -> \
                        (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{\"x\"}, {?}]> : tensor<4x4xi32>
                    %1 = stablehlo.add %0, %0 : tensor<4x4xi32>
                    return %1 : tensor<4x4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_leaves_outputs_open_over_auto_axes() {
        // An output that is computed from the output of a `place`, rather than produced by it, has a type that drops
        // the split of the target along the auto axis `x`, so the compiler chooses its placement along that axis, and
        // the output is annotated open over it, as compilation annotates it.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let target = Placement::new(mesh, sharded).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            move |x: ArrayTracer| {
                let placed = x.place(&target, None).unwrap();
                placed.clone() + placed
            },
            test_vector_type(4),
        )
        .unwrap();
        let options = ProgramLoweringOptions::new().with_input_shardings(vec![replicated]);
        let open = indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> \
                        (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{\"x\"}]> : tensor<4xf32>
                    %1 = stablehlo.add %0, %0 : tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "};
        assert_eq!(traced.to_rendered_mlir("main", &options).unwrap(), open);
        assert_eq!(traced.to_rendered_mlir("main", &options.clone().with_open_outputs(vec![true])).unwrap(), open,);
        assert_eq!(
            traced.to_rendered_mlir("main", &options.with_open_outputs(Vec::new())).unwrap(),
            open.replace("[{?}]", "[{}]"),
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_keeps_effectful_dead_shard_maps() {
        // Pruning keeps a `shard_map` whose ordinary outputs are all dead when its body has observable effects, with no
        // ordinary outputs, so the lowered manual computation only threads the effect token of its body.
        let (traced, sharding) = traced_effectful_dead_shard_map();
        assert_eq!(
            traced
                .to_rendered_mlir(
                    "main",
                    &ProgramLoweringOptions::new()
                        .with_input_shardings((std::slice::from_ref(&sharding)).to_vec())
                        .with_output_shardings((std::slice::from_ref(&sharding)).to_vec())
                )
                .unwrap(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"x\"=2]>
                  func.func @main(%arg0: tensor<4xf64> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> \
                        (tensor<4xf64> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = sdy.manual_computation(%arg0, %0) in_shardings=[<@mesh, [{\"x\"}]>, <@mesh, []>] \
                        out_shardings=[<@mesh, []>] manual_axes={\"x\"} \
                        (%arg1: tensor<2xf64>, %arg2: !stablehlo.token) {
                      %2 = stablehlo.custom_call @ryft.print(%arg1, %arg2) {api_version = 4 : i32, \
                          backend_config = {label = \"body\"}, has_side_effect = true, \
                          operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], \
                          result_layouts = [dense<> : tensor<0xindex>]} : (tensor<2xf64>, !stablehlo.token) \
                          -> !stablehlo.token
                      sdy.return %2 : !stablehlo.token
                    } : (tensor<4xf64>, !stablehlo.token) -> !stablehlo.token
                    return %arg0 : tensor<4xf64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_program_to_rendered_mlir_executes_effectful_dead_shard_maps_on_cpu() {
        let (traced, sharding) = traced_effectful_dead_shard_map();
        let module = traced
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .with_input_shardings((std::slice::from_ref(&sharding)).to_vec())
                    .with_output_shardings((std::slice::from_ref(&sharding)).to_vec()),
            )
            .unwrap();
        let client = cpu_client(2);
        ensure_print_handler_registered(client).unwrap();
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, Some(2)))
            .unwrap();
        let shards = [vec![1.5f64, 2.5], vec![3.5f64, 4.5]];
        let inputs = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .zip(&shards)
            .map(|(device, shard)| {
                let bytes = values_to_bytes(shard.as_slice());
                vec![ExecutionInput {
                    buffer: Arc::new(
                        client.buffer(bytes.as_slice(), BufferType::F64, [2u64], None, device.clone(), None).unwrap(),
                    ),
                    donatable: false,
                }]
            })
            .collect::<Vec<_>>();
        let (outputs, mut lines) = with_captured_prints(|| {
            executable
                .execute(
                    inputs.iter().map(|inputs| ExecutionDeviceInputs { inputs, ..Default::default() }).collect(),
                    Vec::new(),
                    0,
                    None,
                    Some(file!()),
                    None,
                    None,
                    &[],
                )
                .unwrap()
                .block_until_ready()
                .unwrap()
        });

        // Every device returns its own shard and prints it exactly once, although nothing uses the `shard_map` output.
        assert_eq!(outputs.len(), 2);
        for (output, shard) in outputs.iter().zip(&shards) {
            assert_eq!(output.outputs.len(), 1);
            let bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(&values_from_bytes::<f64>(&bytes), shard);
        }
        lines.sort();
        assert_eq!(lines, vec!["body: [1.5, 2.5]".to_string(), "body: [3.5, 4.5]".to_string()]);
    }

    #[test]
    fn test_program_session_constants_from_regions() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let [first, mixed, unreferenced, last] =
            [1.0, 2.0, 3.0, 4.0].map(|value| RetainedArray::new(&f32_scalar(&session, &mesh, value)));

        // The callee ignores its lifted captures, but independently returns the same array as the middle capture.
        let mut callee = ProgramBuilder::<Constant, Operation>::new();
        for _ in 0..3 {
            callee.add_input(Type::Array(mixed.r#type().into_owned()));
        }
        let output = callee.add_constant(Constant::Session(mixed.clone()));
        let callee = Arc::new(callee.build(vec![output], vec![Placeholder; 3], vec![Placeholder]).unwrap());

        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let [first_id, mixed_id, _, last_id] =
            [&first, &mixed, &unreferenced, &last].map(|value| builder.add_constant(Constant::Session(value.clone())));
        let region = builder.intern_callee(&callee, None).unwrap();
        let output = builder
            .add_instruction(
                Operation::Call(CallOperation::new(3)),
                vec![region],
                vec![last_id, mixed_id, first_id],
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![], vec![Placeholder]).unwrap();
        let entry = program.entry_region_ref();

        // Both groups follow first occurrence in the atom table, even though the call passes its inputs in reverse.
        // A constant with no reads remains a hidden input when the program has not been pruned.
        let local = ProgramSessionConstants::from_regions(entry.emitted_regions(None, CallTraversal::Skip), None);
        assert_eq!(local.inputs, vec![unreferenced.clone()]);
        assert_eq!(local.forwarded, vec![first.clone(), mixed.clone(), last.clone()]);

        let whole = ProgramSessionConstants::from_regions(entry.emitted_regions(None, CallTraversal::Enter), None);
        assert_eq!(whole.inputs, vec![mixed, unreferenced]);
        assert_eq!(whole.forwarded, vec![first, last]);
    }

    #[test]
    fn test_program_session_constants_from_regions_assigns_the_same_inputs_on_every_process() {
        // Two single-device clients stand for two processes. Each independently places its default scalar on global
        // device 0 twice. Only the owner can reuse the original device buffer.
        let clients = [0, 1].map(|process| {
            let options = CpuClientOptions { device_count: Some(1), process_id: Some(process) };
            let client = load_cpu_plugin().unwrap().client(ClientOptions::CPU(options)).unwrap();
            let client: &'static ryft_pjrt::Client<'static> = Box::leak(Box::new(client));
            client
        });
        let first_device = Device::from_pjrt(&clients[0].addressable_devices().unwrap()[0]).unwrap();
        let modules = clients.map(|client| {
            let session = session_with_remote_devices(client, peer_devices(client, &clients));
            let scalar = session.place(4f32, PlacementOptions::default()).unwrap();
            let first = session.place(scalar.clone(), PlacementOptions::to(first_device)).unwrap();
            let second = session.place(scalar.clone(), PlacementOptions::to(first_device)).unwrap();
            let copy = session.place(first.clone(), PlacementOptions::default().with_may_alias(false)).unwrap();
            if client.process_index().unwrap() == 0 {
                let buffer = scalar.shards()[0].buffer().unwrap();
                assert!(Arc::ptr_eq(first.shards()[0].buffer().unwrap(), buffer));
                assert!(Arc::ptr_eq(second.shards()[0].buffer().unwrap(), buffer));
            } else {
                assert_eq!(first.addressable_shards().count(), 0);
                assert_eq!(second.addressable_shards().count(), 0);
            }
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let first_clone = first.clone();
            let outputs = [&scalar, &first, &second, &first_clone, &copy]
                .map(|array| builder.add_constant(Constant::Session(RetainedArray::new(array))))
                .to_vec();
            let program = builder.build::<Vec<Constant>, Vec<Constant>>(outputs, vec![], vec![Placeholder; 5]).unwrap();
            let regions = program.entry_region_ref().emitted_regions(None, CallTraversal::Enter);
            let constants = ProgramSessionConstants::from_regions(regions, None);
            assert_eq!(constants.inputs, [&scalar, &first, &second, &copy].map(RetainedArray::new));
            assert!(constants.forwarded.is_empty());

            // One array declared with one type twice is one constant, and declared with two types (e.g., by a function
            // compiled for a mesh of another topology, refer to `imported_constant`) two, on the process that holds its
            // buffers and on the one that holds none alike.
            let other_mesh = LogicalMesh::new(vec![MeshAxis::new("other", 1, MeshAxisType::Auto).unwrap()]).unwrap();
            let other_type = first.r#type().into_owned().with_sharding(Sharding::replicated(other_mesh, 0)).unwrap();
            let declared = RetainedArray::new(&first).with_declared_type(other_type.clone());
            assert_eq!(declared, RetainedArray::new(&first).with_declared_type(other_type));
            assert_ne!(declared, RetainedArray::new(&first));
            let mut declared_builder = ProgramBuilder::<Constant, Operation>::new();
            let declared_outputs = [declared.clone(), RetainedArray::new(&first), declared.clone()]
                .map(|constant| declared_builder.add_constant(Constant::Session(constant)))
                .to_vec();
            let declared_program = declared_builder
                .build::<Vec<Constant>, Vec<Constant>>(declared_outputs, vec![], vec![Placeholder; 3])
                .unwrap();
            let regions = declared_program.entry_region_ref().emitted_regions(None, CallTraversal::Enter);
            assert_eq!(
                ProgramSessionConstants::from_regions(regions, None),
                ProgramSessionConstants { inputs: vec![declared, RetainedArray::new(&first)], forwarded: vec![] },
            );

            let captures = constants.inputs.iter().map(|constant| constant.r#type().into_owned()).collect::<Vec<_>>();
            let output_types = vec![scalar.r#type().into_owned(); 5];
            program
                .to_rendered_mlir(
                    "main",
                    &ProgramLoweringOptions::new()
                        .without_pruning()
                        .with_capture_types(captures)
                        .with_input_types(&())
                        .with_output_types(&output_types)
                        .without_boundary_shardings(),
                )
                .unwrap()
        });

        assert_eq!(modules[0], modules[1]);
        assert_eq!(
            modules[0],
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
                    return %arg0, %arg1, %arg2, %arg1, %arg3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_session_constants_from_regions_ignores_dormant_regions() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let constant = RetainedArray::new(&f32_scalar(&session, &domain_mesh(client, "x", 1), 2.0));
        let scalar_type = constant.r#type().into_owned();
        let with_constant = |extended| {
            scalar_linear_region(&scalar_type, extended, |builder, _, value| {
                let constant = builder.add_constant(Constant::Session(constant.clone()));
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![value, constant], None).unwrap()[0]
            })
        };

        // A session constant that only the dormant transpose region of a `linear_call` refers to is never lowered, so
        // it does not become a hidden input of the entry function, unlike one that its forward region refers to.
        let plain_region = with_constant(false);
        let constant_region = with_constant(true);
        let dormant = linear_call_program(&plain_region, &constant_region);
        let regions = dormant.entry_region_ref().emitted_regions(None, CallTraversal::Enter);
        assert_eq!(
            ProgramSessionConstants::from_regions(regions, None),
            ProgramSessionConstants { inputs: vec![], forwarded: vec![] }
        );

        let live = linear_call_program(&constant_region, &plain_region);
        let regions = live.entry_region_ref().emitted_regions(None, CallTraversal::Enter);
        assert_eq!(
            ProgramSessionConstants::from_regions(regions, None),
            ProgramSessionConstants { inputs: vec![constant], forwarded: vec![] }
        );
    }

    #[test]
    fn test_program_session_constants_from_regions_skips_bodies_that_the_target_replaces() {
        // A session constant that only the body of a named composite call refers to becomes a hidden input where
        // lowering emits the body, and not on a target whose backend implementation replaces the whole call
        // (`ryft.tests.xla_named_square` version 1 on CUDA, refer to `omits_regions`), which never lowers the body,
        // as the collection of the module mesh and the validation of placements skip it.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let constant = RetainedArray::new(&f32_scalar(&session, &domain_mesh(client, "x", 1), 2.0));
        let scalar_type = Type::Array(constant.r#type().into_owned());
        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(scalar_type.clone());
            let constant = builder.add_constant(Constant::Session(constant.clone()));
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, constant], None).unwrap()[0];
            Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
        };
        let key = CompositeKey::new("ryft.tests.xla_named_square".to_string(), 1);
        let named = NamedComposite::new(key, |types: &[Type]| Ok(types.to_vec()), |inputs| Ok(vec![inputs[0].clone()]));
        let rules = CustomRuleRegistration::new(CustomRuleDefinition::new("adding").with_named_composite(named));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(scalar_type);
        let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
        let regions = vec![builder.intern_callee(&body, None).unwrap()];
        let output = builder.add_instruction(operation, regions, vec![input], None).unwrap()[0];
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder]);
        let program = program.unwrap();
        for (platform, expected) in [("cuda", vec![]), ("cpu", vec![constant])] {
            let regions = program.entry_region_ref().emitted_regions(Some(platform), CallTraversal::Enter);
            assert_eq!(
                ProgramSessionConstants::from_regions(regions, Some(platform)),
                ProgramSessionConstants { inputs: expected, forwarded: vec![] }
            );
        }
    }

    #[test]
    fn test_program_boundary_aliases() {
        // The entry function takes one capture and two public inputs of the same type, and returns one output of
        // another type followed by two outputs of that type.
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let other_type = ArrayType::new_static(DataType::F32, [2]);
        let input_types = vec![r#type.clone(); 3];
        let output_types = [other_type, r#type.clone(), r#type];
        let boundary_mapping = MlirProgramBoundaryMapping::new(&input_types, &output_types);
        let shardings = [None, None, None];
        let aliases = |donated_inputs: &[bool]| {
            ProgramBoundary::aliases(
                &[],
                donated_inputs,
                &[],
                1,
                &boundary_mapping,
                &input_types,
                &output_types,
                &shardings,
                &shardings,
            )
        };

        // Each donated input is paired with the first output of its type that no earlier input was paired with, and
        // the capture is never paired.
        assert_eq!(aliases(&[]), Ok(HashMap::new()));
        assert_eq!(aliases(&[true, false]), Ok(HashMap::from([(1, 1)])));
        assert_eq!(aliases(&[false, true]), Ok(HashMap::from([(2, 1)])));
        assert_eq!(aliases(&[true, true]), Ok(HashMap::from([(1, 1), (2, 2)])));

        // The donation flags must cover the public inputs exactly.
        assert_eq!(
            aliases(&[true]),
            Err(Error::Execution(ExecutionError::DonationFlagCountMismatch { expected_count: 2, actual_count: 1 })),
        );
    }

    #[test]
    fn test_entry_alias_attribute_executes_in_place_on_cpu() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[MlirSize::Static(4)], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let function_block = context.block(&[(tensor_type, location)]);
                {
                    let input = function_block.argument(0).unwrap();
                    let mut function_block_ref = function_block.as_ref();
                    let doubled =
                        function_block_ref.append_operation(stable_hlo::add(input, input, location).unwrap()).unwrap();
                    function_block_ref
                        .append_operation(func::r#return(&[doubled.result(0).unwrap()], location).unwrap())
                        .unwrap();
                }
                let mut function_region = context.region();
                function_region.append_block(function_block).unwrap();
                func::func(
                    "main",
                    func::FuncAttributes {
                        arguments: vec![TypeAndAttributes {
                            r#type: tensor_type.as_ref(),
                            attributes: Some(HashMap::from([
                                ("mhlo.sharding".into(), context.string_attribute("{replicated}").as_ref()),
                                (
                                    "tf.aliasing_output".into(),
                                    context.integer_attribute(context.signless_integer_type(64), 0).as_ref(),
                                ),
                            ])),
                        }],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    function_region,
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        let module = module.to_string();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32> {mhlo.sharding = "{replicated}", tf.aliasing_output = 0 : i64}) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );

        let client = cpu_client(1);
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, None))
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        let input = client
            .buffer(values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(), BufferType::F32, [4], None, device, None)
            .unwrap();
        input.ready().unwrap().r#await().unwrap();

        // The input is ready and remains alive in `inputs` until execution completes, so comparing its opaque address
        // with the synchronized output address does not dereference either device pointer.
        let input_pointer = unsafe { input.unsafe_pointer().unwrap() };
        let inputs = [ExecutionInput { buffer: Arc::new(input), donatable: true }];
        let execution = executable
            .execute(
                vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
                &[],
            )
            .unwrap();
        let mut device_outputs = execution.block_until_ready().unwrap().remove(0);
        let output = device_outputs.outputs.remove(0);
        let output_pointer = unsafe { output.unsafe_pointer().unwrap() };

        // `tf.aliasing_output` is a hint rather than a guarantee, because the copy insertion of XLA may materialize a
        // fresh output. This pointer equality is therefore a deliberate canary pinned to the CPU plugin version that
        // the repository pins. If a plugin upgrade breaks it while the numeric assertion below still passes,
        // re-evaluate whether the alias hint is still honored rather than assuming a correctness bug.
        assert_eq!(output_pointer, input_pointer);
        let output = output.copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<f32>(output.as_slice()), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_program_pruned_for_lowering() {
        // The callee `f(x) = (x, place(x, CPU 1))` names devices only through the `place` that computes its second
        // output, and its call records their mesh. A program that returns both outputs keeps that mesh. A program that
        // drops the second output loses the `place` to pruning, so its call is lowered as one that names no devices.
        let vector_type = test_vector_type(4);
        let device = Device::new(1, 0);
        let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
        let callee = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let operation = PlaceOperation::new(PlacementRequest::from(device), None);
            let placed = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            Arc::new(builder.build(vec![input, placed], vec![Placeholder], vec![Placeholder; 2]).unwrap())
        };
        let program = |output_indices: &[usize]| {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let regions = vec![builder.intern_callee(&callee, None).unwrap()];
            let call = Operation::Call(CallOperation::new(0).with_boundary(Some(&mesh), Vec::new(), Vec::new()));
            let outputs = builder.add_instruction(call, regions, vec![input], None).unwrap().to_vec();
            let outputs = output_indices.iter().map(|&index| outputs[index]).collect::<Vec<_>>();
            let output_count = outputs.len();
            builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; output_count])
                .unwrap()
        };
        assert_eq!(
            program(&[0, 1]).pruned_for_lowering(None).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4], %2:f32[4] = call [devices=[1]] %0 [
                    callee={
                        lambda %0:f32[4] .
                        let %1:f32[4] = place [target=device(1)] %0
                        in (%0, %1)
                    },
                ]
                in (%1, %2)"},
        );
        assert_eq!(
            program(&[0]).pruned_for_lowering(None).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4] = call %0 [
                    callee={
                        lambda %0:f32[4] .
                        in (%0)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_program_contains_unresolved_state() {
        let stateful =
            |extended| scalar_linear_region(&ArrayType::scalar(DataType::F32), extended, add_reference_state);

        // State in the dormant transpose region of a `linear_call` is never lowered, unlike state in its forward
        // region, and neither is state in the body of a scan that admits no iteration, unlike state in one that
        // iterates.
        assert!(!linear_call_program(&stateful(false), &stateful(true)).contains_unresolved_state());
        assert!(linear_call_program(&stateful(true), &stateful(false)).contains_unresolved_state());
        assert!(!stateful_scan_program(0).contains_unresolved_state());
        assert!(stateful_scan_program(1).contains_unresolved_state());
    }

    #[test]
    fn test_program_contains_unresolved_references() {
        let stateful =
            |extended| scalar_linear_region(&ArrayType::scalar(DataType::F32), extended, add_reference_state);

        // References in the dormant transpose region of a `linear_call` are never lowered, unlike references in its
        // forward region, and neither are references in the body of a scan that admits no iteration.
        assert!(!linear_call_program(&stateful(false), &stateful(true)).contains_unresolved_references());
        assert!(linear_call_program(&stateful(true), &stateful(false)).contains_unresolved_references());
        assert!(!stateful_scan_program(0).contains_unresolved_references());
        assert!(stateful_scan_program(1).contains_unresolved_references());
    }

    #[test]
    fn test_program_has_placement_requests() {
        let sharding = |axis_type: MeshAxisType| {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap()
        };
        let (auto_sharding, explicit_sharding) = (sharding(MeshAxisType::Auto), sharding(MeshAxisType::Explicit));
        let device = Device::new(0, 0);
        let has_requests = |traced: TracedProgram<ArrayType, ArrayType>| {
            (traced.has_placement_requests(None, true), traced.has_placement_requests(None, false))
        };

        // A `place` with a device target and a `constrain_sharding` always count.
        let placed = trace(|x: ArrayTracer| x.place(&device, None).unwrap(), test_vector_type(4)).unwrap();
        assert_eq!(has_requests(placed), (true, true));
        let constrained =
            trace(|x: ArrayTracer| x.constrain_sharding(&auto_sharding).unwrap(), test_vector_type(4)).unwrap();
        assert_eq!(has_requests(constrained), (true, true));

        // A `place` with an in-place target names no devices, so it counts only among the memory requests.
        let moved = trace(|x: ArrayTracer| x.place(&Memory::Host { pinned: true }, None).unwrap(), test_vector_type(4))
            .unwrap();
        assert_eq!(has_requests(moved), (true, false));

        // A `place` without a target, a `reshard`, and operations that place nothing request no placement.
        let kept = trace(|x: ArrayTracer| x.place(&None::<Placement>, None).unwrap(), test_vector_type(4)).unwrap();
        assert_eq!(has_requests(kept), (false, false));
        let resharded = trace(|x: ArrayTracer| x.reshard(&explicit_sharding).unwrap(), test_vector_type(4)).unwrap();
        assert_eq!(has_requests(resharded), (false, false));
        let computed = trace(|x: ArrayTracer| x.sin().unwrap(), test_vector_type(4)).unwrap();
        assert_eq!(has_requests(computed), (false, false));
    }

    #[test]
    fn test_program_output_sharding_sources() {
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        let split = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let vector_type = test_vector_type(4);

        // An output that a `place` produces, also through an operation that returns its input unchanged, has the
        // placement of the target, an output that returns an input is placed where that input enters, and the
        // program determines no sharding for an output that it computes.
        let target = Placement::new(mesh.clone(), split.clone()).unwrap();
        let traced: TracedProgram<ArrayType, (ArrayType, ArrayType, ArrayType)> = trace(
            move |x: ArrayTracer| {
                let placed = x.place(&target, None).unwrap().stop_gradient().unwrap();
                let computed = x.sin().unwrap();
                (placed, x, computed)
            },
            vector_type.clone(),
        )
        .unwrap();
        assert_eq!(
            traced.output_sharding_sources(3, |constant| constant.sharding().clone()),
            vec![Some(OutputShardingSource::Determined(split.clone())), Some(OutputShardingSource::Input(0)), None],
        );
        assert_eq!(
            traced.output_sharding_sources(1, |constant| constant.sharding().clone()),
            vec![Some(OutputShardingSource::Determined(split.clone()))],
        );

        // The boundary of the callee `f(x) = (x, x)` takes precedence over its body. An output of a call with a
        // sharding override has that override, and an output that returns a callee input is placed where the executable
        // of the callee places that input, which is replicated over the mesh that the call names. The same output of a
        // call that names no devices is followed back to the input of the call.
        let callee = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            Arc::new(builder.build(vec![input, input], vec![Placeholder], vec![Placeholder; 2]).unwrap())
        };
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(vector_type.into());
        let regions = vec![builder.intern_callee(&callee, None).unwrap()];
        let call = CallOperation::new(0).with_boundary(Some(&mesh), Vec::new(), vec![Some(split.clone())]);
        let placed =
            builder.add_instruction(Operation::Call(call), regions.clone(), vec![input], None).unwrap().to_vec();
        let call = Operation::Call(CallOperation::new(0));
        let followed = builder.add_instruction(call, regions, vec![input], None).unwrap()[1];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![placed[0], placed[1], followed],
                vec![Placeholder],
                vec![Placeholder; 3],
            )
            .unwrap();
        assert_eq!(
            program.output_sharding_sources(3, |constant| constant.sharding().clone()),
            vec![
                Some(OutputShardingSource::Determined(split)),
                Some(OutputShardingSource::Determined(replicated)),
                Some(OutputShardingSource::Input(0)),
            ],
        );
    }

    #[test]
    fn test_program_output_sharding_sources_follow_operations_that_keep_the_placement_of_their_input() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let split = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let vector_type = test_vector_type(4);
        let input = Array::from_host_buffer(
            &session,
            vector_type.clone(),
            mesh.clone(),
            Some(replicated),
            values_to_bytes(&[1f32, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        let target = Placement::new(mesh.clone(), split.clone()).unwrap();

        // Every operation that returns its input together with its placement keeps the split that a preceding `place`
        // gives its input along an auto mesh axis, which the type of the output does not record, so that the output
        // of the program is split as well (refer to `ProgramLoweringAnalysisExtension::output_sharding_sources`).
        let operations: [(&str, fn(CompileTracer) -> CompileTracer); 5] = [
            ("place without a target", |x| x.place(&None::<Placement>, None).unwrap()),
            ("place with an in-place target", |x| x.place(&Memory::Host { pinned: true }, None).unwrap()),
            ("transfer_to_memory", |x| x.transfer_to_memory(Memory::Host { pinned: true }).unwrap()),
            ("stop_gradient", |x| x.stop_gradient().unwrap()),
            ("tag", |x| x.tag("tagged").unwrap()),
        ];
        for (name, operation) in operations {
            let function: CompiledFunction<ArrayType, ArrayType> = compile(
                |x: CompileTracer| operation(x.place(&target, None).unwrap()),
                vector_type.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            let output = function.executable().execute(&session, input.clone()).unwrap();
            assert_eq!((name, output.sharding()), (name, &split));
        }
    }

    #[test]
    fn test_sharding_dimensions_with_unconstrained_replicated() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let dimensions = vec![
            ShardingDimension::sharded(["x"]),
            ShardingDimension::unconstrained(),
            ShardingDimension::replicated(),
        ];
        let sharding = Sharding::new(mesh, dimensions).unwrap();
        assert_eq!(
            sharding.dimensions_with_unconstrained_replicated().to_vec(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        );
    }

    #[test]
    fn test_program_lower_inline() {
        let array_type = test_vector_type(2);
        let program =
            trace(|(left, right): (ArrayTracer, ArrayTracer)| left * right, (array_type.clone(), array_type.clone()))
                .unwrap();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = array_type.lower(&context, location).unwrap();
        let block = context.block(&[(tensor_type, location), (tensor_type, location)]);
        let left = block.argument(0).unwrap().as_ref();
        let right = block.argument(1).unwrap().as_ref();
        let outputs = program
            .lower_inline(
                &[],
                &[left, right],
                None,
                &ProgramLoweringContext::new(),
                &mut EffectTokens::default(),
                block.as_ref(),
                location.as_ref(),
            )
            .unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(operations.len(), 1);
        let multiply = operations[0];
        assert_eq!(multiply.name().as_str(), Ok("stablehlo.multiply"));
        assert_eq!(multiply.operand(0).unwrap().value().unwrap(), left);
        assert_eq!(multiply.operand(1).unwrap().value().unwrap(), right);
        assert_eq!(outputs, vec![multiply.result(0).unwrap().as_ref()]);
    }

    #[test]
    fn test_program_lower_inline_rejects_input_count_mismatch() {
        let program = trace(|x: ArrayTracer| x.sin().unwrap(), test_vector_type(2)).unwrap();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let block = context.block_with_no_arguments();
        assert!(matches!(
            program.lower_inline(
                &[],
                &[],
                None,
                &ProgramLoweringContext::new(),
                &mut EffectTokens::default(),
                block.as_ref(),
                location.as_ref(),
            ),
            Err(Error::Core(error)) if matches!(
                error.as_ref(),
                CoreError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
            ),
        ));
        assert!(block.is_empty());
    }

    #[test]
    fn test_region_ref_emitted_regions() {
        let vector = test_vector_type(4);
        let callee = Arc::new(unproject_plain_program(neg_branch(vector.clone())));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(vector.into());
        let first = add_call(&mut builder, &callee, vec![input]);
        let second = add_call(&mut builder, &callee, vec![first]);
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![second], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let entry = program.entry_region_ref();
        let shared = entry.instructions()[0].regions()[0];
        assert_eq!(entry.instructions()[1].regions(), &[shared]);
        assert_eq!(
            entry.emitted_regions(None, CallTraversal::Enter).map(RegionRef::id).collect::<Vec<_>>(),
            vec![entry.id(), shared],
        );
        assert_eq!(
            entry.emitted_regions(None, CallTraversal::Skip).map(RegionRef::id).collect::<Vec<_>>(),
            vec![entry.id()],
        );
    }

    #[test]
    fn test_region_placement_requests_for_lowering() {
        // A `place` in the body of a named composite call is a request where lowering emits the body, and not for a
        // target whose backend implementation replaces the whole call (i.e., `ryft.tests.xla_named_square` version 1
        // on CUDA, refer to `omits_regions`), so that the eager placement of the enclosing operation does not read it.
        // The requests are retained for each platform separately.
        let vector_type = test_vector_type(4);
        let other = LogicalMesh::new(vec![MeshAxis::new("other", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_devices = DeviceMesh::new(other.clone(), vec![Device::new(2, 0), Device::new(3, 0)]).unwrap();
        let placement = Placement::new(other_devices.clone(), Sharding::replicated(other, 1)).unwrap();
        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let operation = PlaceOperation::new(PlacementRequest::from(placement), None);
            let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
        };
        let key = CompositeKey::new("ryft.tests.xla_named_square".to_string(), 1);
        let named = NamedComposite::new(key, |types: &[Type]| Ok(types.to_vec()), |inputs| Ok(vec![inputs[0].clone()]));
        let rules = CustomRuleRegistration::new(CustomRuleDefinition::new("placing").with_named_composite(named));
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(vector_type.into());
        let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
        let regions = vec![builder.intern_callee(&body, None).unwrap()];
        let output = builder.add_instruction(operation, regions, vec![input], None).unwrap()[0];
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder]);
        let program = program.unwrap();
        let requests = |target_platform| {
            RegionPlacementRequests::for_lowering(program.entry_region_ref(), target_platform)
                .requests()
                .to_vec()
        };
        let placed = vec![(RequestedPlacement::Devices(other_devices), PLACE_OPERATION_NAME)];
        assert_eq!(requests(None), placed);
        assert_eq!(requests(Some("cuda")), vec![]);
        assert_eq!(requests(Some("cpu")), placed);
        assert_eq!(requests(Some("cuda")), vec![]);
    }

    #[test]
    fn test_region_placement_requests_for_lowering_of_nested_calls() {
        // `f(x) = place(x, CPU 0)` wrapped in 40 functions that each call the previous one. The call of each wrapper
        // names devices because the request in the innermost callee does, which the analysis of a wrapper reads from
        // the facts of its callee that it derives in the same pass over the pruned program, rather than by analyzing
        // the callee of every nested call again, which took time exponential in the depth of the nesting (i.e., about
        // 4.5 seconds for 20 wrappers in a debug build), and would not finish for these 40. Staging nests as deeply as
        // the functions do, so the test runs on a thread with a larger stack.
        let depth = 40;
        let requests = std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || {
                let device = Device::new(0, 0);
                let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
                let scalar_type = ArrayType::scalar(DataType::F32);
                let options = Options::new(mesh.clone());
                let placing = |x: CompileTracer| x.place(&device, None).unwrap();
                let mut function: StagedFunction<ArrayType, ArrayType> =
                    stage(placing, scalar_type.clone(), options.clone()).unwrap();
                for _ in 0..depth {
                    function =
                        stage(|x: CompileTracer| function.call(x).unwrap(), scalar_type.clone(), options.clone())
                            .unwrap();
                }
                let program = function.source_program().program();
                let requested =
                    RegionPlacementRequests::for_lowering(program.entry_region_ref(), None).requested_mesh();
                assert_eq!(requested.unwrap().map(|(requested, _)| requested), Some(mesh.clone()));
                assert_eq!(program.pruned_for_lowering(None).unwrap().to_string(), program.to_string());
                (mesh, RegionPlacementRequests::for_lowering(program.entry_region_ref(), None).requests().to_vec())
            })
            .unwrap()
            .join()
            .unwrap();
        let (mesh, requests) = requests;
        assert_eq!(requests.len(), depth + 1);
        assert_eq!(requests[0], (RequestedPlacement::Devices(mesh.clone()), CALL_OPERATION_NAME));
        assert_eq!(requests[depth], (RequestedPlacement::Devices(mesh), PLACE_OPERATION_NAME));
        assert!(requests[1..depth].iter().all(|request| *request == requests[0]));
    }

    #[test]
    fn test_region_placement_requests_for_lowering_skips_the_bodies_of_scans_without_iterations() {
        // A `place` in the body of a scan is a request when the length of the scan admits an iteration, and not on any
        // target when its length is zero, because lowering then emits nothing of its body (refer to `omits_regions`).
        let vector_type = test_vector_type(4);
        let elsewhere = Device::new(1, 0);
        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(vector_type.clone().into());
            let operation = PlaceOperation::new(PlacementRequest::from(elsewhere), None);
            let output = builder.add_instruction(operation, Vec::new(), vec![carry], None).unwrap()[0];
            Arc::new(builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder]).unwrap())
        };
        let program = |length: usize| {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let regions = vec![builder.intern_callee(&body, None).unwrap()];
            let scan = ScanOperation::<Type>::new(1, length);
            let output = builder.add_instruction(scan, regions, vec![input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let requests = |length, target_platform| {
            RegionPlacementRequests::for_lowering(program(length).entry_region_ref(), target_platform)
                .requests()
                .to_vec()
        };
        let device_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![elsewhere]).unwrap();
        let placed = vec![(RequestedPlacement::Devices(device_mesh), PLACE_OPERATION_NAME)];
        assert_eq!(requests(0, None), vec![]);
        assert_eq!(requests(0, Some("cpu")), vec![]);
        assert_eq!(requests(1, None), placed);
        assert_eq!(requests(1, Some("cpu")), placed);
    }

    #[test]
    fn test_region_placement_requests_requested_mesh() {
        let vector_type = test_vector_type(4);
        let first = Device::new(0, 0);
        let second = Device::new(1, 0);
        let device_mesh = |device: Device| DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
        let program = |targets: Vec<PlacementRequest>| {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let mut value = builder.add_input(vector_type.clone().into());
            for target in targets {
                value = builder
                    .add_instruction(PlaceOperation::new(target, None), Vec::new(), vec![value], None)
                    .unwrap()[0];
            }
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![value], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let requested = |program: &FlatProgram<Session>| {
            RegionPlacementRequests::for_lowering(program.entry_region_ref(), None)
                .requested_mesh()
                .map(|requested| requested.map(|(mesh, _)| mesh))
        };
        let memory = PlacementRequest::InPlace { memory: Some(Memory::Device), layout: None };

        // A request that only moves a value to another memory names no devices, and requests that name the same
        // devices form one mesh, which a device target names as the canonical mesh of that device.
        assert_eq!(requested(&program(vec![memory.clone()])), Ok(None));
        assert_eq!(
            requested(&program(vec![memory, PlacementRequest::from(first), PlacementRequest::from(first)])),
            Ok(Some(device_mesh(first))),
        );

        // Requests that name different devices do not form one mesh.
        assert_eq!(
            requested(&program(vec![PlacementRequest::from(first), PlacementRequest::from(second)])),
            Err(Error::Lowering(LoweringError::UnsupportedOperation {
                operation: PLACE_OPERATION_NAME.to_string(),
                reason: "it requests a placement onto the devices [1], while another placement request of the same \
                         program names the devices [0], and the two do not form one mesh"
                    .to_string(),
            })),
        );

        // A device target and a named mesh of the same device select the named mesh in either order, and a device
        // target of an input that is tracked over a mesh of one device keeps that logical mesh, as the type rule of
        // `place` does.
        let named = LogicalMesh::new(vec![MeshAxis::new("named", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let named_mesh = DeviceMesh::new(named.clone(), vec![second]).unwrap();
        let named_target =
            PlacementRequest::from(Placement::new(named_mesh.clone(), Sharding::replicated(named.clone(), 1)).unwrap());
        for targets in [
            vec![named_target.clone(), PlacementRequest::from(second)],
            vec![PlacementRequest::from(second), named_target.clone()],
        ] {
            assert_eq!(requested(&program(targets)), Ok(Some(named_mesh.clone())));
        }
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let named_type = vector_type.clone().with_sharding(Sharding::replicated(named, 1)).unwrap();
        let input = builder.add_input(named_type.into());
        let output = builder
            .add_instruction(PlaceOperation::new(PlacementRequest::from(second), None), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(requested(&program), Ok(Some(named_mesh)));
    }

    #[test]
    fn test_merge_requested_placement() {
        let canonical = DeviceMesh::new(LogicalMesh::single_device(), vec![Device::new(0, 0)]).unwrap();
        let named = |name: &str| {
            let mesh = LogicalMesh::new(vec![MeshAxis::new(name, 1, MeshAxisType::Explicit).unwrap()]).unwrap();
            DeviceMesh::new(mesh, vec![Device::new(0, 0)]).unwrap()
        };
        let (first_named, second_named) = (named("first"), named("second"));
        let elsewhere = DeviceMesh::new(LogicalMesh::single_device(), vec![Device::new(1, 0)]).unwrap();

        // The first request that names devices is recorded with the name of its operation.
        let mut requested = None;
        assert_eq!(merge_requested_placement(&mut requested, Cow::Borrowed(&canonical), PLACE_OPERATION_NAME), Ok(()));
        assert_eq!(requested, Some((Cow::Borrowed(&canonical), PLACE_OPERATION_NAME)));

        // A named mesh of the same device replaces the canonical mesh and is kept when the canonical mesh or another
        // named mesh of that device follows it.
        assert_eq!(merge_requested_placement(&mut requested, Cow::Borrowed(&first_named), CALL_OPERATION_NAME), Ok(()));
        assert_eq!(requested, Some((Cow::Borrowed(&first_named), CALL_OPERATION_NAME)));
        assert_eq!(merge_requested_placement(&mut requested, Cow::Borrowed(&canonical), PLACE_OPERATION_NAME), Ok(()));
        assert_eq!(
            merge_requested_placement(&mut requested, Cow::Borrowed(&second_named), PLACE_OPERATION_NAME),
            Ok(()),
        );
        assert_eq!(requested, Some((Cow::Borrowed(&first_named), CALL_OPERATION_NAME)));

        // A request that names other devices does not form one mesh with the recorded request, which is kept.
        assert_eq!(
            merge_requested_placement(&mut requested, Cow::Borrowed(&elsewhere), PLACE_OPERATION_NAME),
            Err(Error::Lowering(LoweringError::UnsupportedOperation {
                operation: PLACE_OPERATION_NAME.to_string(),
                reason: "it requests a placement onto the devices [1], while another placement request of the same \
                         program names the devices [0], and the two do not form one mesh"
                    .to_string(),
            })),
        );
        assert_eq!(requested, Some((Cow::Borrowed(&first_named), CALL_OPERATION_NAME)));
    }

    #[test]
    fn test_entry_output_shardings() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = |dimension: ShardingDimension| Sharding::new(mesh.clone(), vec![dimension]).unwrap();
        let split = sharded(ShardingDimension::sharded(["x"]));
        let replicated = sharded(ShardingDimension::replicated());
        let open = sharded(ShardingDimension::unconstrained());
        let plain_type = test_vector_type(4);
        let split_type = plain_type.clone().with_sharding(split.clone()).unwrap();

        // Output sharding overrides are used whatever the program and the output types state, with their unconstrained
        // dimensions replicated.
        assert_eq!(
            entry_output_shardings(
                std::slice::from_ref(&split_type),
                Some(std::slice::from_ref(&open)),
                &[Some(OutputShardingSource::Determined(split.clone()))],
                &[],
            ),
            Ok(vec![Some(replicated.clone())]),
        );

        // Without overrides, an output has the placement that the program determines for it or that of the input
        // that it returns, with its unconstrained dimensions replicated, and otherwise the sharding of its type, if
        // any.
        assert_eq!(
            entry_output_shardings(
                &[plain_type.clone(), plain_type.clone(), plain_type.clone(), split_type, plain_type.clone()],
                None,
                &[
                    Some(OutputShardingSource::Determined(split.clone())),
                    Some(OutputShardingSource::Input(0)),
                    Some(OutputShardingSource::Input(1)),
                    None,
                    None,
                ],
                &[Some(open), None],
            ),
            Ok(vec![Some(split.clone()), Some(replicated.clone()), None, Some(split), None]),
        );

        // A placement that the program gives an output keeps only the reduction state that the output type records.
        let pending = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let pending_type = plain_type.clone().with_sharding(pending.clone()).unwrap();
        assert_eq!(
            entry_output_shardings(
                &[plain_type, pending_type],
                None,
                &[
                    Some(OutputShardingSource::Determined(pending.clone())),
                    Some(OutputShardingSource::Determined(pending.clone())),
                ],
                &[],
            ),
            Ok(vec![Some(replicated), Some(pending)]),
        );
    }

    #[test]
    fn test_entry_outputs_with_compiler_chosen_placement() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let annotation = Sharding::replicated(mesh.clone(), 1);
        let plain_type = test_vector_type(4);
        let open_type = plain_type
            .clone()
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::unconstrained()]).unwrap())
            .unwrap();
        let open = |output_type: &ArrayType,
                    output_shardings: Option<&[Sharding]>,
                    source: Option<OutputShardingSource>,
                    annotation: Option<Sharding>| {
            entry_outputs_with_compiler_chosen_placement(
                std::slice::from_ref(output_type),
                output_shardings,
                &[source],
                &[annotation],
            )
        };

        // An output is open when its annotation is over a mesh with an auto axis along which the compiler has a choice
        // and nothing fixes its placement.
        assert_eq!(open(&plain_type, None, None, Some(annotation.clone())), vec![true]);
        assert_eq!(open(&plain_type, None, None, Some(Sharding::replicated(explicit_mesh, 1))), vec![false]);
        assert_eq!(open(&plain_type, None, None, None), vec![false]);

        // An output sharding override and a placement that the program gives the output fix it.
        let overrides = Some(std::slice::from_ref(&annotation));
        assert_eq!(open(&plain_type, overrides, None, Some(annotation.clone())), vec![false]);
        let source = Some(OutputShardingSource::Input(0));
        assert_eq!(open(&plain_type, None, source, Some(annotation.clone())), vec![false]);

        // Reduction state and manual variation in the annotation fix it, and so does an unconstrained dimension in the
        // output type, which the runtime placement of the output reads as replicated.
        let unreduced = annotation.clone().with_unreduced_axes(["a"]).unwrap();
        assert_eq!(open(&plain_type, None, None, Some(unreduced)), vec![false]);
        let reduced = annotation.clone().with_reduced_axes(["a"]).unwrap();
        assert_eq!(open(&plain_type, None, None, Some(reduced)), vec![false]);
        let varying = annotation.clone().with_varying_manual_axes(["m"]).unwrap();
        assert_eq!(open(&plain_type, None, None, Some(varying)), vec![false]);
        assert_eq!(open(&open_type, None, None, Some(annotation)), vec![false]);
    }
}
