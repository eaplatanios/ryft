//! Functional kernel calls whose block references are confined to an attached body region.
//!
//! A kernel is an ordinary Ryft program, its _body_, that runs once per point of a logical [`Grid`] and that accesses
//! arrays only through references to block windows of those arrays. The call that applies a kernel owns the full array
//! signature, and the body sees only the window of each array that belongs to the current grid point. Each
//! [`KernelParameter`] declares the full array type of one parameter, its [`KernelParameterAccess`], and the
//! [`BlockMapping`] that selects its window for every grid point, and it derives the logical block reference type
//! through which the body accesses that window. Compiler, launch, and schedule metadata are intentionally absent, so
//! one definition serves every compiler adapter and the interpreter.
//!
//! # Call Boundary
//!
//! [`KernelCallOperation`] is the higher-order [`KERNEL_CALL_OPERATION_NAME`] operation, which has exactly one attached
//! body region. The body inputs are the block references in parameter order, then one dimension coordinate per grid
//! axis, and then the scalar prefetched rank-0 integer arrays, whose values the block mappings receive as dimensions
//! after the grid coordinates. The body returns no values, because its stores produce the results of the call. Outside
//! the call, every parameter that is not write-only contributes an ordinary array input, the scalar prefetched arrays
//! follow those inputs, and every parameter that is not read-only contributes an ordinary array result. A read-write
//! parameter therefore contributes one input and one result that may share storage (refer to
//! [`KernelCallOperation::aliases`]), and no reference or input mutation is visible outside the call. A mapping or
//! scalar prefetch conversion that may fail is an observable assertion of the call itself, and
//! [`KernelDefinition::effects`] summarizes it together with the effects of the body.
//!
//! # Definitions and Validation
//!
//! [`KernelDefinition`] pairs a call with an immutable body that it validated before retaining either.
//! [`KernelDefinition::new`] and the tracing constructors [`KernelDefinition::trace`] and
//! [`KernelDefinition::trace_with_prefetch`] reject a body whose interface does not match the call signature, a body
//! that contains a `shard_map` anywhere in its attached region closure, and a body whose reference accesses
//! [`validate_kernel_body`] rejects against the declared parameter accesses. This boundary validation does not grant
//! execution. [`VerifiedKernel`](crate::compilation::VerifiedKernel) additionally verifies initialization, bounds,
//! complete output coverage, and the disjointness of mutable windows across grid points before a compiler adapter
//! admits a definition. Interpretation in [`interpretation`](crate::interpretation) checks initialization, output
//! coverage, and races as well before it replays a body.
//!
//! Dynamic signatures are specialized before or after tracing. [`KernelParameter::from_refined_types`] fixes full array
//! shapes before a body is traced, while [`KernelDefinition::specialize_grid`] and
//! [`KernelDefinition::specialize_prefetch`] bind grid extents and scalar prefetched values of an existing definition
//! without retracing its body. These functions and [`KernelDefinition::simplified`] produce a new definition that is
//! validated again, so no validation remains attached to changed semantics. Construction and validation failures are
//! reported as [`KernelError`]s.
//!
//! # Semantic Keys
//!
//! [`KernelDefinition::semantic_key`] encodes the call and body of a definition into the same wire records as its
//! portable source document (refer to [`serialization`](crate::serialization)), projects out what is not semantic
//! (i.e., diagnostic provenance, nominal dimension names, and grid axis names), and serializes the result canonically
//! after a header that carries [`KERNEL_KEY_VERSION`]. The key numbers nominal dimension identities in occurrence
//! order, retains exact constant bytes and floating-point bits, and rejects operations whose identity is local to the
//! process. It covers every valid definition, including features outside the transport schema, which key-only records
//! encode. Execution integrations combine it with the complete configuration key of a compiler adapter before reusing
//! compiled work.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt::Display;
use std::hash::Hash;

use ryft_core::{
    Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayReferenceTransform, ArraySliceAxis, ArrayType,
    ArrayTypeRefinements, Dimension, DimensionBounds, DimensionFromScalar, DimensionFromScalarOperation, DimensionType,
    DimensionVariable, EagerContext, EffectClass, EffectClasses, Effects, EffectsSummary, FlatProgram,
    InputRegionProvenance, InstructionId, LogicalMesh, Operation, OperationFormatter, PadOperation, Placeholder,
    ProgramBuilder, ProgramError, ReferenceAccessDescriptor, ReferenceAccessMode, ReferenceAccessOperation,
    ReferenceType, RegionInterface, RegionRef, RegionSlot, Sharding, Tracer, TracingContext, Type, TypeError,
    TypeIdentityRenaming, TypeRefinements, Typed, TypedProgram,
};
use thiserror::Error;

use crate::grids::{Grid, GridError};
use crate::mappings::{BlockMapping, BlockMappingError, BoundaryPolicy};
use crate::operations::{KernelExtension, KernelOperation, NoKernelExtension};
use crate::serialization::encode_semantic_key;
use crate::validation::{
    KernelBoundaryContract, KernelParameterAccess, KernelReferenceSummary, KernelValidationError, validate_kernel_body,
};

/// Invalid kernel signature or body boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum KernelError {
    /// Ordinary tracing or region finalization rejected the program.
    #[error(transparent)]
    Program(#[from] ProgramError),

    /// Canonical type inference rejected the call or its body interface.
    #[error(transparent)]
    Type(#[from] TypeError),

    /// A concrete grid extent violates its declared bounds or identity relationships.
    #[error(transparent)]
    Grid(#[from] GridError),

    /// The canonical mapping verifier rejected a [`BlockMapping`].
    #[error(transparent)]
    Mapping(#[from] BlockMappingError),

    /// Reference access or lifetime validation by [`validate_kernel_body`] rejected the body.
    #[error(transparent)]
    Validation(#[from] KernelValidationError),

    /// The body applies an operation that kernels do not support anywhere in their attached region closure. A
    /// `shard_map` is the only such operation, because a kernel body runs once per grid point of one device, so it
    /// cannot contain a manual SPMD computation over a device mesh.
    #[error("kernel body instruction {instruction} applies `{operation}`, which kernel bodies do not support")]
    UnsupportedOperation {
        /// Name of the unsupported operation.
        operation: &'static str,

        /// Instruction that applies the unsupported operation.
        instruction: InstructionId,
    },

    /// The block and full array have different logical ranks.
    #[error("kernel parameter has array rank {array_rank} but block rank {block_rank}")]
    ParameterRank {
        /// Rank of the full array type.
        array_rank: usize,

        /// Rank of the block shape of the parameter mapping.
        block_rank: usize,
    },

    /// Mapping arguments must correspond to grid coordinates followed by scalar prefetched dimensions.
    #[error("kernel parameter {parameter} mapping takes {actual} inputs but the call requires {expected}")]
    MappingInputCount {
        /// Parameter whose mapping takes the wrong number of inputs.
        parameter: usize,

        /// Number of grid dimensions plus the number of scalar prefetched arrays.
        expected: usize,

        /// Number of inputs of the mapping program.
        actual: usize,
    },

    /// Independent grid axes, or a grid axis and a scalar prefetched dimension, cannot impose equality through a shared
    /// mapping dimension identity.
    #[error(
        "kernel parameter {parameter} mapping axes {first_axis} and {second_axis} share a dimension identity but may \
         differ"
    )]
    MappingInputIdentity {
        /// Parameter whose mapping equates independent coordinates.
        parameter: usize,

        /// First logical axis using the identity.
        first_axis: usize,

        /// Later mapping input using the same identity, which is a logical axis or a scalar prefetched dimension
        /// following the grid coordinates.
        second_axis: usize,
    },

    /// A coordinate type excludes a grid point that the call may execute.
    #[error("kernel parameter {parameter} mapping axis {axis} admits {actual}, but the grid requires {expected}")]
    MappingInputBounds {
        /// Parameter whose mapping excludes a coordinate.
        parameter: usize,

        /// Logical grid axis supplying the coordinate.
        axis: usize,

        /// Coordinate interval required by the declared grid envelope.
        expected: DimensionBounds,

        /// Coordinate interval admitted by the mapping input.
        actual: DimensionBounds,
    },
}

/// Full logical array, declared access, and window mapping for one body reference input.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelParameter {
    /// Full array type at the functional outer boundary.
    r#type: ArrayType,

    /// Access permitted on this parameter's body reference.
    access: KernelParameterAccess,

    /// Pure mapping from grid points to windows, with an explicit boundary policy.
    mapping: BlockMapping,

    /// Canonical reference type derived from the full array and logical block by the array slice transform rules.
    body_type: ArrayIrType,
}

impl KernelParameter {
    /// Creates a parameter whose block rank agrees with its full array rank. The reference type is derived by the
    /// canonical [`ArrayReferenceTransform::Slice`] type rules, which preserve the layout of a whole window and
    /// validate the sharding of a partial window. Masked windows have a fixed logical block type derived by slicing the
    /// valid origin window and applying the canonical padding type rules of [`PadOperation`]. Their private tile
    /// storage has no required physical layout. Invalid elements are accessible only through masked memory operations
    /// with an explicit fallback, and padding does not initialize them.
    ///
    /// The full array type is physical. It has no sharding, because a kernel runs on one device and every type in its
    /// body follows from its parameters. How an argument is placed is a property of the call (refer to
    /// [`KernelCallOperation`]), which accepts placed inputs of this type.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::ParameterRank`] when the block rank of `mapping` differs from the rank of `type`, and
    /// [`KernelError::Type`] when `type` has a sharding, when a masked window has a full array type without a static
    /// shape, or when the slice or padding type rules reject the block.
    pub fn new(r#type: ArrayType, access: KernelParameterAccess, mapping: BlockMapping) -> Result<Self, KernelError> {
        if r#type.rank() != mapping.block_shape().len() {
            return Err(KernelError::ParameterRank {
                array_rank: r#type.rank(),
                block_rank: mapping.block_shape().len(),
            });
        }
        if r#type.sharding().is_some() {
            return Err(TypeError::invalid(format!(
                "kernel parameter type `{type}` has a sharding, but a kernel runs on one device and its call accepts \
                 placed inputs",
            ))
            .into());
        }
        let referent = if mapping.boundary_policy() == BoundaryPolicy::Masked {
            let shape = r#type
                .static_shape()
                .ok_or_else(|| TypeError::invalid("masked kernel windows require a static full array shape"))?;
            let valid_shape = mapping
                .block_shape()
                .iter()
                .zip(shape.dimensions())
                .map(|(&block, &extent)| block.min(extent))
                .collect::<Vec<_>>();
            let transform = ArrayReferenceTransform::Slice {
                axes: valid_shape.iter().map(|&extent| ArraySliceAxis::new(0, extent, 1)).collect(),
            };
            let valid_type = transform.output_type(&r#type)?;
            let padding = PadOperation::<ArrayType>::new(
                vec![0; r#type.rank()],
                mapping
                    .block_shape()
                    .iter()
                    .zip(&valid_shape)
                    .map(|(&block, &valid)| i64::try_from(block - valid).unwrap())
                    .collect(),
                vec![0; r#type.rank()],
            )?;
            padding
                .infer_output_types(
                    &[valid_type, ArrayType::scalar(r#type.data_type()).with_memory(r#type.memory())],
                    &[],
                )?
                .remove(0)
                .with_layout(None)
        } else {
            let transform = ArrayReferenceTransform::Slice {
                axes: mapping.block_shape().iter().map(|&extent| ArraySliceAxis::new(0, extent, 1)).collect(),
            };
            transform.output_type(&r#type)?
        };
        let body_type = ArrayIrType::Reference(ReferenceType::new(referent));
        Ok(Self { r#type, access, mapping, body_type })
    }

    /// Specializes declared full array shapes before constructing parameters or tracing a body. Canonical
    /// [`ArrayTypeRefinements`] checks enforce bounds and agreement of repeated dimension identities across all
    /// parameters, including outputs. Every supplied actual shape must be static, and unresolved output dimensions are
    /// rejected rather than inferred from unrelated inputs or runtime contents.
    ///
    /// Actual types must refine the declarations, including declared data type, memory, layout, and sharding
    /// constraints. Only their shapes are copied, so additional actual layout or sharding information does not become a
    /// declaration or specialization dependency. Canonical slice and padding inference then validates each fixed block
    /// type. This constructs a concrete signature for subsequent tracing and does not restage a retained dynamic body.
    ///
    /// # Parameters
    ///
    ///   - `parameters`: Full type declaration, access, and mapping for each parameter in body reference order.
    ///   - `actual_types`: Explicit concrete types in the same order, including write-only output types.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::Type`] when the actual types do not refine the declarations or when an actual shape is
    /// not static, and every error of [`Self::new`] for the specialized parameters.
    pub fn from_refined_types(
        parameters: Vec<(ArrayType, KernelParameterAccess, BlockMapping)>,
        actual_types: &[ArrayType],
    ) -> Result<Vec<Self>, KernelError> {
        ArrayTypeRefinements::new(parameters.iter().map(|(r#type, _, _)| r#type), actual_types)?;
        parameters
            .into_iter()
            .zip(actual_types)
            .enumerate()
            .map(|(parameter, ((declared, access, mapping), actual))| {
                let shape = actual.static_shape().ok_or_else(|| {
                    TypeError::invalid(format!(
                        "kernel parameter {parameter} specialization requires a static full shape",
                    ))
                })?;
                Self::new(declared.with_shape(shape), access, mapping)
            })
            .collect()
    }

    /// Returns the declared access for the body window.
    pub fn access(&self) -> KernelParameterAccess {
        self.access
    }

    /// Returns the mapping that selects a window for each grid point.
    pub fn mapping(&self) -> &BlockMapping {
        &self.mapping
    }

    /// Returns the canonical logical reference type of this parameter inside the body. Windows that cover the whole
    /// array retain the complete array type, and partial windows use the existing slice rules for layout, sharding, and
    /// memory. Masked windows use a private padded tile with no required physical layout.
    pub fn body_type(&self) -> ArrayIrType {
        self.body_type.clone()
    }
}

impl Typed for KernelParameter {
    type Type = ArrayType;

    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

/// Canonical name for the higher-order [`KernelCallOperation`].
pub const KERNEL_CALL_OPERATION_NAME: &str = "kernel_call";

/// Version of the encoding of kernel semantic keys, which [`KernelDefinition::semantic_key`] writes as the only version
/// in its header. The version stays at 1 until the first release of `ryft-kernels`, and unreleased encoding changes
/// reuse it. After that release, every incompatible change to the encoded semantics increments it. This version is
/// independent of the version of serialized kernel sources and of adapter artifact and executable persistence versions.
pub const KERNEL_KEY_VERSION: u32 = 1;

/// Higher-order operation with one attached kernel body that preserves its block references.
///
/// Body reference inputs follow the order of [`Self::parameters`]. Outer inputs omit write-only parameters, and outer
/// results omit read-only parameters. A read-write parameter therefore contributes one ordinary input and one ordinary
/// result. This functional alias relationship does not expose a reference state slot or permit an input mutation
/// visible outside the call. The body returns no values, because its stores produce the declared array results.
///
/// The parameter and scalar prefetch types are physical and have no sharding, because a kernel runs on one device.
/// Placement belongs to the call. An input may have a sharding that places it over a mesh and that records manual
/// variation, as long as it carries no pending reduction and every such input uses the same mesh. A kernel cannot be
/// assumed to produce the same result on every device, so when an input varies along manual axes, every result varies
/// along them too. A write-only result has its declared type, placed replicated on the mesh of the inputs when it
/// varies, which is how a custom call types a result without a declared sharding. The result of a read-write
/// parameter is the storage of its input and has exactly the type of that input, which is what the alias of a custom
/// call requires, so that input must itself vary along every manual axis along which another input varies. Whether a
/// placed input is the local shard that one device can run the kernel on is decided where a kernel is compiled for a
/// backend and not here.
///
/// The call declares an [`EffectClass::OrderedAssertion`] of its own exactly when one of the following may fail:
///
///   - a mapping program (e.g., a division by a prefetched dimension that may be zero),
///   - the conversion of a scalar prefetched value to the dimension type of a mapping input that receives it, which
///     fails for a negative or out of bounds value, or
///   - the window checks of [`BlockMapping::evaluate`], which reject a window that exceeds its full array under
///     [`BoundaryPolicy::InBounds`] and a window limit that exceeds the maximum dimension extent under either policy.
///     A window is treated as able to fail unless [`BlockMapping::has_total_windows`] proves the checks from the
///     bounds of the mapping output types and the full array type of its parameter.
///
/// Such a failure is observable even when the results of the call are unused, so simplification retains the call and
/// preserves its order relative to other ordered effects. A call whose mapping programs are total, that converts no
/// scalar prefetched value, and whose windows are proven (e.g., whole array windows, or tiles whose grid fits the
/// array) declares no effect, so it is eliminated like pure work when its body is pure as well.
/// [`KernelDefinition::effects`] combines these effects with those of the body.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelCallOperation {
    /// Logical execution grid, independent of native launch dimensions.
    grid: Grid,

    /// Logical array parameters in body input order.
    parameters: Vec<KernelParameter>,

    /// Canonical logical coordinate types, in grid axis order after the reference inputs.
    coordinate_types: Vec<DimensionType>,

    /// Rank-0 integer arrays read before mapping evaluation and supplied as private body values.
    prefetch_types: Vec<ArrayType>,
}

impl KernelCallOperation {
    /// Creates a call with one mapping coordinate per logical grid dimension. Full body validation belongs to
    /// [`KernelDefinition::new`], and constructing metadata alone does not make an executable kernel.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::new_with_prefetch`].
    pub fn new(grid: Grid, parameters: Vec<KernelParameter>) -> Result<Self, KernelError> {
        Self::new_with_prefetch(grid, parameters, vec![])
    }

    /// Creates a call with scalar prefetched arrays after its ordinary array inputs. Each mapping takes grid
    /// coordinates followed by checked dimensions converted from these rank-0 integer arrays. Body inputs append the
    /// original scalar arrays after references and coordinates. Values must be explicitly specialized before
    /// executable qualification, which preserves the ordinary array ABI at native call boundaries. A prefetched
    /// dimension cannot share its identity with a grid coordinate, because specialization binds it to a constant.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::Type`] when a prefetched type is not a rank-0 integer array type or has a sharding,
    /// [`KernelError::MappingInputCount`] when a mapping does not take one input per grid dimension and prefetched
    /// array, [`KernelError::MappingInputIdentity`] when a mapping shares one dimension identity between coordinates
    /// that may differ, and [`KernelError::MappingInputBounds`] when a mapping input excludes a grid coordinate.
    pub fn new_with_prefetch(
        grid: Grid,
        parameters: Vec<KernelParameter>,
        prefetch_types: Vec<ArrayType>,
    ) -> Result<Self, KernelError> {
        for r#type in &prefetch_types {
            DimensionFromScalarOperation::new(DimensionVariable::new("prefetch", DimensionBounds::unbounded()).into())
                .infer_output_types(&[ArrayIrType::Array(r#type.clone())], &[])?;
            if r#type.sharding().is_some() {
                return Err(TypeError::invalid(format!(
                    "scalar prefetch type `{type}` has a sharding, but a kernel runs on one device and its call \
                     accepts placed inputs",
                ))
                .into());
            }
        }
        let maximum_extents = grid
            .dimensions()
            .iter()
            .map(|dimension| match dimension.extent() {
                Dimension::Static(extent) => *extent,
                Dimension::Dynamic(variable) => variable.bounds().upper().unwrap() - 1,
            })
            .collect::<Vec<_>>();
        for (parameter, metadata) in parameters.iter().enumerate() {
            let actual = metadata.mapping.program().input_types().len();
            let expected = grid.dimensions().len() + prefetch_types.len();
            if actual != expected {
                return Err(KernelError::MappingInputCount { parameter, expected, actual });
            }
            // A permanently empty grid has no coordinate bindings to validate. Otherwise every mapping must admit
            // the full coordinate envelope, including zero for dimensions whose runtime extent can also be zero.
            if !maximum_extents.contains(&0) {
                for (axis, (r#type, &maximum)) in
                    metadata.mapping.program().input_types().iter().zip(&maximum_extents).enumerate()
                {
                    let ArrayIrType::Dimension(r#type) = r#type else { unreachable!() };
                    for (first_axis, first_type) in
                        metadata.mapping.program().input_types().iter().take(axis).enumerate()
                    {
                        let ArrayIrType::Dimension(first_type) = first_type else { unreachable!() };
                        if first_type.variable().is_some()
                            && first_type.variable() == r#type.variable()
                            && (maximum > 1 || maximum_extents[first_axis] > 1)
                        {
                            return Err(KernelError::MappingInputIdentity { parameter, first_axis, second_axis: axis });
                        }
                    }
                    let expected = DimensionBounds::non_negative(Some(maximum)).unwrap();
                    let actual = r#type.bounds();
                    if !actual.contains_bounds(expected) {
                        return Err(KernelError::MappingInputBounds { parameter, axis, expected, actual });
                    }
                }
            }
            // Specialization binds every prefetched dimension to a constant, which cannot preserve a runtime equality
            // with a remaining grid coordinate, so `BlockMapping::specialize` rejects such a shared identity for every
            // prefetched value, even on an empty grid.
            let input_types = metadata.mapping.program().input_types();
            let (coordinate_types, prefetch_input_types) = input_types.split_at(grid.dimensions().len());
            for (offset, prefetch_type) in prefetch_input_types.iter().enumerate() {
                let ArrayIrType::Dimension(prefetch_type) = prefetch_type else { unreachable!() };
                for (first_axis, r#type) in coordinate_types.iter().enumerate() {
                    let ArrayIrType::Dimension(r#type) = r#type else { unreachable!() };
                    if prefetch_type.variable().is_some() && r#type.variable() == prefetch_type.variable() {
                        return Err(KernelError::MappingInputIdentity {
                            parameter,
                            first_axis,
                            second_axis: grid.dimensions().len() + offset,
                        });
                    }
                }
            }
        }
        let coordinate_types = maximum_extents
            .iter()
            .enumerate()
            .map(|(axis, &extent)| {
                DimensionType::new(
                    format!("program_{axis}"),
                    DimensionBounds::non_negative(Some(extent.max(1))).unwrap(),
                )
            })
            .collect();
        Ok(Self { grid, parameters, coordinate_types, prefetch_types })
    }

    /// Reconstructs checked source metadata while retaining coordinate identities and bounds from specialization.
    pub(crate) fn from_parts(
        grid: Grid,
        parameters: Vec<KernelParameter>,
        prefetch_types: Vec<ArrayType>,
        coordinate_types: Vec<DimensionType>,
    ) -> Result<Self, KernelError> {
        let mut operation = Self::new_with_prefetch(grid, parameters, prefetch_types)?;
        if coordinate_types.len() != operation.coordinate_types.len() {
            return Err(TypeError::invalid("coordinate type count differs from the grid rank").into());
        }
        for (axis, (actual, required)) in coordinate_types.iter().zip(&operation.coordinate_types).enumerate() {
            if actual.bounds().upper().is_none() || !actual.bounds().contains_bounds(required.bounds()) {
                return Err(TypeError::invalid(format!("coordinate {axis} bounds do not admit the grid")).into());
            }
            if actual.variable().is_some()
                && coordinate_types[..axis].iter().any(|earlier| earlier.variable() == actual.variable())
            {
                return Err(
                    TypeError::invalid(format!("coordinate {axis} repeats an earlier coordinate identity")).into()
                );
            }
        }
        operation.coordinate_types = coordinate_types;
        Ok(operation)
    }

    /// Returns the logical grid.
    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// Returns parameter metadata in body input order.
    pub fn parameters(&self) -> &[KernelParameter] {
        &self.parameters
    }

    /// Returns coordinate types in logical grid axis order. Each identity belongs to this call definition and admits
    /// indices from zero up to the maximum possible extent. Empty axes use `[0, 1)` because no invocation occurs.
    pub fn coordinate_types(&self) -> &[DimensionType] {
        &self.coordinate_types
    }

    /// Returns the rank-0 integer array types in scalar prefetch order.
    pub fn prefetch_types(&self) -> &[ArrayType] {
        &self.prefetch_types
    }

    /// Returns the body signature, which consists of the block references, then the dimension coordinates, and then
    /// the scalar prefetched arrays.
    pub fn body_input_types(&self) -> Vec<ArrayIrType> {
        self.parameters
            .iter()
            .map(KernelParameter::body_type)
            .chain(self.coordinate_types.iter().cloned().map(ArrayIrType::Dimension))
            .chain(self.prefetch_types.iter().cloned().map(ArrayIrType::Array))
            .collect()
    }

    /// Returns the declared outer array input types, excluding write-only outputs and appending scalar prefetch. An
    /// actual input may additionally be placed (refer to the type documentation).
    pub fn input_types(&self) -> Vec<ArrayIrType> {
        self.parameters
            .iter()
            .filter(|parameter| parameter.access != KernelParameterAccess::WriteOnly)
            .map(|parameter| ArrayIrType::Array(parameter.r#type.clone()))
            .chain(self.prefetch_types.iter().cloned().map(ArrayIrType::Array))
            .collect()
    }

    /// Returns the declared result types in parameter order, which are the result types of a call on inputs without a
    /// sharding. [`Self::output_types_for`] returns the result types for placed inputs.
    pub fn output_types(&self) -> Vec<ArrayIrType> {
        self.parameters
            .iter()
            .filter(|parameter| parameter.access != KernelParameterAccess::ReadOnly)
            .map(|parameter| ArrayIrType::Array(parameter.r#type.clone()))
            .collect()
    }

    /// Returns the result types of this call for the provided actual `input_types`, which may be placed (refer to the
    /// type documentation). The result of a read-write parameter has the type of its input. A write-only result has
    /// its declared type, replicated on the mesh of the inputs and varying along each manual axis along which an input
    /// varies when there is such an axis.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] when an input is not its declared array type up to its sharding, when an input carries
    /// a pending reduction, when placed inputs use different meshes, or when the input of a read-write parameter does
    /// not vary along a manual axis along which another input varies.
    pub fn output_types_for(&self, input_types: &[ArrayIrType]) -> Result<Vec<ArrayIrType>, TypeError> {
        let mismatch = || {
            TypeError::invalid(format!("`{KERNEL_CALL_OPERATION_NAME}` input types do not match its logical signature"))
        };
        let declared = self.input_types();
        if input_types.len() != declared.len() {
            return Err(mismatch());
        }
        let mut mesh = None::<&LogicalMesh>;
        let mut varying_axes = BTreeSet::new();
        for (actual, declared) in input_types.iter().zip(&declared) {
            let (ArrayIrType::Array(actual), ArrayIrType::Array(declared)) = (actual, declared) else {
                return Err(mismatch());
            };
            if actual.data_type() != declared.data_type()
                || actual.shape() != declared.shape()
                || actual.memory() != declared.memory()
                || actual.layout() != declared.layout()
            {
                return Err(mismatch());
            }
            let Some(sharding) = actual.sharding() else { continue };
            if !sharding.unreduced_axes().is_empty() || !sharding.reduced_axes().is_empty() {
                return Err(TypeError::invalid(format!(
                    "`{KERNEL_CALL_OPERATION_NAME}` input type `{actual}` carries a pending reduction",
                )));
            }
            if mesh.is_some_and(|mesh| mesh != sharding.mesh()) {
                return Err(TypeError::invalid(format!(
                    "`{KERNEL_CALL_OPERATION_NAME}` placed inputs must use the same mesh",
                )));
            }
            mesh = Some(sharding.mesh());
            varying_axes.extend(sharding.varying_manual_axes().iter().cloned());
        }
        let mut inputs = input_types.iter();
        let mut outputs = Vec::new();
        for (index, parameter) in self.parameters.iter().enumerate() {
            match parameter.access {
                KernelParameterAccess::ReadOnly => {
                    inputs.next();
                }
                KernelParameterAccess::ReadWrite => {
                    // The result is the storage of the input, so it has the type of the input. The body may store a
                    // value that differs on each device into that storage, which the input must therefore admit.
                    let Some(ArrayIrType::Array(input)) = inputs.next() else { return Err(mismatch()) };
                    let varies = |axis: &String| {
                        input.sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(axis))
                    };
                    if let Some(axis) = varying_axes.iter().find(|axis| !varies(axis)) {
                        return Err(TypeError::invalid(format!(
                            "`{KERNEL_CALL_OPERATION_NAME}` read-write parameter {index} receives `{input}`, which \
                             does not vary along manual axis `{axis}` like another input, so its result could not \
                             hold a value that differs on each device",
                        )));
                    }
                    outputs.push(ArrayIrType::Array(input.clone()));
                }
                KernelParameterAccess::WriteOnly => {
                    let sharding = mesh
                        .filter(|_| !varying_axes.is_empty())
                        .map(|mesh| {
                            Sharding::replicated(mesh.clone(), parameter.r#type.rank())
                                .with_varying_manual_axes(varying_axes.clone())
                        })
                        .transpose()
                        .map_err(|error| TypeError::invalid(error.to_string()))?;
                    let r#type = parameter.r#type.clone().with_sharding(sharding);
                    outputs.push(ArrayIrType::Array(r#type.map_err(|error| TypeError::invalid(error.to_string()))?));
                }
            }
        }
        Ok(outputs)
    }

    /// Returns `(output, input)` alias candidates for read-write parameters, which are local to this operation. The
    /// execution backend must honor its ordinary donation and liveness rules before reusing storage, because these are
    /// not external state slots.
    pub fn aliases(&self) -> Vec<(usize, usize)> {
        let mut aliases = Vec::new();
        let mut input = 0;
        let mut output = 0;
        for parameter in &self.parameters {
            match parameter.access {
                KernelParameterAccess::ReadOnly => input += 1,
                KernelParameterAccess::WriteOnly => output += 1,
                KernelParameterAccess::ReadWrite => {
                    aliases.push((output, input));
                    input += 1;
                    output += 1;
                }
            }
        }
        aliases
    }

    /// Derives the body access declaration (i.e., the [`KernelBoundaryContract`]) from the parameter metadata without
    /// storing a second access table.
    pub fn boundary_contract(&self) -> KernelBoundaryContract {
        KernelBoundaryContract::new(
            self.parameters
                .iter()
                .map(|parameter| Some(parameter.access))
                .chain(self.coordinate_types.iter().map(|_| None))
                .chain(self.prefetch_types.iter().map(|_| None))
                .collect(),
        )
    }

    /// Specializes bounded grid extents while retaining the original coordinate identities and body signature.
    /// Mapping programs and their bounds remain unchanged, and the concrete grid becomes part of semantic identity.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::Grid`] when [`Grid::specialize`] rejects `extents`.
    pub fn specialize_grid(&self, extents: &[usize]) -> Result<Self, KernelError> {
        Ok(Self { grid: self.grid.specialize(extents)?, ..self.clone() })
    }

    /// Specializes scalar prefetched values into every mapping, checking the dimension bounds of each mapping. Body
    /// constants must be bound as well, so use [`KernelDefinition::specialize_prefetch`] for an attached definition.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::Type`] when the number or types of `values` differ from [`Self::prefetch_types`],
    /// [`KernelError::Program`] when a value does not convert to a dimension within the bounds of its mapping input,
    /// and [`KernelError::Mapping`] when [`BlockMapping::specialize`] rejects the converted dimensions.
    pub fn specialize_prefetch(&self, values: &[Array]) -> Result<Self, KernelError> {
        if values.len() != self.prefetch_types.len() {
            return Err(TypeError::invalid(format!(
                "kernel scalar prefetch value count must be {} but is {}",
                self.prefetch_types.len(),
                values.len(),
            ))
            .into());
        }
        for (position, (value, expected)) in values.iter().zip(&self.prefetch_types).enumerate() {
            if value.r#type().as_ref() != expected {
                return Err(TypeError::invalid(format!(
                    "kernel scalar prefetch {position} expected type `{expected}` but received `{}`",
                    value.r#type(),
                ))
                .into());
            }
        }
        let parameters = self
            .parameters
            .iter()
            .map(|parameter| {
                let dimensions = parameter
                    .mapping
                    .program()
                    .input_types()
                    .iter()
                    .skip(self.grid.dimensions().len())
                    .zip(values)
                    .map(|(r#type, value)| {
                        let ArrayIrType::Dimension(r#type) = r#type else { unreachable!() };
                        value.to_dimension(r#type.to_variable())
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                Ok(KernelParameter { mapping: parameter.mapping.specialize(&dimensions)?, ..parameter.clone() })
            })
            .collect::<Result<Vec<_>, KernelError>>()?;
        Ok(Self { parameters, prefetch_types: vec![], ..self.clone() })
    }

    /// Validates the exact attached body against the type signature and boundary contract of this call. A `shard_map`
    /// anywhere in the attached region closure of the body is rejected first, with
    /// [`KernelError::UnsupportedOperation`]. The region access policy of this call, which ordinary program analysis
    /// applies to the body, admits every access that does not consume its parameter. The generic reference analysis
    /// that [`validate_kernel_body`] runs already rejects every consumption of a body input, so direct definitions and
    /// eager invocation enforce the same policy without checking it again.
    pub(crate) fn validate_body<Extension>(
        &self,
        region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
    ) -> Result<KernelReferenceSummary, KernelError>
    where
        Extension: ReferenceAccessOperation<Type = ArrayIrType, Transform = ArrayReferenceTransform>,
    {
        // Every attached region of the body executes as part of the kernel, including the regions of instructions whose
        // results are unused, so the whole closure is inspected.
        if let Some((instruction, operation)) = region.instructions_in_closure().find_map(|(id, instruction)| {
            matches!(instruction.operation(), KernelOperation::Portable(ArrayIrOperation::ShardMap(_)))
                .then(|| (id, instruction.operation().name()))
        }) {
            return Err(KernelError::UnsupportedOperation { operation, instruction });
        }
        self.infer_output_types(&self.input_types(), std::slice::from_ref(region.interface()))?;
        Ok(validate_kernel_body(region, &self.boundary_contract())?)
    }

    /// Collects nominal identity occurrences in metadata order so rendering preserves relationships between grid
    /// extents, full array dimensions, and mapping values without exposing allocation addresses.
    pub(crate) fn identity_occurrences(&self) -> Vec<DimensionVariable> {
        let mut identities = Vec::new();
        for dimension in self.grid.dimensions() {
            if let Dimension::Dynamic(variable) = dimension.extent() {
                identities.push(variable.clone());
            }
        }
        identities.extend(self.coordinate_types.iter().filter_map(DimensionType::variable).cloned());
        identities.extend(
            self.prefetch_types
                .iter()
                .flat_map(|r#type| r#type.identities().map(|(_, identity)| identity.clone())),
        );
        for parameter in &self.parameters {
            identities.extend(parameter.r#type.identities().map(|(_, identity)| identity.clone()));
            for region in parameter.mapping.program().regions() {
                for atom in region.atoms() {
                    identities.extend(atom.r#type().identities().map(|(_, identity)| identity.clone()));
                }
            }
        }
        identities
    }

    /// Returns the effect classes of the mapping programs together with those of converting each scalar prefetched
    /// value to the dimension type of every mapping input that receives it. These cover every failure that
    /// [`Self::specialize_prefetch`] can raise, and [`Operation::effects`] adds the window checks to them.
    pub(crate) fn mapping_program_effects(&self) -> EffectClasses {
        self.parameters.iter().fold(EffectClasses::NONE, |classes, parameter| {
            let program = parameter.mapping.program();
            program.input_types().iter().skip(self.grid.dimensions().len()).fold(
                classes.union(program.effects().classes()),
                |classes, r#type| {
                    let ArrayIrType::Dimension(r#type) = r#type else { unreachable!() };
                    classes.union(DimensionFromScalarOperation::new(r#type.clone()).effects().classes())
                },
            )
        })
    }
}

impl Display for KernelCallOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for KernelCallOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        KERNEL_CALL_OPERATION_NAME
    }

    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("body")] }
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        if region_interfaces.len() != 1 {
            return Err(TypeError::invalid(format!("`{KERNEL_CALL_OPERATION_NAME}` requires exactly one body region")));
        }
        let output_types = self.output_types_for(input_types)?;
        let expected = self.body_input_types();
        if region_interfaces[0].input_types() != expected {
            return Err(TypeError::invalid(format!(
                "`{KERNEL_CALL_OPERATION_NAME}` body input types do not match its logical block references and \
                 coordinates",
            )));
        }
        if !region_interfaces[0].output_types().is_empty() {
            return Err(TypeError::invalid(format!(
                "`{KERNEL_CALL_OPERATION_NAME}` body must publish results through its declared output references",
            )));
        }
        Ok(output_types)
    }

    fn region_capture_input_count(&self, region_index: usize) -> Option<usize> {
        (region_index == 0).then_some(0)
    }

    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        if region_index == 0 && input_index < self.parameters.len() + self.coordinate_types.len() {
            InputRegionProvenance::Local
        } else {
            InputRegionProvenance::None
        }
    }

    fn allows_reference_access_through_region_input(&self, region_index: usize, mode: ReferenceAccessMode) -> bool {
        // Non-atomic accumulation is admitted on every grid, including grids with parallel axes. This is sound because
        // the launch window qualification of `validate_kernel_initialization` counts an accumulated parameter as
        // mutated and, unless every access of that parameter is atomic, rejects mutable windows of grid points that the
        // grid does not order and that overlap with `KernelInitializationError::OverlappingWindows`. So two grid points
        // that may run concurrently never accumulate into the same element.
        region_index == 0 && !mode.is_consuming()
    }

    fn effects(&self) -> Cow<'_, Effects> {
        // The mappings of every grid point run before its body, after every scalar prefetched value is converted to the
        // dimension type of each mapping input that receives it, and the window that a mapping selects is then checked
        // against its full array. A failure of any of these is observable even when the results of the call are unused,
        // so their assertions belong to the call itself. The body effects reach the call through its attached
        // computation region, and a mapping has no references, so the reference state of the body stays the only state
        // of the call.
        let mut classes = self.mapping_program_effects();
        if self
            .parameters
            .iter()
            .any(|parameter| !parameter.mapping.has_total_windows(parameter.r#type.shape()))
        {
            classes = classes.union(EffectClasses::single(EffectClass::OrderedAssertion));
        }
        if classes.is_empty() { Cow::Borrowed(Effects::empty()) } else { Cow::Owned(Effects::explicit(classes)) }
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        let parameters = self
            .parameters
            .iter()
            .map(|parameter| {
                let program =
                    parameter.mapping.program().rename_type_identities(renaming).map_err(TypeError::custom)?;
                let mapping = BlockMapping::new(
                    program,
                    parameter.mapping.block_shape().to_vec(),
                    parameter.mapping.boundary_policy(),
                )
                .map_err(TypeError::custom)?;
                Ok(KernelParameter {
                    r#type: parameter.r#type.rename_identities(renaming)?,
                    access: parameter.access,
                    mapping,
                    body_type: parameter.body_type.rename_identities(renaming)?,
                })
            })
            .collect::<Result<Vec<_>, TypeError>>()?;
        Ok(Self {
            grid: self.grid.rename_type_identities(renaming).map_err(TypeError::custom)?,
            parameters,
            prefetch_types: self
                .prefetch_types
                .iter()
                .map(|r#type| r#type.rename_identities(renaming))
                .collect::<Result<_, _>>()?,
            coordinate_types: self
                .coordinate_types
                .iter()
                .map(|coordinate| coordinate.rename_identities(renaming))
                .collect::<Result<_, _>>()?,
        })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("grid", &self.grid)?;
            for (index, r#type) in self.prefetch_types.iter().enumerate() {
                operation.field(&format!("prefetch_{index}"), r#type)?;
            }
            let mut identities = Vec::new();
            let occurrences = self
                .identity_occurrences()
                .into_iter()
                .map(|identity| {
                    if let Some(index) = identities.iter().position(|existing| existing == &identity) {
                        index
                    } else {
                        identities.push(identity);
                        identities.len() - 1
                    }
                })
                .collect::<Vec<_>>();
            if !occurrences.is_empty() {
                operation.field("identities", format!("{occurrences:?}"))?;
                for (index, identity) in identities.iter().enumerate() {
                    operation.field(&format!("identity_{index}_bounds"), identity.bounds())?;
                }
            }
            for (index, parameter) in self.parameters.iter().enumerate() {
                operation.field(
                    &format!("parameter_{index}"),
                    format!(
                        "{} {} {:?} {:?}",
                        parameter.access,
                        parameter.r#type,
                        parameter.mapping.block_shape(),
                        parameter.mapping.boundary_policy(),
                    ),
                )?;
                operation.program(&format!("mapping_{index}"), parameter.mapping.program())?;
            }
            Ok(())
        })
    }
}

impl ReferenceAccessOperation for KernelCallOperation {
    type Transform = ArrayReferenceTransform;

    fn base_input_count(&self) -> usize {
        self.input_types().len()
    }

    fn reference_access_descriptor(
        &self,
        _input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
        // The call takes ordinary arrays at its outer boundary, so it accesses no reference of the enclosing program.
        None
    }

    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        _transforms: Vec<ArrayReferenceTransform>,
    ) -> Result<Self, ProgramError> {
        Err(ProgramError::MalformedProgram(format!(
            "operation `{}` does not support the requested transforms at input {input_index}",
            self.name(),
        )))
    }
}

/// Immutable body and its validated logical call boundary. Rewrites produce a new [`FlatProgram`] and must construct
/// a new definition so reference validation cannot remain attached to changed semantics.
///
/// This boundary validation does not yet grant execution, because initialization, output coverage, and parallel races
/// are checked by the kernel execution verifier (i.e., [`VerifiedKernel`](crate::compilation::VerifiedKernel)). The
/// body remains an ordinary Ryft program, with no adapter dependencies.
#[derive(Clone, Debug)]
pub struct KernelDefinition<Extension: Operation<Type = ArrayIrType> = NoKernelExtension> {
    /// Full logical call metadata.
    operation: KernelCallOperation,

    /// Immutable ordinary program containing the kernel's reference operations.
    body: FlatProgram<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,

    /// Reference analysis and access decisions for this exact body.
    references: KernelReferenceSummary,
}

impl<Extension> KernelDefinition<Extension>
where
    Extension: ReferenceAccessOperation<Type = ArrayIrType, Transform = ArrayReferenceTransform>,
{
    /// Validates the actual body and its call interface before retaining either. Reference constants, reference
    /// outputs, undeclared accesses, incompatible block types, and `shard_map` instructions anywhere in the attached
    /// region closure of the body fail before a definition becomes observable.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::UnsupportedOperation`] for a `shard_map` in the body, [`KernelError::Type`] when the body
    /// interface does not match the call signature, and [`KernelError::Validation`] when [`validate_kernel_body`]
    /// rejects the reference accesses of the body.
    pub fn new(
        operation: KernelCallOperation,
        body: FlatProgram<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
    ) -> Result<Self, KernelError> {
        let references = operation.validate_body(body.entry_region_ref())?;
        Ok(Self { operation, body, references })
    }

    /// Traces a body through ordinary Ryft staging and validates the finalized program. The closure receives a
    /// tuple of reference tracers in parameter order and dimension tracers in grid axis order. Stores publish the
    /// declared outputs. The closure runs once during staging, and runtime branches and loops use canonical staged
    /// control flow capabilities. The canonical tuple parameter machinery flattens the finished body signature.
    ///
    /// Foreign tracers, escaped builders, discarded captures, and poisoned operations retain the existing tracing
    /// diagnostics. No partially constructed definition is returned after either tracing or validation fails.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::Program`] when tracing or `function` fails, and the errors of [`Self::new`] when the
    /// traced body fails validation.
    pub fn trace<F>(operation: KernelCallOperation, function: F) -> Result<Self, KernelError>
    where
        F: FnOnce((Vec<KernelTracer<Extension>>, Vec<KernelTracer<Extension>>)) -> Result<(), ProgramError>,
    {
        Self::trace_with_prefetch(operation, |(references, coordinates, _prefetched)| {
            function((references, coordinates))
        })
    }

    /// Traces references, grid coordinates, and scalar prefetched arrays as three canonical parameter groups.
    /// Prefetched values remain private arrays in the body until explicit specialization binds their constants.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::Program`] when tracing or `function` fails, and the errors of [`Self::new`] when the
    /// traced body fails validation.
    pub fn trace_with_prefetch<F>(operation: KernelCallOperation, function: F) -> Result<Self, KernelError>
    where
        F: FnOnce(
            (Vec<KernelTracer<Extension>>, Vec<KernelTracer<Extension>>, Vec<KernelTracer<Extension>>),
        ) -> Result<(), ProgramError>,
    {
        let input_types = (
            operation.parameters().iter().map(KernelParameter::body_type).collect::<Vec<_>>(),
            operation.coordinate_types().iter().cloned().map(ArrayIrType::Dimension).collect::<Vec<_>>(),
            operation.prefetch_types().iter().cloned().map(ArrayIrType::Array).collect::<Vec<_>>(),
        );
        let (_, body) = TracingContext::<ArrayIrValue<Array>, KernelOperation<Extension>>::trace(
            |inputs| {
                function(inputs)?;
                Ok(Vec::<KernelTracer<Extension>>::new())
            },
            input_types,
        )?;
        Self::new(operation, body.into_flat_program())
    }

    /// Specializes a bounded grid without retracing the body. Existing coordinate identities and reference types
    /// remain valid, and the resulting definition is checked against its specialized call. Execution still verifies
    /// initialization and complete output coverage for the selected extents before allocating memory.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`KernelCallOperation::specialize_grid`] and of [`Self::new`].
    pub fn specialize_grid(&self, extents: &[usize]) -> Result<Self, KernelError> {
        Self::new(self.operation.specialize_grid(extents)?, self.body.clone())
    }

    /// Binds scalar prefetched arrays to canonical mapping and body constants without executing body effects. The
    /// ordinary program splicer preserves nested regions and validates type identities, and the resulting definition
    /// retains the functional boundary of arrays with these specialized inputs removed. Exact constant values
    /// participate in [`Self::semantic_key`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`KernelCallOperation::specialize_prefetch`], [`KernelError::Program`] when splicing the
    /// body fails, and the errors of [`Self::new`].
    pub fn specialize_prefetch(&self, values: &[Array]) -> Result<Self, KernelError> {
        let operation = self.operation.specialize_prefetch(values)?;
        let remaining = self.body.input_types().len() - values.len();
        let mut builder = ProgramBuilder::new();
        let mut inputs = self
            .body
            .input_types()
            .into_owned()
            .into_iter()
            .take(remaining)
            .map(|r#type| builder.add_input(r#type))
            .collect::<Vec<_>>();
        inputs.extend(values.iter().cloned().map(|value| builder.add_constant(ArrayIrValue::Array(value))));
        let outputs = builder.splice_program(&self.body, &inputs)?;
        Self::new(operation, builder.build(outputs, vec![Placeholder; remaining], vec![])?)
    }

    /// Removes dead work with the existing program simplifier and validates the resulting body boundary again.
    /// Observable effects and their relative order survive, including reference writes whose results are unused.
    /// This returns a new definition, and any previous executable verification applies only to the original definition.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::Program`] when simplification fails, and the errors of [`Self::new`].
    pub fn simplified(&self) -> Result<Self, KernelError> {
        Self::new(self.operation.clone(), self.body.clone().into_simplified()?)
    }
}

impl<Extension: Operation<Type = ArrayIrType>> KernelDefinition<Extension> {
    /// Returns the logical call metadata.
    pub fn operation(&self) -> &KernelCallOperation {
        &self.operation
    }

    /// Returns the immutable body program.
    pub fn body(&self) -> &FlatProgram<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>> {
        &self.body
    }

    /// Returns the reference summary for this exact immutable body.
    pub fn references(&self) -> &KernelReferenceSummary {
        &self.references
    }

    /// Returns the observable effects of one application of this definition, which is the authoritative summary for
    /// every consumer that decides whether a call may be eliminated, replaced, or reordered (e.g., a pure fallback
    /// or a compiler embedding). It combines the effects of the call itself (refer to [`KernelCallOperation`], whose
    /// mappings, scalar prefetch conversions, and window checks may fail) with those of the body, which is exactly
    /// what an enclosing program derives for an instruction that applies the call. The reference state of the body is
    /// local to the call and appears as [`EffectClass::OrderedState`] without an explicit declaration, which is the
    /// state that an embedding may discharge at the call boundary.
    pub fn effects(&self) -> EffectsSummary {
        self.operation.effects().summary().union(self.body.effects())
    }

    /// Returns a versioned structural key of this definition, which identifies its semantics for caching.
    ///
    /// The key is derived from the same wire records as [`KernelDefinition::to_source`], encoded in the key mode of
    /// the serializer and written as compact JSON (refer to [`serialization`](crate::serialization) for the format).
    /// Key mode projects out what is not semantic. Nominal dimension names are dropped, and identities are numbered
    /// in order of occurrence, while bounds and repeated identity relationships remain significant. Grid axis names
    /// and instruction provenance are excluded. Typed literal arrays retain their exact logical bytes in both atom and
    /// operation payloads, including mapping programs, and floating-point metadata retains its IEEE bits. Adapter
    /// options, schedules, and invocation values do not belong to this key.
    ///
    /// This is an encoding of cache identity, not a round trip executable serialization format, and it covers every
    /// valid definition rather than only the transport subset. Features that [`KernelDefinition::to_source`] rejects
    /// are encoded through key-only records. Nested kernel calls, custom calls, attached composites, gathers,
    /// scatters, the shardings of distributed array types, requested output shardings, sharding changes and
    /// constraints, and dropout are encoded structurally, and a registered extension provides its exact versioned
    /// payload through [`KernelExtension::semantic_key`]. Shardings keep their mesh axis names, which are semantic.
    /// Operations without a wire record are transitionally encoded by their diagnostic renderings after canonical
    /// identity renaming. Definitions whose identity is local to the process return an error instead of inheriting it
    /// from diagnostic formatting.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] when the body contains a reference constant, an extension without an exact semantic
    /// identity, or an operation whose identity is local to the process (e.g., `rematerialize` operations and
    /// `composite` calls with retained rules that are not named composites).
    pub fn semantic_key(&self) -> Result<String, TypeError>
    where
        Extension: KernelExtension,
    {
        encode_semantic_key(self)
    }
}

/// Tracer of a kernel body, which [`KernelDefinition::trace`] and [`KernelDefinition::trace_with_prefetch`] pass to
/// their tracing closures.
type KernelTracer<Extension> = Tracer<TracingContext<ArrayIrValue<Array>, KernelOperation<Extension>>>;

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayOperation, ArrayReferenceTransformIndex, AtomId, CompositeAttribute, CompositeJvpRule, CompositeKey,
        CompositeOperation, CompositeTransposeOperation, ConditionOperation, ConstantOperation, Context,
        CustomCallAttribute, CustomCallOperation, CustomRuleDefinition, CustomRuleRegistration, DataType,
        DimensionError, DimensionMulOperation, DimensionValue, EffectClass, EffectClasses, GatherDimensionNumbers,
        GatherMode, GatherOperation, Layout, LogicalMesh, MAX_DIMENSION_EXTENT, Memory, MeshAxis, MeshAxisType,
        NamedComposite, NothingSavable, PrintOperation, Provenance, ProvenanceScope, ReduceOperation, ReductionKind,
        ReferenceRead, ReferenceReadOperation, ReferenceWrite, ReferenceWriteOperation, RegionId,
        RematerializeOperation, ReshapeOperation, ResidualPolicyReference, ScatterDimensionNumbers, ScatterMode,
        ScatterOperation, ScatterReductionKind, Shape, ShardMapBoundary, ShardMapOperation, Sharding,
        ShardingDimension, TiledLayout, UnreducedKind, ZeroOperation,
    };

    use crate::authoring::{MappingExpression, ParameterWindow, mapped_call, whole_array_parameter};
    use crate::grids::{GridDimension, GridExecution};
    use crate::initialization::KernelInitializationError;
    use crate::memory::{MaskedStoreOperation, WindowMaskOperation};
    use crate::operations::tests::TestExtension;

    use super::*;

    /// Constant rank-0 mapping used by the scalar call boundary tests.
    fn scalar_mapping() -> BlockMapping {
        let program: FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>> =
            ProgramBuilder::new().build(vec![], vec![], vec![]).unwrap();
        BlockMapping::new(program, vec![], BoundaryPolicy::InBounds).unwrap()
    }

    /// Scalar parameter with a singleton block that covers the whole value.
    fn parameter(access: KernelParameterAccess) -> KernelParameter {
        KernelParameter::new(ArrayType::scalar(DataType::I32), access, scalar_mapping()).unwrap()
    }

    /// Copy body whose input and output references remain internal to a functional call.
    fn copy_definition() -> KernelDefinition {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![parameter(KernelParameterAccess::ReadOnly), parameter(KernelParameterAccess::WriteOnly)],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let input = builder.add_input(operation.parameters()[0].body_type());
        let output = builder.add_input(operation.parameters()[1].body_type());
        let value = builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![input], None).unwrap()[0];
        builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, value], None).unwrap();
        KernelDefinition::new(operation, builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap()).unwrap()
    }

    /// Empty scalar body with independently named mapping and grid identities for semantic key tests.
    fn dynamic_definition(name: &str, upper: usize, shared: bool) -> KernelDefinition {
        let variable = DimensionVariable::new(name, DimensionBounds::non_negative(Some(upper)).unwrap());
        let second = if shared { variable.clone() } else { DimensionVariable::new(name, variable.bounds()) };
        let grid = Grid::new(vec![
            GridDimension::new(Dimension::Dynamic(variable), GridExecution::Parallel),
            GridDimension::new(Dimension::Dynamic(second), GridExecution::Parallel),
        ])
        .unwrap();
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        for _ in 0..2 {
            mapping.add_input(ArrayIrType::Dimension(DimensionType::new(name, DimensionBounds::unbounded())));
        }
        let parameter = KernelParameter::new(
            ArrayType::scalar(DataType::I32),
            KernelParameterAccess::ReadOnly,
            BlockMapping::new(
                mapping.build(vec![], vec![Placeholder; 2], vec![]).unwrap(),
                vec![],
                BoundaryPolicy::InBounds,
            )
            .unwrap(),
        )
        .unwrap();
        let operation = KernelCallOperation::new(grid, vec![parameter]).unwrap();
        let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        for r#type in operation.body_input_types() {
            body.add_input(r#type);
        }
        KernelDefinition::new(operation, body.build(vec![], vec![Placeholder; 3], vec![]).unwrap()).unwrap()
    }

    /// Reads an array window at a prefetched offset and publishes the prefetched scalar through a singleton output.
    fn prefetch_definition() -> KernelDefinition {
        let coordinate = DimensionType::new("coordinate", DimensionBounds::non_negative(Some(1)).unwrap());
        let offset = DimensionType::new("offset", DimensionBounds::non_negative(Some(8)).unwrap());
        let mut input_mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        input_mapping.add_input(coordinate.clone().into());
        let offset_input = input_mapping.add_input(offset.clone().into());
        let input_mapping = BlockMapping::new(
            input_mapping.build(vec![offset_input], vec![Placeholder; 2], vec![Placeholder]).unwrap(),
            vec![2],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let mut output_mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        output_mapping.add_input(coordinate.into());
        output_mapping.add_input(offset.into());
        let output_mapping = BlockMapping::new(
            output_mapping.build(vec![], vec![Placeholder; 2], vec![]).unwrap(),
            vec![],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let operation = KernelCallOperation::new_with_prefetch(
            Grid::new(vec![GridDimension::new(Dimension::Static(1), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, vec![8]),
                    KernelParameterAccess::ReadOnly,
                    input_mapping,
                )
                .unwrap(),
                KernelParameter::new(
                    ArrayType::scalar(DataType::I32),
                    KernelParameterAccess::WriteOnly,
                    output_mapping,
                )
                .unwrap(),
            ],
            vec![ArrayType::scalar(DataType::I32)],
        )
        .unwrap();
        KernelDefinition::trace_with_prefetch(operation, |(references, _coordinates, values)| {
            references[1].write(&values[0])
        })
        .unwrap()
    }

    /// Read-only call with no results and an empty body whose single window starts at `start`, a mapping expression
    /// over one signed scalar prefetched value.
    fn prefetch_mapping_definition(start: MappingExpression) -> KernelDefinition {
        let operation = mapped_call(
            Grid::new(vec![]).unwrap(),
            vec![(
                ArrayType::new_static(DataType::I32, vec![4]),
                KernelParameterAccess::ReadOnly,
                ParameterWindow::ElementStarts {
                    block_shape: vec![1],
                    boundary: BoundaryPolicy::InBounds,
                    starts: vec![start],
                },
            )],
            vec![ArrayType::scalar(DataType::I32)],
        )
        .unwrap();
        KernelDefinition::trace_with_prefetch(operation, |_| Ok(())).unwrap()
    }

    /// Applies `definition` to the inputs of a program that leaves every result of the call unused.
    fn unused_call_program(
        definition: &KernelDefinition,
    ) -> FlatProgram<EagerContext<ArrayIrValue<Array>, KernelOperation>> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let inputs = definition
            .operation()
            .input_types()
            .into_iter()
            .map(|r#type| builder.add_input(r#type))
            .collect::<Vec<_>>();
        let body = builder.import_region(definition.body().entry_region_ref());
        builder
            .add_instruction(KernelOperation::Call(definition.operation().clone()), vec![body], inputs.clone(), None)
            .unwrap();
        builder.build(vec![], vec![Placeholder; inputs.len()], vec![]).unwrap()
    }

    /// Gathers one element of an `f32[2]` input at the start index held by an `i32[1, 1]` input with `operation`.
    fn gather_definition(operation: GatherOperation) -> KernelDefinition {
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::new_static(DataType::F32, vec![2]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
                whole_array_parameter(
                    ArrayType::new_static(DataType::I32, vec![1, 1]),
                    KernelParameterAccess::ReadOnly,
                )
                .unwrap(),
                whole_array_parameter(ArrayType::new_static(DataType::F32, vec![1]), KernelParameterAccess::WriteOnly)
                    .unwrap(),
            ],
        )
        .unwrap();
        KernelDefinition::trace(call, |(references, _)| {
            let input = references[0].read()?;
            let indices = references[1].read()?;
            let gathered = references[0].context().bind(
                ArrayIrOperation::Array(ArrayOperation::Gather(operation)),
                vec![],
                &[input, indices],
            )?;
            references[2].write(&gathered[0])
        })
        .unwrap()
    }

    /// Scatters an `f32[2]` update into an `f32[4]` input at the start indices held by an `i32[2, 1]` input with
    /// `operation`.
    fn scatter_definition(operation: ScatterOperation) -> KernelDefinition {
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::new_static(DataType::F32, vec![4]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
                whole_array_parameter(
                    ArrayType::new_static(DataType::I32, vec![2, 1]),
                    KernelParameterAccess::ReadOnly,
                )
                .unwrap(),
                whole_array_parameter(ArrayType::new_static(DataType::F32, vec![2]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
                whole_array_parameter(ArrayType::new_static(DataType::F32, vec![4]), KernelParameterAccess::WriteOnly)
                    .unwrap(),
            ],
        )
        .unwrap();
        KernelDefinition::trace(call, |(references, _)| {
            let inputs = references[..3].iter().map(|reference| reference.read()).collect::<Result<Vec<_>, _>>()?;
            let scattered = references[0].context().bind(
                ArrayIrOperation::Array(ArrayOperation::Scatter(operation)),
                vec![],
                &inputs,
            )?;
            references[3].write(&scattered[0])
        })
        .unwrap()
    }

    /// Shared dynamic extent across an input and output, with a fixed masked block of four elements.
    fn refinement_parameters() -> Vec<(ArrayType, KernelParameterAccess, BlockMapping)> {
        let extent = DimensionVariable::new("extent", DimensionBounds::non_negative(Some(5)).unwrap());
        let declared = ArrayType::new(DataType::I32, Shape::new(vec![extent.into()]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let zero = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let mapping = BlockMapping::new(
            builder.build(vec![zero], vec![], vec![Placeholder]).unwrap(),
            vec![4],
            BoundaryPolicy::Masked,
        )
        .unwrap();
        vec![
            (declared.clone(), KernelParameterAccess::ReadOnly, mapping.clone()),
            (declared, KernelParameterAccess::WriteOnly, mapping),
        ]
    }

    #[test]
    fn test_kernel_error_display() {
        // Wrapped errors render exactly as the errors that they wrap.
        let program = ProgramError::MalformedProgram("malformed body".to_owned());
        assert_eq!(KernelError::Program(program.clone()).to_string(), program.to_string());
        let r#type = TypeError::invalid("invalid body type");
        assert_eq!(KernelError::Type(r#type.clone()).to_string(), r#type.to_string());
        let grid = GridError::EmptyName;
        assert_eq!(KernelError::Grid(grid.clone()).to_string(), grid.to_string());
        let mapping = BlockMappingError::UnsupportedOperation { operation: "reference_read" };
        assert_eq!(KernelError::Mapping(mapping.clone()).to_string(), mapping.to_string());
        let validation = KernelValidationError::UndeclaredReferenceParameter { input_index: 0 };
        assert_eq!(KernelError::Validation(validation.clone()).to_string(), validation.to_string());

        let instruction = InstructionId::new(RegionId::new(0), 1);
        assert_eq!(
            KernelError::UnsupportedOperation { operation: "shard_map", instruction }.to_string(),
            "kernel body instruction ^0[1] applies `shard_map`, which kernel bodies do not support",
        );
        assert_eq!(
            KernelError::ParameterRank { array_rank: 1, block_rank: 0 }.to_string(),
            "kernel parameter has array rank 1 but block rank 0",
        );
        assert_eq!(
            KernelError::MappingInputCount { parameter: 0, expected: 1, actual: 0 }.to_string(),
            "kernel parameter 0 mapping takes 0 inputs but the call requires 1",
        );
        assert_eq!(
            KernelError::MappingInputIdentity { parameter: 0, first_axis: 0, second_axis: 1 }.to_string(),
            "kernel parameter 0 mapping axes 0 and 1 share a dimension identity but may differ",
        );
        let expected = DimensionBounds::non_negative(Some(2)).unwrap();
        let actual = DimensionBounds::non_negative(Some(1)).unwrap();
        assert_eq!(
            KernelError::MappingInputBounds { parameter: 0, axis: 1, expected, actual }.to_string(),
            "kernel parameter 0 mapping axis 1 admits [0, 1), but the grid requires [0, 2)",
        );
    }

    #[test]
    fn test_kernel_parameter_new() {
        let parameter = parameter(KernelParameterAccess::ReadOnly);
        assert_eq!(parameter.r#type().as_ref(), &ArrayType::scalar(DataType::I32));
        let error = KernelParameter::new(
            ArrayType::new_static(DataType::I32, [2]),
            KernelParameterAccess::ReadOnly,
            scalar_mapping(),
        )
        .unwrap_err();
        assert_eq!(error, KernelError::ParameterRank { array_rank: 1, block_rank: 0 });
        assert_eq!(error.to_string(), "kernel parameter has array rank 1 but block rank 0");
    }

    #[test]
    fn test_kernel_parameter_new_masked_tiles() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let zero = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let mapping = BlockMapping::new(
            builder.build(vec![zero], vec![], vec![Placeholder]).unwrap(),
            vec![4],
            BoundaryPolicy::Masked,
        )
        .unwrap();
        // Full, partial, empty, and oversized full array windows all have the same fixed logical block type.
        for extent in [8, 4, 2, 0] {
            let full_type = ArrayType::new_static(DataType::F32, [extent])
                .with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![])));
            let parameter =
                KernelParameter::new(full_type.clone(), KernelParameterAccess::ReadOnly, mapping.clone()).unwrap();
            assert_eq!(parameter.r#type().as_ref(), &full_type);
            assert_eq!(
                parameter.body_type(),
                ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [4]))),
            );
        }
    }

    #[test]
    fn test_kernel_parameter_new_rejects_shardings() {
        // A kernel runs on one device, so its parameter types are physical, whatever the placement.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let r#type = ArrayType::scalar(DataType::I32).with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        assert_eq!(
            KernelParameter::new(r#type, KernelParameterAccess::ReadOnly, scalar_mapping()),
            Err(KernelError::Type(TypeError::invalid(
                "kernel parameter type `i32[][sharding={mesh<['x'=2:manual]>, []}]` has a sharding, but a kernel runs \
                 on one device and its call accepts placed inputs",
            ))),
        );
    }

    #[test]
    fn test_kernel_parameter_from_refined_types() {
        for extent in [0, 2, 4] {
            let actual = ArrayType::new_static(DataType::I32, [extent]);
            let parameters =
                KernelParameter::from_refined_types(refinement_parameters(), &[actual.clone(), actual.clone()])
                    .unwrap();
            assert_eq!(parameters[0].r#type().as_ref(), &actual);
            assert_eq!(parameters[1].r#type().as_ref(), &actual);
            assert_eq!(parameters[0].access(), KernelParameterAccess::ReadOnly);
            assert_eq!(parameters[1].access(), KernelParameterAccess::WriteOnly);
            assert_eq!(parameters[0].mapping().block_shape(), &[4]);
            assert_eq!(
                parameters[1].body_type(),
                ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, [4]))),
            );
        }
        let actual = ArrayType::new_static(DataType::I32, [4]);
        let parameters =
            KernelParameter::from_refined_types(refinement_parameters(), &[actual.clone(), actual]).unwrap();
        let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), parameters).unwrap();
        let definition: KernelDefinition =
            KernelDefinition::trace(call, |(references, _)| references[1].write(&references[0].read()?)).unwrap();
        let input = Array::vector(vec![3i32, 1, 4, 1]).unwrap();
        assert_eq!(definition.interpret(vec![input.clone()], 1), Ok(vec![input]));
        assert_eq!(KernelParameter::from_refined_types(vec![], &[]).unwrap().len(), 0);
    }

    #[test]
    fn test_kernel_parameter_from_refined_types_preserves_metadata() {
        let mut declarations = refinement_parameters();
        let memory = Memory::Host { pinned: true };
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        declarations[0].0 = declarations[0].0.clone().with_memory(memory).with_layout(layout.clone());
        declarations[1].0 = declarations[1].0.clone().with_memory(memory);
        let actual = ArrayType::new_static(DataType::I32, [2]).with_memory(memory).with_layout(layout);
        let expected_output = ArrayType::new_static(DataType::I32, [2]).with_memory(memory);
        let parameters = KernelParameter::from_refined_types(declarations, &[actual.clone(), actual.clone()]).unwrap();
        assert_eq!(parameters[0].r#type().as_ref(), &actual);
        assert_eq!(parameters[1].r#type().as_ref(), &expected_output);
        assert_eq!(
            parameters[1].body_type(),
            ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, [4]).with_memory(memory))),
        );
        let declared = refinement_parameters();
        let wrong_memory = ArrayType::new_static(DataType::I32, [2]).with_memory(memory);
        let error = KernelParameter::from_refined_types(declared, &[wrong_memory.clone(), wrong_memory]).unwrap_err();
        assert_eq!(
            error,
            KernelError::Type(TypeError::invalid("type i32[2]@Host[Pinned] does not refine declared type i32[extent]")),
        );
    }

    #[test]
    fn test_kernel_parameter_from_refined_types_checks_complete_signature() {
        let first = ArrayType::new_static(DataType::I32, [2]);
        let second = ArrayType::new_static(DataType::I32, [3]);
        let error = KernelParameter::from_refined_types(refinement_parameters(), &[first.clone(), second]).unwrap_err();
        assert_eq!(
            error,
            KernelError::Type(
                DimensionError::InputDimensionMismatch { dimension: "extent".to_owned(), expected: 2, actual: 3 }
                    .into(),
            ),
        );
        let error = KernelParameter::from_refined_types(
            refinement_parameters(),
            &vec![ArrayType::new_static(DataType::I32, [5]); 2],
        )
        .unwrap_err();
        assert_eq!(
            error,
            KernelError::Type(
                DimensionError::BindingOutOfBounds {
                    variable: "extent".to_owned(),
                    value: 5,
                    bounds: DimensionBounds::non_negative(Some(5)).unwrap(),
                }
                .into(),
            ),
        );
        let mut distinct = refinement_parameters();
        distinct[1].0 = ArrayType::new(
            DataType::I32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::non_negative(Some(5)).unwrap()).into()]),
        );
        let parameters =
            KernelParameter::from_refined_types(distinct, &[first.clone(), ArrayType::new_static(DataType::I32, [3])])
                .unwrap();
        assert_eq!(parameters[0].r#type().as_ref(), &first);
        assert_eq!(parameters[1].r#type().as_ref(), &ArrayType::new_static(DataType::I32, [3]));
        let error = KernelParameter::from_refined_types(refinement_parameters(), &[first]).unwrap_err();
        assert_eq!(
            error,
            KernelError::Type(TypeError::invalid("declared type count 2 does not match actual type count 1")),
        );
    }

    #[test]
    fn test_kernel_parameter_from_refined_types_rejects_unresolved_output() {
        let mut declarations = refinement_parameters();
        let output = DimensionVariable::new("output", DimensionBounds::non_negative(Some(5)).unwrap());
        declarations[1].0 = ArrayType::new(DataType::I32, Shape::new(vec![output.into()]));
        let actual = vec![ArrayType::new_static(DataType::I32, [2]), declarations[1].0.clone()];
        let error = KernelParameter::from_refined_types(declarations, &actual).unwrap_err();
        assert_eq!(
            error,
            KernelError::Type(TypeError::invalid("kernel parameter 1 specialization requires a static full shape")),
        );
    }

    #[test]
    fn test_kernel_parameter_access() {
        assert_eq!(parameter(KernelParameterAccess::ReadWrite).access(), KernelParameterAccess::ReadWrite);
    }

    #[test]
    fn test_kernel_parameter_mapping() {
        assert_eq!(parameter(KernelParameterAccess::ReadOnly).mapping().boundary_policy(), BoundaryPolicy::InBounds);
    }

    #[test]
    fn test_kernel_parameter_body_type() {
        assert_eq!(
            parameter(KernelParameterAccess::ReadOnly).body_type(),
            ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::I32))),
        );
    }

    #[test]
    fn test_kernel_parameter_body_type_preserves_canonical_layout() {
        let full_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let zero = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let program = builder.build(vec![zero, zero], vec![], vec![Placeholder; 2]).unwrap();
        let whole = KernelParameter::new(
            full_type.clone(),
            KernelParameterAccess::ReadOnly,
            BlockMapping::new(program.clone(), vec![2, 3], BoundaryPolicy::InBounds).unwrap(),
        )
        .unwrap();
        assert_eq!(whole.body_type(), ArrayIrType::Reference(ReferenceType::new(full_type.clone())));

        let partial = KernelParameter::new(
            full_type.clone(),
            KernelParameterAccess::ReadOnly,
            BlockMapping::new(program, vec![1, 3], BoundaryPolicy::InBounds).unwrap(),
        )
        .unwrap();
        let transform =
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 1, 1), ArraySliceAxis::new(0, 3, 1)] };
        assert_eq!(
            partial.body_type(),
            ArrayIrType::Reference(ReferenceType::new(transform.output_type(&full_type).unwrap())),
        );
    }

    #[test]
    fn test_kernel_parameter_type() {
        assert_eq!(parameter(KernelParameterAccess::ReadOnly).r#type().into_owned(), ArrayType::scalar(DataType::I32));
    }

    #[test]
    fn test_kernel_call_operation_new() {
        let operation = copy_definition().operation().clone();
        assert_eq!(
            operation.to_string(),
            indoc! {"
                kernel_call [
                    grid=[],
                    parameter_0=read-only i32[] [] InBounds,
                    mapping_0={
                        lambda  .
                        in ()
                    },
                    parameter_1=write-only i32[] [] InBounds,
                    mapping_1={
                        lambda  .
                        in ()
                    },
                ]"},
        );
        let grid = Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap();
        let error = KernelCallOperation::new(grid, vec![parameter(KernelParameterAccess::ReadOnly)]).unwrap_err();
        assert_eq!(error, KernelError::MappingInputCount { parameter: 0, expected: 1, actual: 0 });
        assert_eq!(error.to_string(), "kernel parameter 0 mapping takes 0 inputs but the call requires 1");
    }

    #[test]
    fn test_kernel_call_operation_new_coordinate_bounds() {
        // Builds a call over a grid of one axis with `extent` points whose mapping coordinate admits `bounds`.
        let call = |extent: usize, bounds: DimensionBounds| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            builder.add_input(ArrayIrType::Dimension(DimensionType::new("coordinate", bounds)));
            let mapping = BlockMapping::new(
                builder.build(vec![], vec![Placeholder], vec![]).unwrap(),
                vec![],
                BoundaryPolicy::InBounds,
            )
            .unwrap();
            let grid = Grid::new(vec![GridDimension::new(Dimension::Static(extent), GridExecution::Parallel)]).unwrap();
            KernelCallOperation::new(
                grid,
                vec![
                    KernelParameter::new(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadOnly, mapping)
                        .unwrap(),
                ],
            )
        };

        // Coordinate bounds that admit the whole grid are accepted, and an empty grid binds no coordinate at all.
        let grid_bounds = DimensionBounds::non_negative(Some(2)).unwrap();
        assert_eq!(call(2, grid_bounds).map(|operation| operation.coordinate_types()[0].bounds()), Ok(grid_bounds));
        assert_eq!(
            call(0, DimensionBounds::positive(Some(3)).unwrap())
                .map(|operation| operation.coordinate_types()[0].bounds()),
            Ok(DimensionBounds::non_negative(Some(1)).unwrap()),
        );

        // Coordinate bounds that exclude a grid point are rejected.
        let narrow_bounds = DimensionBounds::non_negative(Some(1)).unwrap();
        assert_eq!(
            call(2, narrow_bounds).unwrap_err(),
            KernelError::MappingInputBounds { parameter: 0, axis: 0, expected: grid_bounds, actual: narrow_bounds },
        );
        let positive_bounds = DimensionBounds::positive(Some(3)).unwrap();
        assert_eq!(
            call(2, positive_bounds).unwrap_err(),
            KernelError::MappingInputBounds { parameter: 0, axis: 0, expected: grid_bounds, actual: positive_bounds },
        );
    }

    #[test]
    fn test_kernel_call_operation_new_rejects_coordinate_identity_aliasing() {
        // Builds a call over a grid with the given `extents` whose mapping uses one identity for both coordinates.
        let call = |extents: [usize; 2]| {
            let coordinate = DimensionType::new("coordinate", DimensionBounds::non_negative(None).unwrap());
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            builder.add_input(ArrayIrType::Dimension(coordinate.clone()));
            builder.add_input(ArrayIrType::Dimension(coordinate));
            let mapping = BlockMapping::new(
                builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap(),
                vec![],
                BoundaryPolicy::InBounds,
            )
            .unwrap();
            KernelCallOperation::new(
                Grid::new(
                    extents
                        .into_iter()
                        .map(|extent| GridDimension::new(Dimension::Static(extent), GridExecution::Parallel))
                        .collect(),
                )
                .unwrap(),
                vec![
                    KernelParameter::new(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadOnly, mapping)
                        .unwrap(),
                ],
            )
        };
        let coordinate_bounds = |operation: KernelCallOperation| {
            operation.coordinate_types().iter().map(DimensionType::bounds).collect::<Vec<_>>()
        };

        // A shared identity is rejected when the two coordinates may differ.
        let error = call([2, 2]).unwrap_err();
        assert_eq!(error, KernelError::MappingInputIdentity { parameter: 0, first_axis: 0, second_axis: 1 });
        assert_eq!(
            error.to_string(),
            "kernel parameter 0 mapping axes 0 and 1 share a dimension identity but may differ",
        );
        assert_eq!(
            call([1, 2]).unwrap_err(),
            KernelError::MappingInputIdentity { parameter: 0, first_axis: 0, second_axis: 1 },
        );

        // A shared identity is accepted when both coordinates are always zero or when the grid is empty.
        let singleton_bounds = DimensionBounds::non_negative(Some(1)).unwrap();
        assert_eq!(call([1, 1]).map(coordinate_bounds), Ok(vec![singleton_bounds, singleton_bounds]));
        assert_eq!(
            call([0, 2]).map(coordinate_bounds),
            Ok(vec![singleton_bounds, DimensionBounds::non_negative(Some(2)).unwrap()]),
        );
    }

    #[test]
    fn test_kernel_call_operation_new_with_prefetch() {
        let definition = prefetch_definition();
        assert_eq!(
            definition.operation().input_types(),
            vec![ArrayType::new_static(DataType::I32, vec![8]).into(), ArrayType::scalar(DataType::I32).into()],
        );
        assert_eq!(
            definition.operation().body_input_types(),
            vec![
                ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, vec![2]))),
                ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::I32))),
                ArrayIrType::Dimension(definition.operation().coordinate_types()[0].clone()),
                ArrayType::scalar(DataType::I32).into(),
            ],
        );
        let invalid = ArrayType::new_static(DataType::I32, vec![1]);
        assert_eq!(
            KernelCallOperation::new_with_prefetch(Grid::new(vec![]).unwrap(), vec![], vec![invalid.clone()])
                .unwrap_err(),
            KernelError::Type(TypeError::invalid(format!(
                "`dimension_from_scalar` input must be a rank-0 integer array but has type `{invalid}`",
            ))),
        );
    }

    #[test]
    fn test_kernel_call_operation_new_with_prefetch_rejects_shardings() {
        // A scalar prefetched input is physical, like every parameter.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let scalar = ArrayType::scalar(DataType::I32).with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        assert_eq!(
            KernelCallOperation::new_with_prefetch(Grid::new(vec![]).unwrap(), vec![], vec![scalar]),
            Err(KernelError::Type(TypeError::invalid(
                "scalar prefetch type `i32[][sharding={mesh<['x'=2:manual]>, []}]` has a sharding, but a kernel runs \
                 on one device and its call accepts placed inputs",
            ))),
        );
    }

    #[test]
    fn test_kernel_call_operation_new_with_prefetch_rejects_prefetch_identity_aliasing() {
        // A prefetched dimension that shares its identity with a grid coordinate can never be specialized, because
        // binding it to a constant cannot preserve the runtime equality with the remaining coordinate.
        let coordinate = DimensionType::new("coordinate", DimensionBounds::non_negative(Some(1)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        builder.add_input(coordinate.clone().into());
        builder.add_input(coordinate.into());
        let mapping = BlockMapping::new(
            builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap(),
            vec![],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let error = KernelCallOperation::new_with_prefetch(
            Grid::new(vec![GridDimension::new(Dimension::Static(1), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadOnly, mapping)
                    .unwrap(),
            ],
            vec![ArrayType::scalar(DataType::I32)],
        )
        .unwrap_err();
        assert_eq!(error, KernelError::MappingInputIdentity { parameter: 0, first_axis: 0, second_axis: 1 });
        assert_eq!(
            error.to_string(),
            "kernel parameter 0 mapping axes 0 and 1 share a dimension identity but may differ",
        );
    }

    #[test]
    fn test_kernel_call_operation_from_parts() {
        let operation = KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap();
        assert_eq!(KernelCallOperation::from_parts(Grid::new(vec![]).unwrap(), vec![], vec![], vec![]), Ok(operation));
        let coordinate = DimensionType::new("coordinate", DimensionBounds::non_negative(Some(1)).unwrap());
        assert_eq!(
            KernelCallOperation::from_parts(Grid::new(vec![]).unwrap(), vec![], vec![], vec![coordinate]),
            Err(KernelError::Type(TypeError::invalid("coordinate type count differs from the grid rank"))),
        );
    }

    #[test]
    fn test_kernel_call_operation_grid() {
        assert_eq!(copy_definition().operation().grid(), &Grid::new(vec![]).unwrap());
    }

    #[test]
    fn test_kernel_call_operation_parameters() {
        let definition = copy_definition();
        assert_eq!(
            definition.operation().parameters().iter().map(KernelParameter::access).collect::<Vec<_>>(),
            vec![KernelParameterAccess::ReadOnly, KernelParameterAccess::WriteOnly],
        );
    }

    #[test]
    fn test_kernel_call_operation_coordinate_types() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![
                GridDimension::new(Dimension::Static(4), GridExecution::Parallel),
                GridDimension::new(Dimension::Static(0), GridExecution::Parallel),
            ])
            .unwrap(),
            vec![],
        )
        .unwrap();
        assert_eq!(
            operation.coordinate_types().iter().map(DimensionType::bounds).collect::<Vec<_>>(),
            vec![DimensionBounds::non_negative(Some(4)).unwrap(), DimensionBounds::non_negative(Some(1)).unwrap()],
        );
        assert_eq!(operation.input_types(), Vec::<ArrayIrType>::new());
        assert_eq!(operation.boundary_contract(), KernelBoundaryContract::new(vec![None, None]));
    }

    #[test]
    fn test_kernel_call_operation_prefetch_types() {
        assert_eq!(prefetch_definition().operation().prefetch_types(), &[ArrayType::scalar(DataType::I32)]);
    }

    #[test]
    fn test_kernel_call_operation_body_input_types() {
        let definition = dynamic_definition("extent", 4, true);
        let operation = definition.operation();
        assert_eq!(operation.body_input_types(), definition.body().input_types().into_owned());
        assert_eq!(operation.body_input_types()[0], operation.parameters()[0].body_type());
        assert_eq!(
            &operation.body_input_types()[1..],
            operation.coordinate_types().iter().cloned().map(ArrayIrType::Dimension).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_kernel_call_operation_input_types() {
        assert_eq!(
            copy_definition().operation().input_types(),
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::I32))],
        );
    }

    #[test]
    fn test_kernel_call_operation_output_types() {
        assert_eq!(
            copy_definition().operation().output_types(),
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::I32))],
        );
    }

    #[test]
    fn test_kernel_call_operation_output_types_for() {
        // The call has a read-only, a write-only, and a read-write `i32` scalar parameter, in this order, so it takes
        // two inputs and returns the write-only result followed by the read-write result.
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                parameter(KernelParameterAccess::ReadOnly),
                parameter(KernelParameterAccess::WriteOnly),
                parameter(KernelParameterAccess::ReadWrite),
            ],
        )
        .unwrap();
        let scalar = ArrayType::scalar(DataType::I32);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let placed = |sharding: Sharding| ArrayIrType::Array(scalar.clone().with_sharding(sharding).unwrap());
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let varying = replicated.clone().with_varying_manual_axes(["x"]).unwrap();

        // Inputs without a sharding give the declared result types.
        let unplaced = ArrayIrType::Array(scalar.clone());
        assert_eq!(operation.output_types_for(&[unplaced.clone(), unplaced.clone()]), Ok(operation.output_types()));

        // The result of the read-write parameter is the storage of its input and has the type of that input, placement
        // included. A placement alone does not reach the write-only result.
        assert_eq!(
            operation.output_types_for(&[unplaced.clone(), placed(replicated.clone())]),
            Ok(vec![unplaced.clone(), placed(replicated.clone())]),
        );
        assert_eq!(
            operation.output_types_for(&[placed(replicated.clone()), unplaced.clone()]),
            Ok(operation.output_types()),
        );

        // Every result varies along the manual axes along which an input varies. The write-only result is replicated on
        // the mesh of the inputs, and the read-write input must already vary, since its storage receives the result.
        assert_eq!(
            operation.output_types_for(&[placed(varying.clone()), placed(varying.clone())]),
            Ok(vec![placed(varying.clone()), placed(varying.clone())]),
        );
        assert_eq!(
            operation.output_types_for(&[unplaced.clone(), placed(varying.clone())]),
            Ok(vec![placed(varying.clone()), placed(varying.clone())]),
        );
        for input in [unplaced.clone(), placed(replicated.clone())] {
            assert_eq!(
                operation.output_types_for(&[placed(varying.clone()), input.clone()]),
                Err(TypeError::invalid(format!(
                    "`kernel_call` read-write parameter 2 receives `{input}`, which does not vary along manual axis \
                     `x` like another input, so its result could not hold a value that differs on each device",
                ))),
            );
        }

        // Only the sharding of an input is free, a pending reduction is not a placement, and placed inputs share one
        // mesh.
        let mismatch = TypeError::invalid("`kernel_call` input types do not match its logical signature");
        assert_eq!(operation.output_types_for(std::slice::from_ref(&unplaced)), Err(mismatch.clone()));
        assert_eq!(
            operation.output_types_for(&[ArrayIrType::Array(ArrayType::scalar(DataType::F32)), unplaced.clone()]),
            Err(mismatch),
        );
        assert_eq!(
            operation.output_types_for(&[placed(replicated.clone().with_unreduced_axes(["x"]).unwrap()), unplaced]),
            Err(TypeError::invalid(
                "`kernel_call` input type `i32[][sharding={mesh<['x'=2:manual]>, [], unreduced={'x'}}]` carries a \
                 pending reduction",
            )),
        );
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        assert_eq!(
            operation.output_types_for(&[placed(replicated), placed(Sharding::replicated(other_mesh, 0))]),
            Err(TypeError::invalid("`kernel_call` placed inputs must use the same mesh")),
        );
    }

    #[test]
    fn test_kernel_call_operation_aliases() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                parameter(KernelParameterAccess::ReadOnly),
                parameter(KernelParameterAccess::WriteOnly),
                parameter(KernelParameterAccess::ReadWrite),
            ],
        )
        .unwrap();
        assert_eq!(operation.aliases(), vec![(1, 1)]);
        assert_eq!(copy_definition().operation().aliases(), vec![]);
    }

    #[test]
    fn test_kernel_call_operation_boundary_contract() {
        assert_eq!(
            copy_definition().operation().boundary_contract(),
            KernelBoundaryContract::new(vec![
                Some(KernelParameterAccess::ReadOnly),
                Some(KernelParameterAccess::WriteOnly),
            ]),
        );
    }

    #[test]
    fn test_kernel_call_operation_specialize_grid() {
        let definition = dynamic_definition("count", 4, true);
        let operation = definition.operation();
        let specialized = operation.specialize_grid(&[2, 2]).unwrap();
        assert_eq!(specialized.body_input_types(), operation.body_input_types());
        assert_eq!(specialized.coordinate_types(), operation.coordinate_types());
        assert_eq!(specialized.grid().dimensions()[0].extent(), &Dimension::Static(2));
        assert!(matches!(operation.grid().dimensions()[0].extent(), Dimension::Dynamic(_)));
        assert_eq!(
            operation.specialize_grid(&[2, 3]).unwrap_err(),
            KernelError::Grid(GridError::Dimension(DimensionError::InputDimensionMismatch {
                dimension: "count".to_owned(),
                expected: 2,
                actual: 3,
            })),
        );
    }

    #[test]
    fn test_kernel_call_operation_specialize_prefetch() {
        let definition = prefetch_definition();
        let specialized = definition.operation().specialize_prefetch(&[Array::scalar(2i32).unwrap()]).unwrap();
        assert_eq!(specialized.prefetch_types(), &[] as &[ArrayType]);
        assert_eq!(specialized.coordinate_types(), definition.operation().coordinate_types());
        assert_eq!(specialized.input_types(), vec![ArrayType::new_static(DataType::I32, vec![8]).into()]);
        assert_eq!(
            specialized.parameters()[0]
                .mapping()
                .evaluate(&[DimensionValue::constant(0).unwrap()], &[8])
                .unwrap()
                .starts(),
            &[2],
        );
    }

    #[test]
    fn test_kernel_call_operation_specialize_prefetch_checks_values() {
        let definition = prefetch_definition();
        let operation = definition.operation();
        assert_eq!(
            operation.specialize_prefetch(&[]).unwrap_err(),
            KernelError::Type(TypeError::invalid("kernel scalar prefetch value count must be 1 but is 0")),
        );
        assert_eq!(
            operation.specialize_prefetch(&[Array::scalar(1i64).unwrap()]).unwrap_err(),
            KernelError::Type(TypeError::invalid(
                "kernel scalar prefetch 0 expected type `i32[]` but received `i64[]`",
            )),
        );
        let ArrayIrType::Dimension(offset) = &operation.parameters()[0].mapping().program().input_types()[1] else {
            unreachable!()
        };
        for value in [-1i32, 8] {
            let array = Array::scalar(value).unwrap();
            let expected = array.to_dimension(offset.variable().unwrap().clone()).unwrap_err();
            assert_eq!(operation.specialize_prefetch(&[array]).unwrap_err(), KernelError::Program(expected));
        }
        assert_eq!(
            operation.specialize_prefetch(&[Array::scalar(0i32).unwrap()]).unwrap().prefetch_types(),
            &[] as &[ArrayType],
        );
    }

    #[test]
    fn test_kernel_call_operation_name() {
        assert_eq!(copy_definition().operation().name(), "kernel_call");
    }

    #[test]
    fn test_kernel_call_operation_region_slots() {
        assert_eq!(copy_definition().operation().region_slots(), &[RegionSlot::computation("body")]);
    }

    #[test]
    fn test_kernel_call_operation_type_inference() {
        let definition = copy_definition();
        let operation = definition.operation();
        assert_eq!(
            operation.infer_output_types(&operation.input_types(), std::slice::from_ref(definition.body().interface())),
            Ok(operation.output_types()),
        );
        assert_eq!(
            operation.infer_output_types(&operation.input_types(), &[]),
            Err(TypeError::invalid("`kernel_call` requires exactly one body region")),
        );
        assert_eq!(
            operation.infer_output_types(&[], std::slice::from_ref(definition.body().interface())),
            Err(TypeError::invalid("`kernel_call` input types do not match its logical signature")),
        );
        let interface = RegionInterface::new(vec![], vec![], EffectClasses::NONE);
        assert_eq!(
            operation.infer_output_types(&operation.input_types(), &[interface]),
            Err(TypeError::invalid(
                "`kernel_call` body input types do not match its logical block references and coordinates",
            )),
        );
    }

    #[test]
    fn test_kernel_call_operation_region_capture_input_count() {
        assert_eq!(copy_definition().operation().region_capture_input_count(0), Some(0));
    }

    #[test]
    fn test_kernel_call_operation_input_region_provenance() {
        let definition = dynamic_definition("extent", 4, true);
        let operation = definition.operation();
        // The reference and both generated coordinates share the same local provenance category.
        assert_eq!(operation.input_region_provenance(0, 0), InputRegionProvenance::Local);
        assert_eq!(operation.input_region_provenance(0, 1), InputRegionProvenance::Local);
        assert_eq!(operation.input_region_provenance(0, 2), InputRegionProvenance::Local);
        assert_eq!(operation.input_region_provenance(0, 3), InputRegionProvenance::None);
        assert_eq!(operation.input_region_provenance(1, 0), InputRegionProvenance::None);
    }

    #[test]
    fn test_kernel_call_operation_allows_reference_access_through_region_input() {
        let parallel = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap(),
            vec![],
        )
        .unwrap();
        let sequential = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Sequential)]).unwrap(),
            vec![],
        )
        .unwrap();
        for mode in [
            ReferenceAccessMode::Read,
            ReferenceAccessMode::Write,
            ReferenceAccessMode::ReadWrite,
            ReferenceAccessMode::Accumulate,
            ReferenceAccessMode::AtomicAccumulate,
        ] {
            assert!(parallel.allows_reference_access_through_region_input(0, mode));
            assert!(sequential.allows_reference_access_through_region_input(0, mode));
            assert!(!parallel.allows_reference_access_through_region_input(1, mode));
        }
        assert!(!sequential.allows_reference_access_through_region_input(0, ReferenceAccessMode::Consume));
    }

    #[test]
    fn test_kernel_call_operation_effects() {
        let assertion = Effects::explicit(EffectClasses::single(EffectClass::OrderedAssertion));

        // Calls without mappings convert no scalar prefetched value, and whole array mappings are total.
        let empty = KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap();
        assert_eq!(empty.effects().as_ref(), Effects::empty());
        let prefetch_types = vec![ArrayType::scalar(DataType::I32)];
        let unused_prefetch =
            KernelCallOperation::new_with_prefetch(Grid::new(vec![]).unwrap(), vec![], prefetch_types).unwrap();
        assert_eq!(unused_prefetch.effects().as_ref(), Effects::empty());
        assert_eq!(copy_definition().operation().effects().as_ref(), Effects::empty());

        // A mapping that may fail and a required conversion of a scalar prefetched value are both assertions.
        let division = prefetch_mapping_definition(MappingExpression::Constant(0) / MappingExpression::Prefetch(0));
        assert_eq!(division.operation().effects().as_ref(), &assertion);
        let offset = prefetch_mapping_definition(MappingExpression::Prefetch(0));
        assert_eq!(offset.operation().effects().as_ref(), &assertion);
        let coordinate = mapped_call(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap(),
            vec![(
                ArrayType::new_static(DataType::I32, vec![4]),
                KernelParameterAccess::ReadOnly,
                ParameterWindow::ElementStarts {
                    block_shape: vec![1],
                    boundary: BoundaryPolicy::InBounds,
                    starts: vec![MappingExpression::GridCoordinate(0) - MappingExpression::Constant(1)],
                },
            )],
            vec![],
        )
        .unwrap();
        assert_eq!(coordinate.effects().as_ref(), &assertion);

        // Explicit specialization rebuilds the mappings with exact effects and removes the conversions.
        let two = Array::scalar(2i32).unwrap();
        let specialized = division.specialize_prefetch(std::slice::from_ref(&two)).unwrap();
        assert_eq!(specialized.operation().effects().as_ref(), Effects::empty());
        let specialized = offset.specialize_prefetch(std::slice::from_ref(&two)).unwrap();
        assert_eq!(specialized.operation().effects().as_ref(), Effects::empty());
    }

    #[test]
    fn test_kernel_call_operation_effects_division_by_zero() {
        // A read-only call with no results whose mapping computes `0 / d` fails for a prefetched `d` of zero, when it
        // is interpreted directly and after simplification of an enclosing program that discards the call.
        let definition = prefetch_mapping_definition(MappingExpression::Constant(0) / MappingExpression::Prefetch(0));
        let program = unused_call_program(&definition);
        let simplified = program.clone().into_simplified().unwrap();
        assert_eq!(simplified.to_string(), program.to_string());
        let input = Array::vector(vec![1i32, 2, 3, 4]).unwrap();
        let zero = Array::scalar(0i32).unwrap();
        let violation = ProgramError::custom(DimensionError::RequirementViolation {
            message: "`dimension_div` requires right > 0; got left=0, right=0".to_owned(),
        });
        let expected = KernelError::Mapping(BlockMappingError::Program(violation));
        let direct = definition.interpret(vec![input.clone(), zero.clone()], 1);
        assert_eq!(direct, Err(ProgramError::custom(expected.clone())));
        let inputs = vec![ArrayIrValue::Array(input), ArrayIrValue::Array(zero.clone())];
        assert_eq!(program.interpret(inputs.clone()), Err(ProgramError::custom(expected.clone())));
        assert_eq!(simplified.interpret(inputs), Err(ProgramError::custom(expected.clone())));
        assert_eq!(definition.specialize_prefetch(&[zero]).unwrap_err(), expected);
    }

    #[test]
    fn test_kernel_call_operation_effects_invalid_conversion() {
        let definition = prefetch_mapping_definition(MappingExpression::Prefetch(0));
        let program = unused_call_program(&definition);
        let simplified = program.clone().into_simplified().unwrap();
        assert_eq!(simplified.to_string(), program.to_string());
        let input = Array::vector(vec![1i32, 2, 3, 4]).unwrap();
        let negative = Array::scalar(-1i32).unwrap();
        let expected = KernelError::Program(ProgramError::Concretization {
            message: "cannot extract a concrete `usize` from `i32[]`; value `-1` is out of range".to_owned(),
        });
        let direct = definition.interpret(vec![input.clone(), negative.clone()], 1);
        assert_eq!(direct, Err(ProgramError::custom(expected.clone())));
        let inputs = vec![ArrayIrValue::Array(input), ArrayIrValue::Array(negative.clone())];
        assert_eq!(program.interpret(inputs.clone()), Err(ProgramError::custom(expected.clone())));
        assert_eq!(simplified.interpret(inputs), Err(ProgramError::custom(expected.clone())));
        assert_eq!(definition.specialize_prefetch(&[negative]).unwrap_err(), expected);
    }

    #[test]
    fn test_kernel_call_operation_effects_valid_prefetch() {
        // A valid prefetch succeeds, but the unspecialized call is retained because another value could fail.
        let definition = prefetch_mapping_definition(MappingExpression::Constant(0) / MappingExpression::Prefetch(0));
        let program = unused_call_program(&definition);
        let simplified = program.clone().into_simplified().unwrap();
        assert_eq!(simplified.to_string(), program.to_string());
        let input = Array::vector(vec![1i32, 2, 3, 4]).unwrap();
        let two = Array::scalar(2i32).unwrap();
        assert_eq!(definition.interpret(vec![input.clone(), two.clone()], 1), Ok(vec![]));
        let inputs = vec![ArrayIrValue::Array(input.clone()), ArrayIrValue::Array(two.clone())];
        assert_eq!(program.interpret(inputs.clone()), Ok(vec![]));
        assert_eq!(simplified.interpret(inputs), Ok(vec![]));

        // After explicit specialization, the mapping is a constant start and the unused pure call is eliminated.
        let specialized = definition.specialize_prefetch(&[two]).unwrap();
        let program = unused_call_program(&specialized);
        assert_eq!(
            program.clone().into_simplified().unwrap().to_string(),
            indoc! {"
                lambda %0:i32[4] .
                in ()"},
        );
        assert_eq!(program.interpret(vec![ArrayIrValue::Array(input)]), Ok(vec![]));
    }

    #[test]
    fn test_kernel_call_operation_effects_order() {
        // A failing call and a failing conversion that precedes or follows it report the earlier failure in program
        // order, both directly and after simplification.
        let definition = prefetch_mapping_definition(MappingExpression::Constant(0) / MappingExpression::Prefetch(0));
        let checked = DimensionVariable::new("checked", DimensionBounds::non_negative(Some(10)).unwrap());
        let program = |assertion_first: bool| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let inputs = definition
                .operation()
                .input_types()
                .into_iter()
                .map(|r#type| builder.add_input(r#type))
                .collect::<Vec<_>>();
            let scalar = builder.add_input(ArrayType::scalar(DataType::I32).into());
            let body = builder.import_region(definition.body().entry_region_ref());
            let assertion = KernelOperation::Portable(ArrayIrOperation::DimensionFromScalar(
                DimensionFromScalarOperation::new(checked.clone().into()),
            ));
            let call = KernelOperation::Call(definition.operation().clone());
            if assertion_first {
                builder.add_instruction(assertion.clone(), vec![], vec![scalar], None).unwrap();
            }
            builder.add_instruction(call, vec![body], inputs, None).unwrap();
            if !assertion_first {
                builder.add_instruction(assertion, vec![], vec![scalar], None).unwrap();
            }
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 3], vec![])
                .unwrap()
        };
        let inputs = vec![
            ArrayIrValue::Array(Array::vector(vec![1i32, 2, 3, 4]).unwrap()),
            ArrayIrValue::Array(Array::scalar(0i32).unwrap()),
            ArrayIrValue::Array(Array::scalar(-1i32).unwrap()),
        ];
        let conversion = ProgramError::Concretization {
            message: "cannot extract a concrete `usize` from `i32[]`; value `-1` is out of range".to_owned(),
        };
        let division = ProgramError::custom(KernelError::Mapping(BlockMappingError::Program(ProgramError::custom(
            DimensionError::RequirementViolation {
                message: "`dimension_div` requires right > 0; got left=0, right=0".to_owned(),
            },
        ))));
        for (assertion_first, expected) in [(true, conversion), (false, division)] {
            let program = program(assertion_first);
            let simplified = program.clone().into_simplified().unwrap();
            assert_eq!(simplified.to_string(), program.to_string());
            assert_eq!(program.interpret(inputs.clone()), Err(expected.clone()));
            assert_eq!(simplified.interpret(inputs.clone()), Err(expected));
        }
    }

    #[test]
    fn test_kernel_call_operation_effects_window_failures() {
        // A read-only call with no results whose total mapping selects a window that fails its boundary checks fails
        // when it is interpreted directly and after simplification of an enclosing program that discards the call.
        let input = Array::vector(vec![1i32, 2, 3, 4]).unwrap();
        let assertion = Effects::explicit(EffectClasses::single(EffectClass::OrderedAssertion));
        for (boundary, start, error) in [
            (BoundaryPolicy::InBounds, 4, BlockMappingError::OutOfBounds { axis: 0, start: 4, limit: 5, extent: 4 }),
            (
                BoundaryPolicy::Masked,
                MAX_DIMENSION_EXTENT,
                BlockMappingError::Overflow { boundary: "window limit", axis: 0 },
            ),
        ] {
            let operation = mapped_call(
                Grid::new(vec![]).unwrap(),
                vec![(
                    ArrayType::new_static(DataType::I32, vec![4]),
                    KernelParameterAccess::ReadOnly,
                    ParameterWindow::ElementStarts {
                        block_shape: vec![1],
                        boundary,
                        starts: vec![MappingExpression::Constant(start)],
                    },
                )],
                vec![],
            )
            .unwrap();
            assert_eq!(operation.parameters()[0].mapping().program().effects().classes(), EffectClasses::NONE);
            assert_eq!(operation.effects().as_ref(), &assertion);
            let definition = KernelDefinition::trace(operation, |_| Ok(())).unwrap();
            let program = unused_call_program(&definition);
            let simplified = program.clone().into_simplified().unwrap();
            assert_eq!(simplified.to_string(), program.to_string());
            let expected = ProgramError::custom(KernelInitializationError::Mapping(error));
            assert_eq!(definition.interpret(vec![input.clone()], 1), Err(expected.clone()));
            let inputs = vec![ArrayIrValue::Array(input.clone())];
            assert_eq!(program.interpret(inputs.clone()), Err(expected.clone()));
            assert_eq!(simplified.interpret(inputs), Err(expected));
        }

        // Windows whose bounds prove the checks declare no effect, so an unused pure call is eliminated. Three blocks
        // of two elements exceed an `i32[4]` unless they are masked.
        let tiled = |extent: usize, boundary: BoundaryPolicy| {
            mapped_call(
                Grid::new(vec![GridDimension::new(Dimension::Static(extent), GridExecution::Parallel)]).unwrap(),
                vec![(
                    ArrayType::new_static(DataType::I32, vec![4]),
                    KernelParameterAccess::ReadOnly,
                    ParameterWindow::BlockIndices {
                        block_shape: vec![2],
                        boundary,
                        indices: vec![MappingExpression::GridCoordinate(0)],
                    },
                )],
                vec![],
            )
            .unwrap()
        };
        assert_eq!(tiled(2, BoundaryPolicy::InBounds).effects().as_ref(), Effects::empty());
        assert_eq!(tiled(3, BoundaryPolicy::InBounds).effects().as_ref(), &assertion);
        assert_eq!(tiled(3, BoundaryPolicy::Masked).effects().as_ref(), Effects::empty());
        let whole = mapped_call(
            Grid::new(vec![]).unwrap(),
            vec![(
                ArrayType::new_static(DataType::I32, vec![4]),
                KernelParameterAccess::ReadOnly,
                ParameterWindow::WholeArray,
            )],
            vec![],
        )
        .unwrap();
        assert_eq!(whole.effects().as_ref(), Effects::empty());
        let definition = KernelDefinition::trace(tiled(2, BoundaryPolicy::InBounds), |_| Ok(())).unwrap();
        assert_eq!(
            unused_call_program(&definition).into_simplified().unwrap().to_string(),
            indoc! {"
                lambda %0:i32[4] .
                in ()"},
        );
    }

    #[test]
    fn test_kernel_call_operation_rename_type_identities() {
        let operation = copy_definition().operation().clone();
        let renamed = operation.rename_type_identities(&TypeIdentityRenaming::new()).unwrap();
        assert_eq!(renamed.input_types(), operation.input_types());
        assert_eq!(renamed.to_string(), operation.to_string());
    }

    #[test]
    fn test_kernel_definition_new() {
        let definition = copy_definition();
        assert_eq!(definition.references().parameters().len(), 2);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(parameter(KernelParameterAccess::ReadOnly).body_type());
        let value = builder.add_constant(ArrayIrValue::Array(Array::scalar(1i32).unwrap()));
        builder
            .add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, value], None)
            .unwrap();
        let body = builder.build(vec![], vec![Placeholder], vec![]).unwrap();
        let entry = body.entry_region_ref().id();
        let operation =
            KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![parameter(KernelParameterAccess::ReadOnly)])
                .unwrap();
        assert_eq!(
            KernelDefinition::new(operation, body).unwrap_err(),
            KernelError::Validation(KernelValidationError::DisallowedAccess {
                input_index: 0,
                access: KernelParameterAccess::ReadOnly,
                mode: ReferenceAccessMode::Write,
                operation: "reference_write",
                instruction: InstructionId::new(entry, 0),
            }),
        );
    }

    #[test]
    fn test_kernel_definition_new_rejects_shard_maps() {
        // A kernel body runs on one device, so a `shard_map` anywhere in the attached region closure of its body is
        // rejected, whether it is applied in the body itself or inside a nested region.

        // Identity `shard_map` over an `i32` scalar that is replicated along the manual mesh axis `x`, whose body is
        // attached as a kernel family region. Its global input and output types carry the replicated sharding.
        fn identity_shard_map() -> (ShardMapOperation, FlatProgram<EagerContext<ArrayIrValue<Array>, KernelOperation>>)
        {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
            let boundary = ShardMapBoundary::new(
                mesh.clone(),
                vec![Sharding::replicated(mesh.clone(), 0)],
                vec![Sharding::replicated(mesh, 0)],
                vec!["x".to_string()],
            )
            .unwrap();
            let scalar_type = ArrayType::scalar(DataType::I32);
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let input = builder.add_input(boundary.local_input_type(0, &scalar_type).unwrap().into());
            let body = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
            let operation = ShardMapOperation::from_program(&body, vec![scalar_type.into()], boundary).unwrap();
            (operation, body)
        }

        // Scalar copy kernel whose body also runs `apply` on a constant of the placed type `r#type` between its read
        // and its write. Kernel parameters are physical, so only such a constant can have the global type of a
        // `shard_map` boundary.
        fn definition(
            r#type: &ArrayType,
            apply: impl FnOnce(&mut ProgramBuilder<ArrayIrValue<Array>, KernelOperation>, AtomId),
        ) -> (Result<KernelDefinition, KernelError>, RegionId) {
            let scalar = ArrayType::scalar(DataType::I32);
            let operation = KernelCallOperation::new(
                Grid::new(vec![]).unwrap(),
                vec![
                    KernelParameter::new(scalar.clone(), KernelParameterAccess::ReadOnly, scalar_mapping()).unwrap(),
                    KernelParameter::new(scalar, KernelParameterAccess::WriteOnly, scalar_mapping()).unwrap(),
                ],
            )
            .unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let input = builder.add_input(operation.parameters()[0].body_type());
            let output = builder.add_input(operation.parameters()[1].body_type());
            let value = builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![input], None).unwrap()[0];
            let placed =
                builder.add_constant(ArrayIrValue::Array(Array::from_elements(r#type.clone(), &[0i32]).unwrap()));
            apply(&mut builder, placed);
            builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, value], None).unwrap();
            let body = builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap();
            let entry = body.entry_region_ref().id();
            (KernelDefinition::new(operation, body), entry)
        }

        // Every case stages a well typed body, because the constant that the `shard_map` receives has the global type
        // of the identity `shard_map` boundary.
        let (operation, _) = identity_shard_map();
        assert_eq!(operation.global_output_types(), operation.global_input_types());
        let global_type = <&ArrayType>::try_from(&operation.global_input_types()[0]).unwrap().clone();

        // A `shard_map` of arrays in the body itself.
        let (result, entry) = definition(&global_type, |builder, value| {
            let (operation, body) = identity_shard_map();
            let body = builder.import_program(body);
            builder
                .add_instruction(ArrayIrOperation::ShardMap(operation), vec![body], vec![value], None)
                .unwrap();
        });
        let error = result.unwrap_err();
        assert_eq!(
            error,
            KernelError::UnsupportedOperation { operation: "shard_map", instruction: InstructionId::new(entry, 1) },
        );
        assert_eq!(
            error.to_string(),
            format!(
                "kernel body instruction {} applies `shard_map`, which kernel bodies do not support",
                InstructionId::new(entry, 1),
            ),
        );

        // A `shard_map` inside a nested region of the body is found through the attached region closure, and the error
        // names its instruction in that nested region.
        let mut nested_region = None;
        let (result, _) = definition(&global_type, |builder, value| {
            let branch = |nested: bool| {
                let mut branch = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
                let input = branch.add_input(global_type.clone().into());
                let output = if nested {
                    let (operation, body) = identity_shard_map();
                    let body = branch.import_program(body);
                    branch
                        .add_instruction(ArrayIrOperation::ShardMap(operation), vec![body], vec![input], None)
                        .unwrap()[0]
                } else {
                    input
                };
                branch
                    .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                        vec![output],
                        vec![Placeholder],
                        vec![Placeholder],
                    )
                    .unwrap()
            };
            let true_branch = builder.import_program(branch(true));
            nested_region = Some(true_branch);
            let false_branch = builder.import_program(branch(false));
            let predicate = builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
            builder
                .add_instruction(
                    ConditionOperation::<ArrayIrType>::new(),
                    vec![true_branch, false_branch],
                    vec![predicate, value],
                    None,
                )
                .unwrap();
        });
        assert_eq!(
            result.unwrap_err(),
            KernelError::UnsupportedOperation {
                operation: "shard_map",
                instruction: InstructionId::new(nested_region.unwrap(), 0),
            },
        );
    }

    #[test]
    fn test_kernel_definition_trace() {
        let explicit = copy_definition();
        let traced =
            KernelDefinition::<NoKernelExtension>::trace(explicit.operation().clone(), |(references, _coordinates)| {
                let value = references[0].read()?;
                references[1].write(&value)
            })
            .unwrap();
        assert_eq!(traced.body().to_string(), explicit.body().to_string());
        assert_eq!(traced.semantic_key().unwrap(), explicit.semantic_key().unwrap());
        assert_eq!(traced.body().interface(), explicit.body().interface());
    }

    #[test]
    fn test_kernel_definition_trace_failure() {
        let result = KernelDefinition::<NoKernelExtension>::trace(copy_definition().operation().clone(), |_| {
            Err(ProgramError::MalformedProgram("traceable helper rejected its arguments".to_owned()))
        });
        assert!(matches!(
            result,
            Err(KernelError::Program(ProgramError::MalformedProgram(message)))
                if message == "traceable helper rejected its arguments",
        ));
    }

    #[test]
    fn test_kernel_definition_trace_with_prefetch() {
        let definition = prefetch_definition();
        assert_eq!(definition.body().input_types(), definition.operation().body_input_types());
        assert_eq!(
            definition.body().to_string(),
            indoc! {"
                lambda %0:ref<i32[2]>, %1:ref<i32[]>, %2:dimension<0>, %3:i32[] .
                let () = reference_write %1 %3
                in ()"},
        );
    }

    #[test]
    fn test_kernel_definition_specialize_grid() {
        let definition = dynamic_definition("count", 4, true);
        let specialized = definition.specialize_grid(&[2, 2]).unwrap();
        assert_eq!(specialized.body().to_string(), definition.body().to_string());
        assert_eq!(specialized.body().input_types(), definition.body().input_types());
        assert_ne!(specialized.semantic_key().unwrap(), definition.semantic_key().unwrap());
        assert_eq!(
            specialized.semantic_key().unwrap(),
            dynamic_definition("other", 4, true).specialize_grid(&[2, 2]).unwrap().semantic_key().unwrap(),
        );
        assert_eq!(specialized.interpret(vec![Array::scalar(7i32).unwrap()], 4), Ok(vec![]));
    }

    #[test]
    fn test_kernel_definition_specialize_prefetch() {
        let definition = prefetch_definition();
        let specialized = definition.specialize_prefetch(&[Array::scalar(2i32).unwrap()]).unwrap();
        assert_eq!(specialized.body().input_types(), specialized.operation().body_input_types());
        assert_eq!(
            specialized.interpret(vec![Array::vector(vec![0i32; 8]).unwrap()], 1),
            Ok(vec![Array::scalar(2i32).unwrap()]),
        );
        assert_ne!(
            specialized.semantic_key().unwrap(),
            definition.specialize_prefetch(&[Array::scalar(3i32).unwrap()]).unwrap().semantic_key().unwrap(),
        );
        assert_eq!(
            specialized.semantic_key().unwrap(),
            prefetch_definition()
                .specialize_prefetch(&[Array::scalar(2i32).unwrap()])
                .unwrap()
                .semantic_key()
                .unwrap(),
        );
    }

    #[test]
    fn test_kernel_definition_specialize_prefetch_matches_runtime_binding() {
        let definition = prefetch_definition();
        let input = Array::vector(vec![0i32; 8]).unwrap();
        let offset = Array::scalar(2i32).unwrap();
        let specialized = definition.specialize_prefetch(std::slice::from_ref(&offset)).unwrap();
        assert_eq!(definition.interpret(vec![input.clone(), offset], 1), specialized.interpret(vec![input.clone()], 1));
        let invalid = Array::scalar(8i32).unwrap();
        let expected = definition.specialize_prefetch(std::slice::from_ref(&invalid)).unwrap_err();
        assert_eq!(definition.interpret(vec![input, invalid], 1), Err(ProgramError::custom(expected)));
    }

    #[test]
    fn test_kernel_definition_specialize_prefetch_rebuilds_mappings() {
        // The table window starts at the prefetched block index times the block extent, which may overflow for an
        // unbounded index and so requires a runtime assertion before specialization.
        let operation = mapped_call(
            Grid::new(vec![GridDimension::new(Dimension::Static(1), GridExecution::Parallel)]).unwrap(),
            vec![
                (
                    ArrayType::new_static(DataType::I32, vec![8]),
                    KernelParameterAccess::ReadOnly,
                    ParameterWindow::BlockIndices {
                        block_shape: vec![2],
                        boundary: BoundaryPolicy::InBounds,
                        indices: vec![MappingExpression::Prefetch(0)],
                    },
                ),
                (
                    ArrayType::new_static(DataType::I32, vec![2]),
                    KernelParameterAccess::WriteOnly,
                    ParameterWindow::WholeArray,
                ),
            ],
            vec![ArrayType::scalar(DataType::I32)],
        )
        .unwrap();
        let definition: KernelDefinition =
            KernelDefinition::trace_with_prefetch(operation, |(references, _coordinates, _values)| {
                references[1].write(&references[0].read()?)
            })
            .unwrap();
        assert_eq!(
            definition.operation().parameters()[0].mapping().program().effects().classes(),
            EffectClasses::single(EffectClass::OrderedAssertion),
        );

        // Specialization folds the start to a constant, so every mapping is pure.
        let block = Array::scalar(2i32).unwrap();
        let specialized = definition.specialize_prefetch(std::slice::from_ref(&block)).unwrap();
        assert_eq!(
            specialized.operation().parameters()[0].mapping().program().to_string(),
            indoc! {"
                lambda %0:dimension<0> .
                let %1:dimension<2> = const 2
                    %2:dimension<4> = const 4
                in (%2)"
            },
        );
        assert_eq!(
            specialized
                .operation()
                .parameters()
                .iter()
                .map(|parameter| parameter.mapping().program().effects().classes())
                .collect::<Vec<_>>(),
            vec![EffectClasses::NONE; 2],
        );

        // The runtime prefetch value and its specialization select the same window.
        let table = Array::vector((0..8).collect::<Vec<i32>>()).unwrap();
        let expected = Ok(vec![Array::vector(vec![4i32, 5]).unwrap()]);
        assert_eq!(definition.interpret(vec![table.clone(), block], 1), expected);
        assert_eq!(specialized.interpret(vec![table], 1), expected);
    }

    #[test]
    fn test_kernel_definition_simplified() {
        let operation = copy_definition().operation().clone();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let input = builder.add_input(operation.parameters()[0].body_type());
        let output = builder.add_input(operation.parameters()[1].body_type());
        builder
            .add_instruction(
                ArrayIrOperation::from(ArrayOperation::Zero(ZeroOperation::new(ArrayType::scalar(DataType::I32)))),
                vec![],
                vec![],
                None,
            )
            .unwrap();
        let initial = builder.add_constant(ArrayIrValue::Array(Array::scalar(9i32).unwrap()));
        builder
            .add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, initial], None)
            .unwrap();
        let value = builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![input], None).unwrap()[0];
        builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, value], None).unwrap();
        let definition =
            KernelDefinition::new(operation, builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap()).unwrap();
        let simplified = definition.simplified().unwrap();
        assert_eq!(
            definition.body().to_string(),
            indoc! {"
                lambda %0:ref<i32[]>, %1:ref<i32[]> .
                let %2:i32[] = zero [type=i32[]]
                    %3:i32[] = const 9
                    () = reference_write %1 %3
                    %4:i32[] = reference_read %0
                    () = reference_write %1 %4
                in ()"},
        );
        assert_eq!(
            simplified.body().to_string(),
            indoc! {"
                lambda %0:ref<i32[]>, %1:ref<i32[]> .
                let %2:i32[] = const 9
                    () = reference_write %1 %2
                    %3:i32[] = reference_read %0
                    () = reference_write %1 %3
                in ()"},
        );
        assert_eq!(simplified.body().interface(), definition.body().interface());
        assert_eq!(simplified.body().effects(), definition.body().effects());
        let input = Array::scalar(17i32).unwrap();
        assert_eq!(simplified.interpret(vec![input.clone()], 1), Ok(vec![input]));
        assert_eq!(simplified.simplified().unwrap().semantic_key().unwrap(), simplified.semantic_key().unwrap());
    }

    #[test]
    fn test_kernel_definition_operation() {
        assert_eq!(copy_definition().operation().name(), KERNEL_CALL_OPERATION_NAME);
    }

    #[test]
    fn test_kernel_definition_body() {
        assert_eq!(
            copy_definition().body().to_string(),
            indoc! {"
                lambda %0:ref<i32[]>, %1:ref<i32[]> .
                let %2:i32[] = reference_read %0
                    () = reference_write %1 %2
                in ()"},
        );
    }

    #[test]
    fn test_kernel_definition_references() {
        let definition = copy_definition();
        assert_eq!(
            definition
                .references()
                .parameters()
                .iter()
                .map(|parameter| parameter.as_ref().unwrap().access())
                .collect::<Vec<_>>(),
            vec![KernelParameterAccess::ReadOnly, KernelParameterAccess::WriteOnly],
        );
    }

    #[test]
    fn test_kernel_definition_effects() {
        // The local reference state of a body is derived from its accesses and is never an explicit declaration.
        let copy = copy_definition();
        assert_eq!(copy.effects(), copy.body().effects());
        assert_eq!(copy.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert!(!copy.effects().has_explicit_ordered_state());

        // Mapping and conversion assertions are added to the effects of the body.
        let assertion = EffectClasses::single(EffectClass::OrderedAssertion);
        let division = prefetch_mapping_definition(MappingExpression::Constant(0) / MappingExpression::Prefetch(0));
        assert_eq!(division.body().effects(), EffectsSummary::PURE);
        assert_eq!(division.effects(), Effects::explicit(assertion).summary());
        let prefetch = prefetch_definition();
        assert_eq!(prefetch.effects().classes(), assertion.union(EffectClasses::single(EffectClass::OrderedState)));
        assert!(prefetch.effects().is_retained_when_unused());
        assert!(!prefetch.effects().has_explicit_ordered_state());

        // A specialized total mapping and an empty body leave a pure call.
        let two = Array::scalar(2i32).unwrap();
        assert_eq!(division.specialize_prefetch(&[two]).unwrap().effects(), EffectsSummary::PURE);
    }

    #[test]
    fn test_kernel_definition_semantic_key() {
        let first = dynamic_definition("first", 4, true);
        let renamed = dynamic_definition("second", 4, true);
        let independent = dynamic_definition("first", 4, false);
        let wider = dynamic_definition("first", 5, true);
        let key = first.semantic_key().unwrap();
        assert_eq!(key, first.clone().semantic_key().unwrap());
        assert_eq!(key, renamed.semantic_key().unwrap());
        assert_ne!(key, independent.semantic_key().unwrap());
        assert_ne!(key, wider.semantic_key().unwrap());
        assert!(!key.contains("first"));
        assert!(key.starts_with(&format!("kernel key {KERNEL_KEY_VERSION}\n")));
    }

    #[test]
    fn test_kernel_definition_semantic_key_preserves_constant_bits() {
        let [first, second] = [0x7fc0_0001, 0x7fc0_0002].map(|bits| {
            let operation = KernelCallOperation::new(
                Grid::new(vec![]).unwrap(),
                vec![
                    KernelParameter::new(
                        ArrayType::scalar(DataType::F32),
                        KernelParameterAccess::WriteOnly,
                        scalar_mapping(),
                    )
                    .unwrap(),
                ],
            )
            .unwrap();
            KernelDefinition::<NoKernelExtension>::trace(operation, |(references, _coordinates)| {
                let value = references[0].context().lift(ArrayIrValue::Array(Array::scalar(f32::from_bits(bits))?))?;
                references[0].write(&value)
            })
            .unwrap()
        });
        // Diagnostic floating-point rendering deliberately omits NaN payloads, but semantic identity must retain them.
        assert_eq!(first.body().to_string(), second.body().to_string());
        assert_ne!(first.semantic_key().unwrap(), second.semantic_key().unwrap());
    }

    #[test]
    fn test_kernel_definition_semantic_key_preserves_operation_constant_bits() {
        let definitions = [0x7fc0_0001, 0x7fc0_0002].map(|bits| {
            let mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let mapping =
                BlockMapping::new(mapping.build(vec![], vec![], vec![]).unwrap(), vec![], BoundaryPolicy::InBounds)
                    .unwrap();
            let operation = KernelCallOperation::new(
                Grid::new(vec![]).unwrap(),
                vec![
                    KernelParameter::new(ArrayType::scalar(DataType::F32), KernelParameterAccess::WriteOnly, mapping)
                        .unwrap(),
                ],
            )
            .unwrap();
            let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let reference = body.add_input(operation.body_input_types()[0].clone());
            let value = body
                .add_instruction(
                    ArrayOperation::Constant(ConstantOperation::new(Array::scalar(f32::from_bits(bits)).unwrap())),
                    vec![],
                    vec![],
                    None,
                )
                .unwrap()[0];
            body.add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, value], None).unwrap();
            KernelDefinition::new(operation, body.build(vec![], vec![Placeholder], vec![]).unwrap()).unwrap()
        });
        assert_eq!(definitions[0].body().to_string(), definitions[1].body().to_string());
        assert_eq!(definitions[0].operation().to_string(), definitions[1].operation().to_string());
        assert_ne!(definitions[0].semantic_key().unwrap(), definitions[1].semantic_key().unwrap());
    }

    #[test]
    fn test_kernel_definition_semantic_key_gather_fill_bits() {
        // Two NaN fills that differ only in their payload bits produce distinct results on the fill path, so their
        // gathers must have distinct keys and must not reuse each other's cached results.
        let gather = || GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1]);
        let fill = |bits: u32| {
            gather().with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(f32::from_bits(bits)).unwrap())) })
        };
        let definitions = [0x7fc0_0001, 0x7fc0_0002].map(|bits| gather_definition(fill(bits)));
        let input = Array::vector(vec![1f32, 2f32]).unwrap();
        let out_of_bounds = Array::matrix(1, 1, vec![5i32]).unwrap();
        let gathered_bits = |definition: &KernelDefinition| {
            let outputs = definition.interpret(vec![input.clone(), out_of_bounds.clone()], 1).unwrap();
            outputs[0].elements::<f32>().unwrap()[0].to_bits()
        };
        assert_eq!(definitions.iter().map(gathered_bits).collect::<Vec<_>>(), vec![0x7fc0_0001, 0x7fc0_0002]);
        let keys = definitions.iter().map(|definition| definition.semantic_key().unwrap()).collect::<Vec<_>>();
        assert_ne!(keys[0], keys[1]);
        let mut cache = HashMap::new();
        cache.insert(keys[0].clone(), gathered_bits(&definitions[0]));
        assert_eq!(cache.get(&keys[1]), None);
        cache.insert(keys[1].clone(), gathered_bits(&definitions[1]));
        assert_eq!(cache.get(&keys[0]), Some(&0x7fc0_0001));
        assert_eq!(cache.get(&keys[1]), Some(&0x7fc0_0002));

        // A gather constructed twice has the same key, and every mode and promise has a distinct key.
        assert_eq!(gather_definition(fill(0x7fc0_0001)).semantic_key().unwrap(), keys[0]);
        let distinct = [
            gather(),
            gather().with_mode(GatherMode::Clip),
            gather().with_mode(GatherMode::Fill { value: None }),
            fill(0x7fc0_0001),
            fill(0x7fc0_0002),
            gather().with_indices_are_sorted(true),
            gather().with_unique_indices(true),
        ]
        .into_iter()
        .map(|operation| gather_definition(operation).semantic_key().unwrap())
        .collect::<HashSet<_>>();
        assert_eq!(distinct.len(), 7);
    }

    #[test]
    fn test_kernel_definition_semantic_key_scatter() {
        let scatter = |kind| ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), kind);
        let input = Array::vector(vec![1f32, 2f32, 3f32, 4f32]).unwrap();
        let indices = Array::matrix(2, 1, vec![1i32, 5]).unwrap();
        let updates = Array::vector(vec![10f32, 20f32]).unwrap();

        // Each update semantics and bounds handling produces a distinct result and has a distinct key.
        let variants = [
            (scatter(ScatterReductionKind::Overwrite).with_mode(ScatterMode::Drop), vec![1f32, 10f32, 3f32, 4f32]),
            (scatter(ScatterReductionKind::Add).with_mode(ScatterMode::Drop), vec![1f32, 12f32, 3f32, 4f32]),
            (scatter(ScatterReductionKind::Add).with_mode(ScatterMode::Clip), vec![1f32, 12f32, 3f32, 24f32]),
        ];
        let mut keys = HashSet::new();
        for (operation, expected) in variants {
            let definition = scatter_definition(operation.clone());
            assert_eq!(
                definition.interpret(vec![input.clone(), indices.clone(), updates.clone()], 1),
                Ok(vec![Array::vector(expected).unwrap()]),
            );
            assert_eq!(scatter_definition(operation).semantic_key(), definition.semantic_key());
            keys.insert(definition.semantic_key().unwrap());
        }
        assert_eq!(keys.len(), 3);

        // Every reduction kind and promise has a distinct key.
        let distinct = [
            scatter(ScatterReductionKind::Overwrite),
            scatter(ScatterReductionKind::Add),
            scatter(ScatterReductionKind::Mul),
            scatter(ScatterReductionKind::Min),
            scatter(ScatterReductionKind::Max),
            scatter(ScatterReductionKind::Add).with_indices_are_sorted(true),
            scatter(ScatterReductionKind::Add).with_unique_indices(true),
        ]
        .into_iter()
        .map(|operation| scatter_definition(operation).semantic_key().unwrap())
        .collect::<HashSet<_>>();
        assert_eq!(distinct.len(), 7);
    }

    #[test]
    fn test_kernel_definition_semantic_key_shardings() {
        // Builds a kernel without parameters whose body reshapes an `f32[4]` constant that carries `sharding`. Kernel
        // parameters are physical, so a sharding can only appear on such a body value.
        let definition = |sharding: Option<Sharding>| {
            let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding).unwrap();
            let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let shape = r#type.shape().clone();
            let value = builder.add_constant(ArrayIrValue::Array(Array::from_elements(r#type, &[0.0f32; 4]).unwrap()));
            let reshape = ArrayOperation::<Array>::Reshape(ReshapeOperation::new(shape.clone(), shape));
            builder.add_instruction(reshape, vec![], vec![value], None).unwrap();
            KernelDefinition::new(call, builder.build(vec![], vec![], vec![]).unwrap()).unwrap()
        };
        let mesh = |name: &str, size: usize, r#type: MeshAxisType| {
            LogicalMesh::new(vec![MeshAxis::new(name, size, r#type).unwrap()]).unwrap()
        };
        let sharded = |name: &str, size: usize, r#type: MeshAxisType| {
            Sharding::new(mesh(name, size, r#type), vec![ShardingDimension::sharded([name])]).unwrap()
        };
        let replicated = |r#type: MeshAxisType| Sharding::replicated(mesh("x", 2, r#type), 1);

        // Separately constructed equal shardings share a key, which records the sharding structurally.
        let key = definition(Some(sharded("x", 2, MeshAxisType::Explicit))).semantic_key().unwrap();
        assert_eq!(definition(Some(sharded("x", 2, MeshAxisType::Explicit))).semantic_key().unwrap(), key);
        assert!(key.contains(r#""sharding":{"mesh":[{"name":"x","size":2,"type":"Explicit"}],"#));

        // Mesh axis names, sizes, and types, placements, and reduction state are all semantic.
        let distinct = [
            None,
            Some(sharded("x", 2, MeshAxisType::Explicit)),
            Some(sharded("y", 2, MeshAxisType::Explicit)),
            Some(sharded("x", 4, MeshAxisType::Explicit)),
            Some(sharded("x", 2, MeshAxisType::Manual)),
            Some(replicated(MeshAxisType::Explicit)),
            Some(Sharding::new(mesh("x", 2, MeshAxisType::Explicit), vec![ShardingDimension::Unconstrained]).unwrap()),
            Some(replicated(MeshAxisType::Explicit).with_unreduced_axes(["x"]).unwrap()),
            Some(
                replicated(MeshAxisType::Explicit)
                    .with_unreduced_axes(["x"])
                    .unwrap()
                    .with_unreduced_kind(UnreducedKind::Max)
                    .unwrap(),
            ),
            Some(replicated(MeshAxisType::Explicit).with_reduced_axes(["x"]).unwrap()),
            Some(replicated(MeshAxisType::Manual).with_varying_manual_axes(["x"]).unwrap()),
        ]
        .into_iter()
        .map(|sharding| definition(sharding).semantic_key().unwrap())
        .collect::<HashSet<_>>();
        assert_eq!(distinct.len(), 11);
    }

    #[test]
    fn test_kernel_definition_semantic_key_output_shardings() {
        // Builds a kernel without parameters whose body reduces an `f32[4, 2]` constant sharded along `x` over its
        // first axis with `operation`. Kernel parameters are physical, so only a body value can be sharded.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let definition = |operation: ReduceOperation| {
            let sharding =
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                    .unwrap();
            let input_type = ArrayType::new_static(DataType::F32, [4, 2]).with_sharding(sharding).unwrap();
            let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let input = Array::from_elements(input_type, &[0.0f32; 8]).unwrap();
            let input = builder.add_constant(ArrayIrValue::Array(input));
            builder
                .add_instruction(ArrayOperation::<Array>::Reduce(operation), vec![], vec![input], None)
                .unwrap();
            KernelDefinition::new(call, builder.build(vec![], vec![], vec![]).unwrap()).unwrap()
        };
        let reduce = || ReduceOperation::new(vec![0], ReductionKind::Sum);
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();

        // Separately constructed equal requests share a key.
        let key = definition(reduce().with_output_sharding(replicated.clone()).unwrap()).semantic_key().unwrap();
        assert_eq!(definition(reduce().with_output_sharding(replicated.clone()).unwrap()).semantic_key().unwrap(), key);
        assert!(key.contains(r#"{"KeyOnly":{"OutputSharding":{"operation":{"Reduce":{"axes":[0],"kind":"Sum"}},"#));

        // The absence of a request and distinct requests have distinct keys.
        let distinct = [
            reduce(),
            reduce().with_output_sharding(replicated).unwrap(),
            reduce().with_output_sharding(unreduced).unwrap(),
        ]
        .into_iter()
        .map(|operation| definition(operation).semantic_key().unwrap())
        .collect::<HashSet<_>>();
        assert_eq!(distinct.len(), 3);
    }

    #[test]
    fn test_kernel_definition_semantic_key_preserves_custom_attribute_kinds_and_bits() {
        let definition = |attribute| {
            let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            body.add_instruction(
                ArrayIrOperation::CustomCall(
                    CustomCallOperation::new("test.foreign", vec![]).with_attribute("value", attribute),
                ),
                vec![],
                vec![],
                None,
            )
            .unwrap();
            KernelDefinition::new(
                KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap(),
                body.build(vec![], vec![], vec![]).unwrap(),
            )
            .unwrap()
        };

        // Attributes of different kinds render differently (e.g., strings are quoted and narrow integers carry their
        // width), and their keys differ as well.
        for (attributes, renderings) in [
            (
                [CustomCallAttribute::String("true".to_owned()), CustomCallAttribute::Boolean(true)],
                ["value=\"true\"", "value=true"],
            ),
            ([CustomCallAttribute::String("1".to_owned()), CustomCallAttribute::I64(1)], ["value=\"1\"", "value=1"]),
            ([CustomCallAttribute::I32(1), CustomCallAttribute::I64(1)], ["value=1i32", "value=1"]),
            (
                [CustomCallAttribute::Bytes(vec![0, 255]), CustomCallAttribute::String("bytes [00, ff]".to_owned())],
                ["value=bytes [00, ff]", "value=\"bytes [00, ff]\""],
            ),
        ] {
            let definitions = attributes.map(definition);
            for (definition, rendering) in definitions.iter().zip(renderings) {
                assert_eq!(
                    definition.body().to_string(),
                    formatdoc! {"
                        lambda  .
                        let () = custom_call [target=test.foreign, {rendering}]
                        in ()"},
                );
            }
            assert_ne!(definitions[0].semantic_key().unwrap(), definitions[1].semantic_key().unwrap());
        }

        // Floating-point attributes that differ only in their NaN payload render identically, but the key preserves
        // their bits.
        for attributes in [
            [
                CustomCallAttribute::F64(f64::from_bits(0x7ff8_0000_0000_0001)),
                CustomCallAttribute::F64(f64::from_bits(0x7ff8_0000_0000_0002)),
            ],
            [
                CustomCallAttribute::F32(f32::from_bits(0x7fc0_0001)),
                CustomCallAttribute::F32(f32::from_bits(0x7fc0_0002)),
            ],
        ] {
            let definitions = attributes.map(definition);
            assert_eq!(definitions[0].body().to_string(), definitions[1].body().to_string());
            assert_ne!(definitions[0].semantic_key().unwrap(), definitions[1].semantic_key().unwrap());
        }
    }

    #[test]
    fn test_kernel_definition_semantic_key_exact_extension_payload() {
        let keys = [0, 1, 256, u32::MAX].map(|payload| {
            let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<TestExtension>>::new();
            let extension = TestExtension::HiddenKey { key: payload, exact: true };
            body.add_instruction(KernelOperation::Extension(extension), vec![], vec![], None).unwrap();
            let definition = KernelDefinition::new(
                KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap(),
                body.build(vec![], vec![], vec![]).unwrap(),
            )
            .unwrap();
            let key = definition.semantic_key().unwrap();
            assert_eq!(key, definition.clone().semantic_key().unwrap());
            key
        });
        for (position, key) in keys.iter().enumerate() {
            for other in &keys[position + 1..] {
                assert_ne!(key, other);
            }
        }
    }

    #[test]
    fn test_kernel_definition_semantic_key_extension_dimension_identities() {
        // Builds a kernel over a grid with two dynamic extents of equal bounds, named `names`, whose body holds an
        // extension that carries the extent identity of grid axis `selected` in its payload.
        let key = |names: [&str; 2], selected: usize| {
            let bounds = DimensionBounds::non_negative(Some(8)).unwrap();
            let extents = names.map(|name| DimensionVariable::new(name, bounds));
            let grid = Grid::new(
                extents
                    .iter()
                    .map(|extent| GridDimension::new(Dimension::Dynamic(extent.clone()), GridExecution::Parallel))
                    .collect(),
            )
            .unwrap();
            let operation = KernelCallOperation::new(grid, vec![]).unwrap();
            let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<TestExtension>>::new();
            for r#type in operation.body_input_types() {
                body.add_input(r#type);
            }
            let extension = TestExtension::Dimension(extents[selected].clone());
            body.add_instruction(KernelOperation::Extension(extension), vec![], vec![], None).unwrap();
            let definition =
                KernelDefinition::new(operation, body.build(vec![], vec![Placeholder; 2], vec![]).unwrap()).unwrap();
            definition.semantic_key().unwrap()
        };

        // Independently constructed definitions that differ only in the names of their identities share a key.
        assert_eq!(key(["rows", "columns"], 0), key(["height", "width"], 0));
        assert_eq!(key(["rows", "columns"], 1), key(["height", "width"], 1));

        // Selecting the other of two distinct identities with equal names and bounds changes the key.
        assert_ne!(key(["extent", "extent"], 0), key(["extent", "extent"], 1));
        assert_ne!(key(["rows", "columns"], 0), key(["rows", "columns"], 1));
    }

    #[test]
    fn test_kernel_definition_semantic_key_rejects_extension_without_exact_identity() {
        for payload in [1, 2] {
            let extension = TestExtension::HiddenKey { key: payload, exact: false };
            assert!(matches!(extension, TestExtension::HiddenKey { key, .. } if key == payload));
            let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<TestExtension>>::new();
            body.add_instruction(KernelOperation::Extension(extension), vec![], vec![], None).unwrap();
            let definition = KernelDefinition::new(
                KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap(),
                body.build(vec![], vec![], vec![]).unwrap(),
            )
            .unwrap();
            assert_eq!(
                definition.semantic_key(),
                Err(TypeError::invalid(
                    "kernel extension `hidden_key_extension` has no exact semantic identity contract",
                )),
            );
        }
    }

    #[test]
    fn test_kernel_definition_semantic_key_composites() {
        // Builds a kernel whose body applies `operation` to one scalar constant, with `region_count` attached regions
        // that are each the scalar identity.
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let definition = |operation: ArrayIrOperation<Array>, region_count: usize| {
            let mut identity = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let input = identity.add_input(scalar_type.clone());
            let identity = identity
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![input],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let regions = (0..region_count).map(|_| body.import_program(identity.clone())).collect::<Vec<_>>();
            let input = body.add_constant(ArrayIrValue::Array(Array::scalar(1f32).unwrap()));
            body.add_instruction(KernelOperation::Portable(operation), regions, vec![input], None).unwrap();
            KernelDefinition::new(
                KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap(),
                body.build(vec![], vec![], vec![]).unwrap(),
            )
            .unwrap()
        };

        // A call with attached rules is encoded structurally, because its rules are regions of the body.
        let attached = || CompositeOperation::from_rule_regions(CompositeJvpRule::Absent, false);
        assert_eq!(
            definition(ArrayIrOperation::Composite(attached()), 1).semantic_key().unwrap(),
            definition(ArrayIrOperation::Composite(attached()), 1).semantic_key().unwrap(),
        );

        // Calls and carriers with retained rules have no encoding, whether their rules are registered in this family or
        // lifted from the array member family, because the identity of their rules is local to the process.
        let retained = |name: &str| {
            Err(TypeError::invalid(format!(
                "`{name}` operations with retained rules have no kernel semantic descriptor encoding, because the \
                 identity of their rules is local to the process",
            )))
        };
        let rules = CustomRuleRegistration::new(
            CustomRuleDefinition::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new("retained"),
        );
        assert_eq!(
            definition(ArrayIrOperation::Composite(CompositeOperation::new(rules.reference())), 1).semantic_key(),
            retained("composite"),
        );
        let member_rules =
            CustomRuleRegistration::new(CustomRuleDefinition::<Array, ArrayOperation<Array>>::new("member"));
        assert_eq!(
            definition(
                ArrayIrOperation::from(ArrayOperation::Composite(CompositeOperation::new(member_rules.reference()))),
                1,
            )
            .semantic_key(),
            retained("composite"),
        );
        let carrier = CompositeTransposeOperation::new(
            rules.reference(),
            0,
            vec![scalar_type.clone()],
            vec![scalar_type.clone()],
        );
        assert_eq!(
            definition(ArrayIrOperation::CompositeTranspose(carrier), 0).semantic_key(),
            retained("composite_transpose"),
        );

        // Calls of named composites are encoded by their keys, so separately registered
        // definitions with equal keys encode equally, in either family.
        let named = |attribute: CompositeAttribute| {
            let key = CompositeKey::new("ryft.tests.kernel_named", 1).with_attribute("attribute", attribute);
            CustomRuleRegistration::new(
                CustomRuleDefinition::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new("named")
                    .with_named_composite(NamedComposite::new(
                        key,
                        |types: &[ArrayIrType]| Ok(types.to_vec()),
                        |inputs: &[Tracer<TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>]| {
                            Ok(inputs.to_vec())
                        },
                    )),
            )
        };
        let named_key = |rules: &CustomRuleRegistration<ArrayIrValue<Array>, ArrayIrOperation<Array>>| {
            definition(ArrayIrOperation::Composite(CompositeOperation::new(rules.reference())), 1).semantic_key()
        };
        let (first, second, other) = (named(true.into()), named(true.into()), named(false.into()));
        assert_ne!(first.definition().id(), second.definition().id());
        assert_eq!(named_key(&first).unwrap(), named_key(&second).unwrap());
        assert_ne!(named_key(&first).unwrap(), named_key(&other).unwrap());

        // Semantic descriptors preserve scalar kinds, optional structure, and float bits at every nesting level.
        for [left, right] in [
            [CompositeAttribute::Float(0.0), CompositeAttribute::Float(-0.0)],
            [
                CompositeAttribute::Float(f64::from_bits(0x7ff8_0000_0000_0001)),
                CompositeAttribute::Float(f64::from_bits(0x7ff8_0000_0000_0002)),
            ],
            [
                CompositeAttribute::from(Some((f64::from_bits(0x7ff8_0000_0000_0001), u64::MAX))),
                CompositeAttribute::from(Some((f64::from_bits(0x7ff8_0000_0000_0002), u64::MAX))),
            ],
            [CompositeAttribute::Integer(1), CompositeAttribute::Float(1.0)],
            [CompositeAttribute::from(None::<u64>), CompositeAttribute::from(Some(None::<u64>))],
            [CompositeAttribute::from(vec![1, 2]), CompositeAttribute::from(vec![2, 1])],
        ] {
            let (first, repeated, other) = (named(left.clone()), named(left), named(right));
            assert_eq!(named_key(&first).unwrap(), named_key(&repeated).unwrap());
            assert_ne!(named_key(&first).unwrap(), named_key(&other).unwrap());
        }

        let named_member = || {
            CustomRuleRegistration::new(
                CustomRuleDefinition::<Array, ArrayOperation<Array>>::new("member").with_named_composite(
                    NamedComposite::new(
                        CompositeKey::new("ryft.tests.kernel_named_member", 1),
                        |types: &[ArrayType]| Ok(types.to_vec()),
                        |inputs: &[Tracer<TracingContext<Array, ArrayOperation<Array>>>]| Ok(inputs.to_vec()),
                    ),
                ),
            )
        };
        let named_member_key = |rules: &CustomRuleRegistration<Array, ArrayOperation<Array>>| {
            definition(ArrayIrOperation::from(ArrayOperation::Composite(CompositeOperation::new(rules.reference()))), 1)
                .semantic_key()
        };
        let (first, second) = (named_member(), named_member());
        assert_ne!(first.definition().id(), second.definition().id());
        assert_eq!(named_member_key(&first).unwrap(), named_member_key(&second).unwrap());

        // A carrier with an attached backward rule is an unresolved reverse mode obligation, so it has no encoding
        // either.
        let carrier =
            CompositeTransposeOperation::from_backward_region(0, vec![scalar_type.clone()], vec![scalar_type.clone()]);
        assert_eq!(
            definition(ArrayIrOperation::CompositeTranspose(carrier), 1).semantic_key(),
            Err(TypeError::invalid("`composite_transpose` operations have no kernel semantic descriptor encoding")),
        );
    }

    #[test]
    fn test_kernel_definition_semantic_key_rejects_rematerialize() {
        // Builds a kernel whose body applies `operation` to one scalar constant through an attached scalar identity.
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let definition = |operation: ArrayIrOperation<Array>| {
            let mut identity = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let input = identity.add_input(scalar_type.clone());
            let identity = identity
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![input],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let region = body.import_program(identity);
            let input = body.add_constant(ArrayIrValue::Array(Array::scalar(1f32).unwrap()));
            body.add_instruction(KernelOperation::Portable(operation), vec![region], vec![input], None).unwrap();
            KernelDefinition::new(
                KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap(),
                body.build(vec![], vec![], vec![]).unwrap(),
            )
            .unwrap()
        };

        // Each registration mints a new policy identity that is local to the process, so calls in either family have
        // no encoding.
        let expected = Err(TypeError::invalid(
            "`rematerialize` operations have no kernel semantic descriptor encoding, because the identity of their \
             residual policies is local to the process",
        ));
        let array_operation = ArrayIrOperation::from(ArrayOperation::Rematerialize(RematerializeOperation::new(
            ResidualPolicyReference::new(NothingSavable),
        )));
        assert_eq!(definition(array_operation).semantic_key(), expected);
        let array_ir_operation =
            ArrayIrOperation::Rematerialize(RematerializeOperation::new(ResidualPolicyReference::new(NothingSavable)));
        assert_eq!(definition(array_ir_operation).semantic_key(), expected);
    }

    #[test]
    fn test_kernel_definition_semantic_key_ignores_grid_names() {
        let key = |dimension: GridDimension| {
            let operation = KernelCallOperation::new(Grid::new(vec![dimension]).unwrap(), vec![]).unwrap();
            let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            for r#type in operation.body_input_types() {
                body.add_input(r#type);
            }
            let definition =
                KernelDefinition::new(operation, body.build(vec![], vec![Placeholder], vec![]).unwrap()).unwrap();
            definition.semantic_key().unwrap()
        };
        let unnamed = GridDimension::new(Dimension::Static(2), GridExecution::Parallel);
        let named = key(unnamed.clone().with_name("rows").unwrap());
        assert_eq!(named, key(unnamed.clone()));
        assert_eq!(named, key(unnamed.with_name("columns").unwrap()));
        assert_ne!(named, key(GridDimension::new(Dimension::Static(2), GridExecution::Sequential)));
        assert_ne!(named, key(GridDimension::new(Dimension::Static(3), GridExecution::Parallel)));
    }

    #[test]
    fn test_kernel_definition_semantic_key_ignores_dimension_names() {
        // A dimension arithmetic operation caches the name of its output, which is derived from the names of its
        // operands and survives identity renaming, so the key must not depend on how a mapping names its coordinate.
        let key = |name: &str| {
            let coordinate_type = DimensionType::new(name, DimensionBounds::non_negative(Some(2)).unwrap());
            let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let coordinate = mapping.add_input(ArrayIrType::Dimension(coordinate_type.clone()));
            let size = DimensionValue::constant(1).unwrap();
            let multiplication = DimensionMulOperation::new(&coordinate_type, size.r#type().as_ref()).unwrap();
            let size = mapping.add_constant(ArrayIrValue::Dimension(size));
            let start = mapping
                .add_instruction(ArrayIrOperation::from(multiplication), vec![], vec![coordinate, size], None)
                .unwrap()[0];
            let mapping = BlockMapping::new(
                mapping.build(vec![start], vec![Placeholder], vec![Placeholder]).unwrap(),
                vec![1],
                BoundaryPolicy::InBounds,
            )
            .unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            let parameter = KernelParameter::new(r#type, KernelParameterAccess::WriteOnly, mapping).unwrap();
            let grid = Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap();
            let operation = KernelCallOperation::new(grid, vec![parameter]).unwrap();
            let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let reference = body.add_input(operation.body_input_types()[0].clone());
            body.add_input(operation.body_input_types()[1].clone());
            let value = Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[1.5f32]).unwrap();
            let value = body.add_constant(ArrayIrValue::Array(value));
            body.add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, value], None).unwrap();
            let definition =
                KernelDefinition::new(operation, body.build(vec![], vec![Placeholder; 2], vec![]).unwrap()).unwrap();
            definition.semantic_key().unwrap()
        };
        assert_eq!(key("row"), key("column"));
    }

    #[test]
    fn test_kernel_definition_semantic_key_ignores_provenance() {
        let operation = copy_definition().operation().clone();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let input = builder.add_input(operation.parameters()[0].body_type());
        let output = builder.add_input(operation.parameters()[1].body_type());
        let provenance = Provenance::scope(ProvenanceScope::new("copy"), Provenance::unknown());
        let value = builder
            .add_instruction(ReferenceReadOperation::new(), vec![], vec![input], Some(provenance.clone()))
            .unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, value], Some(provenance))
            .unwrap();
        let definition =
            KernelDefinition::new(operation, builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap()).unwrap();
        assert_ne!(
            definition.body().entry_region().instructions()[0].provenance(),
            copy_definition().body().entry_region().instructions()[0].provenance(),
        );
        assert_eq!(definition.semantic_key().unwrap(), copy_definition().semantic_key().unwrap());
    }

    #[test]
    fn test_kernel_definition_semantic_key_matches_source_roundtrip() {
        // Decoding creates fresh nominal identities that keep their names, and the key matches either way.
        for definition in [
            copy_definition(),
            dynamic_definition("first", 4, true),
            dynamic_definition("first", 4, false),
            dynamic_definition("first", 4, true).specialize_grid(&[2, 2]).unwrap(),
            prefetch_definition(),
            prefetch_definition().specialize_prefetch(&[Array::scalar(2i32).unwrap()]).unwrap(),
        ] {
            let decoded = KernelDefinition::from_source(definition.to_source().unwrap()).unwrap();
            assert_eq!(decoded.semantic_key().unwrap(), definition.semantic_key().unwrap());
        }
    }

    #[test]
    fn test_kernel_definition_semantic_key_unsupported_source_features() {
        // Builds a scalar copy kernel that prints its value with `label` on the way.
        let definition = |label: &str| {
            let operation = copy_definition().operation().clone();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let input = builder.add_input(operation.parameters()[0].body_type());
            let output = builder.add_input(operation.parameters()[1].body_type());
            let value = builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![input], None).unwrap()[0];
            let value = builder
                .add_instruction(ArrayOperation::Print(PrintOperation::new(label)), vec![], vec![value], None)
                .unwrap()[0];
            builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, value], None).unwrap();
            KernelDefinition::new(operation, builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap()).unwrap()
        };

        // The transport schema rejects the print, but the key still encodes it exactly.
        assert!(definition("first").to_source().is_err_and(|error| error.to_string().ends_with("`print`")));
        assert_eq!(definition("first").semantic_key().unwrap(), definition("first").semantic_key().unwrap());
        assert_ne!(definition("first").semantic_key().unwrap(), definition("second").semantic_key().unwrap());
    }

    #[test]
    fn test_kernel_definition_semantic_key_format() {
        let coordinate_type = DimensionType::new("row", DimensionBounds::non_negative(Some(2)).unwrap());
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let coordinate = mapping.add_input(ArrayIrType::Dimension(coordinate_type.clone()));
        let size = DimensionValue::constant(1).unwrap();
        let multiplication = DimensionMulOperation::new(&coordinate_type, size.r#type().as_ref()).unwrap();
        let size = mapping.add_constant(ArrayIrValue::Dimension(size));
        let start = mapping
            .add_instruction(ArrayIrOperation::from(multiplication), vec![], vec![coordinate, size], None)
            .unwrap()[0];
        let parameter = KernelParameter::new(
            ArrayType::new_static(DataType::F32, [2]),
            KernelParameterAccess::WriteOnly,
            BlockMapping::new(
                mapping.build(vec![start], vec![Placeholder], vec![Placeholder]).unwrap(),
                vec![1],
                BoundaryPolicy::InBounds,
            )
            .unwrap(),
        )
        .unwrap();
        let grid = Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap();
        let operation = KernelCallOperation::new(grid, vec![parameter]).unwrap();
        let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = body.add_input(operation.body_input_types()[0].clone());
        body.add_input(operation.body_input_types()[1].clone());
        let value = Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[1.5f32]).unwrap();
        let value = body.add_constant(ArrayIrValue::Array(value));
        body.add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, value], None).unwrap();
        let definition =
            KernelDefinition::new(operation, body.build(vec![], vec![Placeholder; 2], vec![]).unwrap()).unwrap();

        // After the first release of `ryft-kernels`, a deliberate change to this string requires a `KERNEL_KEY_VERSION`
        // bump.
        assert_eq!(
            definition.semantic_key().unwrap(),
            concat!(
                "kernel key 1\n",
                r#"{"version":1,"variables":[{"name":"","lower":0,"upper":2},{"name":"","lower":0,"upper":2},"#,
                r#"{"name":"","lower":0,"upper":2}],"call":{"grid":[{"extent":{"Static":2},"name":null,"#,
                r#""execution":"Parallel"}],"parameters":[{"type":{"data_type":"F32","shape":[{"Static":2}],"#,
                r#""layout":null,"memory":"Device"},"access":"WriteOnly","#,
                r#""mapping":{"regions":[{"atoms":[{"Variable":{"Dimension":{"Dynamic":0}}},"#,
                r#"{"Constant":{"Dimension":{"extent":1}}},{"Variable":{"Dimension":{"Dynamic":1}}}],"inputs":[0],"#,
                r#""outputs":[2],"instructions":[{"operation":{"DimensionMul":{"left":{"Dynamic":0},"#,
                r#""right":{"Static":1}}},"reference_transforms":[],"inputs":[0,1],"outputs":[2],"regions":[],"#,
                r#""provenance":{"nodes":["Unknown"],"root":0}}]}],"entry":0},"block_shape":[1],"#,
                r#""boundary_policy":"InBounds"}],"coordinates":[{"Dynamic":2}],"prefetch":[]},"#,
                r#""body":{"regions":[{"atoms":[{"Variable":{"Reference":{"data_type":"F32","shape":[{"Static":1}],"#,
                r#""layout":null,"memory":"Device"}}},{"Variable":{"Dimension":{"Dynamic":2}}},"#,
                r#"{"Constant":{"Array":{"type":{"data_type":"F32","shape":[{"Static":1}],"layout":null,"#,
                r#""memory":"Device"},"bytes":[0,0,192,63]}}}],"inputs":[0,1],"outputs":[],"#,
                r#""instructions":[{"operation":"ReferenceWrite","reference_transforms":[{"input_index":0,"#,
                r#""transforms":[]}],"inputs":[0,2],"outputs":[],"regions":[],"provenance":{"nodes":["Unknown"],"#,
                r#""root":0}}]}],"entry":0}}"#,
            ),
        );
    }

    #[test]
    fn test_kernel_definition_semantic_key_replaces_historical_key() {
        // The encoder that preceded the structured records (i.e., `KernelDefinition::semantic_key` at the parent of
        // `d13108387`) framed `Debug` renderings by their byte lengths. This is the key that it produced for an empty
        // kernel, reconstructed by hand from the source of that encoder in `crates/ryft-kernels/src/calls.rs`.
        let historical = concat!(
            "kernel key 2; schema 2\n",
            "2:[]",
            "34:call (Grid { dimensions: [] }, [])",
            "8:end call",
            "12:body entry 0",
            "15:region ([], [])",
            "9:end atoms",
            "10:end region",
        );
        let definition: KernelDefinition =
            KernelDefinition::trace(KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap(), |_| Ok(()))
                .unwrap();
        let key = definition.semantic_key().unwrap();
        assert_ne!(key, historical);
        assert!(key.starts_with("kernel key 1\n{"));

        // A cache populated with the historical key misses for the structured key of the same definition.
        let cache = HashMap::from([(historical.to_owned(), "stale artifact")]);
        assert_eq!(cache.get(&key), None);
    }

    #[test]
    fn test_kernel_definition_semantic_key_reference_transforms() {
        let mut keys = Vec::new();
        for index in [0, 1] {
            let call = KernelCallOperation::new(
                Grid::new(vec![]).unwrap(),
                vec![
                    whole_array_parameter(ArrayType::new_static(DataType::I32, [2]), KernelParameterAccess::ReadOnly)
                        .unwrap(),
                ],
            )
            .unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let input = builder.add_input(call.body_input_types()[0].clone());
            builder
                .add_instruction(
                    ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                        axis: 0,
                        index: ArrayReferenceTransformIndex::Static(index),
                    }]),
                    vec![],
                    vec![input],
                    None,
                )
                .unwrap();
            let definition =
                KernelDefinition::new(call, builder.build(vec![], vec![Placeholder], vec![]).unwrap()).unwrap();
            keys.push(definition.semantic_key().unwrap());
        }
        assert_ne!(keys[0], keys[1]);
    }

    #[test]
    fn test_kernel_definition_semantic_key_window_mask() {
        // A body that stores under the validity mask of its output differs from one that stores under a constant mask.
        let mut keys = Vec::new();
        for window_mask in [false, true] {
            let call = KernelCallOperation::new(
                Grid::new(vec![]).unwrap(),
                vec![
                    whole_array_parameter(ArrayType::new_static(DataType::I32, [2]), KernelParameterAccess::WriteOnly)
                        .unwrap(),
                ],
            )
            .unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let output = builder.add_input(call.body_input_types()[0].clone());
            let value = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![5i32, 7]).unwrap()));
            let mask = if window_mask {
                builder.add_instruction(WindowMaskOperation, vec![], vec![output], None).unwrap()[0]
            } else {
                builder.add_constant(ArrayIrValue::Array(Array::vector(vec![true, true]).unwrap()))
            };
            builder
                .add_instruction(MaskedStoreOperation::new(), vec![], vec![output, value, mask], None)
                .unwrap();
            let definition =
                KernelDefinition::new(call, builder.build(vec![], vec![Placeholder], vec![]).unwrap()).unwrap();
            keys.push(definition.semantic_key().unwrap());
        }
        assert_ne!(keys[0], keys[1]);
        assert!(keys[1].contains(r#"{"operation":"WindowMask","#));
    }
}
