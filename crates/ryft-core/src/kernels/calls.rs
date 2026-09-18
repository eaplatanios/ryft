//! Functional kernel calls with reference windows confined to an attached body region.
//!
//! Full array signatures and block mappings belong to the call. The body receives logical block references and
//! canonical dimension coordinates; read-write and write-only parameters publish ordinary arrays at the outer
//! boundary. Compiler and launch metadata are intentionally absent. Construction validates types and reference access;
//! execution additionally requires the initialization, coverage, and race checks performed by the kernel verifier.

use std::borrow::Cow;
use std::fmt::{Display, Write};

use thiserror::Error;

use crate::arrays::{
    Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReferenceView, ArraySliceAxis, ArrayType,
    ArrayTypeRefinements, Dimension, DimensionBounds, DimensionType, DimensionVariable,
};
use crate::contexts::EagerContext;
use crate::kernels::grids::{Grid, GridError, GridExecution};
use crate::kernels::mappings::{BlockMapping, BlockMappingError, BoundaryPolicy};
use crate::kernels::operations::{KernelExtension, KernelOperation, NoKernelExtension};
use crate::kernels::validation::{
    KernelBoundaryContract, KernelParameterAccess, KernelReferenceSummary, KernelValidationError, validate_kernel_body,
};
use crate::operations::attention::AttentionConfiguration;
use crate::operations::custom_call::{CustomCallAttribute, CustomCallOperation};
use crate::operations::{DimensionFromScalar, DimensionFromScalarOperation, PadOperation};
use crate::parameters::Placeholder;
use crate::programs::{
    Atom, FlatProgram, InputRegionProvenance, Operation, OperationFormatter, ProgramBuilder, ProgramError,
    ReferenceAccessMode, ReferenceType, ReferenceViewOperation, RegionInterface, RegionRef, RegionSlot, Type,
    TypeError, TypeIdentityRenaming, TypeRefinements, Typed,
};
use crate::tracing::{Tracer, TracingContext};

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

    /// The canonical mapping verifier rejected a block mapping.
    #[error(transparent)]
    Mapping(#[from] BlockMappingError),

    /// Reference access or lifetime validation rejected the body.
    #[error(transparent)]
    Validation(#[from] KernelValidationError),

    /// Grid ordering does not admit an access performed on a body parameter.
    #[error("kernel grid does not permit `{mode}` access on parameter {parameter}")]
    DisallowedGridAccess {
        /// Parameter whose access requires stronger grid ordering.
        parameter: usize,
        /// Exact canonical access mode rejected by the grid region policy.
        mode: ReferenceAccessMode,
    },

    /// The block and full array have different logical ranks.
    #[error("kernel parameter has array rank {array_rank} but block rank {block_rank}")]
    ParameterRank { array_rank: usize, block_rank: usize },

    /// Mapping arguments must correspond to grid coordinates followed by scalar-prefetched dimensions.
    #[error("kernel parameter {parameter} mapping takes {actual} inputs but the call requires {expected}")]
    MappingInputCount { parameter: usize, expected: usize, actual: usize },

    /// Independent grid axes cannot impose equality through a shared mapping dimension identity.
    #[error(
        "kernel parameter {parameter} mapping axes {first_axis} and {second_axis} share a dimension identity but may \
         differ"
    )]
    MappingInputIdentity {
        /// Parameter whose mapping equates independent coordinates.
        parameter: usize,
        /// First logical axis using the identity.
        first_axis: usize,
        /// Later logical axis using the same identity.
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
#[derive(Clone, Debug)]
pub struct KernelParameter {
    /// Full array type at the functional outer boundary.
    r#type: ArrayType,

    /// Access permitted on this parameter's body reference.
    access: KernelParameterAccess,

    /// Pure grid-to-window mapping and explicit boundary policy.
    mapping: BlockMapping,

    /// Canonical reference type derived from the full array and logical block by the array view implementation.
    body_type: ArrayIrType,
}

impl KernelParameter {
    /// Creates a parameter whose block rank agrees with its full array rank. The reference type is derived by the
    /// canonical array slice/view rules, preserving whole-window layout and validating partial-window sharding.
    /// Masked windows have a fixed logical block type derived by slicing the valid origin window and applying the
    /// canonical padding type rules. Their private tile storage has no required physical layout. Invalid lanes are
    /// accessible only through masked memory operations with an explicit fallback; padding does not initialize them.
    pub fn new(r#type: ArrayType, access: KernelParameterAccess, mapping: BlockMapping) -> Result<Self, KernelError> {
        if r#type.rank() != mapping.block_shape().len() {
            return Err(KernelError::ParameterRank {
                array_rank: r#type.rank(),
                block_rank: mapping.block_shape().len(),
            });
        }
        let referent = if mapping.boundary_policy() == BoundaryPolicy::Masked {
            let shape = r#type
                .static_shape()
                .ok_or_else(|| TypeError::invalid("masked kernel windows require a static operand shape"))?;
            let valid_shape = mapping
                .block_shape()
                .iter()
                .zip(shape.dimensions())
                .map(|(&block, &extent)| block.min(extent))
                .collect::<Vec<_>>();
            let view = ArrayReferenceView::Slice {
                axes: valid_shape.iter().map(|&extent| ArraySliceAxis::new(0, extent, 1)).collect(),
            };
            let valid_type = view.output_type(&r#type)?;
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
            let view = ArrayReferenceView::Slice {
                axes: mapping.block_shape().iter().map(|&extent| ArraySliceAxis::new(0, extent, 1)).collect(),
            };
            view.output_type(&r#type)?
        };
        let body_type = ArrayIrType::Reference(ReferenceType::new(referent));
        Ok(Self { r#type, access, mapping, body_type })
    }

    /// Specializes declared full array shapes before constructing parameters or tracing a body. Canonical
    /// [`ArrayTypeRefinements`] checks enforce bounds and agreement of repeated dimension identities across
    /// all parameters, including outputs. Every supplied actual shape must be static; unresolved output dimensions
    /// are rejected rather than inferred from unrelated inputs or runtime contents.
    ///
    /// Actual types must refine the declarations, including declared dtype, memory, layout, and sharding constraints.
    /// Only their shapes are copied: additional actual layout or sharding information does not become a declaration
    /// or specialization dependency. Canonical slice and padding inference then validates each fixed block type.
    /// This constructs a concrete signature for subsequent tracing; it does not restage a retained dynamic body.
    ///
    /// # Parameters
    ///
    ///   - `parameters`: Full type declaration, access, and mapping for each parameter in body-reference order.
    ///   - `actual_types`: Explicit concrete types in the same order, including write-only output types.
    pub fn from_refined_types(
        parameters: Vec<(ArrayType, KernelParameterAccess, BlockMapping)>,
        actual_types: &[ArrayType],
    ) -> Result<Vec<Self>, KernelError> {
        ArrayTypeRefinements::establish(parameters.iter().map(|(r#type, _, _)| r#type), actual_types)?;
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

    /// Returns the canonical logical reference type of this parameter inside the body. Whole-array windows retain
    /// the complete array type; partial windows use the existing slice rules for layout, sharding, and memory.
    /// Masked windows use a private padded tile with no required physical layout.
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

/// Canonical name for the higher-order kernel call.
pub const KERNEL_CALL_OPERATION_NAME: &str = "kernel_call";

/// Version of the experimental kernel semantic-key schema. An incompatible change to the encoded semantics
/// requires a new version; this is independent of adapter artifact and executable persistence versions.
pub const KERNEL_SCHEMA_VERSION: u32 = 1;

/// Higher-order operation with one attached, reference-preserving kernel body.
///
/// Body reference inputs follow [`Self::parameters`] order. Outer inputs omit write-only parameters; outer results
/// omit read-only parameters. A read-write parameter therefore contributes one ordinary input and one ordinary
/// result. This functional alias relationship does not expose a reference-state slot or permit an input mutation
/// visible outside the call. The body returns no values: its stores produce the declared array results.
#[derive(Clone, Debug)]
pub struct KernelCallOperation {
    /// Logical execution grid, independent of native launch dimensions.
    grid: Grid,

    /// Logical array parameters in body input order.
    parameters: Vec<KernelParameter>,

    /// Canonical logical coordinate types, in grid-axis order after the reference inputs.
    coordinate_types: Vec<DimensionType>,

    /// Rank-zero integer arrays read before mapping evaluation and supplied as private body values.
    prefetch_types: Vec<ArrayType>,
}

impl KernelCallOperation {
    /// Creates a call with one mapping coordinate per logical grid dimension. Full body validation belongs to
    /// [`KernelDefinition::new`]; constructing metadata alone does not make an executable kernel.
    pub fn new(grid: Grid, parameters: Vec<KernelParameter>) -> Result<Self, KernelError> {
        Self::new_with_prefetch(grid, parameters, vec![])
    }

    /// Creates a call with scalar-prefetched arrays after its ordinary array inputs. Each mapping takes grid
    /// coordinates followed by checked dimensions converted from these rank-zero integer arrays. Body inputs append
    /// the original scalar arrays after references and coordinates. Values must be explicitly specialized before
    /// executable qualification; this preserves the ordinary array ABI at native call boundaries.
    pub fn new_with_prefetch(
        grid: Grid,
        parameters: Vec<KernelParameter>,
        prefetch_types: Vec<ArrayType>,
    ) -> Result<Self, KernelError> {
        for r#type in &prefetch_types {
            DimensionFromScalarOperation::new(DimensionVariable::new("prefetch", DimensionBounds::unbounded()))
                .infer_output_types(&[ArrayIrType::Array(r#type.clone())], &[])?;
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
                        if first_type.variable() == r#type.variable()
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
        }
        let coordinate_types = maximum_extents
            .iter()
            .enumerate()
            .map(|(axis, &extent)| {
                DimensionType::new(DimensionVariable::new(
                    format!("program_{axis}"),
                    DimensionBounds::non_negative(Some(extent.max(1))).unwrap(),
                ))
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
            return Err(TypeError::invalid("serialized coordinate count differs from the grid rank").into());
        }
        for (axis, (actual, required)) in coordinate_types.iter().zip(&operation.coordinate_types).enumerate() {
            if actual.bounds().upper().is_none() || !actual.bounds().contains_bounds(required.bounds()) {
                return Err(TypeError::invalid(format!("coordinate {axis} bounds do not admit the grid")).into());
            }
            if coordinate_types[..axis].iter().any(|earlier| earlier.variable() == actual.variable()) {
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

    /// Returns coordinate types in logical grid-axis order. Each identity belongs to this call definition and admits
    /// indices from zero up to the maximum possible extent. Empty axes use `[0, 1)` because no invocation occurs.
    pub fn coordinate_types(&self) -> &[DimensionType] {
        &self.coordinate_types
    }

    /// Returns the rank-zero integer array types in scalar-prefetch order.
    pub fn prefetch_types(&self) -> &[ArrayType] {
        &self.prefetch_types
    }

    /// Returns the body signature: block references, dimension coordinates, then scalar-prefetched arrays.
    pub fn body_input_types(&self) -> Vec<ArrayIrType> {
        self.parameters
            .iter()
            .map(KernelParameter::body_type)
            .chain(self.coordinate_types.iter().cloned().map(ArrayIrType::Dimension))
            .chain(self.prefetch_types.iter().cloned().map(ArrayIrType::Array))
            .collect()
    }

    /// Returns ordinary outer array input types, excluding write-only outputs and appending scalar prefetch.
    pub fn input_types(&self) -> Vec<ArrayIrType> {
        self.parameters
            .iter()
            .filter(|parameter| parameter.access != KernelParameterAccess::WriteOnly)
            .map(|parameter| ArrayIrType::Array(parameter.r#type.clone()))
            .chain(self.prefetch_types.iter().cloned().map(ArrayIrType::Array))
            .collect()
    }

    /// Returns ordinary outer result types in parameter order.
    pub fn output_types(&self) -> Vec<ArrayIrType> {
        self.parameters
            .iter()
            .filter(|parameter| parameter.access != KernelParameterAccess::ReadOnly)
            .map(|parameter| ArrayIrType::Array(parameter.r#type.clone()))
            .collect()
    }

    /// Returns operation-local `(output, input)` alias candidates for read-write parameters. The execution backend
    /// must honor its ordinary donation and liveness rules before reusing storage; these are not external state slots.
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

    /// Derives the body access declaration from the parameter metadata without storing a second access table.
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
    /// Mapping programs and their bounds remain unchanged; the concrete grid becomes part of semantic identity.
    pub fn specialize_grid(&self, extents: &[usize]) -> Result<Self, KernelError> {
        Ok(Self { grid: self.grid.specialize(extents)?, ..self.clone() })
    }

    /// Specializes scalar-prefetched values into every mapping, checking each mapping's dimension bounds. Body
    /// constants must be bound as well; use [`KernelDefinition::specialize_prefetch`] for an attached definition.
    pub fn specialize_prefetch(&self, values: &[Array]) -> Result<Self, KernelError> {
        if values.len() != self.prefetch_types.len() {
            return Err(TypeError::invalid(format!(
                "kernel scalar prefetch expects {} values but received {}",
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
                    .into_iter()
                    .skip(self.grid.dimensions().len())
                    .zip(values)
                    .map(|(r#type, value)| {
                        let ArrayIrType::Dimension(r#type) = r#type else { unreachable!() };
                        value.to_dimension(r#type.variable().clone())
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                Ok(KernelParameter { mapping: parameter.mapping.specialize(&dimensions)?, ..parameter.clone() })
            })
            .collect::<Result<Vec<_>, KernelError>>()?;
        Ok(Self { parameters, prefetch_types: vec![], ..self.clone() })
    }

    /// Validates the exact attached body and applies the same region access policy used by ordinary program analysis.
    /// Direct definitions and eager invocation must not bypass that policy merely because their outer call has not
    /// been inserted into another program yet.
    pub(crate) fn validate_body<Extension>(
        &self,
        region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
    ) -> Result<KernelReferenceSummary, KernelError>
    where
        Extension: ReferenceViewOperation<Type = ArrayIrType, View = ArrayReferenceView>,
    {
        self.infer_output_types(&self.input_types(), &[region.interface()])?;
        let references = validate_kernel_body(region, &self.boundary_contract())?;
        for (parameter, summary) in references.parameters().iter().enumerate() {
            if let Some(summary) = summary {
                for &mode in summary.modes() {
                    if !self.allows_reference_access_through_region_input(0, mode) {
                        return Err(KernelError::DisallowedGridAccess { parameter, mode });
                    }
                }
            }
        }
        Ok(references)
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
        identities.extend(self.coordinate_types.iter().map(|coordinate| coordinate.variable().clone()));
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
            return Err(TypeError::invalid("`kernel_call` requires exactly one body region"));
        }
        if input_types != self.input_types() {
            return Err(TypeError::invalid("`kernel_call` input types do not match its logical signature"));
        }
        let expected = self.body_input_types();
        if region_interfaces[0].input_types() != expected {
            return Err(TypeError::invalid(
                "`kernel_call` body input types do not match its logical block references and coordinates",
            ));
        }
        if !region_interfaces[0].output_types().is_empty() {
            return Err(TypeError::invalid(
                "`kernel_call` body must publish results through its declared output references",
            ));
        }
        Ok(self.output_types())
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
        region_index == 0
            && !mode.is_consuming()
            && (mode != ReferenceAccessMode::Accumulate
                || self.grid.dimensions().iter().all(|dimension| dimension.execution() == GridExecution::Sequential))
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

/// Immutable body and its validated logical call boundary. Rewrites produce a new [`FlatProgram`] and must construct
/// a new definition so reference validation cannot remain attached to changed semantics.
///
/// This boundary validation does not yet grant execution: initialization, output coverage, and parallel races are
/// checked by the kernel execution verifier. The body remains an ordinary Ryft program, with no adapter dependencies.
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
    Extension: ReferenceViewOperation<Type = ArrayIrType, View = crate::arrays::ArrayReferenceView>,
{
    /// Validates the actual body and its call interface before retaining either. Reference constants, reference
    /// outputs, undeclared accesses, and incompatible block types fail before a definition becomes observable.
    pub fn new(
        operation: KernelCallOperation,
        body: FlatProgram<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
    ) -> Result<Self, KernelError> {
        let references = operation.validate_body(body.entry_region_ref())?;
        Ok(Self { operation, body, references })
    }

    /// Traces a body through ordinary Ryft staging and validates the finalized program. The closure receives a
    /// tuple of reference tracers in parameter order and dimension tracers in grid-axis order. Stores publish the
    /// declared outputs. The closure runs once during staging; runtime branches and loops use canonical staged
    /// control-flow capabilities. The canonical tuple parameter machinery flattens the finished body signature.
    ///
    /// Foreign tracers, escaped builders, discarded captures, and poisoned operations retain the existing tracing
    /// diagnostics. No partially constructed definition is returned after either tracing or validation fails.
    pub fn trace<F>(operation: KernelCallOperation, function: F) -> Result<Self, KernelError>
    where
        F: FnOnce(
            (
                Vec<Tracer<TracingContext<ArrayIrValue<Array>, KernelOperation<Extension>>>>,
                Vec<Tracer<TracingContext<ArrayIrValue<Array>, KernelOperation<Extension>>>>,
            ),
        ) -> Result<(), ProgramError>,
    {
        Self::trace_with_prefetch(operation, |(references, coordinates, _prefetched)| {
            function((references, coordinates))
        })
    }

    /// Traces references, grid coordinates, and scalar-prefetched arrays as three canonical parameter groups.
    /// Prefetched values remain private arrays in the body until explicit specialization binds their constants.
    pub fn trace_with_prefetch<F>(operation: KernelCallOperation, function: F) -> Result<Self, KernelError>
    where
        F: FnOnce(
            (
                Vec<Tracer<TracingContext<ArrayIrValue<Array>, KernelOperation<Extension>>>>,
                Vec<Tracer<TracingContext<ArrayIrValue<Array>, KernelOperation<Extension>>>>,
                Vec<Tracer<TracingContext<ArrayIrValue<Array>, KernelOperation<Extension>>>>,
            ),
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
                Ok(Vec::<Tracer<TracingContext<ArrayIrValue<Array>, KernelOperation<Extension>>>>::new())
            },
            input_types,
        )?;
        Self::new(operation, body.into_flat_program())
    }

    /// Specializes a bounded grid without retracing the body. Existing coordinate identities and reference types
    /// remain valid, and the resulting definition is checked against its specialized call. Execution still verifies
    /// initialization and complete output coverage for the selected extents before allocating memory.
    pub fn specialize_grid(&self, extents: &[usize]) -> Result<Self, KernelError> {
        Self::new(self.operation.specialize_grid(extents)?, self.body.clone())
    }

    /// Binds scalar-prefetched arrays to canonical mapping and body constants without executing body effects. The
    /// ordinary program splicer preserves nested regions and validates type identities; the resulting definition
    /// retains the array-only functional boundary with these specialized inputs removed. Exact constant values
    /// participate in [`Self::semantic_key`].
    pub fn specialize_prefetch(&self, values: &[Array]) -> Result<Self, KernelError> {
        let operation = self.operation.specialize_prefetch(values)?;
        let remaining = self.body.input_types().len() - values.len();
        let mut builder = ProgramBuilder::new();
        let mut inputs = self
            .body
            .input_types()
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
    /// This returns a new definition; any previous executable verification applies only to the original definition.
    pub fn simplified(&self) -> Result<Self, KernelError> {
        Self::new(self.operation.clone(), self.body.simplified()?)
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

    /// Returns a versioned structural key for the closed portable kernel family. Nominal dimension names and
    /// allocation addresses are replaced by occurrence-ordered identities; bounds and repeated identity relationships
    /// remain significant. Region boundaries and graph edges are framed independently of diagnostic program rendering.
    /// Typed literal arrays retain exact logical bytes in both atom and operation payloads, including mapping programs;
    /// floating metadata retains its IEEE bits. Instruction provenance, adapter options, schedules, and invocation
    /// values do not belong to this key.
    ///
    /// This is a cache-identity encoding, not a round-trip executable serialization format. The closed payload schema
    /// uses its audited structural fields, with explicit encodings for lossy value formats. Arbitrary extensions and
    /// unsupported opaque descriptors return an error instead of inheriting identity from diagnostic formatting. A
    /// registered extension provides its exact versioned payload through [`KernelExtension::semantic_key`].
    pub fn semantic_key(&self) -> Result<String, TypeError>
    where
        Extension: KernelExtension,
    {
        let mut identities = self.operation.identity_occurrences();
        for region in self.body.regions().iter() {
            for instruction in region.instructions() {
                if let KernelOperation::Call(operation) = instruction.operation() {
                    identities.extend(operation.identity_occurrences());
                }
            }
        }
        for region in self.body.regions() {
            for atom in region.atoms() {
                identities.extend(atom.r#type().identities().map(|(_, identity)| identity.clone()));
            }
        }
        let mut renaming = TypeIdentityRenaming::new();
        for identity in identities {
            if !renaming.replacements().iter().any(|(source, _)| source == &identity) {
                let target =
                    DimensionVariable::new(format!("dimension_{}", renaming.replacements().len()), identity.bounds());
                renaming.insert(identity, target)?;
            }
        }
        let operation = self.operation.rename_type_identities(&renaming)?;
        let body = self.body.rename_type_identities(&renaming).map_err(TypeError::custom)?;
        let bounds = renaming.replacements().iter().map(|(_, target)| target.bounds()).collect::<Vec<_>>();
        let mut key = format!("kernel key 2; schema {KERNEL_SCHEMA_VERSION}\n");
        Self::semantic_field(&mut key, &format!("{bounds:?}"));
        Self::call_semantic_fields(&mut key, &operation)?;
        Self::semantic_field(&mut key, &format!("body entry {}", body.entry().index()));
        for region in body.regions().iter() {
            Self::region_semantic_fields(&mut key, region.atoms(), region.input_ids(), region.output_ids())?;
            for instruction in region.instructions() {
                Self::semantic_field(
                    &mut key,
                    &format!("{:?}", (instruction.inputs(), instruction.outputs(), instruction.regions())),
                );
                match instruction.operation() {
                    KernelOperation::Portable(operation) => Self::portable_semantic_fields(&mut key, operation)?,
                    KernelOperation::Call(operation) => Self::call_semantic_fields(&mut key, operation)?,
                    KernelOperation::Scratch(operation) => {
                        Self::semantic_field(&mut key, &format!("Scratch({operation:?})"))
                    }
                    KernelOperation::TileLoad(operation) => {
                        Self::semantic_field(&mut key, &format!("TileLoad({operation:?})"))
                    }
                    KernelOperation::AsyncCopy(operation) => {
                        Self::semantic_field(&mut key, &format!("AsyncCopy({operation:?})"))
                    }
                    KernelOperation::Wait(operation) => Self::semantic_field(&mut key, &format!("Wait({operation:?})")),
                    KernelOperation::MaskedLoad(operation) => {
                        Self::semantic_field(&mut key, &format!("MaskedLoad({operation:?})"))
                    }
                    KernelOperation::MaskedStore(operation) => {
                        Self::semantic_field(&mut key, &format!("MaskedStore({operation:?})"))
                    }
                    KernelOperation::MaskedSwap(operation) => {
                        Self::semantic_field(&mut key, &format!("MaskedSwap({operation:?})"))
                    }
                    KernelOperation::Extension(operation) => {
                        let payload = operation.semantic_key()?;
                        let mut encoded = String::with_capacity(payload.len().saturating_mul(2));
                        for byte in payload {
                            write!(encoded, "{byte:02x}").unwrap();
                        }
                        Self::semantic_field(&mut key, "extension");
                        Self::semantic_field(&mut key, &encoded);
                    }
                }
            }
            Self::semantic_field(&mut key, "end region");
        }
        Ok(key)
    }

    /// Appends a byte-length-framed UTF-8 field so punctuation and adjacent payloads cannot alias one another.
    fn semantic_field(key: &mut String, value: &str) {
        write!(key, "{}:{value}", value.len()).unwrap();
    }

    /// Encodes the typed atom table and public edges of one region, excluding diagnostic provenance.
    fn region_semantic_fields(
        key: &mut String,
        atoms: &[Atom<ArrayIrValue<Array>>],
        inputs: &[crate::programs::AtomId],
        outputs: &[crate::programs::AtomId],
    ) -> Result<(), TypeError> {
        Self::semantic_field(key, &format!("region {:?}", (inputs, outputs)));
        for atom in atoms {
            Self::semantic_field(key, &format!("type {:?}", atom.r#type()));
            match atom {
                Atom::Variable(_) => Self::semantic_field(key, "variable"),
                Atom::Constant(ArrayIrValue::Array(value)) => {
                    Self::semantic_field(key, &format!("array bytes {:02x?}", value.logical_bytes()));
                }
                Atom::Constant(ArrayIrValue::Dimension(value)) => {
                    Self::semantic_field(key, &format!("dimension {value:?}"));
                }
                Atom::Constant(ArrayIrValue::Reference(_)) => {
                    return Err(TypeError::invalid("reference constants have no persistent kernel semantic identity"));
                }
            }
        }
        Self::semantic_field(key, "end atoms");
        Ok(())
    }

    /// Encodes a call boundary and every mapping program through canonical typed fields and actual region graphs.
    fn call_semantic_fields(key: &mut String, operation: &KernelCallOperation) -> Result<(), TypeError> {
        Self::semantic_field(key, &format!("call {:?}", (operation.grid(), operation.prefetch_types())));
        for parameter in operation.parameters() {
            let mapping = parameter.mapping();
            Self::semantic_field(
                key,
                &format!(
                    "parameter {:?}",
                    (
                        parameter.r#type(),
                        parameter.access(),
                        parameter.body_type(),
                        mapping.block_shape(),
                        mapping.boundary_policy(),
                    )
                ),
            );
            Self::semantic_field(key, &format!("mapping entry {}", mapping.program().entry().index()));
            for region in mapping.program().regions().iter() {
                Self::region_semantic_fields(key, region.atoms(), region.input_ids(), region.output_ids())?;
                for instruction in region.instructions() {
                    Self::semantic_field(
                        key,
                        &format!("{:?}", (instruction.inputs(), instruction.outputs(), instruction.regions())),
                    );
                    Self::portable_semantic_fields(key, instruction.operation())?;
                }
                Self::semantic_field(key, "end region");
            }
            Self::semantic_field(key, "end mapping");
        }
        Self::semantic_field(key, "end call");
        Ok(())
    }

    /// Encodes exact custom-call attribute kinds and bits rather than their untyped display forms.
    fn custom_call_semantic_fields(key: &mut String, operation: &CustomCallOperation) {
        Self::semantic_field(key, operation.target_name());
        for (name, value) in operation.attributes() {
            Self::semantic_field(key, name);
            match value {
                CustomCallAttribute::String(value) => {
                    Self::semantic_field(key, "string");
                    Self::semantic_field(key, value);
                }
                CustomCallAttribute::Bytes(value) => Self::semantic_field(key, &format!("bytes {value:02x?}")),
                CustomCallAttribute::Boolean(value) => Self::semantic_field(key, &format!("boolean {value}")),
                CustomCallAttribute::I64(value) => Self::semantic_field(key, &format!("i64 {value}")),
                CustomCallAttribute::F64(value) => Self::semantic_field(key, &format!("f64 {:016x}", value.to_bits())),
            }
        }
    }

    /// Encodes all floating attention configuration using IEEE bits; optional/default choices remain distinct.
    fn attention_semantic_fields(key: &mut String, configuration: AttentionConfiguration) {
        Self::semantic_field(
            key,
            &format!(
                "attention {:?}",
                (
                    configuration.scale().map(f64::to_bits),
                    configuration.causal(),
                    configuration.local_window(),
                    configuration.implementation(),
                    configuration.return_residual(),
                    configuration.dropout().map(|(rate, seed)| (rate.to_bits(), seed)),
                )
            ),
        );
    }

    /// Encodes the closed canonical portable payload family, with explicit eligibility for lossy/opaque forms.
    fn portable_semantic_fields(key: &mut String, operation: &ArrayIrOperation<Array>) -> Result<(), TypeError> {
        // Unlike the public Operation render contract, this closed structural representation includes typed variant
        // tags and stored fields. Literal bytes and floating payloads below are explicit because Debug is lossy there.
        // No adapter extension or user-supplied formatter can enter this branch.
        Self::semantic_field(key, &format!("portable {operation:?}"));
        match operation {
            ArrayIrOperation::Array(operation) => match operation {
                ArrayOperation::Constant(operation) => {
                    Self::semantic_field(key, &format!("constant bytes {:02x?}", operation.value().logical_bytes()))
                }
                ArrayOperation::CustomCall(operation) => Self::custom_call_semantic_fields(key, operation),
                ArrayOperation::DotProductAttention(operation) => {
                    Self::attention_semantic_fields(key, operation.configuration())
                }
                ArrayOperation::DotProductAttentionBackward(operation) => {
                    Self::attention_semantic_fields(key, operation.configuration())
                }
                ArrayOperation::LinearCall(operation) if operation.is_transpose_only() => {
                    return Err(TypeError::invalid(
                        "transpose-only linear calls have no kernel semantic descriptor encoding",
                    ));
                }
                ArrayOperation::Zero(_)
                | ArrayOperation::ZeroLike(_)
                | ArrayOperation::One(_)
                | ArrayOperation::OneLike(_)
                | ArrayOperation::Iota(_)
                | ArrayOperation::Min(_)
                | ArrayOperation::Max(_)
                | ArrayOperation::Neg(_)
                | ArrayOperation::Add(_)
                | ArrayOperation::Sub(_)
                | ArrayOperation::Mul(_)
                | ArrayOperation::Div(_)
                | ArrayOperation::Abs(_)
                | ArrayOperation::Sign(_)
                | ArrayOperation::Rem(_)
                | ArrayOperation::Pow(_)
                | ArrayOperation::Sqrt(_)
                | ArrayOperation::Rsqrt(_)
                | ArrayOperation::Sin(_)
                | ArrayOperation::Cos(_)
                | ArrayOperation::Tanh(_)
                | ArrayOperation::Atan2(_)
                | ArrayOperation::Exp(_)
                | ArrayOperation::Log(_)
                | ArrayOperation::Logistic(_)
                | ArrayOperation::Floor(_)
                | ArrayOperation::Ceil(_)
                | ArrayOperation::Round(_)
                | ArrayOperation::Log1p(_)
                | ArrayOperation::LogAddExp(_)
                | ArrayOperation::Erf(_)
                | ArrayOperation::Not(_)
                | ArrayOperation::And(_)
                | ArrayOperation::Or(_)
                | ArrayOperation::Xor(_)
                | ArrayOperation::Complex(_)
                | ArrayOperation::Conjugate(_)
                | ArrayOperation::Real(_)
                | ArrayOperation::Imaginary(_)
                | ArrayOperation::Dot(_)
                | ArrayOperation::RaggedDot(_)
                | ArrayOperation::ScaledDot(_)
                | ArrayOperation::Reduce(_)
                | ArrayOperation::LogSumExp(_)
                | ArrayOperation::CumulativeSum(_)
                | ArrayOperation::CumulativeProduct(_)
                | ArrayOperation::CumulativeMax(_)
                | ArrayOperation::CumulativeMin(_)
                | ArrayOperation::CumulativeLogSumExp(_)
                | ArrayOperation::Sort(_)
                | ArrayOperation::RngBitGenerator(_)
                | ArrayOperation::ParallelReduce(_)
                | ArrayOperation::AllGather(_)
                | ArrayOperation::ParallelSumScatter(_)
                | ArrayOperation::ParallelPermute(_)
                | ArrayOperation::AllToAll(_)
                | ArrayOperation::RaggedAllToAll(_)
                | ArrayOperation::AxisIndex(_)
                | ArrayOperation::Reverse(_)
                | ArrayOperation::Transpose(_)
                | ArrayOperation::Reshape(_)
                | ArrayOperation::Broadcast(_)
                | ArrayOperation::Pad(_)
                | ArrayOperation::Concatenate(_)
                | ArrayOperation::Gather(_)
                | ArrayOperation::Scatter(_)
                | ArrayOperation::Slice(_)
                | ArrayOperation::UpdateSlice(_)
                | ArrayOperation::DynamicSlice(_)
                | ArrayOperation::DynamicUpdateSlice(_)
                | ArrayOperation::Compare(_)
                | ArrayOperation::Select(_)
                | ArrayOperation::Condition(_)
                | ArrayOperation::While(_)
                | ArrayOperation::Scan(_)
                | ArrayOperation::ConvertElementType(_)
                | ArrayOperation::TransferToMemory(_)
                | ArrayOperation::Reshard(_)
                | ArrayOperation::ShardingConstraint(_)
                | ArrayOperation::StopGradient(_)
                | ArrayOperation::Tag(_)
                | ArrayOperation::Rematerialize(_)
                | ArrayOperation::Print(_)
                | ArrayOperation::CustomJvp(_)
                | ArrayOperation::CustomVjp(_)
                | ArrayOperation::LinearCall(_) => {}
            },
            ArrayIrOperation::CustomCall(operation) => Self::custom_call_semantic_fields(key, operation),
            ArrayIrOperation::LinearCall(operation) if operation.is_transpose_only() => {
                return Err(TypeError::invalid(
                    "transpose-only linear calls have no kernel semantic descriptor encoding",
                ));
            }
            ArrayIrOperation::Zero(_)
            | ArrayIrOperation::One(_)
            | ArrayIrOperation::Iota(_)
            | ArrayIrOperation::Dimension(_)
            | ArrayIrOperation::Compare(_)
            | ArrayIrOperation::DimensionSize(_)
            | ArrayIrOperation::ReferenceNew(_)
            | ArrayIrOperation::ReferenceRead(_)
            | ArrayIrOperation::ReferenceWrite(_)
            | ArrayIrOperation::ReferenceIndex(_)
            | ArrayIrOperation::ReferenceDynamicIndex(_)
            | ArrayIrOperation::ReferenceSlice(_)
            | ArrayIrOperation::ReferenceSwap(_)
            | ArrayIrOperation::ReferenceAddUpdate(_)
            | ArrayIrOperation::ReferenceAtomicAddUpdate(_)
            | ArrayIrOperation::ReferenceFreeze(_)
            | ArrayIrOperation::DimensionFromScalar(_)
            | ArrayIrOperation::DimensionToScalar(_)
            | ArrayIrOperation::Reshape(_)
            | ArrayIrOperation::Broadcast(_)
            | ArrayIrOperation::Concatenate(_)
            | ArrayIrOperation::Pad(_)
            | ArrayIrOperation::DynamicShapeSlice(_)
            | ArrayIrOperation::RngBitGenerator(_)
            | ArrayIrOperation::AllGather(_)
            | ArrayIrOperation::ParallelSumScatter(_)
            | ArrayIrOperation::AllToAll(_)
            | ArrayIrOperation::RaggedAllToAll(_)
            | ArrayIrOperation::Condition(_)
            | ArrayIrOperation::While(_)
            | ArrayIrOperation::Scan(_)
            | ArrayIrOperation::CustomJvp(_)
            | ArrayIrOperation::CustomVjp(_)
            | ArrayIrOperation::LinearCall(_)
            | ArrayIrOperation::Rematerialize(_) => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayOperation, DataType, DimensionError, DimensionType, DimensionValue, Layout, Memory,
        Shape, TiledLayout,
    };
    use crate::contexts::Context;
    use crate::kernels::grids::{GridDimension, GridExecution};
    use crate::kernels::mappings::BoundaryPolicy;
    use crate::operations::{
        ReferenceAddUpdateOperation, ReferenceAtomicAddUpdateOperation, ReferenceRead, ReferenceReadOperation,
        ReferenceWrite, ReferenceWriteOperation, ZeroOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{EffectClasses, ProgramBuilder};

    use super::*;

    /// Constant rank-zero mapping used by the scalar call boundary tests.
    fn scalar_mapping() -> BlockMapping {
        let program: FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>> =
            ProgramBuilder::new().build(vec![], vec![], vec![]).unwrap();
        BlockMapping::new(program, vec![], BoundaryPolicy::InBounds).unwrap()
    }

    /// Scalar parameter with a singleton whole-value block.
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

    /// Empty scalar body with independently named mapping and grid identities for semantic-key tests.
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
            mapping.add_input(ArrayIrType::Dimension(DimensionType::new(DimensionVariable::new(
                name,
                DimensionBounds::unbounded(),
            ))));
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

    /// Reads an offset-prefetched array window and publishes the prefetched scalar through a singleton output.
    fn prefetch_definition() -> KernelDefinition {
        let coordinate =
            DimensionType::new(DimensionVariable::new("coordinate", DimensionBounds::non_negative(Some(1)).unwrap()));
        let offset =
            DimensionType::new(DimensionVariable::new("offset", DimensionBounds::non_negative(Some(8)).unwrap()));
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
    fn test_kernel_parameter_new() {
        let parameter = parameter(KernelParameterAccess::ReadOnly);
        assert_eq!(parameter.r#type().as_ref(), &ArrayType::scalar(DataType::I32));
        assert!(matches!(
            KernelParameter::new(
                ArrayType::new_static(DataType::I32, [2]),
                KernelParameterAccess::ReadOnly,
                scalar_mapping()
            ),
            Err(KernelError::ParameterRank { array_rank: 1, block_rank: 0 }),
        ));
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
        // Full, partial, empty, and oversized operand windows all have the same fixed logical block type.
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
                    .into()
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
                .into()
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
            ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::I32)),)
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
        let view = ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(0, 1, 1), ArraySliceAxis::new(0, 3, 1)] };
        assert_eq!(
            partial.body_type(),
            ArrayIrType::Reference(ReferenceType::new(view.output_type(&full_type).unwrap()))
        );
    }

    #[test]
    fn test_kernel_parameter_type() {
        assert_eq!(parameter(KernelParameterAccess::ReadOnly).r#type().into_owned(), ArrayType::scalar(DataType::I32));
    }

    #[test]
    fn test_kernel_call_operation_new() {
        let operation = copy_definition().operation().clone();
        assert_eq!(operation.name(), "kernel_call");
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("body")]);
        assert_eq!(operation.region_capture_input_count(0), Some(0));
        let grid = Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap();
        assert!(matches!(
            KernelCallOperation::new(grid, vec![parameter(KernelParameterAccess::ReadOnly)]),
            Err(KernelError::MappingInputCount { parameter: 0, expected: 1, actual: 0 }),
        ));
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
            ReferenceAccessMode::AtomicAccumulate,
        ] {
            assert!(parallel.allows_reference_access_through_region_input(0, mode));
            assert!(sequential.allows_reference_access_through_region_input(0, mode));
            assert!(!parallel.allows_reference_access_through_region_input(1, mode));
        }
        assert!(!parallel.allows_reference_access_through_region_input(0, ReferenceAccessMode::Accumulate));
        assert!(sequential.allows_reference_access_through_region_input(0, ReferenceAccessMode::Accumulate));
        assert!(!sequential.allows_reference_access_through_region_input(0, ReferenceAccessMode::Consume));
    }

    #[test]
    fn test_kernel_call_operation_new_coordinate_bounds() {
        for (extent, bounds, accepted) in [
            (2, DimensionBounds::non_negative(Some(2)).unwrap(), true),
            (2, DimensionBounds::non_negative(Some(1)).unwrap(), false),
            (2, DimensionBounds::positive(Some(3)).unwrap(), false),
            (0, DimensionBounds::positive(Some(3)).unwrap(), true),
        ] {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            builder.add_input(ArrayIrType::Dimension(DimensionType::new(DimensionVariable::new("coordinate", bounds))));
            let mapping = BlockMapping::new(
                builder.build(vec![], vec![Placeholder], vec![]).unwrap(),
                vec![],
                BoundaryPolicy::InBounds,
            )
            .unwrap();
            let grid = Grid::new(vec![GridDimension::new(Dimension::Static(extent), GridExecution::Parallel)]).unwrap();
            let result = KernelCallOperation::new(
                grid,
                vec![
                    KernelParameter::new(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadOnly, mapping)
                        .unwrap(),
                ],
            );
            if accepted {
                assert!(result.is_ok());
            } else {
                let Err(KernelError::MappingInputBounds { parameter, axis, expected, actual }) = result else {
                    panic!("expected rejected coordinate bounds");
                };
                assert_eq!((parameter, axis), (0, 0));
                assert_eq!(expected, DimensionBounds::non_negative(Some(extent)).unwrap());
                assert_eq!(actual, bounds);
            }
        }
    }

    #[test]
    fn test_kernel_call_operation_new_rejects_coordinate_identity_aliasing() {
        for (extents, accepted) in [([2, 2], false), ([1, 2], false), ([1, 1], true), ([0, 2], true)] {
            let coordinate =
                DimensionType::new(DimensionVariable::new("coordinate", DimensionBounds::non_negative(None).unwrap()));
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            builder.add_input(ArrayIrType::Dimension(coordinate.clone()));
            builder.add_input(ArrayIrType::Dimension(coordinate));
            let mapping = BlockMapping::new(
                builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap(),
                vec![],
                BoundaryPolicy::InBounds,
            )
            .unwrap();
            let result = KernelCallOperation::new(
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
            );
            if accepted {
                assert!(result.is_ok());
            } else {
                let error = result.unwrap_err();
                assert!(matches!(
                    error,
                    KernelError::MappingInputIdentity { parameter: 0, first_axis: 0, second_axis: 1 }
                ));
                assert_eq!(
                    error.to_string(),
                    "kernel parameter 0 mapping axes 0 and 1 share a dimension identity but may differ"
                );
            }
        }
    }

    #[test]
    fn test_kernel_call_operation_new_with_prefetch() {
        let definition = prefetch_definition();
        assert_eq!(
            definition.operation().input_types(),
            vec![ArrayType::new_static(DataType::I32, vec![8]).into(), ArrayType::scalar(DataType::I32).into(),]
        );
        assert_eq!(definition.operation().body_input_types().len(), 4);
        assert_eq!(definition.operation().body_input_types()[3], ArrayType::scalar(DataType::I32).into());
        let invalid = ArrayType::new_static(DataType::I32, vec![1]);
        assert_eq!(
            KernelCallOperation::new_with_prefetch(Grid::new(vec![]).unwrap(), vec![], vec![invalid.clone()])
                .unwrap_err(),
            KernelError::Type(TypeError::invalid(format!(
                "`dimension_from_scalar` input must be a rank-0 integer array but has type {invalid}",
            ))),
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
            vec![KernelParameterAccess::ReadOnly, KernelParameterAccess::WriteOnly]
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
            vec![DimensionBounds::non_negative(Some(4)).unwrap(), DimensionBounds::non_negative(Some(1)).unwrap()]
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
        assert_eq!(operation.body_input_types(), definition.body().input_types());
        assert_eq!(operation.body_input_types()[0], operation.parameters()[0].body_type());
        assert_eq!(
            &operation.body_input_types()[1..],
            operation.coordinate_types().iter().cloned().map(ArrayIrType::Dimension).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_kernel_call_operation_input_types() {
        assert_eq!(
            copy_definition().operation().input_types(),
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::I32))]
        );
    }

    #[test]
    fn test_kernel_call_operation_output_types() {
        assert_eq!(
            copy_definition().operation().output_types(),
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::I32))]
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
            ])
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
            KernelError::Grid(GridError::Dimension(crate::arrays::DimensionError::InputDimensionMismatch {
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
            &[2]
        );
    }

    #[test]
    fn test_kernel_call_operation_specialize_prefetch_checks_values() {
        let definition = prefetch_definition();
        let operation = definition.operation();
        assert_eq!(
            operation.specialize_prefetch(&[]).unwrap_err(),
            KernelError::Type(TypeError::invalid("kernel scalar prefetch expects 1 values but received 0",))
        );
        assert_eq!(
            operation.specialize_prefetch(&[Array::scalar(1i64).unwrap()]).unwrap_err(),
            KernelError::Type(TypeError::invalid(
                "kernel scalar prefetch 0 expected type `i32[]` but received `i64[]`",
            ))
        );
        let ArrayIrType::Dimension(offset) = &operation.parameters()[0].mapping().program().input_types()[1] else {
            unreachable!()
        };
        for value in [-1i32, 8] {
            let array = Array::scalar(value).unwrap();
            let expected = array.to_dimension(offset.variable().clone()).unwrap_err();
            assert_eq!(operation.specialize_prefetch(&[array]).unwrap_err(), KernelError::Program(expected));
        }
        assert_eq!(
            operation.specialize_prefetch(&[Array::scalar(0i32).unwrap()]).unwrap().prefetch_types(),
            &[] as &[ArrayType]
        );
    }

    #[test]
    fn test_kernel_call_operation_type_inference() {
        let definition = copy_definition();
        let operation = definition.operation();
        assert_eq!(
            operation.infer_output_types(&operation.input_types(), &[definition.body().interface()]),
            Ok(operation.output_types())
        );
        assert_eq!(
            operation.infer_output_types(&operation.input_types(), &[]),
            Err(TypeError::invalid("`kernel_call` requires exactly one body region"))
        );
        assert_eq!(
            operation.infer_output_types(&[], &[definition.body().interface()]),
            Err(TypeError::invalid("`kernel_call` input types do not match its logical signature"))
        );
        let interface = RegionInterface::new(vec![], vec![], EffectClasses::NONE);
        assert_eq!(
            operation.infer_output_types(&operation.input_types(), &[interface]),
            Err(TypeError::invalid(
                "`kernel_call` body input types do not match its logical block references and coordinates"
            ))
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
        let reference = builder.add_input(parameter(KernelParameterAccess::WriteOnly).body_type());
        builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![reference], None).unwrap();
        let body = builder.build(vec![], vec![Placeholder], vec![]).unwrap();
        let operation =
            KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![parameter(KernelParameterAccess::WriteOnly)])
                .unwrap();
        assert!(matches!(
            KernelDefinition::new(operation, body),
            Err(KernelError::Validation(KernelValidationError::DisallowedAccess {
                input_index: 0,
                access: KernelParameterAccess::WriteOnly,
                mode: ReferenceAccessMode::Read,
                operation: "reference_read",
                ..
            },))
        ));
    }

    #[test]
    fn test_kernel_call_operation_validate_body_grid_access() {
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        mapping.add_input(ArrayIrType::Dimension(DimensionType::new(DimensionVariable::new(
            "coordinate",
            DimensionBounds::non_negative(None).unwrap(),
        ))));
        let mapping = BlockMapping::new(
            mapping.build(vec![], vec![Placeholder], vec![]).unwrap(),
            vec![],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let parameter =
            KernelParameter::new(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadWrite, mapping).unwrap();
        let parallel = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap(),
            vec![parameter.clone()],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(parameter.body_type());
        builder.add_input(parallel.coordinate_types()[0].clone().into());
        let update = builder.add_constant(ArrayIrValue::Array(Array::scalar(1i32).unwrap()));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), vec![], vec![reference, update], None)
            .unwrap();
        let body: FlatProgram<EagerContext<ArrayIrValue<Array>, KernelOperation>> =
            builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap();
        let error = parallel.validate_body(body.entry_region_ref()).unwrap_err();
        assert_eq!(error, KernelError::DisallowedGridAccess { parameter: 0, mode: ReferenceAccessMode::Accumulate });
        assert_eq!(error.to_string(), "kernel grid does not permit `accumulate` access on parameter 0");
        let atomic_body = body
            .map_operations(|operation| {
                Ok(match operation {
                    KernelOperation::Portable(ArrayIrOperation::ReferenceAddUpdate(_)) => {
                        KernelOperation::from(ReferenceAtomicAddUpdateOperation::new())
                    }
                    operation => operation.clone(),
                })
            })
            .unwrap();
        assert_eq!(
            parallel.validate_body(atomic_body.entry_region_ref()).unwrap().parameter(0).unwrap().modes(),
            &std::collections::BTreeSet::from([ReferenceAccessMode::AtomicAccumulate]),
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
                if message == "traceable helper rejected its arguments"
        ));
    }

    #[test]
    fn test_kernel_definition_trace_with_prefetch() {
        let definition = prefetch_definition();
        assert_eq!(definition.body().input_types(), definition.operation().body_input_types());
        assert_eq!(definition.body().entry_region_ref().instructions().len(), 1);
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
            Ok(vec![Array::scalar(2i32).unwrap()])
        );
        assert_ne!(
            specialized.semantic_key().unwrap(),
            definition.specialize_prefetch(&[Array::scalar(3i32).unwrap()]).unwrap().semantic_key().unwrap()
        );
        assert_eq!(
            specialized.semantic_key().unwrap(),
            prefetch_definition()
                .specialize_prefetch(&[Array::scalar(2i32).unwrap()])
                .unwrap()
                .semantic_key()
                .unwrap()
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
        assert_eq!(definition.body().instructions().len(), 4);
        assert_eq!(
            simplified
                .body()
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["reference_write", "reference_read", "reference_write"],
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
        assert_eq!(copy_definition().body().entry_region_ref().instructions().len(), 2);
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
            vec![KernelParameterAccess::ReadOnly, KernelParameterAccess::WriteOnly]
        );
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
        // Diagnostic floating-point rendering deliberately omits NaN payloads; semantic identity must retain them.
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
                    ArrayOperation::Constant(crate::operations::ConstantOperation::new(
                        Array::scalar(f32::from_bits(bits)).unwrap(),
                    )),
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
    fn test_kernel_definition_semantic_key_preserves_custom_attribute_kinds_and_bits() {
        for attributes in [
            [CustomCallAttribute::String("true".to_owned()), CustomCallAttribute::Boolean(true)],
            [CustomCallAttribute::String("1".to_owned()), CustomCallAttribute::I64(1)],
            [CustomCallAttribute::Bytes(vec![0, 255]), CustomCallAttribute::String("bytes [00, ff]".to_owned())],
            [
                CustomCallAttribute::F64(f64::from_bits(0x7ff8_0000_0000_0001)),
                CustomCallAttribute::F64(f64::from_bits(0x7ff8_0000_0000_0002)),
            ],
        ] {
            let definitions = attributes.map(|attribute| {
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
            });
            assert_eq!(definitions[0].body().to_string(), definitions[1].body().to_string());
            assert_ne!(definitions[0].semantic_key().unwrap(), definitions[1].semantic_key().unwrap());
        }
    }

    #[test]
    fn test_kernel_definition_semantic_key_preserves_attention_bits() {
        for configurations in [
            [
                AttentionConfiguration::new().with_scale(f64::from_bits(0x7ff8_0000_0000_0001)),
                AttentionConfiguration::new().with_scale(f64::from_bits(0x7ff8_0000_0000_0002)),
            ],
            [
                AttentionConfiguration::new().with_dropout((0.0, 1)),
                AttentionConfiguration::new().with_dropout((-0.0, 1)),
            ],
        ] {
            let keys = configurations.map(|configuration| {
                let mut key = String::new();
                KernelDefinition::<NoKernelExtension>::attention_semantic_fields(&mut key, configuration);
                key
            });
            assert_ne!(keys[0], keys[1]);
        }
    }

    /// Extension whose diagnostic text intentionally omits semantic payload, demonstrating conservative eligibility.
    #[derive(Clone, Debug)]
    struct HiddenKeyExtension<const EXACT: bool = false>(u32);

    impl<const EXACT: bool> Operation for HiddenKeyExtension<EXACT> {
        type Type = ArrayIrType;
        fn name(&self) -> &'static str {
            "hidden_key_extension"
        }
        fn infer_output_types(
            &self,
            _inputs: &[ArrayIrType],
            _regions: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            Ok(vec![])
        }
    }

    impl KernelExtension for HiddenKeyExtension {}

    impl KernelExtension for HiddenKeyExtension<true> {
        fn semantic_key(&self) -> Result<Vec<u8>, TypeError> {
            let mut key = b"test.hidden_key_extension\0v1\0".to_vec();
            key.extend_from_slice(&self.0.to_le_bytes());
            Ok(key)
        }
    }

    impl<const EXACT: bool> ReferenceViewOperation for HiddenKeyExtension<EXACT> {
        type View = ArrayReferenceView;
        fn reference_view(&self, _output_index: usize) -> Option<ArrayReferenceView> {
            None
        }
        fn validate_reference_view(
            view: &ArrayReferenceView,
            source: &ArrayIrType,
            target: &ArrayIrType,
        ) -> Result<(), crate::programs::ReferenceViewValidationError> {
            crate::arrays::validate_array_reference_view(view, source, target)
        }
        fn reapply_reference_view<C: Context<Type = ArrayIrType, Operation = Self>>(
            _context: &C,
            _view: &ArrayReferenceView,
            _source: C::Value,
            _symbols: &[C::Value],
        ) -> Result<C::Value, ProgramError> {
            Err(ProgramError::UnsupportedOperation { message: "hidden extension has no reference views".to_owned() })
        }
    }

    #[test]
    fn test_kernel_definition_semantic_key_exact_extension_payload() {
        let keys = [0, 1, 256, u32::MAX].map(|payload| {
            let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<HiddenKeyExtension<true>>>::new();
            body.add_instruction(KernelOperation::Extension(HiddenKeyExtension(payload)), vec![], vec![], None)
                .unwrap();
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
    fn test_kernel_definition_semantic_key_rejects_extension_without_exact_identity() {
        for payload in [1, 2] {
            let extension = HiddenKeyExtension(payload);
            assert_eq!(extension.0, payload);
            let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<HiddenKeyExtension>>::new();
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
                ))
            );
        }
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
        assert!(key.starts_with(&format!("kernel key 2; schema {KERNEL_SCHEMA_VERSION}\n")));
    }
}
