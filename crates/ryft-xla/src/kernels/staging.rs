//! XLA staging carriers for portable and enabled adapter kernels with ordinary attached computation regions.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use ryft_core::{
    Array as CoreArray, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayReferenceTransform, Atom,
    AtomId, BatchableOperation, BatchingContext, BatchingDriver, BatchingError, CompositeJvpRule, CompositeOperation,
    ConstantOperation, Context, CotangentAccumulator, CustomCallOperation, CustomRuleSource, DifferentiableOperation,
    DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError, DifferentiationPolicy,
    Domain as CoreDomain, Effects, GuardOperation, InputRegionProvenance, Instruction, InterpretableOperation,
    InterpretationDriver, LogicalMesh, MaybeZero, MeshAxisType, MeshContext, Operation as CoreOperation,
    OperationBoundaryPruning, OperationFoldOutput, OperationProvider, OutputRegionProvenance, PartialEvaluationContext,
    PartialEvaluationDriver, PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation, Placeholder,
    Program as CoreProgram, ProgramError, ReferenceAccessDescriptor, ReferenceAccessMode, ReferenceAccessOperation,
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation, Region, RegionDataFlow, RegionInterface, RegionLiveness, RegionSlot,
    ShardMapBoundary, ShardMapOperation, Sharding, ShardingDimension, Tracer as CoreTracer, TracingContext,
    TransposableOperation, TranspositionContext, TranspositionDriver, Type, TypeError, TypeIdentityRenaming, Typed,
    TypedProgram, Value as CoreValue,
};
use ryft_kernels::{
    KernelCallOperation, KernelCompiler, KernelCompletionProtocol, KernelDefinition,
    KernelExtension as PortableKernelExtension, KernelExtensionMemory, KernelOperation as PortableKernelOperation,
    KernelParameterAccess, KernelSchedule, NoKernelExtension, VerifiedKernel,
};
use ryft_mlir::ValueRef;

use crate::kernels::{
    CompiledKernel, KernelEmbeddingError, KernelOutputEmbedding, placed_custom_call, unbound_kernel_axis_message,
    validate_kernel_sharding,
};
use crate::{
    Constant, Error, FlatProgram, LowerableOperation, LowerableOperationFamily, Lowerer, LoweringError, Operation,
    Target,
};

// TODO(eaplatanios): Review this module.

/// Adapter operation families enabled in this XLA integration.
///
/// Portable operations remain ordinary [`PortableKernelOperation`](ryft_kernels::KernelOperation) variants. This enum
/// owns only the typed conversion between enabled adapter families and XLA staging. It does not choose a compiler or
/// erase extension semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelExtension {
    /// Exact Mosaic GPU instruction semantics, admitted only by an explicitly selected Mosaic compiler.
    #[cfg(feature = "mosaic-gpu")]
    Mosaic(ryft_mosaic::kernels::gpu::GpuOperation),
}

impl From<NoKernelExtension> for KernelExtension {
    fn from(extension: NoKernelExtension) -> Self {
        match extension {}
    }
}

impl TryFrom<KernelExtension> for NoKernelExtension {
    type Error = ProgramError;

    fn try_from(extension: KernelExtension) -> Result<Self, Self::Error> {
        match extension {
            #[cfg(feature = "mosaic-gpu")]
            KernelExtension::Mosaic(_) => Err(ProgramError::UnsupportedOperation {
                message: "portable kernel compiler binding cannot consume a Mosaic GPU extension".to_owned(),
            }),
        }
    }
}

#[cfg(feature = "mosaic-gpu")]
impl From<ryft_mosaic::kernels::gpu::GpuOperation> for KernelExtension {
    fn from(extension: ryft_mosaic::kernels::gpu::GpuOperation) -> Self {
        Self::Mosaic(extension)
    }
}

#[cfg(feature = "mosaic-gpu")]
impl TryFrom<KernelExtension> for ryft_mosaic::kernels::gpu::GpuOperation {
    type Error = ProgramError;

    fn try_from(extension: KernelExtension) -> Result<Self, Self::Error> {
        match extension {
            KernelExtension::Mosaic(extension) => Ok(extension),
        }
    }
}

impl Display for KernelExtension {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl ReferenceAccessOperation for KernelExtension {
    type Transform = ArrayReferenceTransform;

    fn base_input_count(&self) -> usize {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.base_input_count(),
        }
    }

    fn reference_access_descriptor(
        &self,
        input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, Self::Transform>> {
        let _ = input_index;
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.reference_access_descriptor(input_index),
        }
    }

    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<Self::Transform>,
    ) -> Result<Self, ProgramError> {
        let _ = (input_index, &transforms);
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::Mosaic)
            }
        }
    }
}

impl PortableKernelExtension for KernelExtension {
    fn semantic_key(
        &self,
        identity: &mut dyn FnMut(&ryft_core::DimensionVariable) -> usize,
    ) -> Result<Vec<u8>, TypeError> {
        let _ = &identity;
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.semantic_key(identity),
        }
    }

    fn memory_semantics(&self) -> Result<KernelExtensionMemory, TypeError> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.memory_semantics(),
        }
    }

    fn completion_protocol(&self) -> KernelCompletionProtocol {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.completion_protocol(),
        }
    }

    fn replay_cost(&self) -> Option<usize> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.replay_cost(),
        }
    }
}

impl CoreOperation for KernelExtension {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.name(),
        }
    }

    fn defined_type_identities(&self) -> Cow<'_, [<ArrayIrType as Type>::Identity]> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.defined_type_identities(),
        }
    }

    fn region_slots(&self) -> &'static [RegionSlot] {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.region_slots(),
        }
    }

    fn validate_region_interfaces(
        &self,
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<(), ProgramError> {
        let _ = region_interfaces;
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.validate_region_interfaces(region_interfaces),
        }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
        let _ = (input_types, region_interfaces);
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.infer_region_input_types(input_types, region_interfaces),
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        let _ = (input_types, region_interfaces);
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.infer_output_types(input_types, region_interfaces),
        }
    }

    fn fold(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Option<Vec<OperationFoldOutput>>, TypeError> {
        let _ = (input_types, region_interfaces);
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.fold(input_types, region_interfaces),
        }
    }

    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        let _ = (input_index, region_index);
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.input_region_provenance(region_index, input_index),
        }
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        let _ = output_index;
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.output_region_provenance(output_index),
        }
    }

    fn region_data_flow(&self) -> RegionDataFlow<'_> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.region_data_flow(),
        }
    }

    fn prune_boundary(
        &self,
        input_types: &[Self::Type],
        used_outputs: &[bool],
        regions: &mut dyn RegionLiveness,
    ) -> Result<Option<OperationBoundaryPruning<Self>>, ProgramError> {
        let _ = (input_types, used_outputs, &regions);
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => Ok(operation
                .prune_boundary(input_types, used_outputs, regions)?
                .map(|pruning| pruning.map_operation(Self::Mosaic))),
        }
    }

    fn is_zero(&self, output_index: usize) -> bool {
        let _ = output_index;
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.is_zero(output_index),
        }
    }

    fn region_capture_input_count(&self, region_index: usize) -> Option<usize> {
        let _ = region_index;
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.region_capture_input_count(region_index),
        }
    }

    fn reference_output_identity_input(&self, output_index: usize) -> Option<usize> {
        let _ = output_index;
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.reference_output_identity_input(output_index),
        }
    }

    fn allows_reference_access_through_region_input(&self, region_index: usize, mode: ReferenceAccessMode) -> bool {
        let _ = (mode, region_index);
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.allows_reference_access_through_region_input(region_index, mode),
        }
    }

    fn defers_reference_discharge(&self) -> bool {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.defers_reference_discharge(),
        }
    }

    fn effects(&self) -> Cow<'_, Effects> {
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.effects(),
        }
    }

    fn render(&self, formatter: &mut Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let _ = indentation;
        #[cfg(not(feature = "mosaic-gpu"))]
        let _ = &formatter;
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => operation.render(formatter, indentation),
        }
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayIrType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        let _ = renaming;
        match *self {
            #[cfg(feature = "mosaic-gpu")]
            Self::Mosaic(ref operation) => Ok(Self::Mosaic(operation.rename_type_identities(renaming)?)),
        }
    }
}

/// Canonical kernel instruction retained inside the XLA operation family until explicit compiler selection.
///
/// The payload contains only operation metadata. Its computations remain ordinary attached regions of the XLA
/// program, so effect, reference, identity, and structural validation continue to inspect the actual current body.
/// Generic differentiation and reference discharge require a kernel rule supported by the owner and reject this
/// carrier.
/// [`CoreOperation::region_data_flow`](ryft_core::Operation::region_data_flow) forwards the payload's declaration
/// unchanged, including custom rules and opaque carriers that publish references.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelOperation(pub(crate) PortableKernelOperation<KernelExtension>);

impl KernelOperation {
    /// Wraps a canonical portable instruction without changing its semantic contract.
    pub fn new(operation: PortableKernelOperation) -> Self {
        Self::from(operation)
    }

    /// Returns the canonical operation metadata. Computations belong to the containing instruction's regions.
    pub fn operation(&self) -> &PortableKernelOperation<KernelExtension> {
        &self.0
    }
}

impl<Extension: CoreOperation<Type = ArrayIrType> + Into<KernelExtension>> From<PortableKernelOperation<Extension>>
    for KernelOperation
{
    fn from(operation: PortableKernelOperation<Extension>) -> Self {
        // This conversion only widens a supported family and cannot fail. Map the conversion over the result
        // so it also works when no adapters are enabled and KernelExtension has no inhabited variants.
        Self(operation.map_extension(|extension| Ok(Into::into(extension))).unwrap())
    }
}

impl Display for KernelOperation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, formatter)
    }
}

impl CoreOperation for KernelOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn defined_type_identities(&self) -> Cow<'_, [<ArrayIrType as Type>::Identity]> {
        self.0.defined_type_identities()
    }

    fn region_slots(&self) -> &'static [RegionSlot] {
        self.0.region_slots()
    }

    fn validate_region_interfaces(
        &self,
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<(), ProgramError> {
        self.0.validate_region_interfaces(region_interfaces)
    }

    fn infer_region_input_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
        self.0.infer_region_input_types(input_types, region_interfaces)
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        self.0.infer_output_types(input_types, region_interfaces)
    }

    fn fold(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Option<Vec<OperationFoldOutput>>, TypeError> {
        self.0.fold(input_types, region_interfaces)
    }

    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        self.0.input_region_provenance(region_index, input_index)
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        self.0.output_region_provenance(output_index)
    }

    fn region_data_flow(&self) -> RegionDataFlow<'_> {
        self.0.region_data_flow()
    }

    fn prune_boundary(
        &self,
        input_types: &[Self::Type],
        used_outputs: &[bool],
        regions: &mut dyn RegionLiveness,
    ) -> Result<Option<OperationBoundaryPruning<Self>>, ProgramError> {
        Ok(self
            .0
            .prune_boundary(input_types, used_outputs, regions)?
            .map(|pruning| pruning.map_operation(Self)))
    }

    fn is_zero(&self, output_index: usize) -> bool {
        self.0.is_zero(output_index)
    }

    fn region_capture_input_count(&self, region_index: usize) -> Option<usize> {
        self.0.region_capture_input_count(region_index)
    }

    fn reference_output_identity_input(&self, output_index: usize) -> Option<usize> {
        self.0.reference_output_identity_input(output_index)
    }

    fn allows_reference_access_through_region_input(&self, region_index: usize, mode: ReferenceAccessMode) -> bool {
        self.0.allows_reference_access_through_region_input(region_index, mode)
    }

    fn defers_reference_discharge(&self) -> bool {
        self.0.defers_reference_discharge()
    }

    fn effects(&self) -> Cow<'_, Effects> {
        self.0.effects()
    }

    fn render(&self, formatter: &mut Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.0.render(formatter, indentation)
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayIrType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self(self.0.rename_type_identities(renaming)?))
    }
}

impl<C: Context<Type = ArrayIrType>, P: ReferenceDischargePolicy<C>> ReferenceDischargeableOperation<C, P>
    for KernelOperation
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        let _ = (context, driver, inputs);
        Err(ProgramError::MalformedProgram(format!(
            "unselected kernel operation `{}` requires owner reference discharge",
            self.name(),
        )))
    }
}

impl<C: CoreDomain<Type = ArrayIrType>> InterpretableOperation<C> for KernelOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let _ = (context, driver, inputs);
        Err(ProgramError::MalformedProgram(format!(
            "unselected kernel operation `{}` requires XLA compiler selection before interpretation",
            self.name(),
        )))
    }
}

impl<C: Context<Type = ArrayIrType>> PartiallyEvaluatableOperation<C> for KernelOperation
where
    Self: Into<C::Operation>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        let regions = driver.regions().map(|region| region.to_program()).collect::<Vec<_>>();
        if let PortableKernelOperation::Call(call) = &self.0 {
            // The inputs are validated against the original call signature first, because the trailing scalar prefetch
            // values are split off by position.
            let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            call.infer_output_types(&input_types, &driver.region_interfaces())?;
            let prefetch_count = call.prefetch_types().len();
            if prefetch_count != 0 && inputs[inputs.len() - prefetch_count..].iter().all(|input| input.is_known()) {
                return Err(ProgramError::UnsupportedOperation {
                    message: "scalar-prefetched inputs of XLA kernels must be specialized on the host before partial \
                        evaluation"
                        .to_owned(),
                });
            }
            context.residualize(self.clone(), regions, inputs)
        } else {
            context.fold_or_residualize(self.clone(), regions, inputs)
        }
    }
}

impl<C> BatchableOperation<C, ArrayIrBatchingPolicy> for KernelOperation
where
    C: Context<Type = ArrayIrType, Constant = Constant, Operation = Operation> + MeshContext,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
        let PortableKernelOperation::Call(call) = &self.0 else {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("kernel operation `{}` must be batched through its complete kernel call", self.name()),
            });
        };
        let definition = definition_from_body(call, driver.region(0)?)?;
        // One kernel call runs on one device, so a batch that is placed on mesh axes outside of every manual region
        // is batched per device inside a `shard_map` over those axes. Scalar prefetched inputs are not parameters and
        // must be specialized before batching, which the ordinary path reports.
        if call.prefetch_types().is_empty()
            && let bound_axes = context.parent().manual_axes()
            && let Some((mesh, axes)) = unbound_batch_placement(inputs, &bound_axes)?
        {
            return batch_in_shard_map(context.parent(), &definition, inputs, &mesh, axes, &bound_axes);
        }
        let (definition, axes) = definition
            .batched_inputs(inputs, ryft_kernels::DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS)
            .map_err(ProgramError::custom)?;
        let values = stage_kernel(
            context.parent(),
            &definition,
            &inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>(),
        )?;
        values
            .into_iter()
            .zip(axes)
            .map(|(value, axis)| ArrayIrBatch::new(value, axis.map(|axis| axis as isize)))
            .collect::<Result<Vec<_>, _>>()
    }
}

/// Returns the mesh and the mesh axes that the mapped `inputs` of a kernel place their batch axes on when no enclosing
/// manual region binds those axes, and [`None`] when the batch is not placed on mesh axes, when it is already the
/// local batch of an enclosing manual region, or when the mapped inputs do not agree on the placement (which
/// [`KernelDefinition::batched_inputs`] then reports).
///
/// `bound_axes` are the mesh axes that the enclosing manual regions made manual (refer to
/// [`MeshContext::manual_axes`]). A batch dimension that is placed on such axes is a local shard, whether or not the
/// region tracks manual variation, and one that is placed on other axes is a global value.
fn unbound_batch_placement<V: CoreValue<Type = ArrayIrType>>(
    inputs: &[ArrayIrBatch<V>],
    bound_axes: &[(String, ryft_core::NamedAxis)],
) -> Result<Option<(LogicalMesh, Vec<String>)>, BatchingError> {
    let mut placement = None::<Option<(LogicalMesh, Vec<String>)>>;
    for input in inputs {
        let Some(axis) = input.batch_axis_position() else { continue };
        let r#type = input.value().r#type();
        let ArrayIrType::Array(r#type) = r#type.as_ref() else { return Ok(None) };
        let current = match r#type.sharding().map(|sharding| (sharding, &sharding.dimensions()[axis])) {
            Some((sharding, ShardingDimension::Sharded(axes))) => {
                let is_bound = |axis: &&String| {
                    bound_axes.iter().any(|(name, binding)| {
                        name == *axis
                            && matches!(binding, ryft_core::NamedAxis::Mesh { mesh, .. }
                                if mesh.has_same_topology(sharding.mesh()))
                    })
                };
                match axes.iter().filter(is_bound).count() {
                    0 => Some((sharding.mesh().clone(), axes.clone())),
                    bound if bound == axes.len() => None,
                    _ => {
                        return Err(BatchingError::UnsupportedOperation {
                            message: format!(
                                "kernel batch axis is placed on {}, of which an enclosing manual region binds only a \
                                 part",
                                sharding.dimensions()[axis],
                            ),
                        });
                    }
                }
            }
            _ => None,
        };
        match &placement {
            None => placement = Some(current),
            Some(previous) if previous != &current => return Ok(None),
            Some(_) => {}
        }
    }
    Ok(placement.flatten())
}

/// Batches `definition` over packed `inputs` whose batch axes are placed on the `axes` of `mesh`, by staging a
/// `shard_map` over those axes whose body batches the kernel over the local batch of each device. A kernel runs on one
/// device and its compiler cannot partition it, while its batch items are independent, so every device runs the grid
/// of its own items. This follows the batching rule of JAX's `pallas_call`, which binds the batched call inside a
/// `shard_map` over the mesh axes of the batch with the local batch size.
///
/// The parameter types of a kernel are physical, so the body applies the kernel that is batched over the local batch
/// extent to the local shards as they are. Its results vary along the manual axes like its inputs, and the output
/// shardings of the map place the batch axis of every result on those axes. The results are typed over `mesh`, which
/// types the axes either as manual or, for a caller in JAX's explicit sharding mode, as explicit (refer to
/// [`ShardMapBoundary::caller_mesh`]).
///
/// An input that is also placed on a mesh axis outside of `axes` that no enclosing manual region binds (i.e., none of
/// `bound_axes`) is rejected, because one device would hold only a part of it.
fn batch_in_shard_map<C>(
    context: &C,
    definition: &KernelDefinition<KernelExtension>,
    inputs: &[ArrayIrBatch<C::Value>],
    mesh: &LogicalMesh,
    axes: Vec<String>,
    bound_axes: &[(String, ryft_core::NamedAxis)],
) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayIrType, Constant = Constant, Operation = Operation>,
{
    let unsupported = |message: String| BatchingError::UnsupportedOperation { message };

    // The map transfers nothing and converts nothing, so the packed inputs must already be the declared inputs of the
    // kernel up to their batch axes and their shardings, as on the ordinary path.
    let unbatched_types = inputs.iter().map(ArrayIrBatch::unbatched_type).collect::<Vec<_>>();
    if definition.operation().output_types_for(&unbatched_types).is_err() {
        return Err(ProgramError::custom(ryft_kernels::KernelTransformError::Unsupported {
            message: "batched input types differ from the kernel signature".to_owned(),
        })
        .into());
    }
    let explicit_count = axes.iter().filter(|axis| mesh.axis_type(axis) == Some(MeshAxisType::Explicit)).count();
    if explicit_count != 0 && explicit_count != axes.len() {
        return Err(unsupported(format!(
            "kernel batch axis is placed on {}, which mixes explicit and manual mesh axes",
            ShardingDimension::Sharded(axes),
        )));
    }
    let map_mesh = mesh.with_axis_types(&axes, MeshAxisType::Manual).map_err(ProgramError::custom)?;

    // The placement of every input over the mesh of the map, and the batch axis of every parameter.
    let mut input_types = Vec::with_capacity(inputs.len());
    let mut input_shardings = Vec::with_capacity(inputs.len());
    for (index, input) in inputs.iter().enumerate() {
        let r#type = input.value().r#type();
        let ArrayIrType::Array(r#type) = r#type.as_ref() else {
            return Err(unsupported("kernel batching accepts only ordinary array inputs".to_owned()));
        };

        // The map makes only the axes of the batch manual. An input that is also placed on another axis that no
        // enclosing manual region binds would reach the kernel split across devices, and gathering it is a choice
        // with a communication cost that belongs to the caller.
        if let Some(axis) = r#type.sharding().and_then(|sharding| {
            sharding.dimensions().iter().find_map(|dimension| match dimension {
                ShardingDimension::Sharded(names) => names
                    .iter()
                    .find(|name| !axes.contains(name) && !bound_axes.iter().any(|(bound, _)| bound == *name)),
                _ => None,
            })
        }) {
            return Err(unsupported(unbound_kernel_axis_message(index, axis)));
        }
        input_shardings.push(match r#type.sharding() {
            Some(sharding) => {
                Sharding::new(map_mesh.clone(), sharding.dimensions().to_vec()).map_err(ProgramError::custom)?
            }
            None => Sharding::replicated(map_mesh.clone(), r#type.rank()),
        });
        input_types.push(r#type.clone());
    }
    let mut packed = inputs.iter();
    let mut batch_axes = Vec::with_capacity(definition.operation().parameters().len());
    let mut output_shardings = Vec::new();
    let mut output_axes = Vec::new();
    for parameter in definition.operation().parameters() {
        let axis = if parameter.access() == KernelParameterAccess::WriteOnly {
            Some(0)
        } else {
            let input = packed.next().ok_or_else(|| {
                unsupported("kernel batching requires one input for every readable parameter".to_owned())
            })?;
            input.batch_axis_position()
        };
        batch_axes.push(axis);
        if parameter.access() != KernelParameterAccess::ReadOnly {
            // Every writable parameter is mapped, and only the batch axis of its result is placed on the mesh.
            let axis =
                axis.ok_or_else(|| unsupported("writable kernel parameters require a mapped batch axis".to_owned()))?;
            let mut dimensions = vec![ShardingDimension::Replicated; parameter.r#type().rank() + 1];
            *dimensions
                .get_mut(axis)
                .ok_or_else(|| unsupported("kernel batch axis is out of bounds for its parameter".to_owned()))? =
                ShardingDimension::Sharded(axes.clone());
            output_shardings.push(Sharding::new(map_mesh.clone(), dimensions).map_err(ProgramError::custom)?);
            output_axes.push(Some(axis));
        }
    }

    let boundary =
        ShardMapBoundary::new(map_mesh, input_shardings, output_shardings, axes).map_err(ProgramError::custom)?;
    let boundary = if explicit_count != 0 { boundary.with_explicit_caller_axes() } else { boundary };
    // The local shard of an input keeps the memory of that input, which is the memory that the kernel declares.
    let local_input_types = input_types
        .iter()
        .enumerate()
        .map(|(index, r#type)| {
            let local = boundary.local_input_type(index, &boundary.global_input_type(index, r#type)?)?;
            Ok::<_, ryft_core::ShardMapError>(local.with_memory(r#type.memory()))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(ProgramError::custom)?;
    let local_batch_size = inputs
        .iter()
        .zip(&local_input_types)
        .find_map(|(input, r#type)| Some(r#type.static_shape()?.dimensions()[input.batch_axis_position()?]))
        .ok_or_else(|| unsupported("kernel batching requires a mapped input with a static shape".to_owned()))?;
    let local = definition
        .batched(local_batch_size, &batch_axes, ryft_kernels::DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS)
        .map_err(ProgramError::custom)?;
    let body = kernel_program(&local, local_input_types.into_iter().map(ArrayIrType::Array).collect())?;
    let operation =
        ShardMapOperation::from_program(&body, input_types.into_iter().map(ArrayIrType::Array).collect(), boundary)
            .map_err(ProgramError::custom)?;
    let values = context.bind(
        Operation::ShardMap(Box::new(operation)),
        vec![body],
        &inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>(),
    )?;
    values
        .into_iter()
        .zip(output_axes)
        .map(|(value, axis)| ArrayIrBatch::new(value, axis.map(|axis| axis as isize)))
        .collect::<Result<Vec<_>, _>>()
}

impl<C: Context<Type = ArrayIrType>> DifferentiableOperation<C> for KernelOperation {
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let _ = (context, driver, inputs);
        Err(ProgramError::MalformedProgram(format!(
            "unselected kernel operation `{}` has no differentiation rule",
            self.name(),
        ))
        .into())
    }
}

impl<V: CoreValue<Type = ArrayIrType>, O: CoreOperation<Type = ArrayIrType>> TransposableOperation<V, O>
    for KernelOperation
where
    O: OperationProvider<ArrayIrType, GuardOperation<ArrayIrType>, Operation = O>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<CoreTracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<CoreTracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        let _ = (context, driver, inputs, outputs, accumulators);
        Err(ProgramError::MalformedProgram(format!(
            "unselected kernel operation `{}` has no transposition rule",
            self.name(),
        ))
        .into())
    }
}

impl LowerableOperation for KernelOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayIrType],
        output_types: &[ArrayIrType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(LoweringError::UnsupportedOperation {
            operation: self.name().to_string(),
            reason: "unselected kernel operation requires an explicit XLA compiler binding".to_string(),
        }
        .into())
    }
}

/// Binds an unselected kernel with its real body attached to the surrounding XLA program.
///
/// Portable instructions and enabled typed extensions remain visible to ordinary region, effect, and reference
/// traversal. Compiler selection belongs to the enclosing XLA session's immutable compilation options. Staging itself
/// neither invokes an adapter nor acquires native runtime resources. Bind concrete scalar-prefetch values with
/// [`KernelDefinition::specialize_prefetch`] before staging when they are intended as compile-time parameters. XLA
/// capture references cannot be materialized as host scalars by partial evaluation without an explicit transfer.
pub fn stage_kernel<C, Extension>(
    context: &C,
    definition: &KernelDefinition<Extension>,
    inputs: &[C::Value],
) -> Result<Vec<C::Value>, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = Constant, Operation = Operation>,
    Extension: PortableKernelExtension + Into<KernelExtension>,
{
    context.bind(
        Operation::Kernel(KernelOperation::new(PortableKernelOperation::<NoKernelExtension>::Call(
            definition.operation().clone(),
        ))),
        vec![stage_body(definition)?],
        inputs,
    )
}

/// Stages a native kernel primal with an explicit canonical JVP region. The JVP receives primal inputs followed by
/// active tangents and returns primal results followed by their tangents, as checked by a [`CompositeOperation`]
/// with a JVP rule region. The supplied rule owns derivative semantics, and the mutable kernel body is never
/// implicitly differentiated. The primal applies the kernel to inputs of the types of `inputs`, which may be placed
/// (refer to [`KernelCallOperation`]), so the rule region must be staged for those types.
pub fn stage_kernel_with_jvp<C, Extension>(
    context: &C,
    definition: &KernelDefinition<Extension>,
    jvp: &FlatProgram,
    inputs: &[C::Value],
) -> Result<Vec<C::Value>, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = Constant, Operation = Operation>,
    Extension: PortableKernelExtension + Into<KernelExtension>,
{
    context.bind(
        CompositeOperation::<Constant, Operation>::from_rule_regions(CompositeJvpRule::Explicit, false),
        vec![kernel_program(definition, input_types(inputs))?, jvp.clone()],
        inputs,
    )
}

/// Stages a native kernel primal with explicit canonical forward and backward VJP regions. The forward region returns
/// primal outputs and residuals. The backward region receives residuals and output cotangents and returns input
/// cotangents. A [`CompositeOperation`] with reverse mode rules validates these boundaries and retains its contract
/// that supports only reverse mode. A forward rule may itself stage a kernel when its primal must execute natively
/// during differentiation.
pub fn stage_kernel_with_vjp<C, Extension>(
    context: &C,
    definition: &KernelDefinition<Extension>,
    forward: &FlatProgram,
    backward: &FlatProgram,
    inputs: &[C::Value],
) -> Result<Vec<C::Value>, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = Constant, Operation = Operation>,
    Extension: PortableKernelExtension + Into<KernelExtension>,
{
    context.bind(
        CompositeOperation::<Constant, Operation>::from_rule_regions(CompositeJvpRule::Absent, true),
        vec![kernel_program(definition, input_types(inputs))?, forward.clone(), backward.clone()],
        inputs,
    )
}

/// Stages a native kernel primal whose derivative rules are the retained callbacks of a registered custom rule set.
/// The rules are traced lazily, on the first derivative request of each specialization, and ordinary execution only
/// ever runs the native kernel. `rules` is either a [`CustomRuleReference`](ryft_core::CustomRuleReference) to a rule
/// set registered in the XLA operation family, whose rules are written against XLA tracers, or a
/// [`LiftedCustomRules`](ryft_core::LiftedCustomRules) of a rule set registered in the array member family, whose
/// rules are written against array tracers and whose specializations are converted into this family.
pub fn stage_kernel_with_rules<C, Extension, S>(
    context: &C,
    definition: &KernelDefinition<Extension>,
    rules: S,
    inputs: &[C::Value],
) -> Result<Vec<C::Value>, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = Constant, Operation = Operation>,
    Extension: PortableKernelExtension + Into<KernelExtension>,
    S: CustomRuleSource<Constant, Operation>,
    Operation: From<CompositeOperation<Constant, Operation, S>>,
{
    context.bind(CompositeOperation::new(rules), vec![kernel_program(definition, input_types(inputs))?], inputs)
}

/// Uses an explicitly supplied pure equivalent program for differentiation while ordinary execution retains the native
/// kernel primal. This is an AD contract, not recovery from compiler errors, so the caller guarantees equivalence for
/// every admitted input. Complete input and output types must agree with the declared signature, and for placed inputs
/// the fallback is staged again over their types and must give the results of the call. The fallback cannot contain
/// references or effects, and the native call cannot have observable effects beyond its confined local reference state,
/// which [`KernelDefinition::effects`] summarizes together with the assertions of its mappings and scalar prefetch
/// conversions. State that the body declares explicitly (e.g., through an external call) is observable and is rejected,
/// as it is at the compiled embedding boundary. Existing canonical JVP construction derives the fallback rule, so
/// reverse mode uses the same primitive transpose machinery and rematerialization can recompute only the explicitly
/// selected pure derivative computation.
pub fn stage_kernel_with_fallback<C, Extension>(
    context: &C,
    definition: &KernelDefinition<Extension>,
    fallback: &FlatProgram,
    inputs: &[C::Value],
) -> Result<Vec<C::Value>, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = Constant, Operation = Operation>,
    Extension: PortableKernelExtension + Into<KernelExtension>,
{
    let signature =
        || ProgramError::Type(TypeError::invalid("pure kernel fallback must preserve the complete signature"));
    if fallback.input_types() != definition.operation().input_types()
        || fallback.output_types() != definition.operation().output_types()
    {
        return Err(signature());
    }

    // The fallback is declared over the physical signature. For placed inputs it is staged again over their types, so
    // that it and its derivative rule have the interface of this call, and it must then give the results the types
    // that the kernel call gives them.
    let actual_types = input_types(inputs);
    let restaged;
    let fallback = if actual_types == definition.operation().input_types() {
        fallback
    } else {
        let (output_types, program) = TracingContext::<Constant, Operation>::trace(
            |values: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                let context = values[0].context().clone();
                fallback.interpret_in_context(&context, values)
            },
            actual_types.clone(),
        )?;
        if output_types != definition.operation().output_types_for(&actual_types)? {
            return Err(signature());
        }
        restaged = program.into_flat_program();
        &restaged
    };
    // Only the private reference state of the body may be dropped with the primal. State that the body declares
    // explicitly (e.g., an external call) is observable, like every other effect class.
    let effects = definition.effects();
    if effects.has_explicit_ordered_state()
        || effects.classes().into_iter().any(|effect| effect != ryft_core::EffectClass::OrderedState)
    {
        return Err(ProgramError::UnsupportedOperation {
            message: "pure kernel fallback cannot replace observable kernel effects".to_owned(),
        });
    }
    if !fallback.effects().classes().is_empty()
        || fallback
            .regions()
            .iter()
            .flat_map(|region| region.atoms())
            .any(|atom| matches!(atom.r#type().as_ref(), ArrayIrType::Reference(_)))
    {
        return Err(ProgramError::UnsupportedOperation {
            message: "pure kernel fallback cannot contain references or observable effects".to_owned(),
        });
    }
    let jvp = fallback.jvp().map_err(ProgramError::from)?;
    stage_kernel_with_jvp(context, definition, &jvp, inputs)
}

/// PJRT facts for one selected execution device, in the mesh's physical order.
#[derive(Clone, Debug)]
pub struct KernelDeviceFacts {
    /// PJRT device kind, including the implementation's architecture description.
    kind: String,

    /// Typed plugin attributes, including compute capability when the plugin exposes it.
    attributes: BTreeMap<String, ryft_pjrt::Value>,
}

impl KernelDeviceFacts {
    /// Creates the facts of one execution device from its PJRT device `kind` and its typed plugin `attributes`.
    pub fn new<K: Into<String>>(kind: K, attributes: BTreeMap<String, ryft_pjrt::Value>) -> Self {
        Self { kind: kind.into(), attributes }
    }

    /// Returns the PJRT device kind, including the implementation's architecture description.
    #[inline]
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Returns the typed plugin attributes, including compute capability when the plugin exposes it.
    #[inline]
    pub fn attributes(&self) -> &BTreeMap<String, ryft_pjrt::Value> {
        &self.attributes
    }
}

/// Actual execution facts joined with the adapter target before kernel admission and compilation.
///
/// These facts come from the live PJRT client and selected mesh, rather than caller-provided target claims. They
/// enter the selected custom call's configuration identity before the ordinary cache lookup of lowered programs.
#[derive(Clone, Debug)]
pub struct KernelExecutionFacts {
    /// Live PJRT platform name.
    platform_name: String,

    /// Live PJRT platform implementation version.
    platform_version: String,

    /// Live plugin's PJRT C API version.
    pjrt_version: ryft_pjrt::Version,

    /// Whether the live plugin exposes the PJRT FFI extension.
    has_ffi_extension: bool,

    /// Live client attributes, in deterministic name order.
    attributes: BTreeMap<String, ryft_pjrt::Value>,

    /// Actual selected devices, preserving mesh order and multiplicity.
    devices: Vec<KernelDeviceFacts>,
}

impl KernelExecutionFacts {
    /// Creates execution facts from explicit values. Facts that describe a live client come from
    /// [`Self::from_target`]. This constructor serves callers that describe an execution environment without opening
    /// one (e.g., to check whether a kernel target admits it).
    ///
    /// # Parameters
    ///
    ///   - `platform_name`: PJRT platform name.
    ///   - `platform_version`: PJRT platform implementation version.
    ///   - `pjrt_version`: PJRT C API version of the plugin.
    ///   - `has_ffi_extension`: Whether the plugin exposes the PJRT FFI extension.
    ///   - `attributes`: Client attributes.
    ///   - `devices`: Facts of the selected devices, in mesh order and with their multiplicity.
    pub fn new<N: Into<String>, V: Into<String>>(
        platform_name: N,
        platform_version: V,
        pjrt_version: ryft_pjrt::Version,
        has_ffi_extension: bool,
        attributes: BTreeMap<String, ryft_pjrt::Value>,
        devices: Vec<KernelDeviceFacts>,
    ) -> Self {
        Self {
            platform_name: platform_name.into(),
            platform_version: platform_version.into(),
            pjrt_version,
            has_ffi_extension,
            attributes,
            devices,
        }
    }

    /// Selects the facts of the devices of `mesh`, in mesh order, from the [`Target`] of the execution owner,
    /// without loading an additional device runtime.
    pub fn from_target(target: &Target, mesh: &ryft_core::DeviceMesh) -> Result<Self, KernelEmbeddingError> {
        let devices = mesh
            .devices()
            .iter()
            .map(|mesh_device| {
                let device = target.device(mesh_device.id()).ok_or_else(|| KernelEmbeddingError::Invalid {
                    message: format!("kernel execution device {} is not visible to the live client", mesh_device.id()),
                })?;
                Ok(KernelDeviceFacts { kind: device.kind().to_owned(), attributes: device.attributes().clone() })
            })
            .collect::<Result<Vec<_>, KernelEmbeddingError>>()?;
        Ok(Self {
            platform_name: target.platform_name().to_owned(),
            platform_version: target.platform_version().to_owned(),
            pjrt_version: target.pjrt_version(),
            has_ffi_extension: target.has_ffi_extension(),
            attributes: target.attributes().clone(),
            devices,
        })
    }

    /// Encodes platform, plugin and ordered device capabilities using exact attribute variants and floating-point bits.
    /// Profiling and tuning records can hash these same bytes to reject incompatible measurements. Logical placement
    /// and process/device identifiers remain part of the caller's mesh contract, independently of capability identity.
    pub fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
        /// Encodes the finite PJRT attribute family without lossy floating-point formatting.
        fn attributes(values: &BTreeMap<String, ryft_pjrt::Value>) -> serde_json::Value {
            serde_json::Value::Array(
                values
                    .iter()
                    .map(|(name, value)| {
                        let value = match value {
                            ryft_pjrt::Value::Bool(value) => serde_json::json!(["bool", value]),
                            ryft_pjrt::Value::I64(value) => serde_json::json!(["i64", value]),
                            ryft_pjrt::Value::I64List(value) => serde_json::json!(["i64_list", value]),
                            ryft_pjrt::Value::F32(value) => serde_json::json!(["f32", value.to_bits()]),
                            ryft_pjrt::Value::String(value) => serde_json::json!(["string", value]),
                        };
                        serde_json::json!([name, value])
                    })
                    .collect(),
            )
        }
        Ok(serde_json::to_vec(&serde_json::json!([
            1,
            self.platform_name,
            self.platform_version,
            self.pjrt_version.major,
            self.pjrt_version.minor,
            self.has_ffi_extension,
            attributes(&self.attributes),
            self.devices
                .iter()
                .map(|device| serde_json::json!([device.kind, attributes(&device.attributes)]))
                .collect::<Vec<_>>()
        ]))?)
    }

    /// Returns the live PJRT platform name.
    #[inline]
    pub fn platform_name(&self) -> &str {
        &self.platform_name
    }

    /// Returns the live PJRT platform implementation version.
    #[inline]
    pub fn platform_version(&self) -> &str {
        &self.platform_version
    }

    /// Returns the live plugin's PJRT C API version.
    #[inline]
    pub fn pjrt_version(&self) -> ryft_pjrt::Version {
        self.pjrt_version
    }

    /// Returns whether the live plugin exposes the PJRT FFI extension.
    #[inline]
    pub fn has_ffi_extension(&self) -> bool {
        self.has_ffi_extension
    }

    /// Returns the actual selected devices, preserving mesh order and multiplicity.
    #[inline]
    pub fn devices(&self) -> &[KernelDeviceFacts] {
        &self.devices
    }
}

/// XLA integration contract joining a target owned by an adapter with actual PJRT execution facts.
///
/// Implementations belong in the integration that knows both systems, so adapter crates need not depend on XLA.
/// Reject unsupported platforms, device architectures, and plugin ABI requirements before adapter admission.
/// A successful check grants no fallback, implicit target substitution, or unchecked runtime feature support.
pub trait KernelTarget {
    /// Validates this immutable compilation target against all actual selected execution devices and the plugin ABI.
    fn admit_execution(&self, facts: &KernelExecutionFacts) -> Result<(), KernelEmbeddingError>;
}

/// One explicitly enabled compiler and typed embedding selected by the enclosing XLA compilation options.
///
/// The binding is immutable and uses a complete configuration key for equality and hashing. It does not own an
/// executable, runtime registry, or cache. Each selection revalidates the current attached body before invoking the
/// captured typed adapter. There is no fallback to another compiler or host interpretation.
#[derive(Clone)]
pub struct KernelCompilerBinding {
    /// Complete adapter and embedding identity, with unambiguous component lengths.
    configuration: Vec<u8>,

    /// Host verifier work limit, which affects admission work but not compiled kernel semantics.
    maximum_programs: usize,

    /// Typed adapter and embedding retained behind the selection boundary owned by XLA.
    compile: Arc<
        dyn Fn(
                &KernelDefinition<KernelExtension>,
                &KernelExecutionFacts,
            ) -> Result<CustomCallOperation, KernelEmbeddingError>
            + Send
            + Sync,
    >,
}

impl KernelCompilerBinding {
    /// Binds a typed adapter and embedding with their immutable target, options, and schedule.
    ///
    /// The embedding supplies every integration choice through its configuration key, including handler target,
    /// physical mapping, payload schema, and plugin ABI requirements. `maximum_programs` bounds core verification work.
    pub fn new<C, B>(
        compiler: C,
        target: C::Target,
        options: C::Options,
        schedule: KernelSchedule,
        embedding: B,
        maximum_programs: usize,
    ) -> Result<Self, KernelEmbeddingError>
    where
        C: 'static + Send + Sync + KernelCompiler<Error: 'static + Send + Sync>,
        C::Target: 'static + Send + Sync + KernelTarget,
        C::Options: 'static + Send + Sync,
        B: 'static + Send + Sync + KernelOutputEmbedding<C::Output>,
    {
        Self::new_with_extensions::<C, B, NoKernelExtension>(
            compiler,
            target,
            options,
            schedule,
            embedding,
            maximum_programs,
        )
    }

    /// Binds an explicitly selected extension family without changing portable constructor inference.
    ///
    /// Before compilation, the current attached body is reconstructed into this exact family. An operation owned by
    /// another family fails conversion rather than falling back to a different compiler. Validation and embedding
    /// then use the same path through execution facts and immutable configuration as ordinary portable kernels.
    pub fn new_with_extensions<C, B, Extension>(
        compiler: C,
        target: C::Target,
        options: C::Options,
        schedule: KernelSchedule,
        embedding: B,
        maximum_programs: usize,
    ) -> Result<Self, KernelEmbeddingError>
    where
        C: 'static + Send + Sync + KernelCompiler<Extension, Error: 'static + Send + Sync>,
        C::Target: 'static + Send + Sync + KernelTarget,
        C::Options: 'static + Send + Sync,
        B: 'static + Send + Sync + KernelOutputEmbedding<C::Output, Extension>,
        Extension: 'static + Send + Sync + PortableKernelExtension + TryFrom<KernelExtension, Error = ProgramError>,
    {
        let adapter_configuration = compiler
            .configuration_key(&target, &options, &schedule)
            .map_err(|error| KernelEmbeddingError::Compiler(Box::new(error)))?;
        let embedding_configuration = embedding.configuration_key()?;
        let mut configuration = Vec::new();
        for component in [&adapter_configuration, &embedding_configuration] {
            configuration.extend_from_slice(&(component.len() as u64).to_le_bytes());
            configuration.extend_from_slice(component);
        }
        let compile = Arc::new(move |definition: &KernelDefinition<KernelExtension>, facts: &KernelExecutionFacts| {
            target.admit_execution(facts)?;
            let current_configuration = compiler
                .configuration_key(&target, &options, &schedule)
                .map_err(|error| KernelEmbeddingError::Compiler(Box::new(error)))?;
            if current_configuration != adapter_configuration {
                return Err(KernelEmbeddingError::Invalid {
                    message: "selected kernel compiler configuration changed after binding".to_owned(),
                });
            }
            if embedding.configuration_key()? != embedding_configuration {
                return Err(KernelEmbeddingError::Invalid {
                    message: "selected kernel embedding configuration changed after binding".to_owned(),
                });
            }
            let body = definition
                .body()
                .map_operations(|operation| operation.clone().map_extension(Extension::try_from))
                .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?;
            let definition = KernelDefinition::new(definition.operation().clone(), body)
                .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?;
            let verified = VerifiedKernel::new(&definition, maximum_programs)
                .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?;
            let output = verified
                .compile(&compiler, &target, &options, &schedule)
                .map_err(|error| KernelEmbeddingError::Compiler(Box::new(error)))?;
            let facts_configuration = facts.configuration_key()?;
            // The common embedding boundary appends the embedding key exactly once. This component retains
            // compiler and execution facts identity, and the binding separately checks both captured configurations.
            let mut configuration = Vec::new();
            configuration.extend_from_slice(&(adapter_configuration.len() as u64).to_le_bytes());
            configuration.extend_from_slice(&adapter_configuration);
            configuration.extend_from_slice(&(facts_configuration.len() as u64).to_le_bytes());
            configuration.extend_from_slice(&facts_configuration);
            Ok(CompiledKernel::from_output(&verified, &configuration, &output, &embedding)?.custom_call().clone())
        });
        Ok(Self { configuration, maximum_programs, compile })
    }

    /// Returns the complete deterministic adapter and XLA embedding configuration identity.
    pub fn configuration(&self) -> &[u8] {
        &self.configuration
    }

    /// Compiles the current attached body after core verification and exact admission by the selected adapter.
    pub(crate) fn compile(
        &self,
        definition: &KernelDefinition<KernelExtension>,
        facts: &KernelExecutionFacts,
    ) -> Result<CustomCallOperation, KernelEmbeddingError> {
        (self.compile)(definition, facts)
    }
}

impl Debug for KernelCompilerBinding {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KernelCompilerBinding")
            .field("configuration", &self.configuration)
            .field("maximum_programs", &self.maximum_programs)
            .finish_non_exhaustive()
    }
}

impl PartialEq for KernelCompilerBinding {
    fn eq(&self, other: &Self) -> bool {
        self.configuration == other.configuration && self.maximum_programs == other.maximum_programs
    }
}

impl Eq for KernelCompilerBinding {}

impl std::hash::Hash for KernelCompilerBinding {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.configuration.hash(state);
        self.maximum_programs.hash(state);
    }
}

/// Selects reachable kernel calls, discharges their state proven local, and extracts the remaining region closure.
/// Dormant transform rule regions retain their unselected carriers until a transform makes them executable.
pub(crate) fn select_kernels(
    program: &FlatProgram,
    binding: Option<&KernelCompilerBinding>,
    execution_facts: impl FnOnce() -> Result<KernelExecutionFacts, KernelEmbeddingError>,
) -> Result<Option<FlatProgram>, KernelEmbeddingError> {
    // A selected call owns its entire body. Do not separately select descendants that its compiler consumes.
    // Ordinary computation edges remain visible, and dormant rule edges are retained without being compiled.
    let mut executable_regions = HashMap::<_, BTreeMap<String, ryft_core::MeshAxis>>::new();
    let mut pending = vec![(program.entry(), BTreeMap::new())];
    let mut has_kernel = false;
    while let Some((id, mut bound_axes)) = pending.pop() {
        if let Some(previous) = executable_regions.get(&id) {
            // A shared region must be safe under every executable caller, including an unbound caller.
            bound_axes = previous
                .iter()
                .filter(|&(name, axis)| bound_axes.get(name) == Some(axis))
                .map(|(name, axis)| (name.clone(), axis.clone()))
                .collect();
            if previous == &bound_axes {
                continue;
            }
        }
        executable_regions.insert(id, bound_axes.clone());
        let region = program.region_ref(id).unwrap();
        for instruction in region.instructions() {
            if matches!(instruction.operation(), Operation::Kernel(KernelOperation(PortableKernelOperation::Call(_)))) {
                has_kernel = true;
                continue;
            }
            let mut child_axes = bound_axes.clone();
            if let Operation::ShardMap(operation) = instruction.operation() {
                let map = operation.boundary();
                child_axes.extend(map.manual_axes().iter().map(|name| {
                    let index = map.mesh().axis_index(name).unwrap();
                    (name.clone(), map.mesh().axes()[index].clone())
                }));
            }
            pending.extend(
                instruction
                    .regions()
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|&(index, _)| {
                        instruction.operation().region_role(index) == Some(ryft_core::RegionRole::Computation)
                    })
                    .map(|(_, id)| (id, child_axes.clone())),
            );
        }
    }
    if !has_kernel {
        return Ok(None);
    }
    let binding = binding.ok_or_else(|| KernelEmbeddingError::Invalid {
        message: "kernel call has no explicitly enabled XLA compiler binding".to_owned(),
    })?;
    let facts = execution_facts()?;
    let mut selected = false;
    let regions = program
        .regions()
        .iter()
        .enumerate()
        .map(|(index, region)| {
            let instructions = region
                .instructions()
                .iter()
                .map(|instruction| {
                    if let Some(bound_axes) = executable_regions.get(&ryft_core::RegionId::new(index))
                        && let Operation::Kernel(KernelOperation(PortableKernelOperation::Call(operation))) =
                            instruction.operation()
                    {
                        let body = ryft_core::RegionRef::new(program.regions(), instruction.regions()[0])
                            .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?;
                        let definition = definition_from_body(operation, body)
                            .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?;
                        // The parameter types of a kernel are physical, so the placement to validate is that of the
                        // values that the call receives.
                        let input_types = instruction
                            .inputs()
                            .iter()
                            .filter_map(|input| match region.atoms()[input.index()].r#type().as_ref() {
                                ArrayIrType::Array(r#type) => Some(r#type.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        validate_kernel_sharding(&input_types, bound_axes)?;

                        // The compiler sees the physical kernel. The custom call that replaces the kernel call is
                        // declared for the placement of the values that the call receives and must give its results
                        // the types that the kernel call gave them, since the instruction keeps its result atoms.
                        let compiled = binding.compile(&definition, &facts)?;
                        let atom_type = |atom: &AtomId| region.atoms()[atom.index()].r#type().into_owned();
                        let compiled = placed_custom_call(
                            operation,
                            &compiled,
                            &instruction.inputs().iter().map(atom_type).collect::<Vec<_>>(),
                        )
                        .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?;
                        selected = true;
                        return Ok(Instruction::new(
                            compiled.into(),
                            instruction.inputs().to_vec(),
                            instruction.outputs().to_vec(),
                            vec![],
                        )
                        .with_provenance(instruction.provenance().clone()));
                    }
                    Ok(instruction.clone())
                })
                .collect::<Result<Vec<_>, KernelEmbeddingError>>()?;
            Ok(Region::new(
                region.atoms().to_vec(),
                region.input_ids().to_vec(),
                region.output_ids().to_vec(),
                instructions,
            ))
        })
        .collect::<Result<Vec<_>, KernelEmbeddingError>>()?;
    if !selected {
        return Ok(None);
    }
    let arena = ryft_core::RegionArena::from_regions(regions)
        .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?;
    let selected = ryft_core::RegionRef::new(&arena, program.entry())
        .map_err(|error| KernelEmbeddingError::Invalid { message: error.to_string() })?
        .to_program();
    for region in selected.entry_region_ref().computation_regions() {
        if let Some(instruction) = region
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), Operation::Kernel(_)))
        {
            return Err(KernelEmbeddingError::Invalid {
                message: format!(
                    "kernel operation `{}` remains outside a selected kernel call",
                    instruction.operation().name()
                ),
            });
        }
    }
    Ok(Some(selected))
}

/// Converts a verified definition's body into the XLA value universe without hiding its region graph.
///
/// This is a structural conversion of the whole arena, like
/// [`Program::map_operations`](ryft_core::Program::map_operations), so atom and region identifiers stay stable,
/// including shared descendants. Array literal atoms become variables defined by existing constant
/// instructions at the region entrance. No runtime buffers, captures, or references become literal XLA values.
fn stage_body<Extension: PortableKernelExtension + Into<KernelExtension>>(
    definition: &KernelDefinition<Extension>,
) -> Result<FlatProgram, ProgramError> {
    let body = definition.body();
    let regions = body
        .regions()
        .iter()
        .map(|region| {
            let mut instructions = Vec::new();
            let atoms = region
                .atoms()
                .iter()
                .enumerate()
                .map(|(index, atom)| {
                    Ok(match atom {
                        Atom::Variable(r#type) => Atom::Variable(r#type.clone()),
                        Atom::Constant(ArrayIrValue::Dimension(value)) => {
                            Atom::Constant(Constant::Dimension(value.clone()))
                        }
                        Atom::Constant(ArrayIrValue::Array(value)) => {
                            instructions.push(Instruction::new(
                                ConstantOperation::new(value.clone()).into(),
                                vec![],
                                vec![AtomId::new(index)],
                                vec![],
                            ));
                            Atom::Variable(ArrayIrType::Array(value.r#type().into_owned()))
                        }
                        Atom::Constant(ArrayIrValue::Reference(_)) => {
                            return Err(ProgramError::MalformedProgram(
                                "kernel body cannot capture a literal reference".to_owned(),
                            ));
                        }
                    })
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            instructions.extend(region.instructions().iter().map(|instruction| {
                Instruction::new(
                    Operation::Kernel(KernelOperation::from(instruction.operation().clone())),
                    instruction.inputs().to_vec(),
                    instruction.outputs().to_vec(),
                    instruction.regions().to_vec(),
                )
                .with_provenance(instruction.provenance().clone())
            }));
            Ok(Region::new(atoms, region.input_ids().to_vec(), region.output_ids().to_vec(), instructions))
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    CoreProgram::new(
        vec![Placeholder; body.input_count()],
        vec![Placeholder; body.output_count()],
        regions,
        body.entry(),
    )
}

/// Returns the types of `inputs`, which are the types that a kernel call on them is staged for.
fn input_types<V: CoreValue<Type = ArrayIrType>>(inputs: &[V]) -> Vec<ArrayIrType> {
    inputs.iter().map(|input| input.r#type().into_owned()).collect()
}

/// Builds the ordinary functional primal region shared by canonical composite carriers, which applies `definition` to
/// inputs of `input_types`. These are its declared input types up to their shardings (e.g., the local shards of a
/// manual region).
fn kernel_program<Extension: PortableKernelExtension + Into<KernelExtension>>(
    definition: &KernelDefinition<Extension>,
    input_types: Vec<ArrayIrType>,
) -> Result<FlatProgram, ProgramError> {
    let mut builder = ryft_core::ProgramBuilder::new();
    let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
    let body = stage_body(definition)?;
    let body_region = builder.import_region(body.entry_region_ref());
    let outputs = builder
        .add_instruction(
            Operation::Kernel(KernelOperation::new(PortableKernelOperation::<NoKernelExtension>::Call(
                definition.operation().clone(),
            ))),
            vec![body_region],
            inputs.clone(),
            None,
        )?
        .to_vec();
    builder.build(
        outputs,
        vec![Placeholder; inputs.len()],
        vec![Placeholder; definition.operation().output_types().len()],
    )
}

/// Restores the current attached body for the core verifier. Retained operation metadata is never a body proof.
fn definition_from_body(
    operation: &KernelCallOperation,
    body: ryft_core::RegionRef<'_, Constant, Operation>,
) -> Result<KernelDefinition<KernelExtension>, ProgramError> {
    let body = body.to_program();
    let regions = body
        .regions()
        .iter()
        .map(|region| {
            let atoms = region
                .atoms()
                .iter()
                .map(|atom| {
                    Ok(match atom {
                        Atom::Variable(r#type) => Atom::Variable(r#type.clone()),
                        Atom::Constant(Constant::Dimension(value)) => {
                            Atom::Constant(ArrayIrValue::<CoreArray>::Dimension(value.clone()))
                        }
                        Atom::Constant(Constant::Boolean(value)) => {
                            Atom::Constant(ArrayIrValue::Array(CoreArray::scalar(*value)?))
                        }
                        Atom::Constant(Constant::Captured(_) | Constant::Session(_)) => {
                            return Err(ProgramError::MalformedProgram(
                                "kernel body cannot capture an external XLA value".to_owned(),
                            ));
                        }
                    })
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let instructions = region
                .instructions()
                .iter()
                .map(|instruction| {
                    let operation = match instruction.operation() {
                        Operation::Kernel(operation) => operation.0.clone(),
                        Operation::Array(ryft_core::ArrayOperation::Constant(operation)) => {
                            PortableKernelOperation::Portable(
                                ryft_core::ArrayOperation::Constant(operation.clone()).into(),
                            )
                        }
                        operation => {
                            return Err(ProgramError::MalformedProgram(format!(
                                "operation `{}` of the XLA family has no kernel reconstruction",
                                operation.name(),
                            )));
                        }
                    };
                    Ok(Instruction::new(
                        operation,
                        instruction.inputs().to_vec(),
                        instruction.outputs().to_vec(),
                        instruction.regions().to_vec(),
                    )
                    .with_provenance(instruction.provenance().clone()))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            Ok(Region::new(atoms, region.input_ids().to_vec(), region.output_ids().to_vec(), instructions))
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let body = CoreProgram::new(
        vec![Placeholder; body.input_count()],
        vec![Placeholder; body.output_count()],
        regions,
        body.entry(),
    )?;
    KernelDefinition::new(operation.clone(), body).map_err(|error| ProgramError::MalformedProgram(error.to_string()))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::error::Error as _;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayIrOperation, ArrayOperation, ArrayType, AssertOperation, AttachedRegions, CompilationTracer,
        ConditionOperation, CotangentDestinationKind, CustomCallOperation, CustomRuleDefinition,
        CustomRuleRegistration, DataType, Device, DeviceMesh, DimensionBounds, DimensionFromScalarOperation,
        DimensionType, DimensionValue, DimensionVariable, EffectClass, LiftedCustomRules, LogicalMesh, Memory,
        MeshAxis, MeshAxisType, NamedAxis, NothingSavable, PartialEvaluation, PartitionedProgram,
        ProgramBuilder as CoreProgramBuilder, RecursiveBatchingDriver, ReferenceRead, ReferenceReadOperation,
        ReferenceType, ReferenceWrite, ReferenceWriteOperation, RegionDriver, RegionRef, RematerializeOperation,
        ReshapeOperation, ResidualPolicyReference, ReverseModeDifferentiate, ScanOperation, ShardMapBoundary,
        ShardMapOperation, Sharding, ShardingDimension, StagedFunction as CoreStagedFunction, StagingContext,
        TransferToMemoryOperation, TypedProgram, WhileOperation,
    };
    use ryft_kernels::{
        BlockMapping, BoundaryPolicy, Grid, KernelCompilationError, KernelCompiler, KernelParameter,
        KernelParameterAccess, KernelSchedule, MappingExpression, ParameterWindow, TileLoadOperation, VerifiedKernel,
        mapped_call, whole_array_parameter,
    };
    #[cfg(feature = "mosaic-gpu")]
    use ryft_mosaic::kernels::gpu::{
        Compiler as GpuCompiler, GpuOperation, Mma, Options as GpuOptions, Target as GpuTarget,
    };
    use ryft_pjrt::{ClientOptions, CpuClientOptions};

    #[cfg(feature = "mosaic-gpu")]
    use crate::kernels::MosaicGpuEmbedding;
    use crate::kernels::{KernelEmbeddingError, KernelOutputEmbedding};
    use crate::sessions::tests::session_with_client;
    use crate::{ArrayConstant, FromPjrt, JitCallOperation, Options, Platform, ProgramBuilder, Session};

    use super::*;

    /// Typed fixture compiler that admits only a caller-selected target and retains exact option identity.
    struct Compiler;

    impl KernelCompiler for Compiler {
        type Target = FixtureTarget;
        type Options = u32;
        type Output = ();
        type Error = std::io::Error;

        fn admit(
            &self,
            kernel: &VerifiedKernel<'_>,
            target: &FixtureTarget,
            options: &u32,
            schedule: &KernelSchedule,
        ) -> Result<(), KernelCompilationError<std::io::Error>> {
            let _ = (kernel, options, schedule);
            if target.0 {
                Ok(())
            } else {
                Err(KernelCompilationError::Unavailable { message: "fixture target unavailable".to_owned() })
            }
        }

        fn configuration_key(
            &self,
            target: &FixtureTarget,
            options: &u32,
            schedule: &KernelSchedule,
        ) -> Result<Vec<u8>, KernelCompilationError<std::io::Error>> {
            let _ = schedule;
            let mut key = vec![u8::from(target.0)];
            key.extend_from_slice(&options.to_le_bytes());
            Ok(key)
        }

        fn compile(
            &self,
            kernel: &VerifiedKernel<'_>,
            target: &FixtureTarget,
            options: &u32,
            schedule: &KernelSchedule,
        ) -> Result<(), KernelCompilationError<std::io::Error>> {
            let _ = (kernel, target, schedule);
            if *options == u32::MAX {
                return Err(KernelCompilationError::Compiler(std::io::Error::other("fixture compiler failed")));
            }
            Ok(())
        }
    }

    /// Explicit fixture target whose admission checks the actual execution platform first.
    struct FixtureTarget(bool);

    impl KernelTarget for FixtureTarget {
        fn admit_execution(&self, facts: &KernelExecutionFacts) -> Result<(), KernelEmbeddingError> {
            if facts.platform_name != "fixture" {
                return Err(KernelEmbeddingError::Invalid {
                    message: "fixture execution platform mismatch".to_owned(),
                });
            }
            Ok(())
        }
    }

    /// Deferred output written by hand, whose identity includes its actual handler selection.
    struct Embedding(&'static str);

    impl KernelOutputEmbedding<()> for Embedding {
        fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
            Ok(self.0.as_bytes().to_vec())
        }

        fn custom_call(
            &self,
            kernel: &VerifiedKernel<'_>,
            output: &(),
        ) -> Result<CustomCallOperation, KernelEmbeddingError> {
            let _ = output;
            Ok(CustomCallOperation::new(
                self.0,
                vec![kernel.definition().operation().parameters()[0].r#type().into_owned()],
            )
            .with_input_output_alias(0, 0)?)
        }
    }

    /// Deterministic execution facts for the embedding fixture written by hand.
    fn facts() -> Result<KernelExecutionFacts, KernelEmbeddingError> {
        Ok(KernelExecutionFacts {
            platform_name: "fixture".to_owned(),
            platform_version: "1".to_owned(),
            pjrt_version: ryft_pjrt::Version { major: 0, minor: 115 },
            has_ffi_extension: true,
            attributes: Default::default(),
            devices: vec![],
        })
    }

    /// Shares an inert compiler identity for selection tests and for persistence tests that execute ordinary PJRT
    /// programs.
    pub(crate) fn binding(options: u32) -> KernelCompilerBinding {
        KernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            options,
            KernelSchedule::default(),
            Embedding("ryft.test.aot"),
            1,
        )
        .unwrap()
    }

    /// A scalar floating-point identity whose native body stays opaque to derivative construction.
    fn differentiable_definition() -> KernelDefinition {
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        KernelDefinition::trace(call, |(references, _)| {
            references[0].write(&references[0].read()?)?;
            Ok(())
        })
        .unwrap()
    }

    /// Builds the primal program of `definition` over its declared input types.
    fn kernel_primal(definition: &KernelDefinition) -> Result<FlatProgram, ProgramError> {
        kernel_program(definition, definition.operation().input_types())
    }

    /// Compiles scalar derivative programs through the existing XLA session and waits for concrete results.
    fn execute_derivative(program: &FlatProgram, inputs: &[f32]) -> Vec<f32> {
        /// Keeps the traced input and output value lifetimes tied to the same execution client.
        fn replay(
            program: &FlatProgram,
            inputs: Vec<CompilationTracer<Session>>,
        ) -> Result<Vec<CompilationTracer<Session>>, crate::Error> {
            let context = inputs[0].context().clone();
            Ok(program.interpret_in_context(&context, inputs)?)
        }

        let client = crate::tests::execution_client();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![]).unwrap(),
            vec![Device::from_pjrt(client.addressable_devices().unwrap().remove(0)).unwrap()],
        )
        .unwrap();
        let session = session_with_client(client).with_device_mesh(mesh.clone());
        let staged = CoreStagedFunction::<Session, Vec<ArrayIrType>, Vec<ArrayIrType>>::stage(
            |_, _, inputs| replay(program, inputs),
            vec![],
            program.input_types().into_owned(),
            Options::new(mesh.clone()),
        )
        .unwrap();
        let compiled = staged.lower(&session).unwrap().compile(&session).unwrap();
        let inputs = inputs
            .iter()
            .zip(program.input_types().into_owned())
            .map(|(value, r#type)| {
                let r#type = <&ArrayType>::try_from(&r#type).unwrap().clone();
                crate::Value::Array(
                    crate::Array::from_host_buffer(&session, r#type, mesh.clone(), value.to_ne_bytes()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        compiled
            .executable()
            .execute(&session, inputs)
            .unwrap()
            .into_iter()
            .map(|value| {
                let crate::Value::Array(value) = value else { panic!("derivative returned a non-array value") };
                value.block_until_ready().unwrap();
                let bytes = value
                    .addressable_shards()
                    .next()
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                f32::from_ne_bytes(bytes.as_slice().try_into().unwrap())
            })
            .collect()
    }

    /// Creates an ordinary XLA graph with a real attached scalar kernel body.
    fn program() -> FlatProgram {
        let definition = crate::kernels::tests::definition();
        TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                stage_kernel(inputs[0].context(), &definition, &inputs)
            },
            definition.operation().input_types(),
        )
        .unwrap()
        .1
    }

    #[test]
    fn test_kernel_operation_new() {
        let definition = crate::kernels::tests::definition();
        let carrier = KernelOperation::new(PortableKernelOperation::Call(definition.operation().clone()));
        assert_eq!(carrier.operation(), &PortableKernelOperation::Call(definition.operation().clone()));
        assert_eq!(carrier.name(), definition.operation().name());
        assert_eq!(carrier.region_slots(), definition.operation().region_slots());
        assert_eq!(carrier.input_region_provenance(0, 0), InputRegionProvenance::Local);
    }

    #[test]
    fn test_kernel_operation_region_data_flow() {
        let definition = crate::kernels::tests::definition();
        let opaque = KernelOperation::new(PortableKernelOperation::Call(definition.operation().clone()));
        assert!(matches!(opaque.region_data_flow(), RegionDataFlow::Opaque));
        let ordinary =
            KernelOperation::new(PortableKernelOperation::from(ArrayOperation::Condition(ConditionOperation::new())));
        assert!(matches!(ordinary.region_data_flow(), RegionDataFlow::Provenance));

        let recurrent = KernelOperation::new(PortableKernelOperation::Portable(ArrayIrOperation::<CoreArray>::from(
            ArrayOperation::Scan(ScanOperation::new(2, 3usize)),
        )));
        let RegionDataFlow::Custom(expected) = recurrent.0.region_data_flow() else {
            panic!("a scan must declare its custom region data flow");
        };
        let RegionDataFlow::Custom(forwarded) = recurrent.region_data_flow() else {
            panic!("the XLA kernel carrier must retain custom region data flow");
        };
        assert!(std::ptr::eq(forwarded, expected));

        #[cfg(feature = "mosaic-gpu")]
        assert!(matches!(
            KernelExtension::Mosaic(ryft_mosaic::kernels::gpu::GpuOperation::Wgmma).region_data_flow(),
            RegionDataFlow::Opaque,
        ));
    }

    #[test]
    fn test_kernel_operation_partially_evaluate_prefetch() {
        let scalar = ArrayType::scalar(DataType::I32);
        let mut mapping = CoreProgramBuilder::<ArrayIrValue<CoreArray>, ArrayIrOperation<CoreArray>>::new();
        mapping.add_input(DimensionType::new("prefetched", DimensionBounds::non_negative(Some(4)).unwrap()).into());
        let parameter = KernelParameter::new(
            scalar.clone(),
            KernelParameterAccess::WriteOnly,
            BlockMapping::new(
                mapping.build(vec![], vec![Placeholder], vec![]).unwrap(),
                vec![],
                BoundaryPolicy::InBounds,
            )
            .unwrap(),
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace_with_prefetch(
            KernelCallOperation::new_with_prefetch(Grid::new(vec![]).unwrap(), vec![parameter], vec![scalar.clone()])
                .unwrap(),
            |(references, _, values)| references[0].write(&values[0]),
        )
        .unwrap();
        let program = kernel_primal(&definition).unwrap();
        let outer = TracingContext::<Constant, Operation>::new();
        let known = outer.input(scalar.clone().into());
        assert_eq!(
            program.partially_evaluate_in_context(&outer, &[PartialValue::Known(known)]).unwrap_err(),
            ProgramError::UnsupportedOperation {
                message: "scalar-prefetched inputs of XLA kernels must be specialized on the host before partial \
                    evaluation"
                    .to_owned(),
            }
        );
        let residual = program.partially_evaluate_in_context(&outer, &[PartialValue::Unknown(scalar.into())]).unwrap();
        assert_eq!(
            residual.program().to_string(),
            indoc! {"
                lambda %0:i32[] .
                let %1:i32[] = kernel_call [
                    grid=[],
                    prefetch_0=i32[],
                    identities=[0],
                    identity_0_bounds=[0, 4),
                    parameter_0=write-only i32[] [] InBounds,
                    mapping_0={
                        lambda %0:dimension<prefetched ∈ [0, 4)> .
                        in ()
                    },
                ] %0 [
                    body={
                        lambda %0:ref<i32[]>, %1:i32[] .
                        let () = reference_write %0 %1
                        in ()
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_kernel_operation_partially_evaluate_inputs() {
        /// Region driver that exposes the staged body of one kernel call and supports no recursive partial evaluation.
        struct CallBodyDriver<'o> {
            body: &'o FlatProgram,
        }

        impl RegionDriver<Constant, Operation> for CallBodyDriver<'_> {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Constant, Operation>>
            where
                Constant: 'r,
                Operation: 'r,
            {
                std::iter::once(self.body.entry_region_ref())
            }
        }

        impl PartialEvaluationDriver<TracingContext<Constant, Operation>> for CallBodyDriver<'_> {
            fn partially_evaluate_region(
                &self,
                _context: &PartialEvaluationContext<TracingContext<Constant, Operation>>,
                _index: usize,
                _inputs: Vec<PartialEvaluationValue<CoreTracer<TracingContext<Constant, Operation>>>>,
            ) -> Result<Vec<PartialEvaluationValue<CoreTracer<TracingContext<Constant, Operation>>>>, ProgramError>
            {
                panic!("a kernel call never partially evaluates its body")
            }

            fn partially_evaluate_program(
                &self,
                _context: &PartialEvaluationContext<TracingContext<Constant, Operation>>,
                _region: RegionRef<'_, Constant, Operation>,
                _knowledge: &[PartialValue<CoreTracer<TracingContext<Constant, Operation>>>],
            ) -> Result<PartialEvaluation<TracingContext<Constant, Operation>>, ProgramError> {
                panic!("a kernel call never partially evaluates its body")
            }

            fn partition_program(
                &self,
                _context: &PartialEvaluationContext<TracingContext<Constant, Operation>>,
                _region: RegionRef<'_, Constant, Operation>,
                _input_known: &[bool],
            ) -> Result<PartitionedProgram<Constant, Operation>, ProgramError> {
                panic!("a kernel call never partitions its body")
            }
        }

        // A call that copies the element of its `i32[4]` data input at the prefetched start into its `i32[1]` result,
        // whose logical inputs are therefore `[i32[4], i32[]]`.
        let data_type = ArrayType::new_static(DataType::I32, vec![4]);
        let prefetch_type = ArrayType::scalar(DataType::I32);
        let call = mapped_call(
            Grid::new(vec![]).unwrap(),
            vec![
                (
                    data_type.clone(),
                    KernelParameterAccess::ReadOnly,
                    ParameterWindow::ElementStarts {
                        block_shape: vec![1],
                        boundary: BoundaryPolicy::InBounds,
                        starts: vec![MappingExpression::Prefetch(0)],
                    },
                ),
                (
                    ArrayType::new_static(DataType::I32, vec![1]),
                    KernelParameterAccess::WriteOnly,
                    ParameterWindow::WholeArray,
                ),
            ],
            vec![prefetch_type.clone()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace_with_prefetch(call, |(references, _, _)| {
            let value = references[0].read()?;
            references[1].write(&value)
        })
        .unwrap();
        let body = stage_body(&definition).unwrap();
        let driver = CallBodyDriver { body: &body };
        let operation =
            KernelOperation::new(PortableKernelOperation::<NoKernelExtension>::Call(definition.operation().clone()));
        let outer = TracingContext::<Constant, Operation>::new();
        let evaluate = |known: &[bool], types: &[ArrayType]| {
            let context = PartialEvaluationContext::new(outer.clone());
            let inputs = known
                .iter()
                .zip(types)
                .enumerate()
                .map(|(index, (known, r#type))| {
                    if *known {
                        PartialEvaluationValue::known_input(outer.input(r#type.clone().into()))
                    } else {
                        context.unknown_input(r#type.clone().into(), index)
                    }
                })
                .collect::<Vec<_>>();
            let outputs = operation.partially_evaluate(&context, &driver, &inputs)?;
            context.into_evaluation(outputs)
        };
        let mismatch =
            ProgramError::Type(TypeError::invalid("`kernel_call` input types do not match its logical signature"));

        // Inputs that do not match the original signature are rejected before the prefetch values are split off,
        // whether the trailing input is known or not.
        let i64_type = ArrayType::scalar(DataType::I64);
        assert_eq!(evaluate(&[], &[]).map(|_| ()), Err(mismatch.clone()));
        assert_eq!(evaluate(&[true], &[prefetch_type.clone()]).map(|_| ()), Err(mismatch.clone()));
        assert_eq!(evaluate(&[false], &[prefetch_type.clone()]).map(|_| ()), Err(mismatch.clone()));
        assert_eq!(evaluate(&[true], &[data_type.clone()]).map(|_| ()), Err(mismatch.clone()));
        assert_eq!(
            evaluate(&[false, false, true], &[data_type.clone(), data_type.clone(), prefetch_type.clone()]).map(|_| ()),
            Err(mismatch.clone()),
        );
        assert_eq!(evaluate(&[false, true], &[data_type.clone(), i64_type.clone()]).map(|_| ()), Err(mismatch.clone()));
        assert_eq!(evaluate(&[false, false], &[data_type.clone(), i64_type]).map(|_| ()), Err(mismatch.clone()));
        assert_eq!(evaluate(&[true, false], &[prefetch_type.clone(), data_type.clone()]).map(|_| ()), Err(mismatch));

        // A known prefetch value must be specialized on the host instead, whether the data input is known or not.
        let unspecialized = ProgramError::UnsupportedOperation {
            message: "scalar-prefetched inputs of XLA kernels must be specialized on the host before partial \
                evaluation"
                .to_owned(),
        };
        assert_eq!(
            evaluate(&[false, true], &[data_type.clone(), prefetch_type.clone()]).map(|_| ()),
            Err(unspecialized.clone()),
        );
        assert_eq!(
            evaluate(&[true, true], &[data_type.clone(), prefetch_type.clone()]).map(|_| ()),
            Err(unspecialized),
        );

        // An unknown prefetch value leaves the original call and its complete input list residual.
        let residual = evaluate(&[false, false], &[data_type.clone(), prefetch_type.clone()]).unwrap();
        assert_eq!(
            residual.program().to_string(),
            indoc! {"
                lambda %0:i32[4], %1:i32[] .
                let %2:i32[1] = kernel_call [
                    grid=[],
                    prefetch_0=i32[],
                    identities=[0, 0],
                    identity_0_bounds=[0, ∞),
                    parameter_0=read-only i32[4] [1] InBounds,
                    mapping_0={
                        lambda %0:dimension<prefetch_0 ∈ [0, ∞)> .
                        in (%0)
                    },
                    parameter_1=write-only i32[1] [1] InBounds,
                    mapping_1={
                        lambda %0:dimension<prefetch_0 ∈ [0, ∞)> .
                        let %1:dimension<0> = const 0
                        in (%1)
                    },
                ] %0 %1 [
                    body={
                        lambda %0:ref<i32[1]>, %1:ref<i32[1]>, %2:i32[] .
                        let %3:i32[1] = reference_read %0
                            () = reference_write %1 %3
                        in ()
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );

        // A known data input becomes a residual program input after the unknown prefetch value.
        let residual = evaluate(&[true, false], &[data_type, prefetch_type]).unwrap();
        assert_eq!(
            residual.program().to_string(),
            indoc! {"
                lambda %0:i32[], %1:i32[4] .
                let %2:i32[1] = kernel_call [
                    grid=[],
                    prefetch_0=i32[],
                    identities=[0, 0],
                    identity_0_bounds=[0, ∞),
                    parameter_0=read-only i32[4] [1] InBounds,
                    mapping_0={
                        lambda %0:dimension<prefetch_0 ∈ [0, ∞)> .
                        in (%0)
                    },
                    parameter_1=write-only i32[1] [1] InBounds,
                    mapping_1={
                        lambda %0:dimension<prefetch_0 ∈ [0, ∞)> .
                        let %1:dimension<0> = const 0
                        in (%1)
                    },
                ] %1 %0 [
                    body={
                        lambda %0:ref<i32[1]>, %1:ref<i32[1]>, %2:i32[] .
                        let %3:i32[1] = reference_read %0
                            () = reference_write %1 %3
                        in ()
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_kernel_operation_batch() {
        let definition = differentiable_definition();
        let callees = [Arc::new(stage_body(&definition).unwrap())];
        let regions = AttachedRegions::Callees(&callees);
        let driver = RecursiveBatchingDriver::new(&regions);
        let vector = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3]));
        let (_, program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                let parent = inputs[0].context();
                let extent = parent.constant(Constant::Dimension(DimensionValue::constant(3).unwrap()));
                let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), extent);
                let outputs = KernelOperation::new(PortableKernelOperation::<NoKernelExtension>::Call(
                    definition.operation().clone(),
                ))
                .batch(&context, &driver, &[ArrayIrBatch::new(inputs[0].clone(), Some(0))?])?;
                Ok::<_, ProgramError>(outputs.into_iter().map(ArrayIrBatch::into_value).collect::<Vec<_>>())
            },
            vec![vector],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:dimension<3> = const 3
                    %2:f32[3] = kernel_call [
                        grid=[3:parallel],
                        identities=[0, 0],
                        identity_0_bounds=[0, 3),
                        parameter_0=read-write f32[3] [1] InBounds,
                        mapping_0={
                            lambda %0:dimension<batch ∈ [0, 3)> .
                            in (%0)
                        },
                    ] %0 [
                        body={
                            lambda %0:ref<f32[1]>, %1:dimension<batch ∈ [0, 3)> .
                            let %2:f32[] = reference_read [transforms=[index(axis=0, index=0)]] %0
                                () = reference_write [transforms=[index(axis=0, index=0)]] %0 %2
                            in ()
                        },
                    ]
                in (%2)
            "}
            .trim_end(),
        );
        let Operation::Kernel(KernelOperation(PortableKernelOperation::Call(call))) =
            program.instructions()[0].operation()
        else {
            panic!("expected a batched kernel call");
        };
        let body = program.region_ref(program.instructions()[0].regions()[0]).unwrap();
        assert_eq!(definition_from_body(call, body).unwrap().operation().input_types(), program.input_types().as_ref());
    }

    /// Batches `definition`, whose only input is an `f32[3]` item, along axis 1 of a packed `f32[3, 4]` input with
    /// `sharding` in a fresh trace whose enclosing manual regions bind `bound_axes`. Returns the staged program
    /// together with the batch axis of every result, or the batching error.
    fn batch_sharded(
        definition: &KernelDefinition,
        sharding: Sharding,
        bound_axes: Vec<(String, NamedAxis)>,
    ) -> Result<(FlatProgram, Vec<Option<usize>>), BatchingError> {
        let packed = ArrayType::new_static(DataType::F32, [3, 4]).with_sharding(sharding).unwrap();
        batch_packed(definition, packed, bound_axes)
    }

    /// Batches `definition` like [`batch_sharded`] over a packed input in `memory`, in a trace without enclosing
    /// manual regions.
    fn batch_sharded_with_memory(
        definition: &KernelDefinition,
        sharding: Sharding,
        memory: Memory,
    ) -> Result<(FlatProgram, Vec<Option<usize>>), BatchingError> {
        let packed = ArrayType::new_static(DataType::F32, [3, 4]).with_memory(memory).with_sharding(sharding).unwrap();
        batch_packed(definition, packed, Vec::new())
    }

    /// Batches `definition` along axis 1 of one input of type `packed` in a fresh trace whose enclosing manual regions
    /// bind `bound_axes`. Returns the staged program together with the batch axis of every result, or the batching
    /// error.
    fn batch_packed(
        definition: &KernelDefinition,
        packed: ArrayType,
        bound_axes: Vec<(String, NamedAxis)>,
    ) -> Result<(FlatProgram, Vec<Option<usize>>), BatchingError> {
        let callees = [Arc::new(stage_body(definition).unwrap())];
        let regions = AttachedRegions::Callees(&callees);
        let driver = RecursiveBatchingDriver::new(&regions);
        let mut result = None;
        let (_, program) = TracingContext::<Constant, Operation>::trace_with_named_axes(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                let parent = inputs[0].context();
                let extent = parent.constant(Constant::Dimension(DimensionValue::constant(4).unwrap()));
                let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), extent);
                let outputs = KernelOperation::new(PortableKernelOperation::<NoKernelExtension>::Call(
                    definition.operation().clone(),
                ))
                .batch(&context, &driver, &[ArrayIrBatch::new(inputs[0].clone(), Some(1))?]);
                Ok::<_, ProgramError>(match outputs {
                    Ok(outputs) => {
                        result = Some(Ok(outputs.iter().map(ArrayIrBatch::batch_axis_position).collect::<Vec<_>>()));
                        outputs.into_iter().map(ArrayIrBatch::into_value).collect::<Vec<_>>()
                    }
                    Err(error) => {
                        result = Some(Err(error));
                        Vec::new()
                    }
                })
            },
            vec![ArrayIrType::Array(packed)],
            bound_axes,
        )
        .unwrap();
        result.unwrap().map(|axes| (program.into_flat_program(), axes))
    }

    /// Returns a kernel over `f32[3]` items that copies its input into its output, through one read-write parameter
    /// when `in_place` is set and from a read-only parameter into a write-only parameter otherwise.
    fn item_copy_definition(in_place: bool) -> KernelDefinition {
        let item = ArrayType::new_static(DataType::F32, [3]);
        let parameters = if in_place {
            vec![whole_array_parameter(item, KernelParameterAccess::ReadWrite).unwrap()]
        } else {
            vec![
                whole_array_parameter(item.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(item, KernelParameterAccess::WriteOnly).unwrap(),
            ]
        };
        let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), parameters).unwrap();
        KernelDefinition::trace(call, |(references, _)| references[references.len() - 1].write(&references[0].read()?))
            .unwrap()
    }

    #[test]
    fn test_kernel_operation_batch_sharded_batch_axes() {
        // The batch axis is placed on the mesh axis `x`, which no enclosing manual region binds. The rule stages a
        // `shard_map` over `x` whose body batches the kernel over the local batch of each device, because a kernel
        // runs on one device and cannot be partitioned.
        let stage = |axis_type: MeshAxisType, in_place: bool| {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let sharding =
                Sharding::new(mesh, vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])]).unwrap();
            let (program, axes) = batch_sharded(&item_copy_definition(in_place), sharding, Vec::new()).unwrap();
            assert_eq!(axes, vec![Some(if in_place { 1 } else { 0 })]);
            program
        };
        // A caller that types `x` as explicit receives results typed over its own mesh. The body applies the physical
        // kernel to the local shard, and the write-only result gains the batch as its leading dimension, which the
        // output sharding of the map places like the batch axis of the input.
        assert_eq!(
            stage(MeshAxisType::Explicit, false).to_string(),
            indoc! {"
                lambda %0:f32[3, 4][sharding={mesh<['x'=2:explicit]>, [{}, {'x'}]}] .
                let %1:dimension<4> = const 4
                    %2:f32[4, 3][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] = shard_map [
                        mesh=['x'=2:manual],
                        caller_mesh=['x'=2:explicit],
                        input_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}],
                        output_shardings=[{mesh<['x'=2:manual]>, [{'x'}, {}]}],
                        manual_axes=['x'],
                        global_input_types=[f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}]],
                        global_output_types=[f32[4, 3][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}]],
                    ] %0 [
                        body={
                            lambda %0:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                            let %1:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = kernel_call [
                                grid=[2:parallel],
                                identities=[0, 0, 0],
                                identity_0_bounds=[0, 2),
                                parameter_0=read-only f32[3, 2] [3, 1] InBounds,
                                mapping_0={
                                    lambda %0:dimension<batch ∈ [0, 2)> .
                                    let %1:dimension<0> = const 0
                                    in (%1, %0)
                                },
                                parameter_1=write-only f32[2, 3] [1, 3] InBounds,
                                mapping_1={
                                    lambda %0:dimension<batch ∈ [0, 2)> .
                                    let %1:dimension<0> = const 0
                                    in (%0, %1)
                                },
                            ] %0 [
                                body={
                                    lambda %0:ref<f32[3, 1]>, %1:ref<f32[1, 3]>, %2:dimension<batch ∈ [0, 2)> .
                                    let %3:f32[3] = reference_read [transforms=[index(axis=1, index=0)]] %0
                                        () = reference_write [transforms=[index(axis=0, index=0)]] %1 %3
                                    in ()
                                },
                            ]
                            in (%1)
                        },
                    ]
                in (%2)
            "}
            .trim_end(),
        );

        // A read-write parameter keeps the batch axis of its input.
        let manual = stage(MeshAxisType::Manual, true);
        assert_eq!(
            manual.to_string(),
            indoc! {"
                lambda %0:f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}] .
                let %1:dimension<4> = const 4
                    %2:f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}] = shard_map [
                        mesh=['x'=2:manual],
                        input_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}],
                        output_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}],
                        manual_axes=['x'],
                        global_input_types=[f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}]],
                        global_output_types=[f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}]],
                    ] %0 [
                        body={
                            lambda %0:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                            let %1:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = kernel_call [
                                grid=[2:parallel],
                                identities=[0, 0],
                                identity_0_bounds=[0, 2),
                                parameter_0=read-write f32[3, 2] [3, 1] InBounds,
                                mapping_0={
                                    lambda %0:dimension<batch ∈ [0, 2)> .
                                    let %1:dimension<0> = const 0
                                    in (%1, %0)
                                },
                            ] %0 [
                                body={
                                    lambda %0:ref<f32[3, 1]>, %1:dimension<batch ∈ [0, 2)> .
                                    let %2:f32[3] = reference_read [transforms=[index(axis=1, index=0)]] %0
                                        () = reference_write [transforms=[index(axis=1, index=0)]] %0 %2
                                    in ()
                                },
                            ]
                            in (%1)
                        },
                    ]
                in (%2)
            "}
            .trim_end(),
        );

        // The local shards vary only along the axis that the map binds, so the compiler binding selects the kernel
        // inside the map, and the custom call types its result like the kernel call did.
        let binding = binding(1);
        assert_eq!(
            select_kernels(&manual, Some(&binding), facts).unwrap().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}] .
                let %1:dimension<4> = const 4
                    %2:f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}] = shard_map [
                        mesh=['x'=2:manual],
                        input_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}],
                        output_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}],
                        manual_axes=['x'],
                        global_input_types=[f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}]],
                        global_output_types=[f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}]],
                    ] %0 [
                        body={
                            lambda %0:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                            let %1:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = custom_call [
                                target=ryft.test.aot,
                                ryft.kernel.schema=1,
                                ryft.kernel.semantic=\"93588cd1e2b84de11eb6ee99e1bebae728ebb6f7248d90960d547303721a0aa2\",
                                ryft.kernel.configuration=\"515ef62b84d55eb9cf36a09e7cd9bbbc71b81d00380ee8324af729c3e7766e27\",
                                input_output_alias=0->0,
                            ] %0
                            in (%1)
                        },
                    ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            select_kernels(&stage(MeshAxisType::Explicit, true), Some(&binding), facts).unwrap().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[3, 4][sharding={mesh<['x'=2:explicit]>, [{}, {'x'}]}] .
                let %1:dimension<4> = const 4
                    %2:f32[3, 4][sharding={mesh<['x'=2:explicit]>, [{}, {'x'}]}] = shard_map [
                        mesh=['x'=2:manual],
                        caller_mesh=['x'=2:explicit],
                        input_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}],
                        output_shardings=[{mesh<['x'=2:manual]>, [{}, {'x'}]}],
                        manual_axes=['x'],
                        global_input_types=[f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {'x'}]}]],
                        global_output_types=[f32[3, 4][sharding={mesh<['x'=2:explicit]>, [{}, {'x'}]}]],
                    ] %0 [
                        body={
                            lambda %0:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                            let %1:f32[3, 2][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = custom_call [
                                target=ryft.test.aot,
                                ryft.kernel.schema=1,
                                ryft.kernel.semantic=\"93588cd1e2b84de11eb6ee99e1bebae728ebb6f7248d90960d547303721a0aa2\",
                                ryft.kernel.configuration=\"515ef62b84d55eb9cf36a09e7cd9bbbc71b81d00380ee8324af729c3e7766e27\",
                                input_output_alias=0->0,
                            ] %0
                            in (%1)
                        },
                    ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_kernel_operation_batch_bound_and_mixed_batch_axes() {
        // Batches an in-place kernel in a trace whose enclosing manual regions bind the mesh axes named `bound`, and
        // returns the staged operation or the batching error.
        let mesh = |x: MeshAxisType| {
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2, x).unwrap(),
                MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            ])
            .unwrap()
        };
        let batch = |sharding: Sharding, bound: &[&str], checks_manual_variation: bool| {
            let bound_axes = bound
                .iter()
                .map(|name| {
                    let mesh = sharding.mesh().clone();
                    let axis = mesh.axis_index(name).unwrap();
                    (name.to_string(), NamedAxis::Mesh { mesh, axis, size: 2, checks_manual_variation })
                })
                .collect();
            batch_sharded(&item_copy_definition(true), sharding, bound_axes)
                .map(|(program, _)| program.instructions()[0].operation().clone())
        };
        let sharded = |mesh: LogicalMesh, axes: &[&str]| {
            Sharding::new(mesh, vec![ShardingDimension::Replicated, ShardingDimension::sharded(axes.iter().copied())])
                .unwrap()
        };

        // A batch that is placed on mesh axes that an enclosing manual region binds is the local batch of that region,
        // so the kernel is batched in place, whether or not the region tracks manual variation.
        let local = sharded(mesh(MeshAxisType::Manual), &["x"]);
        assert!(matches!(
            batch(local.clone().with_varying_manual_axes(["x"]).unwrap(), &["x"], true),
            Ok(Operation::Kernel(_)),
        ));
        assert!(matches!(batch(local, &["x"], false), Ok(Operation::Kernel(_))));

        // A global batch is wrapped in a `shard_map` over the axes that it is placed on, including when an enclosing
        // manual region binds another axis.
        assert!(matches!(
            batch(sharded(mesh(MeshAxisType::Manual), &["x", "y"]), &[], true),
            Ok(Operation::ShardMap(_))
        ));
        assert!(matches!(batch(sharded(mesh(MeshAxisType::Manual), &["x"]), &["y"], true), Ok(Operation::ShardMap(_))));

        // The item dimension is placed on `y`, which the map over the batch axis `x` does not make manual. One device
        // would hold only a part of each item, unless an enclosing manual region binds `y`.
        let split = Sharding::new(
            mesh(MeshAxisType::Manual),
            vec![ShardingDimension::sharded(["y"]), ShardingDimension::sharded(["x"])],
        )
        .unwrap();
        assert!(matches!(
            batch(split.clone(), &[], true),
            Err(BatchingError::UnsupportedOperation { message }) if message == unbound_kernel_axis_message(0, "y"),
        ));
        assert!(matches!(batch(split, &["y"], true), Ok(Operation::ShardMap(_))));

        // A placement of which an enclosing manual region binds only a part, or that mixes explicit and manual axes,
        // is neither.
        assert!(matches!(
            batch(sharded(mesh(MeshAxisType::Manual), &["x", "y"]), &["x"], true),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "kernel batch axis is placed on {'x', 'y'}, of which an enclosing manual region binds \
                               only a part",
        ));
        assert!(matches!(
            batch(sharded(mesh(MeshAxisType::Explicit), &["x", "y"]), &[], true),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "kernel batch axis is placed on {'x', 'y'}, which mixes explicit and manual mesh axes",
        ));
    }

    #[test]
    fn test_kernel_operation_batch_sharded_constant_store() {
        // The body stores a constant, which is the same on every device, into the window of each batch item. Kernel
        // types are physical, so the constant carries no mesh and fits the local kernel of every device.
        let item = ArrayType::new_static(DataType::F32, [3]);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(item.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(item.clone(), KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let mut body = CoreProgramBuilder::<ArrayIrValue<CoreArray>, PortableKernelOperation>::new();
        body.add_input(call.parameters()[0].body_type());
        let output = body.add_input(call.parameters()[1].body_type());
        let value = CoreArray::from_elements(item, &[1.0f32, 2.0, 3.0]).unwrap();
        let value = body.add_constant(ArrayIrValue::Array(value));
        body.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, value], None).unwrap();
        let definition =
            KernelDefinition::new(call, body.build(vec![], vec![Placeholder; 2], vec![]).unwrap()).unwrap();
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let sharding =
                Sharding::new(mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])])
                    .unwrap();
            let (program, axes) = batch_sharded(&definition, sharding, Vec::new()).unwrap();
            assert_eq!(axes, vec![Some(0)]);
            assert!(matches!(program.instructions()[0].operation(), Operation::ShardMap(_)));
            let output_sharding =
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]).unwrap();
            assert_eq!(
                program.output_types().as_ref(),
                &[ArrayIrType::Array(
                    ArrayType::new_static(DataType::F32, [4, 3]).with_sharding(output_sharding).unwrap()
                )],
            );
        }
    }

    #[cfg(feature = "triton")]
    #[test]
    fn test_kernel_operation_batch_sharded_batch_axes_triton_admission() {
        // The kernel that the staged `shard_map` applies on each device has physical parameter types, so the Triton
        // lowering admits it. Only admission runs here, and nothing is compiled.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])]).unwrap();
        let (program, _) = batch_sharded(&item_copy_definition(false), sharding, Vec::new()).unwrap();
        let body = program.region_ref(program.instructions()[0].regions()[0]).unwrap();
        let Operation::Kernel(KernelOperation(PortableKernelOperation::Call(call))) =
            body.instructions()[0].operation()
        else {
            panic!("expected a kernel call in the body of the staged `shard_map`");
        };
        let local =
            definition_from_body(call, program.region_ref(body.instructions()[0].regions()[0]).unwrap()).unwrap();
        let local = KernelDefinition::new(
            local.operation().clone(),
            local
                .body()
                .map_operations(|operation| operation.clone().map_extension(NoKernelExtension::try_from))
                .unwrap(),
        )
        .unwrap();
        let verified = VerifiedKernel::new(&local, 16).unwrap();
        let admission = ryft_triton::kernels::Compiler::new().admit(
            &verified,
            &ryft_triton::kernels::Target::Cuda { major: 12, minor: 1 },
            &ryft_triton::kernels::Options::default(),
            &KernelSchedule::default(),
        );

        // Admission lowers the kernel before it asks the linked native archive for the target, so an archive without
        // the CUDA target reports that it is unavailable only after the lowering admitted the kernel.
        match admission {
            Ok(()) => {}
            Err(ryft_kernels::KernelCompilationError::Unavailable { message }) => {
                assert_eq!(message, "linked native archive does not include the requested Triton compiler target");
            }
            Err(error) => panic!("Triton admission rejected the local kernel: {error}"),
        }
    }

    #[test]
    fn test_kernel_operation_batch_sharded_memory() {
        // The kernel declares pinned host memory. The packed input must have that memory, and the local kernel of
        // each device keeps it.
        let memory = Memory::Host { pinned: true };
        let definition = |memory: Memory| {
            let item = ArrayType::new_static(DataType::F32, [3]).with_memory(memory);
            let call = KernelCallOperation::new(
                Grid::new(vec![]).unwrap(),
                vec![whole_array_parameter(item, KernelParameterAccess::ReadWrite).unwrap()],
            )
            .unwrap();
            let definition: KernelDefinition =
                KernelDefinition::trace(call, |(references, _)| references[0].write(&references[0].read()?)).unwrap();
            definition
        };
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])]).unwrap();
        let (program, _) = batch_sharded_with_memory(&definition(memory), sharding.clone(), memory).unwrap();
        let body = program.region_ref(program.instructions()[0].regions()[0]).unwrap();
        let Operation::Kernel(KernelOperation(PortableKernelOperation::Call(call))) =
            body.instructions()[0].operation()
        else {
            panic!("expected a kernel call in the body of the staged `shard_map`");
        };
        assert_eq!(call.parameters()[0].r#type().memory(), memory);

        // A packed input in another memory than the kernel declares is not an input of that kernel.
        let error = batch_sharded_with_memory(&definition(Memory::Device), sharding, memory).unwrap_err();
        assert_eq!(
            error.to_string(),
            "kernel transform is unsupported: batched input types differ from the kernel signature",
        );
    }

    #[test]
    fn test_kernel_operation_batch_sharded_prefetch() {
        // The kernel has no parameters and one scalar prefetched input, whose packed values are placed on `x`. A
        // prefetched input is not a parameter, so no kernel can be batched per device, and batching reports that the
        // input must be specialized first.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let scalar = ArrayType::scalar(DataType::I32);
        let definition: KernelDefinition = KernelDefinition::trace_with_prefetch(
            KernelCallOperation::new_with_prefetch(Grid::new(vec![]).unwrap(), vec![], vec![scalar]).unwrap(),
            |_| Ok(()),
        )
        .unwrap();
        let callees = [Arc::new(stage_body(&definition).unwrap())];
        let regions = AttachedRegions::Callees(&callees);
        let driver = RecursiveBatchingDriver::new(&regions);
        let packed = ArrayType::new_static(DataType::I32, [4])
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let mut error = None;
        TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                let parent = inputs[0].context();
                let extent = parent.constant(Constant::Dimension(DimensionValue::constant(4).unwrap()));
                let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), extent);
                error = KernelOperation::new(PortableKernelOperation::<NoKernelExtension>::Call(
                    definition.operation().clone(),
                ))
                .batch(&context, &driver, &[ArrayIrBatch::new(inputs[0].clone(), Some(0))?])
                .err();
                Ok::<_, ProgramError>(Vec::<CoreTracer<TracingContext<Constant, Operation>>>::new())
            },
            vec![ArrayIrType::Array(packed)],
        )
        .unwrap();
        assert_eq!(
            error.map(|error| error.to_string()),
            Some(
                "kernel transform is unsupported: scalar-prefetched inputs must be specialized before batching"
                    .to_owned()
            ),
        );
    }

    #[test]
    fn test_stage_kernel() {
        let program = program();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:i32[] .
                let %1:i32[] = kernel_call [
                    grid=[],
                    parameter_0=read-write i32[] [] InBounds,
                    mapping_0={
                        lambda  .
                        in ()
                    },
                ] %0 [
                    body={
                        lambda %0:ref<i32[]> .
                        let %1:i32[] = reference_read %0
                            () = reference_write %0 %1
                        in ()
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert!(program.effects().classes().contains(EffectClass::OrderedState));
    }

    #[test]
    fn test_stage_kernel_rejects_implicit_differentiation() {
        let program = kernel_primal(&differentiable_definition()).unwrap();
        assert_eq!(
            program.jvp().unwrap_err(),
            DifferentiationError::from(ProgramError::MalformedProgram(
                "unselected kernel operation `kernel_call` has no differentiation rule".to_owned(),
            ))
        );
    }

    #[test]
    fn test_stage_kernel_unused_outputs_preserve_assertions() {
        let logical = crate::kernels::tests::definition().operation().clone();
        let definition = KernelDefinition::<NoKernelExtension>::trace(logical, |(references, _)| {
            let value = references[0].read()?;
            references[0].context().bind(
                ArrayIrOperation::DimensionFromScalar(DimensionFromScalarOperation::new(
                    DimensionVariable::new("checked", DimensionBounds::new(0, Some(10)).unwrap()).into(),
                )),
                vec![],
                std::slice::from_ref(&value),
            )?;
            references[0].write(&value)?;
            Ok(())
        })
        .unwrap();
        let (_, program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                // The complete kernel result is deliberately unused by its enclosing computation.
                stage_kernel(inputs[0].context(), &definition, &inputs)?;
                Ok(Vec::<CoreTracer<TracingContext<Constant, Operation>>>::new())
            },
            definition.operation().input_types(),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:i32[] .
                let %1:i32[] = kernel_call [
                    grid=[],
                    parameter_0=read-write i32[] [] InBounds,
                    mapping_0={
                        lambda  .
                        in ()
                    },
                ] %0 [
                    body={
                        lambda %0:ref<i32[]> .
                        let %1:i32[] = reference_read %0
                            %2:dimension<checked ∈ [0, 10)> = dimension_from_scalar [bounds=[0, 10)] %1
                            () = reference_write %0 %1
                        in ()
                    },
                ]
                in ()
            "}
            .trim_end(),
        );
        assert!(program.effects().classes().contains(EffectClass::OrderedAssertion));
        let simplified = program.clone().into_simplified().unwrap();
        assert_eq!(simplified.to_string(), program.to_string());
        assert!(simplified.effects().classes().contains(EffectClass::OrderedAssertion));
    }

    #[test]
    fn test_stage_kernel_unused_outputs_preserve_mapping_assertions() {
        let definition = crate::kernels::tests::mapping_assertion_definition();
        let (_, program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                // The complete kernel result is deliberately unused by its enclosing computation.
                stage_kernel(inputs[0].context(), &definition, &inputs)?;
                Ok(Vec::<CoreTracer<TracingContext<Constant, Operation>>>::new())
            },
            definition.operation().input_types(),
        )
        .unwrap();
        assert!(program.effects().classes().contains(EffectClass::OrderedAssertion));
        let simplified = program.clone().into_simplified().unwrap();
        assert_eq!(simplified.to_string(), program.to_string());

        // Selection requires an embedding that preserves the assertion of the mapping.
        let binding = KernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            1,
            KernelSchedule::default(),
            Embedding("ryft.test.aot"),
            2,
        )
        .unwrap();
        assert!(matches!(
            select_kernels(&program, Some(&binding), facts),
            Err(KernelEmbeddingError::UnsupportedEffect { effect: EffectClass::OrderedAssertion }),
        ));
    }

    #[test]
    fn test_stage_kernel_with_jvp() {
        let definition = differentiable_definition();
        let scalar = definition.operation().input_types()[0].clone();
        let mut builder = ProgramBuilder::new();
        let primal = builder.add_input(scalar.clone());
        let tangent = builder.add_input(scalar.clone());
        let doubled = builder
            .add_instruction(
                ryft_core::ArrayOperation::Add(ryft_core::AddOperation::<ryft_core::ArrayType>::new()),
                vec![],
                vec![tangent, tangent],
                None,
            )
            .unwrap()[0];
        let rule = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![primal, doubled], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let (_, program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                stage_kernel_with_jvp(inputs[0].context(), &definition, &rule, &inputs)
            },
            vec![scalar],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = composite %0 [
                    primal={
                        lambda %0:f32[] .
                        let %1:f32[] = kernel_call [
                            grid=[],
                            parameter_0=read-write f32[] [] InBounds,
                            mapping_0={
                                lambda  .
                                in ()
                            },
                        ] %0 [
                            body={
                                lambda %0:ref<f32[]> .
                                let %1:f32[] = reference_read %0
                                    () = reference_write %0 %1
                                in ()
                            },
                        ]
                        in (%1)
                    },
                    jvp={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = add %1 %1
                        in (%0, %2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(execute_derivative(&program.jvp().unwrap(), &[3.0, 4.0]), vec![3.0, 8.0]);
    }

    #[test]
    fn test_stage_kernel_with_vjp() {
        let definition = differentiable_definition();
        let scalar = definition.operation().input_types()[0].clone();
        let mut forward = ProgramBuilder::new();
        let input = forward.add_input(scalar.clone());
        let forward = forward
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut backward = ProgramBuilder::new();
        let cotangent = backward.add_input(scalar.clone());
        let doubled = backward
            .add_instruction(
                ryft_core::ArrayOperation::Add(ryft_core::AddOperation::<ryft_core::ArrayType>::new()),
                vec![],
                vec![cotangent, cotangent],
                None,
            )
            .unwrap()[0];
        let tripled = backward
            .add_instruction(
                ryft_core::ArrayOperation::Add(ryft_core::AddOperation::<ryft_core::ArrayType>::new()),
                vec![],
                vec![doubled, cotangent],
                None,
            )
            .unwrap()[0];
        let backward = backward
            .build::<Vec<Constant>, Vec<Constant>>(vec![tripled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let (_, program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                stage_kernel_with_vjp(inputs[0].context(), &definition, &forward, &backward, &inputs)
            },
            vec![scalar],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = composite %0 [
                    primal={
                        lambda %0:f32[] .
                        let %1:f32[] = kernel_call [
                            grid=[],
                            parameter_0=read-write f32[] [] InBounds,
                            mapping_0={
                                lambda  .
                                in ()
                            },
                        ] %0 [
                            body={
                                lambda %0:ref<f32[]> .
                                let %1:f32[] = reference_read %0
                                    () = reference_write %0 %1
                                in ()
                            },
                        ]
                        in (%1)
                    },
                    forward={
                        lambda %0:f32[] .
                        in (%0)
                    },
                    backward={
                        lambda %0:f32[] .
                        let %1:f32[] = add %0 %0
                            %2:f32[] = add %1 %0
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        let context = TracingContext::<Constant, Operation>::new();
        let input = context.input(program.input_types()[0].clone());
        let (_, pullback) = context
            .vjp(|input, ()| program.interpret_in_context(input.context(), vec![input.clone()]), input, ())
            .unwrap();
        let backward = pullback.transposed_program(&[CotangentDestinationKind::Return]).unwrap().into_program();
        assert_eq!(execute_derivative(&backward, &[2.0]), vec![6.0]);
    }

    #[test]
    fn test_stage_kernel_with_rules() {
        // A rule set registered in the XLA family, whose rules are written against XLA tracers, and one registered in
        // the array member family, whose rules are written against array tracers, both double the tangent of the
        // identity kernel.
        let definition = differentiable_definition();
        let scalar = definition.operation().input_types()[0].clone();
        let native = CustomRuleRegistration::new(CustomRuleDefinition::<Constant, Operation>::new("double").with_jvp(
            |primals, tangents| {
                let doubled = tangents[0].context().bind(
                    ryft_core::ArrayOperation::Add(ryft_core::AddOperation::<ryft_core::ArrayType>::new()),
                    Vec::new(),
                    &[tangents[0].clone(), tangents[0].clone()],
                )?;
                Ok((primals.to_vec(), doubled))
            },
        ));
        let member = CustomRuleRegistration::new(
            CustomRuleDefinition::<ArrayConstant, ArrayOperation<ArrayConstant>>::new("double")
                .with_jvp(|primals, tangents| Ok((primals.to_vec(), vec![tangents[0].clone() + tangents[0].clone()]))),
        );
        let (_, native_program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                stage_kernel_with_rules(inputs[0].context(), &definition, native.reference(), &inputs)
            },
            vec![scalar.clone()],
        )
        .unwrap();
        let (_, member_program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                stage_kernel_with_rules(
                    inputs[0].context(),
                    &definition,
                    LiftedCustomRules::Member(member.reference()),
                    &inputs,
                )
            },
            vec![scalar],
        )
        .unwrap();
        // Both calls render the same, because rendering does not show where a call's rules come from, so only the
        // operation variants distinguish the definition registered in this family from the lifted member definition.
        assert!(matches!(native_program.instructions()[0].operation(), Operation::Composite(_)));
        assert!(matches!(member_program.instructions()[0].operation(), Operation::LiftedComposite(_)));
        for program in [native_program, member_program] {
            assert_eq!(
                program.to_string(),
                indoc! {"
                    lambda %0:f32[] .
                    let %1:f32[] = composite [name=\"double\"] %0 [
                        primal={
                            lambda %0:f32[] .
                            let %1:f32[] = kernel_call [
                                grid=[],
                                parameter_0=read-write f32[] [] InBounds,
                                mapping_0={
                                    lambda  .
                                    in ()
                                },
                            ] %0 [
                                body={
                                    lambda %0:ref<f32[]> .
                                    let %1:f32[] = reference_read %0
                                        () = reference_write %0 %1
                                    in ()
                                },
                            ]
                            in (%1)
                        },
                    ]
                    in (%1)
                "}
                .trim_end(),
            );
            assert_eq!(execute_derivative(&program.jvp().unwrap(), &[3.0, 4.0]), vec![3.0, 8.0]);
        }
    }

    #[test]
    fn test_stage_kernel_with_fallback() {
        let definition = differentiable_definition();
        let scalar = definition.operation().input_types()[0].clone();
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(scalar.clone());
        let fallback = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let (_, program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                stage_kernel_with_fallback(inputs[0].context(), &definition, &fallback, &inputs)
            },
            vec![scalar],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = composite %0 [
                    primal={
                        lambda %0:f32[] .
                        let %1:f32[] = kernel_call [
                            grid=[],
                            parameter_0=read-write f32[] [] InBounds,
                            mapping_0={
                                lambda  .
                                in ()
                            },
                        ] %0 [
                            body={
                                lambda %0:ref<f32[]> .
                                let %1:f32[] = reference_read %0
                                    () = reference_write %0 %1
                                in ()
                            },
                        ]
                        in (%1)
                    },
                    jvp={
                        lambda %0:f32[], %1:f32[] .
                        in (%0, %1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(execute_derivative(&program.jvp().unwrap(), &[3.0, 4.0]), vec![3.0, 4.0]);
    }

    #[test]
    fn test_stage_kernel_with_fallback_placed_inputs() {
        // The input is replicated on a mesh. Ordinary staging accepts it, and so does the fallback staging, which
        // stages the fallback and its derivative rule for the placed type. The result has the type of the input.
        let definition = differentiable_definition();
        let scalar = definition.operation().input_types()[0].clone();
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(scalar);
        let fallback = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let placed = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        let placed = ArrayIrType::Array(placed);
        for stage in [true, false] {
            let (output_types, program) = TracingContext::<Constant, Operation>::trace(
                |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                    if stage {
                        stage_kernel(inputs[0].context(), &definition, &inputs)
                    } else {
                        stage_kernel_with_fallback(inputs[0].context(), &definition, &fallback, &inputs)
                    }
                },
                vec![placed.clone()],
            )
            .unwrap();
            assert_eq!(output_types, vec![placed.clone()]);
            assert_eq!(program.instructions().len(), 1);
        }
    }

    #[test]
    fn test_stage_kernel_with_fallback_rejects_changed_signature() {
        let definition = differentiable_definition();
        let scalar = definition.operation().input_types()[0].clone();
        let mut fallback = ProgramBuilder::new();
        let input = fallback.add_input(ArrayType::scalar(DataType::F64).into());
        let fallback = fallback
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                stage_kernel_with_fallback(inputs[0].context(), &definition, &fallback, &inputs)
            },
            vec![scalar],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::Type(TypeError::invalid("pure kernel fallback must preserve the complete signature")),
        );
    }

    #[test]
    fn test_stage_kernel_with_fallback_rejects_effectful_fallback() {
        let definition = differentiable_definition();
        let scalar = definition.operation().input_types()[0].clone();
        let mut fallback = ProgramBuilder::new();
        let input = fallback.add_input(scalar.clone());
        let predicate = fallback
            .add_instruction(ConstantOperation::new(CoreArray::scalar(true).unwrap()), vec![], vec![], None)
            .unwrap()[0];
        fallback
            .add_instruction(AssertOperation::<ArrayIrType>::new("check"), vec![], vec![predicate], None)
            .unwrap();
        let fallback = fallback
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                stage_kernel_with_fallback(inputs[0].context(), &definition, &fallback, &inputs)
            },
            vec![scalar],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::UnsupportedOperation {
                message: "pure kernel fallback cannot contain references or observable effects".to_owned(),
            },
        );
    }

    #[test]
    fn test_stage_kernel_with_fallback_rejects_external_state() {
        // The body calls an external function that declares state. A pure fallback would replace the primal under
        // differentiation, and that call with it, although only the private reference state of a body may be dropped.
        let definition = differentiable_definition();
        let scalar = definition.operation().input_types()[0].clone();
        let mut body = CoreProgramBuilder::<ArrayIrValue<CoreArray>, PortableKernelOperation>::new();
        let inputs = definition
            .body()
            .input_types()
            .into_owned()
            .into_iter()
            .map(|r#type| body.add_input(r#type))
            .collect::<Vec<_>>();
        body.splice_program(definition.body(), &inputs).unwrap();
        body.add_instruction(
            ArrayOperation::<CoreArray>::CustomCall(
                CustomCallOperation::new("test.external", vec![]).with_effect_class(EffectClass::OrderedState),
            ),
            vec![],
            vec![],
            None,
        )
        .unwrap();
        let definition = KernelDefinition::new(
            definition.operation().clone(),
            body.build(vec![], vec![Placeholder], vec![]).unwrap(),
        )
        .unwrap();
        assert!(definition.effects().has_explicit_ordered_state());
        let mut fallback = ProgramBuilder::new();
        let input = fallback.add_input(scalar.clone());
        let fallback = fallback
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                stage_kernel_with_fallback(inputs[0].context(), &definition, &fallback, &inputs)
            },
            vec![scalar],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::UnsupportedOperation {
                message: "pure kernel fallback cannot replace observable kernel effects".to_owned(),
            }
        );
    }

    #[test]
    fn test_stage_kernel_with_fallback_rejects_fallible_tile_loads() {
        // The body loads the tile of extent one at start 4 of its `f32[4]` parameter, which lies beyond the parameter
        // and fails when the kernel runs. An identity fallback would drop that failure under differentiation. A load
        // at start 3 is proved valid and leaves the kernel free of observable effects.
        let stage = |start: usize| {
            let vector = ArrayType::new_static(DataType::F32, [4]);
            let call = KernelCallOperation::new(
                Grid::new(vec![]).unwrap(),
                vec![whole_array_parameter(vector.clone(), KernelParameterAccess::ReadWrite).unwrap()],
            )
            .unwrap();
            let input_types = call.body_input_types();
            let start = DimensionValue::constant(start).unwrap();
            let operation = TileLoadOperation::new(vec![1], BoundaryPolicy::InBounds)
                .unwrap()
                .with_input_types(&[
                    input_types[0].clone(),
                    ArrayIrType::Dimension(start.r#type().into_owned()),
                    ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                ])
                .unwrap();
            let mut body = CoreProgramBuilder::<ArrayIrValue<CoreArray>, PortableKernelOperation>::new();
            let reference = body.add_input(input_types[0].clone());
            let start = body.add_constant(ArrayIrValue::Dimension(start));
            let other = body.add_constant(ArrayIrValue::Array(CoreArray::scalar(-1f32).unwrap()));
            body.add_instruction(operation, vec![], vec![reference, start, other], None).unwrap();
            let definition: KernelDefinition =
                KernelDefinition::new(call, body.build(vec![], vec![Placeholder], vec![]).unwrap()).unwrap();
            let mut fallback = ProgramBuilder::new();
            let input = fallback.add_input(ArrayIrType::Array(vector.clone()));
            let fallback = fallback
                .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap();
            TracingContext::<Constant, Operation>::trace(
                |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                    stage_kernel_with_fallback(inputs[0].context(), &definition, &fallback, &inputs)
                },
                vec![ArrayIrType::Array(vector)],
            )
            .map(|_| ())
        };
        assert_eq!(
            stage(4),
            Err(ProgramError::UnsupportedOperation {
                message: "pure kernel fallback cannot replace observable kernel effects".to_owned(),
            }),
        );
        assert_eq!(stage(3), Ok(()));
    }

    #[test]
    fn test_stage_kernel_with_fallback_rejects_observable_primal_effects() {
        let definition = differentiable_definition();
        let scalar = definition.operation().input_types()[0].clone();
        let mut body = CoreProgramBuilder::<ArrayIrValue<CoreArray>, PortableKernelOperation>::new();
        let inputs = definition
            .body()
            .input_types()
            .into_owned()
            .into_iter()
            .map(|r#type| body.add_input(r#type))
            .collect::<Vec<_>>();
        body.splice_program(definition.body(), &inputs).unwrap();
        let predicate = body.add_constant(ArrayIrValue::Array(CoreArray::scalar(false).unwrap()));
        body.add_instruction(
            PortableKernelOperation::from(ArrayIrOperation::Assert(AssertOperation::new("check"))),
            vec![],
            vec![predicate],
            None,
        )
        .unwrap();
        let definition = KernelDefinition::new(
            definition.operation().clone(),
            body.build(vec![], vec![Placeholder], vec![]).unwrap(),
        )
        .unwrap();
        let mut fallback = ProgramBuilder::new();
        let input = fallback.add_input(scalar.clone());
        let fallback = fallback
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                stage_kernel_with_fallback(inputs[0].context(), &definition, &fallback, &inputs)
            },
            vec![scalar],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::UnsupportedOperation {
                message: "pure kernel fallback cannot replace observable kernel effects".to_owned(),
            }
        );
    }

    #[test]
    fn test_stage_kernel_with_fallback_rejects_mapping_assertions() {
        let definition = crate::kernels::tests::mapping_assertion_definition();
        let vector = definition.operation().input_types()[0].clone();
        let mut fallback = ProgramBuilder::new();
        let input = fallback.add_input(vector.clone());
        let fallback = fallback
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                stage_kernel_with_fallback(inputs[0].context(), &definition, &fallback, &inputs)
            },
            vec![vector],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::UnsupportedOperation {
                message: "pure kernel fallback cannot replace observable kernel effects".to_owned(),
            }
        );
    }

    #[test]
    fn test_kernel_execution_facts_from_target() {
        let client = Platform::Cpu
            .client(Some(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() })))
            .unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = ryft_core::DeviceMesh::new(
            ryft_core::LogicalMesh::new(vec![]).unwrap(),
            devices.iter().map(|device| ryft_core::Device::from_pjrt(device.clone()).unwrap()).collect(),
        )
        .unwrap();
        let facts = KernelExecutionFacts::from_target(&Target::from_client(client).unwrap(), &mesh).unwrap();
        assert_eq!(facts.platform_name, client.platform_name().unwrap());
        assert_eq!(facts.platform_version, client.platform_version().unwrap());
        assert_eq!(facts.pjrt_version, client.version());
        assert_eq!(facts.has_ffi_extension, client.ffi_extension().is_ok());
        assert_eq!(facts.devices.len(), 1);
        assert_eq!(facts.devices[0].kind, devices[0].kind().unwrap());
        assert_eq!(
            facts.devices[0].attributes,
            devices[0].attributes().unwrap().iter().map(|(name, value)| (name.clone(), value.clone())).collect()
        );
    }

    #[test]
    fn test_kernel_execution_facts_configuration_key() {
        let first = facts().unwrap();
        let key = first.configuration_key().unwrap();
        assert_eq!(key, first.clone().configuration_key().unwrap());
        let mut changed = first.clone();
        changed.platform_name = "other".to_owned();
        assert_ne!(key, changed.configuration_key().unwrap());
        changed = first.clone();
        changed.platform_version = "2".to_owned();
        assert_ne!(key, changed.configuration_key().unwrap());
        changed = first.clone();
        changed.pjrt_version.minor += 1;
        assert_ne!(key, changed.configuration_key().unwrap());
        changed = first.clone();
        changed.has_ffi_extension = false;
        assert_ne!(key, changed.configuration_key().unwrap());
        changed = first.clone();
        changed
            .devices
            .push(KernelDeviceFacts { kind: "device".to_owned(), attributes: Default::default() });
        assert_ne!(key, changed.configuration_key().unwrap());
        let before_attribute = changed.configuration_key().unwrap();
        changed.devices[0]
            .attributes
            .insert("compute_capability".to_owned(), ryft_pjrt::Value::String("12.1".to_owned()));
        assert_ne!(before_attribute, changed.configuration_key().unwrap());
        for (first_value, second_value) in [
            (ryft_pjrt::Value::Bool(true), ryft_pjrt::Value::String("true".to_owned())),
            (ryft_pjrt::Value::I64(1), ryft_pjrt::Value::I64List(vec![1])),
            (ryft_pjrt::Value::F32(f32::from_bits(0x7fc0_0001)), ryft_pjrt::Value::F32(f32::from_bits(0x7fc0_0002))),
        ] {
            changed.attributes.insert("value".to_owned(), first_value);
            let first_key = changed.configuration_key().unwrap();
            changed.attributes.insert("value".to_owned(), second_value);
            assert_ne!(first_key, changed.configuration_key().unwrap());
        }
    }

    #[test]
    fn test_kernel_compiler_binding_new() {
        let first = KernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            1,
            KernelSchedule::default(),
            Embedding("first"),
            1,
        )
        .unwrap();
        let same = KernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            1,
            KernelSchedule::default(),
            Embedding("first"),
            1,
        )
        .unwrap();
        let different = KernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            1,
            KernelSchedule::default(),
            Embedding("second"),
            1,
        )
        .unwrap();
        assert_eq!(first, same);
        assert_ne!(first, different);
    }

    #[cfg(feature = "mosaic-gpu")]
    #[test]
    fn test_kernel_compiler_binding_new_with_extensions() {
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(
                    ArrayType::new_static(DataType::F16, vec![64, 16]),
                    KernelParameterAccess::ReadOnly,
                )
                .unwrap(),
                whole_array_parameter(
                    ArrayType::new_static(DataType::F16, vec![16, 8]),
                    KernelParameterAccess::ReadOnly,
                )
                .unwrap(),
                whole_array_parameter(
                    ArrayType::new_static(DataType::F32, vec![64, 8]),
                    KernelParameterAccess::WriteOnly,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let definition = KernelDefinition::<GpuOperation>::trace(call, |(references, _)| {
            let left = references[0].read()?;
            let right = references[1].read()?;
            let output = references[0].context().bind(
                PortableKernelOperation::Extension(GpuOperation::Wgmma),
                vec![],
                &[left, right],
            )?;
            references[2].write(&output[0])
        })
        .unwrap();
        let (_, program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                stage_kernel(inputs[0].context(), &definition, &inputs)
            },
            definition.operation().input_types(),
        )
        .unwrap();
        let body_id = program.instructions()[0].regions()[0];
        assert_eq!(
            program
                .region_ref(body_id)
                .unwrap()
                .instructions()
                .iter()
                .filter(|instruction| {
                    matches!(
                        instruction.operation(),
                        Operation::Kernel(KernelOperation(PortableKernelOperation::Extension(
                            KernelExtension::Mosaic(GpuOperation::Wgmma)
                        )))
                    )
                })
                .count(),
            1
        );
        let restored = definition_from_body(definition.operation(), program.region_ref(body_id).unwrap()).unwrap();
        assert_eq!(restored.semantic_key().unwrap(), definition.semantic_key().unwrap());

        let target = GpuTarget::new(9, 0).unwrap().with_threads_per_block(128).unwrap();
        let options = GpuOptions::default().with_mma(Mma::Wgmma);
        let gpu_facts = || {
            Ok(KernelExecutionFacts {
                platform_name: "CUDA".to_owned(),
                platform_version: "cuda 13020".to_owned(),
                pjrt_version: ryft_pjrt::VERSION,
                has_ffi_extension: true,
                attributes: BTreeMap::new(),
                devices: vec![KernelDeviceFacts {
                    kind: "fixture GPU".to_owned(),
                    attributes: BTreeMap::from([(
                        "compute_capability".to_owned(),
                        ryft_pjrt::Value::String("9.0".to_owned()),
                    )]),
                }],
            })
        };
        let portable = KernelCompilerBinding::new(
            GpuCompiler,
            target.clone(),
            options.clone(),
            KernelSchedule::default(),
            MosaicGpuEmbedding,
            1,
        )
        .unwrap();
        assert!(matches!(select_kernels(&program, Some(&portable), gpu_facts),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "portable kernel compiler binding cannot consume a Mosaic GPU extension"));

        let binding = KernelCompilerBinding::new_with_extensions::<_, _, GpuOperation>(
            GpuCompiler,
            target,
            options,
            KernelSchedule::default(),
            MosaicGpuEmbedding,
            1,
        )
        .unwrap();
        // Identical compiler configuration never bypasses the current body's family admission.
        assert_eq!(portable.configuration(), binding.configuration());
        let selected = select_kernels(&program, Some(&binding), gpu_facts).unwrap().unwrap();
        assert_eq!(selected.instructions().len(), 1);
        assert_eq!(selected.instructions()[0].operation().name(), "custom_call");
        assert_eq!(selected.output_types(), definition.operation().output_types());
        assert!(!selected.effects().classes().contains(EffectClass::OrderedState));
        assert_eq!(selected.regions().len(), 1);
    }

    #[test]
    fn test_kernel_compiler_binding_compile() {
        let binding = binding(1);
        let portable = crate::kernels::tests::definition();
        let staged = stage_body(&portable).unwrap();
        let definition = definition_from_body(portable.operation(), staged.entry_region_ref()).unwrap();
        let first_facts = facts().unwrap();
        let first = binding.compile(&definition, &first_facts).unwrap();
        assert_eq!(
            first.to_string(),
            indoc! {"
                custom_call [
                    target=ryft.test.aot,
                    ryft.kernel.schema=1,
                    ryft.kernel.semantic=\"c0aace572db2a32780947575541c582a8c0e012a86dccb7f08be2cffcc1a97b0\",
                    ryft.kernel.configuration=\"515ef62b84d55eb9cf36a09e7cd9bbbc71b81d00380ee8324af729c3e7766e27\",
                    input_output_alias=0->0,
                ]
            "}
            .trim_end(),
        );
        let mut changed = first_facts.clone();
        changed.platform_version = "2".to_owned();
        let second = binding.compile(&definition, &changed).unwrap();
        assert_ne!(first.attributes(), second.attributes());
        changed.platform_name = "other".to_owned();
        assert!(matches!(binding.compile(&definition, &changed), Err(KernelEmbeddingError::Invalid { message })
            if message == "fixture execution platform mismatch"));
        // Target mismatch wins over adapter rejection, proving the execution check precedes compiler admission.
        let unavailable = KernelCompilerBinding::new(
            Compiler,
            FixtureTarget(false),
            1,
            KernelSchedule::default(),
            Embedding("fixture"),
            1,
        )
        .unwrap();
        assert!(matches!(unavailable.compile(&definition, &changed), Err(KernelEmbeddingError::Invalid { message })
            if message == "fixture execution platform mismatch"));
    }

    #[test]
    fn test_kernel_compiler_binding_compile_rejects_changed_embedding() {
        /// Embedding with externally mutable configuration to exercise the immutable binding contract.
        struct MutableEmbedding {
            /// Configuration shared with the caller after the binding is created.
            configuration: Arc<AtomicUsize>,
        }

        impl KernelOutputEmbedding<()> for MutableEmbedding {
            fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
                Ok(self.configuration.load(Ordering::SeqCst).to_le_bytes().to_vec())
            }

            fn custom_call(
                &self,
                kernel: &VerifiedKernel<'_>,
                output: &(),
            ) -> Result<CustomCallOperation, KernelEmbeddingError> {
                Embedding("fixture").custom_call(kernel, output)
            }
        }

        let configuration = Arc::new(AtomicUsize::new(0));
        let binding = KernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            1,
            KernelSchedule::default(),
            MutableEmbedding { configuration: Arc::clone(&configuration) },
            1,
        )
        .unwrap();
        let portable = crate::kernels::tests::definition();
        let staged = stage_body(&portable).unwrap();
        let definition = definition_from_body(portable.operation(), staged.entry_region_ref()).unwrap();
        assert!(binding.compile(&definition, &facts().unwrap()).is_ok());
        configuration.store(1, Ordering::SeqCst);
        assert!(matches!(
            binding.compile(&definition, &facts().unwrap()),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "selected kernel embedding configuration changed after binding",
        ));
    }

    #[test]
    fn test_kernel_compiler_binding_compile_rejects_changed_compiler() {
        /// Compiler with externally mutable configuration to exercise the immutable binding contract.
        struct MutableCompiler {
            /// Configuration shared with the caller after the binding is created.
            configuration: Arc<AtomicUsize>,
        }

        impl KernelCompiler for MutableCompiler {
            type Target = FixtureTarget;
            type Options = u32;
            type Output = ();
            type Error = std::io::Error;

            fn admit(
                &self,
                kernel: &VerifiedKernel<'_>,
                target: &FixtureTarget,
                options: &u32,
                schedule: &KernelSchedule,
            ) -> Result<(), KernelCompilationError<std::io::Error>> {
                let _ = (kernel, target, options, schedule);
                Ok(())
            }

            fn configuration_key(
                &self,
                target: &FixtureTarget,
                options: &u32,
                schedule: &KernelSchedule,
            ) -> Result<Vec<u8>, KernelCompilationError<std::io::Error>> {
                let _ = (target, options, schedule);
                Ok(self.configuration.load(Ordering::SeqCst).to_le_bytes().to_vec())
            }

            fn compile(
                &self,
                kernel: &VerifiedKernel<'_>,
                target: &FixtureTarget,
                options: &u32,
                schedule: &KernelSchedule,
            ) -> Result<(), KernelCompilationError<std::io::Error>> {
                let _ = (kernel, target, options, schedule);
                Ok(())
            }
        }

        let configuration = Arc::new(AtomicUsize::new(0));
        let binding = KernelCompilerBinding::new(
            MutableCompiler { configuration: Arc::clone(&configuration) },
            FixtureTarget(true),
            1,
            KernelSchedule::default(),
            Embedding("fixture"),
            1,
        )
        .unwrap();
        let portable = crate::kernels::tests::definition();
        let staged = stage_body(&portable).unwrap();
        let definition = definition_from_body(portable.operation(), staged.entry_region_ref()).unwrap();
        binding.compile(&definition, &facts().unwrap()).unwrap();
        configuration.store(1, Ordering::SeqCst);
        assert!(matches!(
            binding.compile(&definition, &facts().unwrap()),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "selected kernel compiler configuration changed after binding",
        ));
    }

    #[test]
    fn test_select_kernels() {
        let program = program();
        assert!(program.effects().classes().contains(EffectClass::OrderedState));
        let binding = binding(1);
        let selected = select_kernels(&program, Some(&binding), facts).unwrap().unwrap();
        assert_eq!(
            selected.to_string(),
            indoc! {"
                lambda %0:i32[] .
                let %1:i32[] = custom_call [
                    target=ryft.test.aot,
                    ryft.kernel.schema=1,
                    ryft.kernel.semantic=\"c0aace572db2a32780947575541c582a8c0e012a86dccb7f08be2cffcc1a97b0\",
                    ryft.kernel.configuration=\"515ef62b84d55eb9cf36a09e7cd9bbbc71b81d00380ee8324af729c3e7766e27\",
                    input_output_alias=0->0,
                ] %0
                in (%1)
            "}
            .trim_end(),
        );
        assert!(selected.effects().classes().is_empty());
        assert_eq!(selected.regions().len(), 1);
    }

    #[test]
    fn test_select_kernels_in_shard_map() {
        // Builds a `shard_map` whose body applies a kernel to its local shard and returns that body and the program,
        // so that independently built equal programs can be compared. With `Some(request)`, the kernel also reshapes
        // the value of its parameter, requesting a replicated output sharding when `request` is `true`.
        let build = |reshape: Option<bool>| {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("device", 2, MeshAxisType::Manual).unwrap()]).unwrap();
            let sharding =
                Sharding::new(mesh.clone(), vec![ShardingDimension::Sharded(vec!["device".to_owned()])]).unwrap();
            let global = ArrayType::new_static(DataType::I32, [4]).with_sharding(sharding.clone()).unwrap();
            let local = ArrayType::new_static(DataType::I32, [2])
                .with_sharding(sharding.clone().with_varying_manual_axes(["device"]).unwrap())
                .unwrap();
            let physical = ArrayType::new_static(DataType::I32, [2]);
            let definition: KernelDefinition = KernelDefinition::trace(
                KernelCallOperation::new(
                    Grid::new(vec![]).unwrap(),
                    vec![whole_array_parameter(physical.clone(), KernelParameterAccess::ReadWrite).unwrap()],
                )
                .unwrap(),
                |(references, _)| {
                    if let Some(request) = reshape {
                        let shape = physical.shape().clone();
                        let operation = ReshapeOperation::new(shape.clone(), shape)
                            .with_output_sharding(request.then(|| Sharding::replicated(mesh.clone(), 1)));
                        let value = references[0].read()?;
                        let operation = ArrayIrOperation::Array(ArrayOperation::Reshape(operation));
                        references[0].context().bind(operation, vec![], std::slice::from_ref(&value))?;
                        references[0].write(&value)?;
                    }
                    Ok(())
                },
            )
            .unwrap();
            let body = kernel_program(&definition, vec![local.into()]).unwrap();
            let operation = ShardMapOperation::from_program(
                &body,
                vec![global.clone().into()],
                ShardMapBoundary::new(mesh, vec![sharding.clone()], vec![sharding], vec!["device".to_owned()]).unwrap(),
            )
            .unwrap();
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(global.into());
            let region = builder.import_region(body.entry_region_ref());
            let outputs = builder
                .add_instruction(Operation::ShardMap(Box::new(operation)), vec![region], vec![input], None)
                .unwrap()
                .to_vec();
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder])
                .unwrap();
            (body, program)
        };
        let (body, program) = build(None);
        let binding = binding(1);
        let selected = select_kernels(&program, Some(&binding), facts).unwrap().unwrap();
        assert_eq!(
            selected.to_string(),
            indoc! {"
                lambda %0:i32[4][sharding={mesh<['device'=2:manual]>, [{'device'}]}] .
                let %1:i32[4][sharding={mesh<['device'=2:manual]>, [{'device'}]}] = shard_map [
                    mesh=['device'=2:manual],
                    input_shardings=[{mesh<['device'=2:manual]>, [{'device'}]}],
                    output_shardings=[{mesh<['device'=2:manual]>, [{'device'}]}],
                    manual_axes=['device'],
                    global_input_types=[i32[4][sharding={mesh<['device'=2:manual]>, [{'device'}]}]],
                    global_output_types=[i32[4][sharding={mesh<['device'=2:manual]>, [{'device'}]}]],
                ] %0 [
                    body={
                        lambda %0:i32[2][sharding={mesh<['device'=2:manual]>, [{'device'}], varying_manual={'device'}}] .
                        let %1:i32[2][sharding={mesh<['device'=2:manual]>, [{'device'}], varying_manual={'device'}}] = custom_call [
                            target=ryft.test.aot,
                            ryft.kernel.schema=1,
                            ryft.kernel.semantic=\"f7cc50a3ef48ecb6d57641d539824244179a0ebd191571eaeb72fbbecf87a870\",
                            ryft.kernel.configuration=\"515ef62b84d55eb9cf36a09e7cd9bbbc71b81d00380ee8324af729c3e7766e27\",
                            input_output_alias=0->0,
                        ] %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert!(matches!(
            select_kernels(&body, Some(&binding), facts),
            Err(KernelEmbeddingError::Invalid { message }) if message == unbound_kernel_axis_message(0, "device"),
        ));

        // An independently built equal program selects the same keyed custom call, so it reuses the same entry.
        let (_, again) = build(None);
        assert_eq!(select_kernels(&again, Some(&binding), facts).unwrap().unwrap().to_string(), selected.to_string());

        // A kernel body with a requested output sharding also has a key, which equal programs share and which differs
        // from the key of the same body without the request.
        let select = |reshape| select_kernels(&build(reshape).1, Some(&binding), facts).unwrap().unwrap().to_string();
        let requested = select(Some(true));
        assert_eq!(select(Some(true)), requested);
        assert_ne!(select(Some(false)), requested);
        assert_ne!(select(Some(false)), selected.to_string());
    }

    #[test]
    fn test_select_kernels_preserves_call_and_rematerialization_regions() {
        let definition = differentiable_definition();
        let body = kernel_primal(&definition).unwrap();
        let scalar = body.input_types()[0].clone();
        let binding = binding(1);
        for (operation, expected) in [
            (
                Operation::JitCall(JitCallOperation::new(0)),
                indoc! {"
                    lambda %0:f32[] .
                    let %1:f32[] = jit_call %0 [
                        callee={
                            lambda %0:f32[] .
                            let %1:f32[] = custom_call [
                                target=ryft.test.aot,
                                ryft.kernel.schema=1,
                                ryft.kernel.semantic=\"365635d74d021c307c30e01c2ce48d0c1bfd3d8fe4cf379ceef283f1521ccebd\",
                                ryft.kernel.configuration=\"515ef62b84d55eb9cf36a09e7cd9bbbc71b81d00380ee8324af729c3e7766e27\",
                                input_output_alias=0->0,
                            ] %0
                            in (%1)
                        },
                    ]
                    in (%1)
                "},
            ),
            (
                Operation::Rematerialize(RematerializeOperation::new(ResidualPolicyReference::new(NothingSavable))),
                indoc! {"
                    lambda %0:f32[] .
                    let %1:f32[] = rematerialize %0 [
                        body={
                            lambda %0:f32[] .
                            let %1:f32[] = custom_call [
                                target=ryft.test.aot,
                                ryft.kernel.schema=1,
                                ryft.kernel.semantic=\"365635d74d021c307c30e01c2ce48d0c1bfd3d8fe4cf379ceef283f1521ccebd\",
                                ryft.kernel.configuration=\"515ef62b84d55eb9cf36a09e7cd9bbbc71b81d00380ee8324af729c3e7766e27\",
                                input_output_alias=0->0,
                            ] %0
                            in (%1)
                        },
                    ]
                    in (%1)
                "},
            ),
        ] {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(scalar.clone());
            let body = builder.import_region(body.entry_region_ref());
            let outputs = builder.add_instruction(operation.clone(), vec![body], vec![input], None).unwrap().to_vec();
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder])
                .unwrap();
            let selected = select_kernels(&program, Some(&binding), facts).unwrap().unwrap();
            assert_eq!(selected.to_string(), expected.trim_end());
        }
    }

    #[test]
    fn test_select_kernels_preserves_condition_and_while_regions() {
        let body = kernel_primal(&differentiable_definition()).unwrap();
        let scalar = body.input_types()[0].clone();
        let mut predicate = ProgramBuilder::new();
        predicate.add_input(scalar.clone());
        let result = predicate
            .add_instruction(ConstantOperation::new(CoreArray::scalar(true).unwrap()), vec![], vec![], None)
            .unwrap()[0];
        let predicate = predicate
            .build::<Vec<Constant>, Vec<Constant>>(vec![result], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let binding = binding(1);
        for (operation, expected) in [
            (
                Operation::Condition(ConditionOperation::new()),
                indoc! {"
                    lambda %0:f32[] .
                    let %1:bool[] = constant [value=true]
                        %2:f32[] = condition %1 %0 [
                            true=^0={
                                lambda %0:f32[] .
                                let %1:f32[] = custom_call [
                                    target=ryft.test.aot,
                                    ryft.kernel.schema=1,
                                    ryft.kernel.semantic=\"365635d74d021c307c30e01c2ce48d0c1bfd3d8fe4cf379ceef283f1521ccebd\",
                                    ryft.kernel.configuration=\"515ef62b84d55eb9cf36a09e7cd9bbbc71b81d00380ee8324af729c3e7766e27\",
                                    input_output_alias=0->0,
                                ] %0
                                in (%1)
                            },
                            false=^0,
                        ]
                    in (%2)
                "},
            ),
            (
                Operation::While(WhileOperation::new().with_iteration_bound(2).unwrap()),
                indoc! {"
                    lambda %0:f32[] .
                    let %1:f32[] = while [iteration_bound=2] %0 [
                        condition={
                            lambda %0:f32[] .
                            let %1:bool[] = constant [value=true]
                            in (%1)
                        },
                        body={
                            lambda %0:f32[] .
                            let %1:f32[] = custom_call [
                                target=ryft.test.aot,
                                ryft.kernel.schema=1,
                                ryft.kernel.semantic=\"365635d74d021c307c30e01c2ce48d0c1bfd3d8fe4cf379ceef283f1521ccebd\",
                                ryft.kernel.configuration=\"515ef62b84d55eb9cf36a09e7cd9bbbc71b81d00380ee8324af729c3e7766e27\",
                                input_output_alias=0->0,
                            ] %0
                            in (%1)
                        },
                    ]
                    in (%1)
                "},
            ),
        ] {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(scalar.clone());
            let body_region = builder.import_region(body.entry_region_ref());
            let (regions, inputs) = if operation.name() == "condition" {
                let choice = builder
                    .add_instruction(ConstantOperation::new(CoreArray::scalar(true).unwrap()), vec![], vec![], None)
                    .unwrap()[0];
                (vec![body_region, body_region], vec![choice, input])
            } else {
                let predicate_region = builder.import_region(predicate.entry_region_ref());
                (vec![predicate_region, body_region], vec![input])
            };
            let outputs = builder.add_instruction(operation.clone(), regions, inputs, None).unwrap().to_vec();
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder])
                .unwrap();
            let selected = select_kernels(&program, Some(&binding), facts).unwrap().unwrap();
            assert_eq!(selected.to_string(), expected.trim_end());
        }
    }

    #[test]
    fn test_select_kernels_preserves_scan_region() {
        let primal = kernel_primal(&differentiable_definition()).unwrap();
        let scalar = primal.input_types()[0].clone();
        let mut body = ProgramBuilder::new();
        body.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body.add_input(scalar.clone());
        let outputs = body.splice_program(&primal, &[carry]).unwrap();
        let body = body
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::new();
        let carry = builder.add_input(scalar);
        let region = builder.import_region(body.entry_region_ref());
        let outputs = builder
            .add_instruction(Operation::Scan(ScanOperation::new(1, 3usize)), vec![region], vec![carry], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        let binding = binding(1);
        let selected = select_kernels(&program, Some(&binding), facts).unwrap().unwrap();
        assert_eq!(
            selected.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = scan [carry_count=1, iteration_count=3, reverse=false] %0 [
                    body={
                        lambda %0:i64[], %1:f32[] .
                        let %2:f32[] = custom_call [
                            target=ryft.test.aot,
                            ryft.kernel.schema=1,
                            ryft.kernel.semantic=\"365635d74d021c307c30e01c2ce48d0c1bfd3d8fe4cf379ceef283f1521ccebd\",
                            ryft.kernel.configuration=\"515ef62b84d55eb9cf36a09e7cd9bbbc71b81d00380ee8324af729c3e7766e27\",
                            input_output_alias=0->0,
                        ] %1
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_select_kernels_preserves_external_reference_order() {
        let body = kernel_primal(&differentiable_definition()).unwrap();
        let ArrayIrType::Array(scalar) = body.input_types()[0].clone() else {
            panic!("expected an array");
        };
        let reference_type = ArrayIrType::Reference(ReferenceType::new(scalar));
        let mut builder = ProgramBuilder::new();
        let reference = builder.add_input(reference_type);
        let value = builder
            .add_instruction(
                ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new(),
                vec![],
                vec![reference],
                None,
            )
            .unwrap()[0];
        let outputs = builder.splice_program(&body, &[value]).unwrap();
        builder
            .add_instruction(
                ReferenceWriteOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new(),
                vec![],
                vec![reference, outputs[0]],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        let binding = binding(1);
        let selected = select_kernels(&program, Some(&binding), facts).unwrap().unwrap();
        assert_eq!(
            selected.to_string(),
            indoc! {"
                lambda %0:ref<f32[]> .
                let %1:f32[] = reference_read %0
                    %2:f32[] = custom_call [
                        target=ryft.test.aot,
                        ryft.kernel.schema=1,
                        ryft.kernel.semantic=\"365635d74d021c307c30e01c2ce48d0c1bfd3d8fe4cf379ceef283f1521ccebd\",
                        ryft.kernel.configuration=\"515ef62b84d55eb9cf36a09e7cd9bbbc71b81d00380ee8324af729c3e7766e27\",
                        input_output_alias=0->0,
                    ] %1
                    () = reference_write %0 %2
                in (%2)
            "}
            .trim_end(),
        );
        assert!(selected.effects().classes().contains(EffectClass::OrderedState));
    }

    #[test]
    fn test_select_kernels_preserves_memory_transfers() {
        let body = kernel_primal(&differentiable_definition()).unwrap();
        let ArrayIrType::Array(scalar) = body.input_types()[0].clone() else {
            panic!("expected an array");
        };
        let host = scalar.with_memory(Memory::Host { pinned: true });
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(ArrayIrType::Array(host));
        let device = builder
            .add_instruction(TransferToMemoryOperation::new(Memory::Device), vec![], vec![input], None)
            .unwrap()[0];
        let result = builder.splice_program(&body, &[device]).unwrap()[0];
        let output = builder
            .add_instruction(TransferToMemoryOperation::new(Memory::Host { pinned: true }), vec![], vec![result], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let binding = binding(1);
        let selected = select_kernels(&program, Some(&binding), facts).unwrap().unwrap();
        assert_eq!(
            selected.to_string(),
            indoc! {"
                lambda %0:f32[]@Host[Pinned] .
                let %1:f32[] = transfer_to_memory [destination=Device] %0
                    %2:f32[] = custom_call [
                        target=ryft.test.aot,
                        ryft.kernel.schema=1,
                        ryft.kernel.semantic=\"365635d74d021c307c30e01c2ce48d0c1bfd3d8fe4cf379ceef283f1521ccebd\",
                        ryft.kernel.configuration=\"515ef62b84d55eb9cf36a09e7cd9bbbc71b81d00380ee8324af729c3e7766e27\",
                        input_output_alias=0->0,
                    ] %1
                    %3:f32[]@Host[Pinned] = transfer_to_memory [destination=Host[Pinned]] %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_select_kernels_missing_binding() {
        assert!(matches!(select_kernels(&program(), None, facts), Err(KernelEmbeddingError::Invalid { message })
            if message == "kernel call has no explicitly enabled XLA compiler binding"));
    }

    #[test]
    fn test_select_kernels_rejects_unselected_kernel_operation() {
        let body = kernel_primal(&differentiable_definition()).unwrap();
        let scalar = body.input_types()[0].clone();
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(scalar);
        let outputs = builder.splice_program(&body, &[input]).unwrap();
        let doubled = builder
            .add_instruction(
                Operation::Kernel(KernelOperation::new(PortableKernelOperation::Portable(
                    ryft_core::ArrayOperation::<CoreArray>::Add(ryft_core::AddOperation::<ryft_core::ArrayType>::new())
                        .into(),
                ))),
                vec![],
                vec![outputs[0], outputs[0]],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            select_kernels(&program, Some(&binding(1)), facts),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "kernel operation `add` remains outside a selected kernel call",
        ));
    }

    #[test]
    fn test_select_kernels_admission_failure() {
        let binding = KernelCompilerBinding::new(
            Compiler,
            FixtureTarget(false),
            1,
            KernelSchedule::default(),
            Embedding("ryft.test.selected"),
            1,
        )
        .unwrap();
        assert!(matches!(select_kernels(&program(), Some(&binding), facts), Err(KernelEmbeddingError::Compiler(error))
            if error.to_string() == "kernel compiler is unavailable: fixture target unavailable"));
    }

    #[test]
    fn test_select_kernels_compiler_failure_does_not_poison_later_selection() {
        let program = program();
        let original = program.to_string();
        let failing = KernelCompilerBinding::new(
            Compiler,
            FixtureTarget(true),
            u32::MAX,
            KernelSchedule::default(),
            Embedding("fixture"),
            1,
        )
        .unwrap();
        let successful = binding(1);
        let failure = select_kernels(&program, Some(&failing), facts).unwrap_err();
        assert!(
            matches!(&failure, KernelEmbeddingError::Compiler(error) if error.to_string() == "fixture compiler failed")
        );
        assert_eq!(failure.source().unwrap().to_string(), "fixture compiler failed");
        assert_eq!(program.to_string(), original);
        let first = select_kernels(&program, Some(&successful), facts).unwrap().unwrap();
        let second = select_kernels(&program, Some(&successful), facts).unwrap().unwrap();
        assert_eq!(
            first.to_string(),
            indoc! {"
                lambda %0:i32[] .
                let %1:i32[] = custom_call [
                    target=ryft.test.aot,
                    ryft.kernel.schema=1,
                    ryft.kernel.semantic=\"c0aace572db2a32780947575541c582a8c0e012a86dccb7f08be2cffcc1a97b0\",
                    ryft.kernel.configuration=\"515ef62b84d55eb9cf36a09e7cd9bbbc71b81d00380ee8324af729c3e7766e27\",
                    input_output_alias=0->0,
                ] %0
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(second.to_string(), first.to_string());
        assert!(first.effects().classes().is_empty());
    }

    #[test]
    fn test_select_kernels_revalidates_current_body_effects() {
        let program = program();
        let regions = program
            .regions()
            .iter()
            .enumerate()
            .map(|(index, region)| {
                let mut instructions = region.instructions().to_vec();
                if index == 0 {
                    instructions.push(Instruction::new(
                        Operation::Kernel(KernelOperation::new(PortableKernelOperation::Portable(
                            ryft_core::ArrayOperation::<CoreArray>::CustomCall(
                                CustomCallOperation::new("test.external", vec![])
                                    .with_effect_class(EffectClass::OrderedIo),
                            )
                            .into(),
                        ))),
                        vec![],
                        vec![],
                        vec![],
                    ));
                }
                Region::new(
                    region.atoms().to_vec(),
                    region.input_ids().to_vec(),
                    region.output_ids().to_vec(),
                    instructions,
                )
            })
            .collect();
        let changed = CoreProgram::new(vec![Placeholder], vec![Placeholder], regions, program.entry()).unwrap();
        let binding = binding(1);
        assert!(matches!(
            select_kernels(&changed, Some(&binding), facts),
            Err(KernelEmbeddingError::UnsupportedEffect { effect: EffectClass::OrderedIo })
        ));
    }

    #[test]
    fn test_stage_body() {
        let definition = crate::kernels::tests::definition();
        let body = stage_body(&definition).unwrap();
        assert_eq!(
            body.to_string(),
            indoc! {"
                lambda %0:ref<i32[]> .
                let %1:i32[] = reference_read %0
                    () = reference_write %0 %1
                in ()
            "}
            .trim_end(),
        );
        let restored = definition_from_body(definition.operation(), body.entry_region_ref()).unwrap();
        assert_eq!(restored.semantic_key().unwrap(), definition.semantic_key().unwrap());
        assert_eq!(restored.body().effects().classes(), definition.body().effects().classes());
    }

    #[test]
    fn test_stage_body_materializes_array_literals() {
        let operation = crate::kernels::tests::definition().operation().clone();
        let mut builder = ryft_core::ProgramBuilder::<ArrayIrValue<CoreArray>, PortableKernelOperation>::new();
        let reference = builder.add_input(operation.body_input_types()[0].clone());
        let literal = builder.add_constant(ArrayIrValue::Array(CoreArray::scalar(42i32).unwrap()));
        builder
            .add_instruction(
                ryft_core::ReferenceWriteOperation::<ryft_core::ArrayType, ArrayIrType, ArrayReferenceTransform>::new(),
                vec![],
                vec![reference, literal],
                None,
            )
            .unwrap();
        let definition =
            KernelDefinition::new(operation, builder.build(vec![], vec![Placeholder], vec![]).unwrap()).unwrap();
        let staged = stage_body(&definition).unwrap();
        assert_eq!(
            staged.to_string(),
            indoc! {"
                lambda %0:ref<i32[]> .
                let %1:i32[] = constant [value=42]
                    () = reference_write %0 %1
                in ()
            "}
            .trim_end(),
        );
        assert_eq!(staged.instructions()[0].outputs(), &[literal]);
        let restored = definition_from_body(definition.operation(), staged.entry_region_ref()).unwrap();
        let portable = KernelDefinition::new(
            restored.operation().clone(),
            restored
                .body()
                .map_operations(|operation| operation.clone().map_extension(NoKernelExtension::try_from))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            portable.interpret(vec![CoreArray::scalar(0i32).unwrap()], 1).unwrap(),
            vec![CoreArray::scalar(42i32).unwrap()]
        );
    }

    #[test]
    fn test_definition_from_body_rejects_unreconstructible_operations() {
        let definition = crate::kernels::tests::definition();
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let output = builder
            .add_instruction(
                ryft_core::ArrayOperation::Add(ryft_core::AddOperation::<ryft_core::ArrayType>::new()),
                vec![],
                vec![input, input],
                None,
            )
            .unwrap()[0];
        let body = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            definition_from_body(definition.operation(), body.entry_region_ref()),
            Err(ProgramError::MalformedProgram(message))
                if message == "operation `add` of the XLA family has no kernel reconstruction",
        ));
    }
}
