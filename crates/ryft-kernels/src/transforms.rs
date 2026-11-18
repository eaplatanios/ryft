//! Contains transforms of kernel definitions, which currently consist of batching.
//!
//! Kernel transforms preserve canonical regions, array metadata, and confined reference effects. Batching turns a
//! [`KernelDefinition`] that processes one item into one that processes a batch of independent items in a single
//! launch. It adds a grid coordinate and disjoint array windows rather than vectorizing mutable instructions.
//!
//! # Batching Model
//!
//! [`KernelDefinition::batched`] appends a parallel grid dimension whose extent is the batch size, together with a
//! matching `batch` coordinate, so every batch item runs as its own programs. Every mapped parameter gains a batch
//! dimension in its full array type and in its block shape, and its block mapping selects a size one window at the
//! batch coordinate along that dimension. The body keeps its instructions and attached regions. Every reference
//! access of a mapped parameter receives a leading static index into its size one window, so the body reaches the
//! original referents through folded access paths, and all other instructions are copied unchanged. Unmapped
//! read-only parameters remain shared by all batch items. Parameter types are physical, so the batch dimension has no
//! placement, and the batched call accepts the packed inputs however their batch axes are placed (refer to
//! [`KernelCallOperation`]). The result of a read-write parameter has the type of its packed input, and a write-only
//! result has no placement.
//!
//! Batching rejects the inputs for which this rewrite is not sound, with a [`KernelTransformError::Unsupported`]
//! that names the rejected contract.
//!
//!   - Every writable parameter must be mapped, including functional read-write aliases, because batch items would
//!     otherwise race on one shared array.
//!   - A mapped reference must only be used through reference accesses, because only an access path can receive the
//!     batch selection. Passing it into an attached region (e.g., as a `condition` operand) or using it outside a
//!     reference access (e.g., in a [`WindowMaskOperation`](crate::memory::WindowMaskOperation), whose window validity
//!     would have the wrong shape) is rejected.
//!   - Mapped parameters must not have explicit layouts, and scalar prefetched inputs must be specialized first.
//!   - [`KernelDefinition::batched_inputs`], which derives the batch axes from packed batched inputs, rejects dynamic
//!     batch extents, because the batch becomes a static grid dimension.
//!
//! The batched call and body are rebuilt through their canonical constructors, and the result is requalified by a new
//! executable proof (i.e., [`VerifiedKernel`]) that checks all transformed accesses.
//!
//! # Composition with Ordinary Batching
//!
//! [`KernelOperation`] implements [`BatchableOperation`], so a kernel call staged in an ordinary program batches
//! under the program transforms of `ryft-core`. The implementation rebuilds the complete definition from the call and
//! its attached body, batches it with [`KernelDefinition::batched_inputs`], and binds the batched call in the parent
//! context. Other kernel operations are only batched as part of their complete kernel call.

use std::collections::BTreeMap;

use ryft_core::{
    Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayReferenceTransform,
    ArrayReferenceTransformIndex, Atom, BatchableOperation, BatchingContext, BatchingDriver, BatchingError, Context,
    Dimension, DimensionBounds, DimensionType, Operation, Placeholder, ProgramBuilder, ProgramError, ShardingDimension,
    Typed, TypedProgram, Value, rewrite_reference_access_transforms, validated_reference_access_descriptors,
};
use thiserror::Error;

use crate::calls::{KernelCallOperation, KernelDefinition, KernelError, KernelParameter};
use crate::compilation::VerifiedKernel;
use crate::grids::{Grid, GridDimension, GridExecution};
use crate::initialization::KernelInitializationError;
use crate::interpretation::DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS;
use crate::mappings::{BlockMapping, BoundaryPolicy};
use crate::operations::{KernelExtension, KernelOperation};
use crate::validation::KernelParameterAccess;

/// A kernel transform cannot preserve the declared boundary or prove its rewritten accesses.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum KernelTransformError {
    /// Canonical construction rejected the rewritten program or types.
    #[error(transparent)]
    Kernel(#[from] KernelError),

    /// Rewritten accesses do not satisfy executable initialization and race requirements.
    #[error(transparent)]
    Initialization(#[from] KernelInitializationError),

    /// The batch placement of the inputs is ambiguous or cannot be inserted into a rewritten parameter type.
    #[error(transparent)]
    Batching(#[from] BatchingError),

    /// The selected transform has no proven rule for this input contract.
    #[error("kernel transform is unsupported: {message}")]
    Unsupported {
        /// Description of the unsupported input contract.
        message: String,
    },
}

impl<Extension: KernelExtension> KernelDefinition<Extension> {
    /// Adds an independent parallel batch coordinate and canonical size one windows for mapped parameters. The body
    /// selects the original referents through folded access paths and executes once per batch item. Unmapped read-only
    /// arrays remain shared, and every writable parameter must be mapped, including functional read-write aliases.
    ///
    /// `axes` follows all parameters, including write-only outputs. Each mapped axis is inserted into the full array
    /// and block shape of its parameter. The full array type is batched with
    /// [`ArrayType::batched`](ryft_core::ArrayType::batched), which preserves the data type and memory while clearing
    /// the layout. Parameter types are physical, so the batch dimension has no placement, and the batched call accepts
    /// packed inputs however their batch is placed (refer to [`KernelCallOperation`]). Explicit source layouts are
    /// rejected because restoring them on the selected referents is not justified. Scalar prefetched inputs must be
    /// specialized first (e.g., with [`KernelDefinition::specialize_prefetch`]). A new executable proof checks all
    /// transformed accesses.
    ///
    /// # Parameters
    ///
    ///   - `batch_size`: Static number of independent items. Zero produces an empty launch.
    ///   - `axes`: Inserted array dimension for each mapped parameter, or `None` for a shared read-only parameter.
    ///   - `maximum_programs`: Enumeration budget for executable qualification when affine proofs do not apply.
    ///
    /// # Errors
    ///
    /// Returns [`KernelTransformError::Unsupported`] when `axes` does not cover every parameter, when the call has
    /// scalar prefetched inputs, when a writable parameter is unmapped, when a mapped parameter has an explicit layout,
    /// or when the body uses a mapped reference outside a reference access. Returns [`KernelTransformError::Batching`]
    /// when a mapped axis is out of bounds, [`KernelTransformError::Kernel`] when canonical construction rejects the
    /// batched call or body, and [`KernelTransformError::Initialization`] when the new executable proof rejects the
    /// transformed accesses.
    pub fn batched(
        &self,
        batch_size: usize,
        axes: &[Option<usize>],
        maximum_programs: usize,
    ) -> Result<Self, KernelTransformError> {
        let call = self.operation();
        if axes.len() != call.parameters().len() {
            return Err(unsupported("batch axes must match every kernel parameter"));
        }
        if !call.prefetch_types().is_empty() {
            return Err(unsupported("scalar-prefetched inputs must be specialized before batching"));
        }
        let mut dimensions = call.grid().dimensions().to_vec();
        dimensions.push(GridDimension::new(Dimension::Static(batch_size), GridExecution::Parallel));
        let grid = Grid::new(dimensions).map_err(KernelError::from)?;
        let coordinate = DimensionType::new("batch", DimensionBounds::non_negative(Some(batch_size.max(1))).unwrap());
        let mut parameters = Vec::with_capacity(axes.len());
        for (index, (parameter, axis)) in call.parameters().iter().zip(axes).enumerate() {
            if axis.is_none() && parameter.access() != KernelParameterAccess::ReadOnly {
                return Err(unsupported(format!("writable parameter {index} requires a mapped batch axis")));
            }
            if axis.is_some() && parameter.r#type().layout().is_some() {
                return Err(unsupported(format!("mapped parameter {index} has an explicit layout")));
            }
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let inputs = parameter
                .mapping()
                .program()
                .input_types()
                .into_owned()
                .into_iter()
                .map(|r#type| builder.add_input(r#type))
                .collect::<Vec<_>>();
            let batch = builder.add_input(ArrayIrType::Dimension(coordinate.clone()));
            let mut starts =
                builder.splice_program(parameter.mapping().program(), &inputs).map_err(KernelError::from)?;
            let mut shape = parameter.mapping().block_shape().to_vec();
            let mut r#type = parameter.r#type().into_owned();
            if let Some(axis) = *axis {
                r#type = r#type.batched(axis, Dimension::Static(batch_size), ShardingDimension::Replicated)?;
                starts.insert(axis, batch);
                shape.insert(axis, 1);
            }
            let program = builder
                .build(starts, vec![Placeholder; inputs.len() + 1], vec![Placeholder; shape.len()])
                .map_err(KernelError::from)?;
            // No invocation exists for an empty batch. Padding preserves the formal selected referent type without
            // constructing an invalid slice with one element of the empty full array.
            let policy = if axis.is_some() && batch_size == 0 {
                BoundaryPolicy::Masked
            } else {
                parameter.mapping().boundary_policy()
            };
            let mapping = BlockMapping::new(program, shape, policy).map_err(KernelError::from)?;
            parameters.push(KernelParameter::new(r#type, parameter.access(), mapping)?);
        }
        let mut coordinates = call.coordinate_types().to_vec();
        coordinates.push(coordinate);
        let operation = KernelCallOperation::from_parts(grid, parameters, vec![], coordinates)?;
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<Extension>>::new();
        let inputs =
            operation.body_input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let source = self.body();
        let mut atoms = vec![None; source.atoms().len()];
        for (&original, &input) in source.input_ids().iter().zip(&inputs[..inputs.len() - 1]) {
            atoms[original.index()] = Some(input);
        }
        for (index, atom) in source.atoms().iter().enumerate() {
            if let Atom::Constant(value) = atom {
                atoms[index] = Some(builder.add_constant(value.clone()));
            }
        }
        let mapped = source
            .input_ids()
            .iter()
            .zip(axes)
            .filter_map(|(&input, axis)| axis.map(|axis| (input, axis)))
            .collect::<BTreeMap<_, _>>();
        let attached = source
            .instructions()
            .iter()
            .flat_map(|instruction| instruction.regions().iter().copied())
            .collect::<Vec<_>>();
        let regions = attached
            .iter()
            .map(|&region| source.region_ref(region))
            .collect::<Result<Vec<_>, _>>()
            .map_err(KernelError::from)?;
        let imported = builder.import_regions(&regions).map_err(KernelError::from)?;
        let regions = attached.into_iter().zip(imported).collect::<BTreeMap<_, _>>();
        for instruction in source.instructions() {
            let mut rewritten = instruction.clone();
            for (input_index, input) in instruction.inputs().iter().enumerate() {
                let Some(&axis) = mapped.get(input) else { continue };
                let descriptors =
                    validated_reference_access_descriptors(rewritten.operation(), rewritten.inputs().len())
                        .map_err(KernelError::from)?;
                let descriptor = descriptors[input_index].as_ref().ok_or_else(|| {
                    unsupported(format!(
                        "operation `{}` uses mapped reference input {input_index} outside a reference access",
                        instruction.operation().name(),
                    ))
                })?;
                let mut transforms =
                    vec![ArrayReferenceTransform::Index { axis, index: ArrayReferenceTransformIndex::Static(0) }];
                transforms.extend_from_slice(descriptor.transforms());
                let bindings = rewritten.inputs()[descriptor.bindings()].to_vec();
                rewritten = rewrite_reference_access_transforms(&rewritten, input_index, transforms, bindings)
                    .map_err(KernelError::from)?;
            }
            let inputs = rewritten.inputs().iter().map(|input| atoms[input.index()].unwrap()).collect();
            let attached = rewritten.regions().iter().map(|region| regions[region]).collect();
            let outputs = builder
                .add_instruction(rewritten.operation().clone(), attached, inputs, Some(rewritten.provenance().clone()))
                .map_err(KernelError::from)?;
            for (&original, &output) in instruction.outputs().iter().zip(outputs) {
                atoms[original.index()] = Some(output);
            }
        }
        let outputs = source.output_ids().iter().map(|output| atoms[output.index()].unwrap()).collect();
        let body = builder.build(outputs, vec![Placeholder; inputs.len()], vec![]).map_err(KernelError::from)?;
        let definition = Self::new(operation, body)?;
        VerifiedKernel::new(&definition, maximum_programs)?;
        Ok(definition)
    }

    /// Derives parameter batch axes from canonical packed input carriers and batches the definition with
    /// [`KernelDefinition::batched`], returning the batched definition together with the batch axis of every output.
    ///
    /// Write-only results acquire a leading batch axis, and read-write results retain the axis of their input. With
    /// entirely replicated inputs, the unchanged definition and replicated result axes are returned. Kernel batching
    /// adds a grid dimension rather than a sealed program boundary, so it does not use the shared boundary transport of
    /// structurally batched programs. A dynamic batch extent is therefore rejected, because the batch becomes a static
    /// grid dimension. Every batch item uses the same statically shaped block windows.
    ///
    /// The packed inputs may be placed. Their unbatched types must be the declared input types up to their shardings,
    /// and the batched call accepts the packed inputs as they are, because the parameter types of a kernel are
    /// physical and placement belongs to the call. The results are typed by the batched call. A read-write result
    /// has the type of its packed input, and a write-only result has no placement.
    ///
    /// # Errors
    ///
    /// Returns [`KernelTransformError::Unsupported`] when the unbatched input types differ from the kernel signature,
    /// when a batched input is not an ordinary array, has a dynamic batch extent, or has a batch extent that differs
    /// from that of another input, or when the batched call does not accept the packed input types (e.g., because of
    /// an explicit packed layout). Otherwise, returns the errors of [`KernelDefinition::batched`].
    pub fn batched_inputs<V: Value<Type = ArrayIrType>>(
        &self,
        inputs: &[ArrayIrBatch<V>],
        maximum_programs: usize,
    ) -> Result<(Self, Vec<Option<usize>>), KernelTransformError> {
        let unbatched_types = inputs.iter().map(ArrayIrBatch::unbatched_type).collect::<Vec<_>>();
        if self.operation().output_types_for(&unbatched_types).is_err() {
            return Err(unsupported("batched input types differ from the kernel signature"));
        }
        let mut batch_size = None;
        for input in inputs {
            if let Some(axis) = input.batch_axis_position() {
                let r#type = input.value().r#type();
                let ArrayIrType::Array(r#type) = r#type.as_ref() else {
                    return Err(unsupported("kernel batching accepts only ordinary array inputs"));
                };
                let shape = r#type.static_shape().ok_or_else(|| {
                    unsupported("kernel batching requires a static batch extent, which becomes a static grid dimension")
                })?;
                let extent = shape.dimensions()[axis];
                if batch_size.is_some_and(|previous| previous != extent) {
                    return Err(unsupported("kernel inputs have different batch extents"));
                }
                batch_size = Some(extent);
            }
        }
        let Some(batch_size) = batch_size else {
            return Ok((self.clone(), vec![None; self.operation().output_types().len()]));
        };
        let input_types = inputs.iter().map(|input| input.value().r#type().into_owned()).collect::<Vec<_>>();
        let mut inputs = inputs.iter();
        let axes = self
            .operation()
            .parameters()
            .iter()
            .map(|parameter| {
                if parameter.access() == KernelParameterAccess::WriteOnly {
                    Some(0)
                } else {
                    inputs.next().unwrap().batch_axis_position()
                }
            })
            .collect::<Vec<_>>();
        let output_axes = self
            .operation()
            .parameters()
            .iter()
            .zip(&axes)
            .filter_map(|(parameter, axis)| (parameter.access() != KernelParameterAccess::ReadOnly).then_some(*axis))
            .collect();
        let batched = self.batched(batch_size, &axes, maximum_programs)?;
        if batched.operation().output_types_for(&input_types).is_err() {
            return Err(unsupported("batched kernel parameter types differ from the packed input types"));
        }
        Ok((batched, output_axes))
    }
}

impl<C, Extension> BatchableOperation<C, ArrayIrBatchingPolicy> for KernelOperation<Extension>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = Self>,
    Extension: KernelExtension,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
        // Batching rewrites the grid, the parameters, and the body of a kernel call together, so only a complete call
        // can be batched. Its attached body region is the definition body.
        let Self::Call(call) = self else {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("kernel operation `{}` must be batched through its complete kernel call", self.name()),
            });
        };
        let definition =
            KernelDefinition::new(call.clone(), driver.region(0)?.to_program()).map_err(ProgramError::custom)?;
        let (definition, axes) = definition
            .batched_inputs(inputs, DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS)
            .map_err(|error| match error {
                KernelTransformError::Batching(error) => error,
                error => ProgramError::custom(error).into(),
            })?;
        let values = context.parent().bind(
            definition.operation().clone(),
            vec![definition.body().clone()],
            &inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>(),
        )?;
        values
            .into_iter()
            .zip(axes)
            .map(|(value, axis)| ArrayIrBatch::new(value, axis.map(|axis| axis as isize)))
            .collect::<Result<Vec<_>, _>>()
    }
}

/// Creates a diagnostic for an unsupported transform input contract.
fn unsupported(message: impl Into<String>) -> KernelTransformError {
    KernelTransformError::Unsupported { message: message.into() }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArraySliceAxis, ArrayType, AttachedRegions, ConditionOperation, DataType, DimensionValue, DimensionVariable,
        EagerContext, Layout, LogicalMesh, MeshAxis, MeshAxisType, RecursiveBatchingDriver, ReferenceRead,
        ReferenceWrite, ReferenceWriteOperation, Shape, Sharding, ShardingDimension, TiledLayout, Tracer,
        TracingContext,
    };

    use crate::authoring::whole_array_parameter;
    use crate::memory::{
        AsyncCopyOperation, MaskedStoreOperation, WINDOW_MASK_OPERATION_NAME, WaitOperation, WindowMaskOperation,
    };

    use super::*;

    /// Returns an `f32` array type of the provided static `shape` placed on `dimensions` of a mesh with the explicit
    /// axes `x` and `y` and the manual axis `m`, each of size 2.
    fn sharded_type(shape: &[usize], dimensions: Vec<ShardingDimension>) -> ArrayType {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        ArrayType::new_static(DataType::F32, shape.to_vec())
            .with_sharding(Sharding::new(mesh, dimensions).unwrap())
            .unwrap()
    }

    /// Returns a kernel over `f32[2]` items with a mapped input, a shared read-only input, a write-only output, and a
    /// read-write accumulator, whose body copies the input into the output and the shared input into the accumulator.
    fn copy_definition() -> KernelDefinition {
        let item = ArrayType::new_static(DataType::F32, [2]);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(item.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(item.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(item.clone(), KernelParameterAccess::WriteOnly).unwrap(),
                whole_array_parameter(item, KernelParameterAccess::ReadWrite).unwrap(),
            ],
        )
        .unwrap();
        KernelDefinition::trace(call, |(references, _)| {
            references[2].write(&references[0].read()?)?;
            references[3].write(&references[1].read()?)
        })
        .unwrap()
    }

    #[test]
    fn test_kernel_transform_error_display() {
        assert_eq!(
            KernelTransformError::Kernel(KernelError::Program(ProgramError::MalformedProgram(
                "missing entry region".to_owned(),
            )))
            .to_string(),
            "encountered malformed program: missing entry region",
        );
        assert_eq!(
            KernelTransformError::Initialization(KernelInitializationError::Program(ProgramError::MalformedProgram(
                "missing entry region".to_owned(),
            )))
            .to_string(),
            "encountered malformed program: missing entry region",
        );
        assert_eq!(
            KernelTransformError::Unsupported { message: "batch axes must match every kernel parameter".to_owned() }
                .to_string(),
            "kernel transform is unsupported: batch axes must match every kernel parameter",
        );
    }

    #[test]
    fn test_kernel_definition_batched() {
        let scalar = ArrayType::scalar(DataType::I32);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(scalar.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(scalar, KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(call, |(references, _)| {
            references[1].write(&references[0].read()?)?;
            Ok(())
        })
        .unwrap();
        let batched = definition.batched(3, &[Some(0), Some(0)], 10).unwrap();
        assert_eq!(batched.operation().grid().dimensions().len(), 1);
        assert_eq!(
            batched.body().to_string(),
            indoc! {"
                lambda %0:ref<i32[1]>, %1:ref<i32[1]>, %2:dimension<batch ∈ [0, 3)> .
                let %3:i32[] = reference_read [transforms=[index(axis=0, index=0)]] %0
                    () = reference_write [transforms=[index(axis=0, index=0)]] %1 %3
                in ()"},
        );
        assert_eq!(
            batched.operation().output_types(),
            vec![ArrayIrType::Array(ArrayType::new_static(DataType::I32, [3]))],
        );
        let input = Array::from_elements(ArrayType::new_static(DataType::I32, [3]), &[2i32, 5, 9]).unwrap();
        assert_eq!(batched.interpret(vec![input.clone()], 10).unwrap(), vec![input]);
        let shared = definition.batched(3, &[None, Some(0)], 10).unwrap();
        let input = Array::from_elements(ArrayType::scalar(DataType::I32), &[7i32]).unwrap();
        assert_eq!(
            shared.interpret(vec![input], 10).unwrap(),
            vec![Array::from_elements(ArrayType::new_static(DataType::I32, [3]), &[7i32, 7, 7]).unwrap()],
        );
        let empty = definition.batched(0, &[Some(0), Some(0)], 0).unwrap();
        let input = Array::from_elements(ArrayType::new_static(DataType::I32, [0]), &[] as &[i32]).unwrap();
        assert_eq!(empty.interpret(vec![input.clone()], 0).unwrap(), vec![input]);
    }

    #[test]
    fn test_kernel_definition_batched_independent_copy_transforms() {
        let vector = ArrayType::new_static(DataType::I32, [2]);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(vector.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(vector, KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let source = builder.add_input(call.parameters()[0].body_type());
        let destination = builder.add_input(call.parameters()[1].body_type());
        let selection = ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1)] };
        let token = builder
            .add_instruction(
                AsyncCopyOperation::new()
                    .with_source_transforms(vec![selection.clone()])
                    .with_destination_transforms(vec![selection.clone()]),
                vec![],
                vec![source, destination],
                None,
            )
            .unwrap()[0];
        builder.add_instruction(WaitOperation, vec![], vec![token], None).unwrap();
        let body = builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap();
        let definition = KernelDefinition::new(call, body).unwrap();
        let batched = definition.batched(3, &[Some(0), Some(1)], 10).unwrap();
        assert_eq!(
            batched.body().to_string(),
            indoc! {"
                lambda %0:ref<i32[1, 2]>, %1:ref<i32[2, 1]>, %2:dimension<batch ∈ [0, 3)> .
                let %3:ref<token[]> = async_copy [
                    source_transforms=[index(axis=0, index=0), slice(axes=[0:2])],
                    destination_transforms=[index(axis=1, index=0), slice(axes=[0:2])],
                ] %0 %1
                    () = wait %3
                in ()"},
        );
        let input = Array::from_elements(ArrayType::new_static(DataType::I32, [3, 2]), &[1i32, 2, 3, 4, 5, 6]).unwrap();
        let output =
            Array::from_elements(ArrayType::new_static(DataType::I32, [2, 3]), &[1i32, 3, 5, 2, 4, 6]).unwrap();
        assert_eq!(batched.interpret(vec![input], 10).unwrap(), vec![output]);
    }

    #[test]
    fn test_kernel_definition_batched_rejects_shared_writes() {
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(call, |(references, _)| {
            references[0].write(&references[0].read()?)?;
            Ok(())
        })
        .unwrap();
        assert!(matches!(
            definition.batched(2, &[None], 10),
            Err(KernelTransformError::Unsupported { message })
                if message == "writable parameter 0 requires a mapped batch axis",
        ));
        assert!(matches!(
            definition.batched(2, &[], 10),
            Err(KernelTransformError::Unsupported { message })
                if message == "batch axes must match every kernel parameter",
        ));
    }

    #[test]
    fn test_kernel_definition_batched_rejects_non_access_reference_uses() {
        // A mapped reference can only be batched by prefixing an access path. Passing it into an operation that carries
        // regions, such as `condition`, is not an access, so no path exists to receive the batch selection.
        let scalar = ArrayType::scalar(DataType::I32);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(scalar.clone(), KernelParameterAccess::WriteOnly).unwrap()],
        )
        .unwrap();
        let mut branch = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = branch.add_input(call.parameters()[0].body_type());
        let value = branch.add_input(ArrayIrType::Array(scalar));
        branch
            .add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, value], None)
            .unwrap();
        let branch = branch
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(call.parameters()[0].body_type());
        let predicate = builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
        let value = builder.add_constant(ArrayIrValue::Array(Array::scalar(7i32).unwrap()));
        let branch = builder.import_region(branch.entry_region_ref());
        builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![branch, branch],
                vec![predicate, reference, value],
                None,
            )
            .unwrap();
        let body = builder.build(vec![], vec![Placeholder], vec![]).unwrap();
        let definition = KernelDefinition::new(call, body).unwrap();
        assert!(matches!(
            definition.batched(2, &[Some(0)], 10),
            Err(KernelTransformError::Unsupported { message })
                if message == "operation `condition` uses mapped reference input 1 outside a reference access",
        ));
    }

    #[test]
    fn test_kernel_definition_batched_rejects_window_masks() {
        // The validity of a mapped window is not a reference access, so no access path can receive the batch
        // selection and the widened window's validity would have the wrong shape. Batching therefore rejects it.
        let vector = ArrayType::new_static(DataType::I32, [2]);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(vector, KernelParameterAccess::WriteOnly).unwrap()],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let output = builder.add_input(call.parameters()[0].body_type());
        let value = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![5i32, 7]).unwrap()));
        let mask = builder.add_instruction(WindowMaskOperation, vec![], vec![output], None).unwrap()[0];
        builder
            .add_instruction(MaskedStoreOperation::new(), vec![], vec![output, value, mask], None)
            .unwrap();
        let body = builder.build(vec![], vec![Placeholder], vec![]).unwrap();
        let definition = KernelDefinition::new(call, body).unwrap();
        assert_eq!(definition.interpret(vec![], 1), Ok(vec![Array::vector(vec![5i32, 7]).unwrap()]));
        assert!(matches!(
            definition.batched(2, &[Some(0)], 10),
            Err(KernelTransformError::Unsupported { message })
                if message == format!(
                    "operation `{WINDOW_MASK_OPERATION_NAME}` uses mapped reference input 0 outside a reference access",
                ),
        ));
    }

    #[test]
    fn test_kernel_definition_batched_inputs() {
        let vector = ArrayType::new_static(DataType::I32, [2]);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(vector.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(vector, KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition =
            KernelDefinition::trace(call, |(references, _)| references[1].write(&references[0].read()?)).unwrap();
        let input = Array::from_elements(ArrayType::new_static(DataType::I32, [2, 3]), &[1i32, 2, 3, 4, 5, 6]).unwrap();
        let (batched, axes) = definition
            .batched_inputs(&[ArrayIrBatch::new(ArrayIrValue::Array(input.clone()), Some(1)).unwrap()], 3)
            .unwrap();
        assert_eq!(axes, vec![Some(0)]);
        assert_eq!(
            batched.interpret(vec![input], 3).unwrap(),
            vec![Array::from_elements(ArrayType::new_static(DataType::I32, [3, 2]), &[1i32, 4, 2, 5, 3, 6]).unwrap()],
        );
        let value =
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[7i32, 9]).unwrap());
        let (unmapped, axes) = definition.batched_inputs(&[ArrayIrBatch::new(value, None).unwrap()], 1).unwrap();
        assert_eq!(axes, vec![None]);
        assert_eq!(unmapped.semantic_key().unwrap(), definition.semantic_key().unwrap());

        // Batching clears the layout of a mapped parameter, so a packed input with an explicit layout could not be
        // passed to the batched call.
        let layout = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let r#type = ArrayType::new_static(DataType::I32, [2, 3]).with_layout(layout);
        let input = Array::from_elements(r#type, &[1i32, 2, 3, 4, 5, 6]).unwrap();
        assert!(matches!(
            definition.batched_inputs(&[ArrayIrBatch::new(ArrayIrValue::Array(input), Some(1)).unwrap()], 3),
            Err(KernelTransformError::Unsupported { message })
                if message == "batched kernel parameter types differ from the packed input types",
        ));
    }

    #[test]
    fn test_kernel_definition_batched_inputs_rejects_dynamic_batch_extents() {
        // The batch becomes a static grid dimension of the batched kernel, so a dynamic batch extent is rejected.
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let vector = ArrayType::new_static(DataType::I32, [4]);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(vector.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(vector, KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition =
            KernelDefinition::trace(call, |(references, _)| references[1].write(&references[0].read()?)).unwrap();
        let items = DimensionVariable::new("items", DimensionBounds::new(0, Some(5)).unwrap());
        TestContext::trace(
            |(_, values): (Tracer<TestContext>, Tracer<TestContext>)| {
                assert!(matches!(
                    definition.batched_inputs(&[ArrayIrBatch::new(values.clone(), Some(0)).unwrap()], 1),
                    Err(KernelTransformError::Unsupported { message })
                        if message == "kernel batching requires a static batch extent, which becomes a static grid \
                                       dimension",
                ));
                Ok(values)
            },
            (
                ArrayIrType::Dimension(DimensionType::from(Dimension::Dynamic(items.clone()))),
                ArrayIrType::Array(ArrayType::new(
                    DataType::I32,
                    Shape::new(vec![Dimension::Dynamic(items), Dimension::Static(4)]),
                )),
            ),
        )
        .unwrap();
    }

    #[test]
    fn test_kernel_definition_batched_inputs_placed() {
        // The kernel is physical, and the packed inputs are placed over a mesh in three different ways. The batched
        // call accepts them as they are. The read-write result is the storage of its packed input and keeps that
        // type, and the write-only result has no placement.
        let definition = copy_definition();
        let item = sharded_type(&[2], vec![ShardingDimension::sharded(["y"])]);
        let shared = Array::from_elements(item, &[5.0f32, 6.0]).unwrap();
        for axis in ["x", "m"] {
            let input_type =
                sharded_type(&[2, 2], vec![ShardingDimension::sharded([axis]), ShardingDimension::sharded(["y"])]);
            let accumulator_type =
                sharded_type(&[2, 2], vec![ShardingDimension::sharded(["y"]), ShardingDimension::sharded([axis])]);
            let input = Array::from_elements(input_type, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
            let accumulator = Array::from_elements(accumulator_type.clone(), &[0.0f32; 4]).unwrap();
            let inputs = [
                ArrayIrBatch::new(ArrayIrValue::Array(input.clone()), Some(0)).unwrap(),
                ArrayIrBatch::new(ArrayIrValue::Array(shared.clone()), None).unwrap(),
                ArrayIrBatch::new(ArrayIrValue::Array(accumulator.clone()), Some(1)).unwrap(),
            ];
            let (batched, axes) = definition.batched_inputs(&inputs, 4).unwrap();
            assert_eq!(axes, vec![Some(0), Some(1)]);
            let matrix = ArrayType::new_static(DataType::F32, [2, 2]);
            assert_eq!(
                batched.operation().input_types(),
                vec![
                    ArrayIrType::Array(matrix.clone()),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
                    ArrayIrType::Array(matrix.clone()),
                ],
            );
            assert_eq!(
                batched.semantic_key().unwrap(),
                definition.batched(2, &[Some(0), None, Some(0), Some(1)], 4).unwrap().semantic_key().unwrap()
            );

            // The host interpreter executes the placed arrays as ordinary host arrays.
            let copied = Array::from_elements(matrix, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
            let accumulated = Array::from_elements(accumulator_type, &[5.0f32, 5.0, 6.0, 6.0]).unwrap();
            let outputs = batched.interpret(vec![input.clone(), shared.clone(), accumulator], 4).unwrap();
            assert_eq!(outputs, vec![copied.clone(), accumulated.clone()]);

            // Ordinary batching binds the batched call on the packed inputs.
            let parent = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
            let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
                parent,
                ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
            );
            let callees = [Arc::new(definition.body().clone())];
            let regions = AttachedRegions::Callees(&callees);
            let driver = RecursiveBatchingDriver::new(&regions);
            let outputs =
                KernelOperation::Call(definition.operation().clone()).batch(&context, &driver, &inputs).unwrap();
            assert_eq!(
                outputs,
                vec![
                    ArrayIrBatch::new(ArrayIrValue::Array(copied), Some(0)).unwrap(),
                    ArrayIrBatch::new(ArrayIrValue::Array(accumulated), Some(1)).unwrap(),
                ],
            );
        }

        // Placed inputs must share one mesh, like the inputs of any other operation.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other = ArrayType::new_static(DataType::F32, [2, 2])
            .with_sharding(
                Sharding::new(other_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let input_type =
            sharded_type(&[2, 2], vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])]);
        let batch = |r#type: ArrayType, axis: isize| {
            ArrayIrBatch::new(ArrayIrValue::Array(Array::from_elements(r#type, &[0.0f32; 4]).unwrap()), Some(axis))
                .unwrap()
        };
        let shared = ArrayIrBatch::new(ArrayIrValue::Array(shared), None).unwrap();
        assert!(matches!(
            definition.batched_inputs(&[batch(input_type, 0), shared, batch(other, 0)], 4),
            Err(KernelTransformError::Unsupported { message })
                if message == "batched input types differ from the kernel signature",
        ));
    }

    #[test]
    fn test_kernel_operation_batch() {
        let scalar = ArrayType::scalar(DataType::I32);
        let definition: KernelDefinition = KernelDefinition::trace(
            KernelCallOperation::new(
                Grid::new(vec![]).unwrap(),
                vec![
                    whole_array_parameter(scalar.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                    whole_array_parameter(scalar, KernelParameterAccess::WriteOnly).unwrap(),
                ],
            )
            .unwrap(),
            |(references, _)| references[1].write(&references[0].read()?),
        )
        .unwrap();
        let parent = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            parent,
            ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
        );
        let callees = [Arc::new(definition.body().clone())];
        let regions = AttachedRegions::Callees(&callees);
        let driver = RecursiveBatchingDriver::new(&regions);
        let value = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::I32, [3]), &[2i32, 5, 9]).unwrap(),
        );
        let outputs = KernelOperation::Call(definition.operation().clone())
            .batch(&context, &driver, &[ArrayIrBatch::new(value.clone(), Some(0)).unwrap()])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].value(), &value);
        assert_eq!(outputs[0].batch_axis_position(), Some(0));
    }
}
