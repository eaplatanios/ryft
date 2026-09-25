//! Kernel transforms preserve canonical regions, array metadata, and confined reference effects.
//!
//! Batching adds a grid coordinate and disjoint array windows rather than vectorizing mutable instructions. Static
//! specialization uses the existing program splicer; derivative rules belong to the ordinary custom JVP/VJP carriers.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::arrays::{
    Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayReferenceTransform,
    ArrayReferenceTransformIndex, Dimension, DimensionBounds, DimensionType,
};
use crate::batching::{BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::Context;
use crate::kernels::calls::{KernelCallOperation, KernelDefinition, KernelError, KernelParameter};
use crate::kernels::compilation::VerifiedKernel;
use crate::kernels::grids::{Grid, GridDimension, GridExecution};
use crate::kernels::initialization::KernelInitializationError;
use crate::kernels::interpretation::DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS;
use crate::kernels::mappings::{BlockMapping, BoundaryPolicy};
use crate::kernels::operations::{KernelExtension, KernelOperation};
use crate::kernels::validation::KernelParameterAccess;
use crate::parameters::Placeholder;
use crate::programs::{
    Atom, Operation, ProgramBuilder, ProgramError, Typed, Value, rewrite_reference_access_transforms,
    validated_reference_access_descriptors,
};

/// A kernel transform cannot preserve the declared boundary or prove its rewritten accesses.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum KernelTransformError {
    /// Canonical construction rejected the rewritten program or types.
    #[error(transparent)]
    Kernel(#[from] KernelError),

    /// Rewritten accesses do not satisfy executable initialization and race requirements.
    #[error(transparent)]
    Initialization(#[from] KernelInitializationError),

    /// The selected transform has no proven rule for this input contract.
    #[error("kernel transform is unsupported: {message}")]
    Unsupported {
        /// Exact unsupported contract.
        message: String,
    },
}

impl<Extension: KernelExtension> KernelDefinition<Extension> {
    /// Adds an independent parallel batch coordinate and canonical size-one windows for mapped parameters. The body
    /// selects the original referents through folded access paths and executes once per batch item. Unmapped read-only
    /// arrays remain shared; every writable parameter must be mapped, including functional read-write aliases.
    ///
    /// `axes` follows all parameters, including write-only outputs. Each mapped axis is inserted into that parameter's
    /// full array and block shape. Canonical type insertion preserves dtype, memory, and sharding while clearing
    /// layout; explicit source layouts are rejected because restoring them on the selected referents is not justified.
    /// Scalar-prefetched inputs must be specialized first. A new executable proof checks all transformed accesses.
    ///
    /// # Parameters
    ///
    ///   - `batch_size`: Static number of independent items; zero produces an empty launch.
    ///   - `axes`: Inserted array dimension for each mapped parameter, or `None` for a shared read-only parameter.
    ///   - `maximum_programs`: Enumeration budget for executable qualification when affine proofs do not apply.
    pub fn batched(
        &self,
        batch_size: usize,
        axes: &[Option<usize>],
        maximum_programs: usize,
    ) -> Result<Self, KernelTransformError> {
        let call = self.operation();
        let unsupported = |message: String| KernelTransformError::Unsupported { message };
        if axes.len() != call.parameters().len() {
            return Err(unsupported("batch axes must match every kernel parameter".to_owned()));
        }
        if !call.prefetch_types().is_empty() {
            return Err(unsupported("scalar-prefetched inputs must be specialized before batching".to_owned()));
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
                .into_iter()
                .map(|r#type| builder.add_input(r#type))
                .collect::<Vec<_>>();
            let batch = builder.add_input(ArrayIrType::Dimension(coordinate.clone()));
            let mut starts =
                builder.splice_program(parameter.mapping().program(), &inputs).map_err(KernelError::from)?;
            let mut shape = parameter.mapping().block_shape().to_vec();
            let mut r#type = parameter.r#type().into_owned();
            if let Some(axis) = *axis {
                r#type =
                    r#type.with_inserted_dimension(axis, Dimension::Static(batch_size)).map_err(KernelError::from)?;
                starts.insert(axis, batch);
                shape.insert(axis, 1);
            }
            let program = builder
                .build(starts, vec![Placeholder; inputs.len() + 1], vec![Placeholder; shape.len()])
                .map_err(KernelError::from)?;
            // No invocation exists for an empty batch. Padding preserves the formal selected referent type without
            // constructing an invalid one-element slice of the empty full operand.
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
                let mut transforms = vec![ArrayReferenceTransform::Index { axis, index: ArrayReferenceTransformIndex::Static(0) }];
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

    /// Derives parameter batch axes from canonical packed input carriers and batches the definition. Write-only
    /// results acquire a leading batch axis; read-write results retain their input's axis. With entirely replicated
    /// inputs the unchanged definition and replicated result axes are returned. Dynamic and ragged packed extents
    /// require a separate proven mapping rule and are rejected before any kernel executes.
    pub fn batched_inputs<V: Value<Type = ArrayIrType>>(
        &self,
        inputs: &[ArrayIrBatch<V>],
        maximum_programs: usize,
    ) -> Result<(Self, Vec<Option<usize>>), KernelTransformError> {
        let unsupported = |message: &str| KernelTransformError::Unsupported { message: message.to_owned() };
        if inputs.iter().map(ArrayIrBatch::unbatched_type).collect::<Vec<_>>() != self.operation().input_types() {
            return Err(unsupported("batched input types differ from the kernel signature"));
        }
        let mut batch_size = None;
        for input in inputs {
            if !input.ragged_axes().is_empty() {
                return Err(unsupported("ragged kernel batching requires specialized block mappings"));
            }
            if let Some(axis) = input.batch_axis_position() {
                let r#type = input.value().r#type();
                let ArrayIrType::Array(r#type) = r#type.as_ref() else {
                    return Err(unsupported("kernel batching accepts only ordinary array inputs"));
                };
                let shape = r#type.static_shape().ok_or_else(|| unsupported("kernel batch extents must be static"))?;
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
        Ok((self.batched(batch_size, &axes, maximum_programs)?, output_axes))
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
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        let Self::Call(call) = self else {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("kernel operation `{}` must be batched through its complete kernel call", self.name()),
            });
        };
        let definition =
            KernelDefinition::new(call.clone(), driver.region(0)?.to_program()).map_err(ProgramError::custom)?;
        let (definition, axes) = definition
            .batched_inputs(inputs, DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS)
            .map_err(ProgramError::custom)?;
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
            .map(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArraySliceAxis, ArrayType, DataType};
    use crate::kernels::authoring::whole_array_parameter;
    use crate::kernels::memory::{AsyncCopyOperation, WaitOperation};
    use crate::operations::{ConditionOperation, ReferenceRead, ReferenceWrite, ReferenceWriteOperation};
    use crate::programs::ReferenceAccessOperation;

    use super::*;

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
        assert_eq!(batched.body().instructions().len(), definition.body().instructions().len());
        assert_eq!(
            batched.body().instructions()[0].operation().reference_access_descriptor(0).unwrap().transforms(),
            &[ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) }],
        );
        assert_eq!(
            batched.operation().output_types(),
            vec![ArrayIrType::Array(ArrayType::new_static(DataType::I32, [3]))]
        );
        let input = Array::from_elements(ArrayType::new_static(DataType::I32, [3]), &[2i32, 5, 9]).unwrap();
        assert_eq!(batched.interpret(vec![input.clone()], 10).unwrap(), vec![input]);
        let shared = definition.batched(3, &[None, Some(0)], 10).unwrap();
        let input = Array::from_elements(ArrayType::scalar(DataType::I32), &[7i32]).unwrap();
        assert_eq!(
            shared.interpret(vec![input], 10).unwrap(),
            vec![Array::from_elements(ArrayType::new_static(DataType::I32, [3]), &[7i32, 7, 7]).unwrap(),]
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
        let operation = batched.body().instructions()[0].operation();
        assert_eq!(
            operation.reference_access_descriptor(0).unwrap().transforms(),
            &[ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) }, selection.clone(),],
        );
        assert_eq!(
            operation.reference_access_descriptor(1).unwrap().transforms(),
            &[ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Static(0) }, selection,],
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
        // A mapped reference can only be batched by prefixing an access path. Passing it into a region-carrying
        // operation such as `condition` is not an access, so no path exists to receive the batch selection.
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
            vec![Array::from_elements(ArrayType::new_static(DataType::I32, [3, 2]), &[1i32, 4, 2, 5, 3, 6],).unwrap()]
        );
        let value =
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[7i32, 9]).unwrap());
        let (unmapped, axes) = definition.batched_inputs(&[ArrayIrBatch::new(value, None).unwrap()], 1).unwrap();
        assert_eq!(axes, vec![None]);
        assert_eq!(unmapped.semantic_key().unwrap(), definition.semantic_key().unwrap());
    }

    #[test]
    fn test_kernel_operation_batch() {
        use std::sync::Arc;

        use crate::arrays::DimensionValue;
        use crate::batching::RecursiveBatchingDriver;
        use crate::contexts::EagerContext;
        use crate::programs::CalleeRegionDriver;

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
        let regions = CalleeRegionDriver::new(&callees);
        let driver = RecursiveBatchingDriver::new(&regions);
        let value = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::I32, [3]), &[2i32, 5, 9]).unwrap(),
        );
        let (outputs, evidence) = KernelOperation::Call(definition.operation().clone())
            .batch(&context, &driver, &[ArrayIrBatch::new(value.clone(), Some(0)).unwrap()])
            .unwrap()
            .into_parts();
        assert!(evidence.is_empty());
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].value(), &value);
        assert_eq!(outputs[0].batch_axis_position(), Some(0));
    }
}
