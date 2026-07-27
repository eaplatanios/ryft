use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::backends::arrays::ArrayOperation;
use crate::backends::dimensions::{DimensionOperation, DimensionValue};
use crate::contexts::EagerContext;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::effects::Effects;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::{Operation, OperationProjection};
use crate::programs::regions::{EmptyRegionDriver, OutputRegionProvenance, RegionInterface, RegionSlot};
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{Value, ValueProjection};
use crate::types::{ArrayProgramType, ArrayType, DimensionType, DimensionVariable};

// TODO(eaplatanios): Review this module.

/// Closed [`Operation`] family for array programs that contain both ordinary arrays and first-class runtime
/// dimensions. This dispatcher preserves the homogeneous operation contracts of [`ArrayOperation`] and
/// [`DimensionOperation`]: it selects the member family, projects the composite type boundary once, delegates to that
/// family, and lifts the inferred result types back into [`ArrayProgramType`].
///
/// Operations whose signatures mix arrays and dimensions are intentionally not represented here. They are introduced
/// as explicit outer-family variants when their operand contracts are migrated, because no homogeneous member family
/// can express such a signature.
#[derive(Clone, Debug)]
pub enum ArrayProgramOperation<A: Value<Type = ArrayType>> {
    /// Homogeneous array operation.
    Array(ArrayOperation<A>),

    /// Homogeneous first-class-dimension operation.
    Dimension(DimensionOperation<DimensionValue>),
}

impl<A: Value<Type = ArrayType>> Display for ArrayProgramOperation<A> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<A: Value<Type = ArrayType>> From<ArrayOperation<A>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: ArrayOperation<A>) -> Self {
        Self::Array(operation)
    }
}

impl<A: Value<Type = ArrayType>> From<DimensionOperation<DimensionValue>> for ArrayProgramOperation<A> {
    #[inline]
    fn from(operation: DimensionOperation<DimensionValue>) -> Self {
        Self::Dimension(operation)
    }
}

impl<A: Value<Type = ArrayType>> OperationProjection<ArrayType> for ArrayProgramOperation<A> {
    type Projected = ArrayOperation<A>;
}

impl<A: Value<Type = ArrayType>> OperationProjection<DimensionType> for ArrayProgramOperation<A> {
    type Projected = DimensionOperation<DimensionValue>;
}

/// Projects the complete inference boundary for one homogeneous inner operation while preserving region effects.
fn project_operation_boundary<T: Type>(
    input_types: &[ArrayProgramType],
    region_interfaces: &[RegionInterface<ArrayProgramType>],
) -> Result<(Vec<T>, Vec<RegionInterface<T>>), TypeError>
where
    for<'t> &'t T: TryFrom<&'t ArrayProgramType, Error = TypeError>,
{
    Ok((
        input_types.iter().map(|r#type| <&T>::try_from(r#type).cloned()).collect::<Result<_, _>>()?,
        region_interfaces
            .iter()
            .map(|interface| {
                Ok(RegionInterface::new(
                    interface
                        .input_types()
                        .iter()
                        .map(|r#type| <&T>::try_from(r#type).cloned())
                        .collect::<Result<_, _>>()?,
                    interface
                        .output_types()
                        .iter()
                        .map(|r#type| <&T>::try_from(r#type).cloned())
                        .collect::<Result<_, _>>()?,
                    interface.effects(),
                ))
            })
            .collect::<Result<Vec<_>, TypeError>>()?,
    ))
}

impl<A: Value<Type = ArrayType>> Operation<ArrayProgramType> for ArrayProgramOperation<A> {
    #[inline]
    fn name(&self) -> &'static str {
        match self {
            Self::Array(operation) => operation.name(),
            Self::Dimension(operation) => operation.name(),
        }
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        match self {
            Self::Array(operation) => operation.region_slots(),
            Self::Dimension(operation) => operation.region_slots(),
        }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[ArrayProgramType],
        region_interfaces: &[RegionInterface<ArrayProgramType>],
    ) -> Result<Vec<Option<Vec<ArrayProgramType>>>, TypeError> {
        match self {
            Self::Array(operation) => {
                let (input_types, region_interfaces) = project_operation_boundary(input_types, region_interfaces)?;
                Ok(operation
                    .infer_region_input_types(&input_types, &region_interfaces)?
                    .into_iter()
                    .map(|types| types.map(|types| types.into_iter().map(Into::into).collect()))
                    .collect())
            }
            Self::Dimension(operation) => {
                let (input_types, region_interfaces) = project_operation_boundary(input_types, region_interfaces)?;
                Ok(operation
                    .infer_region_input_types(&input_types, &region_interfaces)?
                    .into_iter()
                    .map(|types| types.map(|types| types.into_iter().map(Into::into).collect()))
                    .collect())
            }
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayProgramType],
        region_interfaces: &[RegionInterface<ArrayProgramType>],
    ) -> Result<Vec<ArrayProgramType>, TypeError> {
        match self {
            Self::Array(operation) => {
                let (input_types, region_interfaces) = project_operation_boundary(input_types, region_interfaces)?;
                Ok(operation
                    .infer_output_types(&input_types, &region_interfaces)?
                    .into_iter()
                    .map(Into::into)
                    .collect())
            }
            Self::Dimension(operation) => {
                let (input_types, region_interfaces) = project_operation_boundary(input_types, region_interfaces)?;
                Ok(operation
                    .infer_output_types(&input_types, &region_interfaces)?
                    .into_iter()
                    .map(Into::into)
                    .collect())
            }
        }
    }

    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        match self {
            Self::Array(operation) => operation.output_region_provenance(output_index),
            Self::Dimension(operation) => operation.output_region_provenance(output_index),
        }
    }

    #[inline]
    fn is_zero(&self, output_index: usize) -> bool {
        match self {
            Self::Array(operation) => operation.is_zero(output_index),
            Self::Dimension(operation) => operation.is_zero(output_index),
        }
    }

    #[inline]
    fn effects(&self) -> Effects {
        match self {
            Self::Array(operation) => operation.effects(),
            Self::Dimension(operation) => operation.effects(),
        }
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        match self {
            Self::Array(operation) => Ok(Self::Array(operation.rename_type_identities(renaming)?)),
            Self::Dimension(operation) => Ok(Self::Dimension(operation.rename_type_identities(renaming)?)),
        }
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Array(operation) => operation.render(formatter, indentation),
            Self::Dimension(operation) => operation.render(formatter, indentation),
        }
    }
}

/// Interprets one homogeneous operation family using its native eager domain and lifts the results back into the
/// composite value family. Common operation arities stay on the stack; only wider operations allocate an input vector.
fn interpret_homogeneous_operation<
    T: Type,
    V: Value<Type = T>,
    O: Operation<T> + InterpretableOperation<EagerContext<V, O>>,
    A: Value<Type = ArrayType>,
>(
    operation: &O,
    inputs: &[ArrayProgramValue<A>],
) -> Result<Vec<ArrayProgramValue<A>>, ProgramError>
where
    ArrayProgramValue<A>: ValueProjection<T, Projected = V>,
{
    let context = EagerContext::<V, O>::new();
    let interpret = |inputs: &[V]| operation.interpret(&context, &EmptyRegionDriver, inputs);
    let outputs = match inputs {
        [] => interpret(&[]),
        [input] => {
            let inputs = [<ArrayProgramValue<A> as ValueProjection<T>>::into_projected(input.clone())?];
            interpret(&inputs)
        }
        [left, right] => {
            let inputs = [
                <ArrayProgramValue<A> as ValueProjection<T>>::into_projected(left.clone())?,
                <ArrayProgramValue<A> as ValueProjection<T>>::into_projected(right.clone())?,
            ];
            interpret(&inputs)
        }
        inputs => {
            let inputs = inputs
                .iter()
                .cloned()
                .map(<ArrayProgramValue<A> as ValueProjection<T>>::into_projected)
                .collect::<Result<Vec<_>, _>>()?;
            interpret(&inputs)
        }
    }?;
    Ok(outputs.into_iter().map(<ArrayProgramValue<A> as ValueProjection<T>>::from_projected).collect())
}

impl<A: Value<Type = ArrayType>> InterpretableOperation<EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>>
    for ArrayProgramOperation<A>
where
    ArrayOperation<A>: InterpretableOperation<EagerContext<A, ArrayOperation<A>>>,
    DimensionOperation<DimensionValue>:
        InterpretableOperation<EagerContext<DimensionValue, DimensionOperation<DimensionValue>>>,
{
    fn interpret<D: InterpretationDriver<EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>>>(
        &self,
        _context: &EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>,
        driver: &D,
        inputs: &[ArrayProgramValue<A>],
    ) -> Result<Vec<ArrayProgramValue<A>>, ProgramError> {
        if !self.region_slots().is_empty() || driver.region_count() != 0 {
            return Err(ProgramError::MalformedProgram(format!(
                "projected operation `{}` cannot carry regions",
                self.name(),
            )));
        }
        match self {
            Self::Array(operation) => interpret_homogeneous_operation(operation, inputs),
            Self::Dimension(operation) => interpret_homogeneous_operation(operation, inputs),
        }
    }
}

/// [`Value`]-level counterpart to [`ArrayProgramType`] that is used by [`Program`](crate::Program)s that may contain
/// both [`ArrayType`]-typed [`Value`]s and [`DimensionValue`]. `A` is the concrete array representation selected by the
/// owning backend. Dimensions use the common [`DimensionValue`] which is a checked host representation, so that eager
/// dimension arithmetic remains host integer work and does not allocate arrays or dispatch to device backends.
///
/// This type allows us to keep arrays and checked host-side dimensions in one storage universe while
/// [`ValueProjection`] lets homogeneous [`Operation`](crate::Operation) machinery borrow or consume only
/// the member that it understands.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum ArrayProgramValue<A: Value<Type = ArrayType>> {
    /// Ordinary backend [`ArrayType`]-typed [`Value`].
    Array(A),

    /// Checked host-side runtime [`DimensionValue`].
    Dimension(DimensionValue),
}

impl<A: Value<Type = ArrayType>> Display for ArrayProgramValue<A> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Array(value) => Display::fmt(value, formatter),
            Self::Dimension(value) => Display::fmt(value, formatter),
        }
    }
}

// TODO(eaplatanios): Review from here onwards.

impl<A: Value<Type = ArrayType>> Typed for ArrayProgramValue<A> {
    type Type = ArrayProgramType;

    fn r#type(&self) -> Cow<'_, ArrayProgramType> {
        Cow::Owned(match self {
            Self::Array(value) => ArrayProgramType::Array(value.r#type().into_owned()),
            Self::Dimension(value) => ArrayProgramType::Dimension(value.r#type().clone()),
        })
    }
}

impl<A: Value<Type = ArrayType>> Value for ArrayProgramValue<A> {
    type DispatchDomain = EagerContext<Self>;
    type ExecutionDomain = EagerContext<Self, ArrayProgramOperation<A>>;

    #[inline]
    fn dispatch_domain(&self) -> Self::DispatchDomain {
        EagerContext::new()
    }

    #[inline]
    fn execution_domain(&self) -> Self::ExecutionDomain {
        EagerContext::new()
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        match self {
            Self::Array(value) => Ok(Self::Array(value.rename_type_identities(renaming)?)),
            Self::Dimension(value) => Ok(Self::Dimension(value.rename_type_identities(renaming)?)),
        }
    }
}

impl<A: Value<Type = ArrayType>> ValueProjection<ArrayType> for ArrayProgramValue<A> {
    type Projected = A;
    type ProjectedRef<'v>
        = &'v A
    where
        Self: 'v;

    #[inline]
    fn from_projected(value: A) -> Self {
        Self::Array(value)
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<&'v A, TypeError>
    where
        ArrayType: 'v,
    {
        match self {
            Self::Array(value) => Ok(value),
            Self::Dimension(_) => Err(TypeError::invalid("expected array type but got dimension type")),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<A, TypeError> {
        match self {
            Self::Array(value) => Ok(value),
            Self::Dimension(_) => Err(TypeError::invalid("expected array type but got dimension type")),
        }
    }
}

impl<A: Value<Type = ArrayType>> ValueProjection<DimensionType> for ArrayProgramValue<A> {
    type Projected = DimensionValue;
    type ProjectedRef<'v>
        = &'v DimensionValue
    where
        Self: 'v;

    #[inline]
    fn from_projected(value: DimensionValue) -> Self {
        Self::Dimension(value)
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<&'v DimensionValue, TypeError>
    where
        DimensionType: 'v,
    {
        match self {
            Self::Array(_) => Err(TypeError::invalid("expected dimension type but got array type")),
            Self::Dimension(value) => Ok(value),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<DimensionValue, TypeError> {
        match self {
            Self::Array(_) => Err(TypeError::invalid("expected dimension type but got array type")),
            Self::Dimension(value) => Ok(value),
        }
    }
}

impl<A: Value<Type = ArrayType>> From<A> for ArrayProgramValue<A> {
    #[inline]
    fn from(value: A) -> Self {
        Self::Array(value)
    }
}

impl<A: Value<Type = ArrayType>> From<DimensionValue> for ArrayProgramValue<A> {
    #[inline]
    fn from(value: DimensionValue) -> Self {
        Self::Dimension(value)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::contexts::{Context, StagingContext};
    use crate::differentiation::DifferentiationTracer;
    use crate::operations::constants::{ConstantOperation, ZeroOperation};
    use crate::operations::control_flow::ConditionOperation;
    use crate::operations::dimensions::{DimensionAddOperation, DimensionRequirementOperation};
    use crate::operations::math::AddOperation;
    use crate::partial::PartialTracer;
    use crate::programs::effects::{Effect, Effects};
    use crate::programs::operations::OperationProjection;
    use crate::tracing::{Tracer, TracingContext};
    use crate::types::{DataType, Dimension, DimensionBounds, DimensionVariable, Shape};

    use super::*;

    #[test]
    fn test_array_program_value_projection() {
        let array = Array::vector((0..4096).map(|value| value as f32).collect());
        let payload = array.values().as_ptr();
        let stored = ArrayProgramValue::Array(array);

        let projected = <ArrayProgramValue<Array> as ValueProjection<ArrayType>>::projected(&stored).unwrap();
        assert_eq!(projected.values().as_ptr(), payload);
        assert_eq!(
            <ArrayProgramValue<Array> as ValueProjection<DimensionType>>::projected(&stored),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );

        let projected = <ArrayProgramValue<Array> as ValueProjection<ArrayType>>::into_projected(stored).unwrap();
        assert_eq!(projected.values().as_ptr(), payload);
    }

    #[test]
    fn test_array_program_dimension_projection() {
        let variable = DimensionVariable::new("extent", DimensionBounds::positive(Some(9)).unwrap());
        let dimension = DimensionValue::new(DimensionType::new(variable), 4).unwrap();
        let stored = ArrayProgramValue::<Array>::Dimension(dimension.clone());

        assert_eq!(<ArrayProgramValue<Array> as ValueProjection<DimensionType>>::projected(&stored), Ok(&dimension),);
        assert_eq!(
            <ArrayProgramValue<Array> as ValueProjection<ArrayType>>::projected(&stored),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(<ArrayProgramValue<Array> as ValueProjection<DimensionType>>::into_projected(stored), Ok(dimension),);
    }

    #[test]
    fn test_array_program_type_projection() {
        let array = ArrayType::new(DataType::F32, Shape::scalar());
        let stored = ArrayProgramType::from(array.clone());
        assert_eq!(<&ArrayType>::try_from(&stored), Ok(&array));
        assert_eq!(
            <&DimensionType>::try_from(&stored),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );

        let dimension =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(9)).unwrap()));
        let stored = ArrayProgramType::from(dimension.clone());
        assert_eq!(<&DimensionType>::try_from(&stored), Ok(&dimension));
        assert_eq!(
            <&ArrayType>::try_from(&stored),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
    }

    #[test]
    fn test_array_program_operation() {
        fn assert_projection<T: Type, O: Operation<T>, C: OperationProjection<T, Projected = O>>() {}

        assert_projection::<ArrayType, ArrayOperation<Array>, ArrayProgramOperation<Array>>();
        assert_projection::<DimensionType, DimensionOperation<DimensionValue>, ArrayProgramOperation<Array>>();

        let array_type = ArrayType::scalar(DataType::F32);
        let array_operation = ArrayProgramOperation::<Array>::from(ArrayOperation::Add(AddOperation));
        assert!(matches!(array_operation, ArrayProgramOperation::Array(ArrayOperation::Add(_))));
        assert_eq!(array_operation.name(), "add");
        assert_eq!(array_operation.to_string(), "add");
        assert_eq!(
            array_operation.infer_output_types(&[array_type.clone().into(), array_type.clone().into()], &[],),
            Ok(vec![array_type.clone().into()]),
        );

        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
        let dimension_operation = ArrayProgramOperation::<Array>::from(DimensionOperation::Add(
            DimensionAddOperation::new(&left_type, &right_type).unwrap(),
        ));
        assert!(matches!(dimension_operation, ArrayProgramOperation::Dimension(DimensionOperation::Add(_)),));
        assert_eq!(dimension_operation.name(), "dimension_add");
        let result_types = dimension_operation
            .infer_output_types(&[left_type.clone().into(), right_type.clone().into()], &[])
            .unwrap();
        let [ArrayProgramType::Dimension(result_type)] = result_types.as_slice() else {
            panic!("expected one dimension result type");
        };
        assert_eq!(result_type.bounds(), DimensionBounds::new(2, Some(17)).unwrap());
        let requirement = ArrayProgramOperation::<Array>::from(DimensionOperation::Requirement(
            DimensionRequirementOperation::equal(&left_type, &right_type),
        ));
        assert_eq!(requirement.effects(), Effects::single(Effect::OrderedAssertion));

        // Every wrong-kind path uses the same checked type projection and therefore reports the canonical diagnostic.
        assert_eq!(
            array_operation.infer_output_types(&[left_type.clone().into(), right_type.clone().into()], &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            dimension_operation.infer_output_types(&[array_type.clone().into(), array_type.clone().into()], &[]),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );

        // Region projection preserves the complete higher-order interface, including effects, before delegating to
        // the homogeneous condition operation.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let interface = RegionInterface::new(
            vec![array_type.clone().into()],
            vec![array_type.clone().into()],
            Effects::single(Effect::OrderedIo),
        );
        let (_, projected_interfaces) =
            project_operation_boundary::<ArrayType>(&[], std::slice::from_ref(&interface)).unwrap();
        assert_eq!(projected_interfaces[0].effects(), Effects::single(Effect::OrderedIo));
        let condition = ArrayProgramOperation::<Array>::from(ArrayOperation::Condition(ConditionOperation::new()));
        assert_eq!(
            condition.infer_output_types(
                &[predicate_type.into(), array_type.clone().into()],
                &[interface.clone(), interface],
            ),
            Ok(vec![array_type.clone().into()]),
        );
        assert_eq!(
            condition.infer_region_input_types(
                &[ArrayType::scalar(DataType::Boolean).into(), array_type.clone().into()],
                &[
                    RegionInterface::new(vec![array_type.clone().into()], vec![], Effects::PURE),
                    RegionInterface::new(vec![array_type.clone().into()], vec![], Effects::PURE),
                ],
            ),
            Ok(vec![None, None]),
        );
        assert_eq!(condition.region_slots(), ConditionOperation::<Array>::new().region_slots());
        assert_eq!(
            condition.output_region_provenance(0),
            ConditionOperation::<Array>::new().output_region_provenance(0),
        );

        // Identity-bearing payloads are renamed by their owning homogeneous family.
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let zero = ArrayProgramOperation::<Array>::from(ArrayOperation::Zero(ZeroOperation::new(dynamic_type)));
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source, target.clone()).unwrap();
        let ArrayProgramOperation::Array(ArrayOperation::Zero(zero)) = zero.rename_type_identities(&renaming).unwrap()
        else {
            panic!("expected a renamed array zero operation");
        };
        assert_eq!(zero.r#type().shape().dimensions(), &[Dimension::Dynamic(target)]);

        let renamed_left = DimensionVariable::new("renamed_left", bounds);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(left_type.variable().clone(), renamed_left.clone()).unwrap();
        let ArrayProgramOperation::Dimension(DimensionOperation::Add(add)) =
            dimension_operation.rename_type_identities(&renaming).unwrap()
        else {
            panic!("expected a renamed dimension addition operation");
        };
        assert_eq!(add.left_type().variable(), &renamed_left);
        assert_eq!(add.right_type(), &right_type);
    }

    #[test]
    fn test_array_program_operation_interpretation() {
        let context = EagerContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        assert_eq!(
            context.bind(
                ArrayOperation::Add(AddOperation),
                Vec::new(),
                &[
                    ArrayProgramValue::Array(Array::vector(vec![1.0, 2.0])),
                    ArrayProgramValue::Array(Array::vector(vec![3.0, 4.0])),
                ],
            ),
            Ok(vec![ArrayProgramValue::Array(Array::vector(vec![4.0, 6.0]))]),
        );

        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
        let operation = DimensionOperation::Add(DimensionAddOperation::new(&left_type, &right_type).unwrap());
        let result = context
            .bind(
                operation,
                Vec::new(),
                &[
                    ArrayProgramValue::Dimension(DimensionValue::new(left_type, 3).unwrap()),
                    ArrayProgramValue::Dimension(DimensionValue::new(right_type, 4).unwrap()),
                ],
            )
            .unwrap();
        let [ArrayProgramValue::Dimension(result)] = result.as_slice() else {
            panic!("expected one dimension result");
        };
        assert_eq!(result.extent(), 7);

        let condition = ArrayProgramOperation::<Array>::from(ArrayOperation::Condition(ConditionOperation::new()));
        assert_eq!(
            condition.interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::MalformedProgram("projected operation `condition` cannot carry regions".to_string(),)),
        );
    }

    #[test]
    fn test_array_program_operation_tracing_has_only_explicit_dependencies() {
        type TestContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        let context = TestContext::new();
        let array = context.input(ArrayType::scalar(DataType::F32).into());
        let array_atom = array.atom_id().unwrap();
        let array = <Tracer<TestContext> as ValueProjection<ArrayType>>::into_projected(array).unwrap();
        array.dispatch_domain().bind(AddOperation, Vec::new(), &[array.clone(), array]).unwrap();

        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
        let left = context.input(left_type.clone().into());
        let right = context.input(right_type.clone().into());
        let left_atom = left.atom_id().unwrap();
        let right_atom = right.atom_id().unwrap();
        let left = <Tracer<TestContext> as ValueProjection<DimensionType>>::into_projected(left).unwrap();
        let right = <Tracer<TestContext> as ValueProjection<DimensionType>>::into_projected(right).unwrap();
        left.dispatch_domain()
            .bind(DimensionAddOperation::new(&left_type, &right_type).unwrap(), Vec::new(), &[left, right])
            .unwrap();

        let builder = context.builder().borrow();
        let [array_instruction, dimension_instruction] = builder.instructions() else {
            panic!("expected one array instruction and one dimension instruction");
        };
        assert_eq!(array_instruction.inputs(), &[array_atom, array_atom]);
        assert!(array_instruction.regions().is_empty());
        assert!(matches!(array_instruction.operation(), ArrayProgramOperation::Array(ArrayOperation::Add(_))));
        assert_eq!(dimension_instruction.inputs(), &[left_atom, right_atom]);
        assert!(dimension_instruction.regions().is_empty());
        assert!(matches!(
            dimension_instruction.operation(),
            ArrayProgramOperation::Dimension(DimensionOperation::Add(_)),
        ));
    }

    #[test]
    fn test_symbolic_value_projection_preserves_ssa_identity() {
        type TestContext = TracingContext<ArrayProgramValue<Array>, ConstantOperation<ArrayProgramValue<Array>>>;

        let context = TestContext::new();
        let tracer = context.input(ArrayProgramType::Array(ArrayType::scalar(DataType::F32)));
        let atom = tracer.atom_id().unwrap();
        let projected = <Tracer<TestContext> as ValueProjection<ArrayType>>::projected(&tracer).unwrap();
        assert_eq!(projected.value().atom_id(), Ok(atom));
        let projected = <Tracer<TestContext> as ValueProjection<ArrayType>>::into_projected(tracer).unwrap();
        assert_eq!(projected.value().atom_id(), Ok(atom));
        assert_eq!(<Tracer<TestContext> as ValueProjection<ArrayType>>::from_projected(projected).atom_id(), Ok(atom),);

        fn assert_projection<V: ValueProjection<ArrayType>>() {}
        assert_projection::<PartialTracer<TestContext>>();
        assert_projection::<DifferentiationTracer<TestContext>>();
    }
}
