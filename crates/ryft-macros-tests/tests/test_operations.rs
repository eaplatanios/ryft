use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

use indoc::indoc;
use pretty_assertions::assert_eq;
use ryft::{
    AddOperation, Array, ArrayBatch, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue,
    ArrayOperation, ArrayReferenceDischarge, ArrayType, AxisIndexOperation, BatchAxis, BatchableOperation,
    BatchingContext, CollectiveAxes, CollectiveOperation, ConstantOperation, Context, CotangentAccumulator, DataType,
    DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationPolicy, Dimension, DimensionAddOperation, DimensionBounds, DimensionOperation,
    DimensionType, DimensionValue, DimensionVariable, Domain, EagerContext, EffectClass, EffectClasses, Effects,
    EmptyRegionDriver, ErasedOperation, GuardOperation, InputRegionProvenance, InterpretableOperation,
    InterpretationDriver, MaybeZero, MemberDifferentiableOperation, MemberOperation, MemberTransposableOperation,
    NegOperation, Operation, OperationBoundaryPruning, OperationFoldOutput, OperationPayloadProjection,
    OperationProjection, OperationProvider, OutputRegionProvenance, ParallelPermuteOperation, ParallelVaryOperation,
    PartialEvaluationInput, PartialEvaluationOutput, PartialValue, Placeholder, ProgramBuilder, ProgramError,
    ReferenceAccessMode, ReferenceDischargeContext, ReferenceDischargeValue, ReferenceDischargeableOperation,
    ReferenceFreezeOperation, ReferenceType, RegionDataFlow, RegionDataFlowBoundary, RegionDataFlowRule,
    RegionDataFlowSources, RegionInterface, RegionLiveness, RegionRole, RegionSlot, Shape, StagingContext,
    TagOperation, Tracer, TracingContext, TransposableOperation, TranspositionContext, TranspositionDriver, TypeError,
    TypeIdentityRenaming, Typed, Value, ZeroOperation, check_count,
};

/// Payload that overrides every optional [`Operation`] method with a distinctive answer, so that generated forwarding
/// is observable. `ID` distinguishes otherwise identical payloads held by different variants of one family.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProbeOperation<const ID: u8> {
    /// Whether [`Operation::rename_type_identities`] produced this payload.
    renamed: bool,
}

impl<const ID: u8> ProbeOperation<ID> {
    /// Creates a payload that has not been renamed.
    fn new() -> Self {
        Self { renamed: false }
    }
}

impl<const ID: u8> Operation for ProbeOperation<ID> {
    type Type = ArrayType;

    fn name(&self) -> &'static str {
        "probe"
    }

    fn defined_type_identities(&self) -> Cow<'_, [DimensionVariable]> {
        Cow::Owned(vec![DimensionVariable::new("probe", DimensionBounds::positive(None).unwrap())])
    }

    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("body")] }
    }

    fn validate_region_interfaces(
        &self,
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<(), ProgramError> {
        Err(ProgramError::UnsupportedOperation { message: format!("probe {ID} region interfaces") })
    }

    fn infer_region_input_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<Option<Vec<ArrayType>>>, TypeError> {
        Ok(vec![Some(input_types.to_vec())])
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        Ok(input_types.to_vec())
    }

    fn fold(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Option<Vec<OperationFoldOutput>>, TypeError> {
        Ok(Some((0..input_types.len()).map(OperationFoldOutput::Input).collect()))
    }

    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        if region_index == 0 {
            InputRegionProvenance::Input { index: input_index + 1 }
        } else {
            InputRegionProvenance::Local
        }
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![OutputRegionProvenance { region_index: 0, output_index }]
    }

    fn region_data_flow(&self) -> RegionDataFlow<'_> {
        RegionDataFlow::Custom(self)
    }

    fn prune_boundary(
        &self,
        input_types: &[ArrayType],
        used_outputs: &[bool],
        _regions: &mut dyn RegionLiveness,
    ) -> Result<Option<OperationBoundaryPruning<Self>>, ProgramError> {
        Ok(Some(OperationBoundaryPruning {
            operation: self.clone(),
            kept_inputs: vec![false; input_types.len()],
            kept_outputs: used_outputs.to_vec(),
        }))
    }

    fn is_zero(&self, output_index: usize) -> bool {
        output_index == usize::from(ID)
    }

    fn region_capture_input_count(&self, region_index: usize) -> Option<usize> {
        (region_index == 0).then_some(2)
    }

    fn reference_output_identity_input(&self, output_index: usize) -> Option<usize> {
        Some(output_index + 1)
    }

    fn allows_reference_access_through_region_input(&self, _region_index: usize, mode: ReferenceAccessMode) -> bool {
        mode == ReferenceAccessMode::Read
    }

    fn defers_reference_discharge(&self) -> bool {
        true
    }

    fn effects(&self) -> Cow<'_, Effects> {
        Cow::Owned(Effects::explicit(EffectClasses::single(EffectClass::OrderedIo)))
    }

    fn rename_type_identities(&self, _renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self { renamed: true })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        write!(formatter, "probe [id={ID}, indentation={indentation}]")
    }
}

impl<const ID: u8> RegionDataFlowRule for ProbeOperation<ID> {
    fn input_sources(
        &self,
        _region_index: usize,
        _input_index: usize,
        _boundary: RegionDataFlowBoundary<'_>,
    ) -> Result<RegionDataFlowSources, ProgramError> {
        Ok(RegionDataFlowSources::Unknown)
    }

    fn output_sources(
        &self,
        _output_index: usize,
        _boundary: RegionDataFlowBoundary<'_>,
    ) -> Result<RegionDataFlowSources, ProgramError> {
        Ok(RegionDataFlowSources::Unknown)
    }

    fn execution_demands(
        &self,
        _used_outputs: &[bool],
        _boundary: RegionDataFlowBoundary<'_>,
        _regions: &mut dyn RegionLiveness,
    ) -> Result<Vec<Option<Vec<bool>>>, ProgramError> {
        Ok(Vec::new())
    }
}

/// Forward mode rules that report which of the two rules ran, so that generated dispatch to each is observable.
impl<const ID: u8, C: Context<Type = ArrayType>> DifferentiableOperation<C> for ProbeOperation<ID> {
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        _context: &DifferentiationContext<C, P>,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation { message: format!("probe {ID} jvp") }.into())
    }

    fn jvp_for_transpose<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        _context: &DifferentiationContext<C, P>,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation { message: format!("probe {ID} jvp_for_transpose") }.into())
    }
}

/// Explicitly typed family that holds the same probe directly and boxed. The zero variant provides the zero tangents
/// that every differentiation context requests.
#[derive(Clone, Debug, Operation)]
#[ryft(type(ArrayType), constant(Array), dispatch(differentiation))]
enum ProbeFamily {
    Probe(ProbeOperation<0>),
    Boxed(Box<ProbeOperation<1>>),
    Zero(ZeroOperation<ArrayType>),
}

/// Homogeneous array family whose operation and constant types are inferred from its single value parameter.
#[derive(Clone, Debug, Operation)]
#[ryft(dispatch(identity))]
enum ArrayFamily<V: Value<Type = ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    Add(AddOperation<ArrayType>),
    Tag(TagOperation<ArrayType>),
    Constant(ConstantOperation<V>),
}

/// Homogeneous array family that selects the transform dispatchers. Its tag payload forwards every transform to its
/// input unchanged, so the family needs no operations for aligning batched or cotangent values.
#[derive(Clone, Debug, Operation)]
#[ryft(dispatch(batching, differentiation, transposition))]
enum UnaryFamily<V: Value<Type = ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    Tag(TagOperation<ArrayType>),
    Constant(ConstantOperation<V>),
}

/// Cotangent addition, which transposition requires of every family but never stages for a single contribution.
impl<V: Value<Type = ArrayType>> OperationProvider<ArrayType, AddOperation<ArrayType>> for UnaryFamily<V> {
    type Operation = Self;

    fn provide(_request: AddOperation<ArrayType>, _input_types: &[&ArrayType]) -> Result<Self, ProgramError> {
        Err(ProgramError::UnsupportedOperation { message: "the unary family has no addition".to_string() })
    }
}

/// Heterogeneous family whose member variants are both projected classes, built from production member families.
#[derive(Clone, Debug, Operation)]
#[ryft(type(ArrayIrType), constant(ArrayIrValue<A>), members(ArrayType, structural(DimensionType)))]
enum ArrayIrFamily<A: Value<Type = ArrayType>> {
    #[ryft(projected(ArrayType), collective)]
    Array(ArrayOperation<A>),

    #[ryft(projected(DimensionType, structural))]
    Dimension(DimensionOperation<DimensionValue>),
}

/// Mixed member payload whose parent instruction interleaves its two array data inputs with two first-class dimension
/// inputs. No production payload arranges its inputs this way, so this fixture is what pins that the generated mixed
/// dispatchers classify inputs individually instead of splitting on the first dimension input.
#[derive(Clone, Debug)]
struct InterleavedOperation;

impl Operation for InterleavedOperation {
    type Type = ArrayType;

    fn name(&self) -> &'static str {
        "interleaved"
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl MemberOperation<ArrayIrType> for InterleavedOperation {
    fn infer_parent_region_input_types(
        &self,
        _input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
        Ok(vec![None; region_interfaces.len()])
    }

    fn infer_parent_output_types(
        &self,
        input_types: &[ArrayIrType],
        _region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        // The parent boundary is `(array, dimension, array, dimension) -> array`, so the payload's own homogeneous rule
        // sees the two array inputs and the dimension inputs only select the result shape.
        let arrays = input_types.iter().filter_map(|r#type| <&ArrayType>::try_from(r#type).ok()).cloned();
        Ok(self.infer_output_types(&arrays.collect::<Vec<_>>(), &[])?.into_iter().map(Into::into).collect())
    }

    fn rename_parent_type_identities(
        &self,
        _renaming: &TypeIdentityRenaming<DimensionVariable>,
    ) -> Result<Self, TypeError> {
        Ok(self.clone())
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<InterleavedOperation>>> MemberDifferentiableOperation<C>
    for InterleavedOperation
{
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        _context: &DifferentiationContext<C, P>,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation { message: "mixed jvp".to_string() }.into())
    }

    fn jvp_for_transpose_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        _context: &DifferentiationContext<C, P>,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation { message: "mixed jvp_for_transpose".to_string() }.into())
    }
}

/// Homogeneous transpose rule that submits the output cotangent to every unknown array input.
impl<V: Value<Type = ArrayType>, O: Operation<Type = ArrayType> + From<AddOperation<ArrayType>>>
    TransposableOperation<V, O> for InterleavedOperation
where
    O: OperationProvider<ArrayType, GuardOperation<ArrayType>, Operation = O>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, 2, DifferentiationError);
        for (input, accumulator) in inputs.iter().zip(accumulators) {
            if matches!(input, PartialValue::Unknown(_)) {
                accumulator.accumulate(context, outputs[0].clone())?;
            }
        }
        Ok(())
    }
}

/// Structural mixed member payload whose parent instruction produces one array output and one first-class dimension
/// output, so its generated forward mode arm must select a zero tangent universe per output.
#[derive(Clone, Debug)]
struct ConstructorOperation {
    /// Array output type of the constructed instruction.
    array_type: ArrayType,

    /// First-class dimension output type of the constructed instruction.
    dimension_type: DimensionType,
}

impl Operation for ConstructorOperation {
    type Type = ArrayType;

    fn name(&self) -> &'static str {
        "constructor"
    }

    fn infer_output_types(
        &self,
        _input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        Ok(vec![self.array_type.clone()])
    }
}

impl MemberOperation<ArrayIrType> for ConstructorOperation {
    fn infer_parent_region_input_types(
        &self,
        _input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
        Ok(vec![None; region_interfaces.len()])
    }

    fn infer_parent_output_types(
        &self,
        _input_types: &[ArrayIrType],
        _region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        Ok(vec![self.array_type.clone().into(), self.dimension_type.clone().into()])
    }

    fn rename_parent_type_identities(
        &self,
        _renaming: &TypeIdentityRenaming<DimensionVariable>,
    ) -> Result<Self, TypeError> {
        Ok(self.clone())
    }
}

impl<V: Value<Type = ArrayType>, O: Operation<Type = ArrayType>> TransposableOperation<V, O> for ConstructorOperation
where
    O: OperationProvider<ArrayType, GuardOperation<ArrayType>, Operation = O>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TranspositionContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        _accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        Ok(())
    }
}

/// Homogeneous array member family of [`MixedFamily`], in which its projected boundary and its delegated mixed
/// transpose rules run. It holds every mixed payload, because mixed transposition stages each one in this family.
#[derive(Clone, Debug, Operation)]
enum MixedMemberFamily<V: Value<Type = ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    Add(AddOperation<ArrayType>),
    Interleaved(InterleavedOperation),
    Constructor(ConstructorOperation),
    Probe(ProbeOperation<2>),
    Constant(ConstantOperation<V>),
}

/// The fixture only transposes its interleaved payload, which is all that mixed transposition delegates to it.
impl<V: Value<Type = ArrayType>> TransposableOperation<V, Self> for MixedMemberFamily<V> {
    fn transpose<D: TranspositionDriver<V, Self>>(
        &self,
        context: &mut TranspositionContext<V, Self>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, Self>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, Self>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        match self {
            Self::Interleaved(operation) => operation.transpose(context, driver, inputs, outputs, accumulators),
            _ => {
                Err(ProgramError::UnsupportedOperation { message: format!("cannot transpose `{}`", self.name()) }
                    .into())
            }
        }
    }
}

/// Parent-universe forward mode rules of the projected member, which stage the member instruction with constant
/// outputs and report the reverse mode rule as unsupported, so that each is observable.
impl<A, C> MemberDifferentiableOperation<C> for MixedMemberFamily<A>
where
    A: Value<Type = ArrayType>,
    C: Context<Type = ArrayIrType, Operation: From<MixedMemberFamily<A>>>,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        context
            .primal()
            .bind(self.clone(), Vec::new(), primals.as_slice())?
            .into_iter()
            .map(DifferentiationDual::new_with_zero_tangent)
            .collect()
    }

    fn jvp_for_transpose_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        _context: &DifferentiationContext<C, P>,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation { message: "projected jvp_for_transpose".to_string() }.into())
    }
}

/// Parent-universe transpose rule of the projected member, which contributes nothing so that it is distinguishable
/// from the homogeneous interleaved rule that mixed transposition delegates to.
impl<A: Value<Type = ArrayType>, V: Value<Type = ArrayIrType>> MemberTransposableOperation<V, MixedFamily<A>>
    for MixedMemberFamily<A>
{
    fn transpose_in_parent<D: TranspositionDriver<V, MixedFamily<A>>>(
        &self,
        _context: &mut TranspositionContext<V, MixedFamily<A>>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, MixedFamily<A>>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, MixedFamily<A>>>>],
        _accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        Ok(())
    }
}

/// Heterogeneous family with mixed variants in both roles, whose data universe defaults to the single computational
/// member that the family declares, next to computational and structural projected members.
#[derive(Clone, Debug, Operation)]
#[ryft(type(ArrayIrType), constant(ArrayIrValue<A>), members(ArrayType, structural(DimensionType)))]
#[ryft(dispatch(differentiation, transposition))]
enum MixedFamily<A: Value<Type = ArrayType>> {
    #[ryft(mixed)]
    Interleaved(InterleavedOperation),

    #[ryft(mixed(structural))]
    Constructor(ConstructorOperation),

    #[ryft(projected(ArrayType))]
    Array(MixedMemberFamily<A>),

    #[ryft(projected(DimensionType, structural))]
    Dimension(DimensionOperation<DimensionValue>),
}

/// Member zero constructor that the structural mixed forward mode arm stages for array outputs.
impl<A: Value<Type = ArrayType>> From<ZeroOperation<ArrayType>> for MixedFamily<A> {
    fn from(operation: ZeroOperation<ArrayType>) -> Self {
        Self::Array(MixedMemberFamily::Zero(operation))
    }
}

/// Cotangent addition, which accumulates contributions to one input in the member family.
impl<A: Value<Type = ArrayType>> From<AddOperation<ArrayIrType>> for MixedFamily<A> {
    fn from(_operation: AddOperation<ArrayIrType>) -> Self {
        Self::Array(MixedMemberFamily::Add(AddOperation::new()))
    }
}

impl<A: Value<Type = ArrayType>> OperationProvider<ArrayIrType, ZeroOperation<ArrayIrType>> for MixedFamily<A> {
    type Operation = Self;

    fn provide(request: ZeroOperation<ArrayIrType>, _input_types: &[&ArrayIrType]) -> Result<Self, ProgramError> {
        Ok(Self::from(ZeroOperation::new(<&ArrayType>::try_from(request.r#type())?.clone())))
    }
}

/// Returns the static array type and first-class dimension type that the mixed fixtures consume and produce. The array
/// type is static because reconstructing a mixed instruction from type metadata alone requires that its shape not live
/// in runtime identity references.
fn mixed_types() -> (ArrayType, DimensionType) {
    let dimension_type = DimensionType::new("items", DimensionBounds::new(1, Some(8)).unwrap());
    (ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])), dimension_type)
}

/// Region liveness for direct [`Operation::prune_boundary`] calls on payloads that never consult it.
struct NoRegionLiveness;

impl RegionLiveness for NoRegionLiveness {
    fn used_region_inputs(&mut self, _region_index: usize, _used_outputs: &[bool]) -> Result<Vec<bool>, ProgramError> {
        Ok(Vec::new())
    }
}

#[test]
fn test_operation() {
    let r#type = ArrayType::scalar(DataType::F32);
    let renaming = TypeIdentityRenaming::new();

    // Every method forwards to the active payload, including through a boxed payload, whose pruned and renamed
    // operations are boxed back into their variant.
    for (operation, id) in
        [(ProbeFamily::from(ProbeOperation::<0>::new()), 0), (ProbeFamily::from(ProbeOperation::<1>::new()), 1)]
    {
        assert_eq!(operation.name(), "probe");
        assert_eq!(operation.defined_type_identities().len(), 1);
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("body")]);
        assert_eq!(operation.region_role(0), Some(RegionRole::Computation));
        assert_eq!(
            operation.validate_region_interfaces(&[]),
            Err(ProgramError::UnsupportedOperation { message: format!("probe {id} region interfaces") }),
        );
        assert_eq!(operation.infer_region_input_types(&[r#type.clone()], &[]), Ok(vec![Some(vec![r#type.clone()])]));
        assert_eq!(operation.infer_output_types(&[r#type.clone()], &[]), Ok(vec![r#type.clone()]));
        assert_eq!(operation.fold(&[r#type.clone()], &[]), Ok(Some(vec![OperationFoldOutput::Input(0)])));
        assert_eq!(operation.input_region_provenance(0, 1), InputRegionProvenance::Input { index: 2 });
        assert_eq!(operation.input_region_provenance(1, 0), InputRegionProvenance::Local);
        assert_eq!(
            operation.output_region_provenance(3),
            vec![OutputRegionProvenance { region_index: 0, output_index: 3 }]
        );
        assert!(!operation.is_zero(1 - id));
        assert!(operation.is_zero(id));
        assert_eq!(operation.region_capture_input_count(0), Some(2));
        assert_eq!(operation.region_capture_input_count(1), None);
        assert_eq!(operation.reference_output_identity_input(2), Some(3));
        assert!(operation.allows_reference_access_through_region_input(0, ReferenceAccessMode::Read));
        assert!(!operation.allows_reference_access_through_region_input(0, ReferenceAccessMode::Write));
        assert!(operation.defers_reference_discharge());
        assert_eq!(operation.effects().into_owned(), Effects::explicit(EffectClasses::single(EffectClass::OrderedIo)));
        assert_eq!(operation.to_string(), format!("probe [id={id}, indentation=0]"));
        let pruning = operation.prune_boundary(&[r#type.clone()], &[true], &mut NoRegionLiveness).unwrap().unwrap();
        assert_eq!((pruning.kept_inputs, pruning.kept_outputs), (vec![false], vec![true]));
        match (pruning.operation, operation.rename_type_identities(&renaming).unwrap()) {
            (ProbeFamily::Probe(pruned), ProbeFamily::Probe(renamed)) if id == 0 => {
                assert_eq!((pruned.renamed, renamed.renamed), (false, true));
            }
            (ProbeFamily::Boxed(pruned), ProbeFamily::Boxed(renamed)) if id == 1 => {
                assert_eq!((pruned.renamed, renamed.renamed), (false, true));
            }
            (pruned, renamed) => panic!("pruning and renaming must keep variant {id}: {pruned:?}, {renamed:?}"),
        }
    }

    // A custom data flow rule is borrowed from the payload itself rather than from a copy.
    let operation = ProbeFamily::from(ProbeOperation::<1>::new());
    let (RegionDataFlow::Custom(rule), ProbeFamily::Boxed(payload)) = (operation.region_data_flow(), &operation) else {
        panic!("the boxed probe declares a custom data flow rule");
    };
    assert!(std::ptr::addr_eq(rule, &**payload));
}

#[test]
fn test_operation_identity() {
    type Family = ArrayFamily<Array>;

    /// Asserts at compile time that `T` implements the complete operation identity contract.
    fn assert_identity<T: Eq + Hash>() {}

    /// Returns the hash of `value` under the standard library's default hasher.
    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    // The generated bounds constrain the payload types rather than `V`, so the family is identity-comparable even
    // though `Array` itself implements neither `Eq` nor `Hash`.
    assert_identity::<Family>();

    // Operations of the same variant with equal payloads are equal and hash identically, so they work as map keys.
    let zero = |data_type: DataType| Family::from(ZeroOperation::new(ArrayType::scalar(data_type)));
    let add = Family::from(AddOperation::new());
    assert_eq!(zero(DataType::F64), zero(DataType::F64));
    assert_eq!(hash_of(&zero(DataType::F64)), hash_of(&zero(DataType::F64)));
    let operations = HashMap::from([(zero(DataType::F64), "zero"), (add.clone(), "add")]);
    assert_eq!(operations.get(&zero(DataType::F64)), Some(&"zero"));
    assert_eq!(operations.get(&add), Some(&"add"));
    assert_eq!(operations.get(&zero(DataType::F32)), None);

    // Different payloads and different variants are unequal.
    assert_ne!(zero(DataType::F64), zero(DataType::F32));
    assert_ne!(zero(DataType::F64), add);

    // Payload identity is delegated as-is, so constants keep their bitwise literal identity.
    let constant = |value: f64| Family::from(ConstantOperation::new(Array::scalar(value).unwrap()));
    assert_eq!(constant(f64::NAN), constant(f64::NAN));
    assert_eq!(hash_of(&constant(f64::NAN)), hash_of(&constant(f64::NAN)));
    assert_ne!(constant(-0.0), constant(0.0));
}

#[test]
fn test_operation_member_variants() {
    type Family = MixedFamily<Array>;

    // Mixed members infer through their payloads' parent boundaries, which accept interleaved input arrangements.
    let (array_type, dimension_type) = mixed_types();
    let array_type = ArrayIrType::from(array_type);
    let dimension_type = ArrayIrType::from(dimension_type);
    let operation = Family::from(InterleavedOperation);
    let input_types = [array_type.clone(), dimension_type.clone(), array_type.clone(), dimension_type.clone()];
    assert_eq!(operation.name(), "interleaved");
    assert_eq!(operation.infer_output_types(&input_types, &[]), Ok(vec![array_type.clone()]));

    // Projected members validate, infer, fold, and prune in their member universe and lift their results back into
    // the family universe, and a boundary outside that universe is rejected before the payload sees it.
    let operation = Family::Array(MixedMemberFamily::Probe(ProbeOperation::new()));
    assert_eq!(operation.name(), "probe");
    assert_eq!(operation.to_string(), "probe [id=2, indentation=0]");
    let interface = RegionInterface::new(vec![array_type.clone()], vec![array_type.clone()], EffectClasses::NONE);
    assert_eq!(
        operation.validate_region_interfaces(&[interface]),
        Err(ProgramError::UnsupportedOperation { message: "probe 2 region interfaces".to_string() }),
    );
    assert_eq!(operation.infer_output_types(&[array_type.clone()], &[]), Ok(vec![array_type.clone()]));
    assert_eq!(
        operation.infer_region_input_types(&[array_type.clone()], &[]),
        Ok(vec![Some(vec![array_type.clone()])])
    );
    assert_eq!(operation.fold(&[array_type.clone()], &[]), Ok(Some(vec![OperationFoldOutput::Input(0)])));
    assert!(matches!(
        operation.infer_output_types(&[dimension_type], &[]),
        Err(TypeError::Invalid { message }) if message == "expected array type but got dimension type",
    ));
    let pruning = operation.prune_boundary(&[array_type], &[true], &mut NoRegionLiveness).unwrap().unwrap();
    assert!(matches!(pruning.operation, Family::Array(MixedMemberFamily::Probe(_))));
    assert!(matches!(
        operation.rename_type_identities(&TypeIdentityRenaming::new()),
        Ok(Family::Array(MixedMemberFamily::Probe(ProbeOperation { renamed: true }))),
    ));
    let (RegionDataFlow::Custom(rule), Family::Array(MixedMemberFamily::Probe(payload))) =
        (operation.region_data_flow(), &operation)
    else {
        panic!("the projected probe declares a custom data flow rule");
    };
    assert!(std::ptr::addr_eq(rule, payload));

    // Each projected variant, in either role, names its payload family as the canonical projection into its member.
    fn assert_projection<O: OperationProjection<T, Projected = P>, T: ryft::Type, P: Operation<Type = T>>() {}
    assert_projection::<Family, ArrayType, MixedMemberFamily<Array>>();
    assert_projection::<Family, DimensionType, DimensionOperation<DimensionValue>>();
}

#[test]
fn test_operation_type_inference() {
    /// Payload that ties the second value parameter of `WhereBoundFamily` to one of its variants.
    #[derive(Clone, Debug)]
    struct CapturedOperation<V>(PhantomData<fn() -> V>);

    impl<V: Value<Type = ArrayType>> Operation for CapturedOperation<V> {
        type Type = ArrayType;

        fn name(&self) -> &'static str {
            "captured"
        }

        fn infer_output_types(
            &self,
            _input_types: &[ArrayType],
            _region_interfaces: &[RegionInterface<ArrayType>],
        ) -> Result<Vec<ArrayType>, TypeError> {
            Ok(Vec::new())
        }
    }

    impl<V: Value<Type = ArrayType>, C: Domain<Type = ArrayType>> InterpretableOperation<C> for CapturedOperation<V> {
        fn interpret<D: InterpretationDriver<C>>(
            &self,
            _context: &C,
            _driver: &D,
            _inputs: &[C::Value],
        ) -> Result<Vec<C::Value>, ProgramError> {
            Ok(Vec::new())
        }
    }

    /// Family whose `Value<Type = ArrayType>` bounds appear only in its where-clause and that has two value
    /// parameters, the first of which is both its runtime value type and its stored constant type.
    #[derive(Clone, Debug, Operation)]
    enum WhereBoundFamily<V, C = V>
    where
        V: Value<Type = ArrayType>,
        C: Value<Type = ArrayType>,
    {
        Add(AddOperation<ArrayType>),
        Constant(ConstantOperation<V>),
        Captured(CapturedOperation<C>),
    }

    fn assert_operation_type<O: Operation<Type = ArrayType>>() {}
    assert_operation_type::<WhereBoundFamily<Array>>();

    let mut builder = ProgramBuilder::<Array, WhereBoundFamily<Array>>::new();
    let x = builder.add_input(ArrayType::scalar(DataType::F32));
    let one = builder
        .add_instruction(ConstantOperation::new(Array::scalar(1f32).unwrap()), Vec::new(), Vec::new(), None)
        .unwrap()[0];
    let y = builder.add_instruction(AddOperation::new(), Vec::new(), vec![x, one], None).unwrap()[0];
    let program = builder.build::<Array, Array>(vec![y], Placeholder, Placeholder).unwrap();
    assert_eq!(program.interpret(Array::scalar(2f32).unwrap()), Ok(Array::scalar(3f32).unwrap()));

    /// Family with a single value parameter whose where-clause bounds an associated type of that parameter, which the
    /// generated interpretation dispatcher renames onto its separate runtime value type.
    #[derive(Clone, Debug, Operation)]
    enum AssociatedBoundFamily<V: Value<Type = ArrayType>>
    where
        V::Type: Clone,
    {
        Add(AddOperation<ArrayType>),
        Constant(ConstantOperation<V>),
    }

    let mut builder = ProgramBuilder::<Array, AssociatedBoundFamily<Array>>::new();
    let x = builder.add_input(ArrayType::scalar(DataType::F32));
    let one = builder
        .add_instruction(ConstantOperation::new(Array::scalar(1f32).unwrap()), Vec::new(), Vec::new(), None)
        .unwrap()[0];
    let y = builder.add_instruction(AddOperation::new(), Vec::new(), vec![x, one], None).unwrap()[0];
    let program = builder.build::<Array, Array>(vec![y], Placeholder, Placeholder).unwrap();
    assert_eq!(program.interpret(Array::scalar(2f32).unwrap()), Ok(Array::scalar(3f32).unwrap()));
}

#[test]
fn test_operation_crate_path() {
    // Generated code names only items at the root of `ryft-core`, so any path that re-exports that root works.
    #[derive(Clone, Debug, Operation)]
    #[ryft(crate = "ryft::core", dispatch(identity, discharge, batching, differentiation, transposition))]
    enum ReexportedFamily<V: Value<Type = ArrayType>> {
        Zero(ZeroOperation<ArrayType>),
        Constant(ConstantOperation<V>),
    }

    let operation = ReexportedFamily::<Array>::from(ZeroOperation::new(ArrayType::scalar(DataType::F32)));
    assert_eq!(operation.name(), "zero");
}

#[test]
fn test_operation_discharge_references() {
    /// Family with a native payload, a projected member, and a bare generic extension, each of which discharges
    /// differently.
    #[derive(Clone, Debug, Operation)]
    #[ryft(type(ArrayIrType), constant(ArrayIrValue<Array>), members(ArrayType, structural(DimensionType)))]
    #[ryft(dispatch(discharge))]
    enum DischargeFamily<E: Operation<Type = ArrayIrType>> {
        Freeze(ReferenceFreezeOperation<ArrayType, ArrayIrType>),

        #[ryft(projected(ArrayType))]
        Array(ArrayOperation<Array>),

        Extension(E),
    }

    type Family = DischargeFamily<ReferenceFreezeOperation<ArrayType, ArrayIrType>>;

    let context = ReferenceDischargeContext::<EagerContext<ArrayIrValue<Array>, Family>, ArrayReferenceDischarge>::new(
        EagerContext::new(),
    );
    let array = |value: f32| ArrayIrValue::Array(Array::scalar(value).unwrap());
    let reference = |value: f32| {
        let r#type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        ReferenceDischargeValue::from(context.bind_discharged(r#type, array(value)).unwrap())
    };

    // Native variants and generic extensions delegate to their payloads' own rules, which consume the reference that
    // the replay of member variants below rejects.
    let freeze = Family::from(ReferenceFreezeOperation::new());
    assert_eq!(
        freeze.discharge_references(&context, &EmptyRegionDriver, &[reference(2.0)]),
        Ok(vec![ReferenceDischargeValue::Value(array(2.0))]),
    );
    let extension = Family::Extension(ReferenceFreezeOperation::new());
    assert_eq!(
        extension.discharge_references(&context, &EmptyRegionDriver, &[reference(3.0)]),
        Ok(vec![ReferenceDischargeValue::Value(array(3.0))]),
    );

    // Member variants replay the complete family verbatim, which rejects a live reference handle as an input.
    let add = Family::Array(ArrayOperation::from(AddOperation::new()));
    let values = [ReferenceDischargeValue::Value(array(2.0)), ReferenceDischargeValue::Value(array(3.0))];
    assert_eq!(
        add.discharge_references(&context, &EmptyRegionDriver, &values),
        Ok(vec![ReferenceDischargeValue::Value(array(5.0))]),
    );
    let reference = reference(4.0);
    assert_eq!(
        add.discharge_references(&context, &EmptyRegionDriver, &[reference.clone(), values[1].clone()]),
        Err(ProgramError::MalformedProgram(format!(
            "reference discharge expected a value input 0 of `add` but received {reference}",
        ))),
    );
}

#[test]
fn test_operation_interpret() {
    // `f(x) = tag(x + 1)` stages its constant through the family's own constant payload.
    let mut builder = ProgramBuilder::<Array, ArrayFamily<Array>>::new();
    let x = builder.add_input(ArrayType::scalar(DataType::F32));
    let one = ConstantOperation::new(Array::scalar(1f32).unwrap());
    let one = builder.add_instruction(one, Vec::new(), Vec::new(), None).unwrap()[0];
    let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![x, one], None).unwrap()[0];
    let y = builder.add_instruction(TagOperation::new("y"), Vec::new(), vec![sum], None).unwrap()[0];
    let program = builder.build::<Array, Array>(vec![y], Placeholder, Placeholder).unwrap();

    // Native variants interpret through their payload rules in an eager context.
    assert_eq!(program.interpret(Array::scalar(2f32).unwrap()), Ok(Array::scalar(3f32).unwrap()));

    // A family with one value parameter also interprets in contexts whose runtime values differ from its stored
    // constants, such as a tracing context, which restages every instruction.
    let context = TracingContext::<Array, ArrayFamily<Array>>::new();
    let input = context.input(ArrayType::scalar(DataType::F32));
    let output = program.interpret_in_context(&context, input).unwrap();
    let builder = context.builder().borrow().clone();
    let traced = builder.build::<Array, Array>(vec![output.atom_id().unwrap()], Placeholder, Placeholder).unwrap();
    assert_eq!(
        traced.to_string(),
        indoc! {"
            lambda %0:f32[] .
            let %1:f32[] = constant [value=1.0]
                %2:f32[] = add %0 %1
                %3:f32[] = tag [key=y] %2
            in (%3)"},
    );

    // Projected members, in either role, interpret in their member's eager value family, and their results lift back
    // into the family's values.
    let context = EagerContext::<ArrayIrValue<Array>, ArrayIrFamily<Array>>::new();
    let array = |value: f32| ArrayIrValue::Array(Array::scalar(value).unwrap());
    let operation = ArrayIrFamily::Array(ArrayOperation::from(AddOperation::new()));
    assert_eq!(operation.interpret(&context, &EmptyRegionDriver, &[array(2.0), array(3.0)]), Ok(vec![array(5.0)]));
    let dimension = |extent| DimensionValue::constant(extent).unwrap();
    let add = DimensionAddOperation::new(dimension(3).r#type().as_ref(), dimension(4).r#type().as_ref()).unwrap();
    let operation = ArrayIrFamily::<Array>::from(DimensionOperation::Add(add));
    let inputs = [ArrayIrValue::Dimension(dimension(3)), ArrayIrValue::Dimension(dimension(4))];
    assert_eq!(
        operation.interpret(&context, &EmptyRegionDriver, &inputs),
        Ok(vec![ArrayIrValue::Dimension(dimension(7))]),
    );
}

#[test]
fn test_operation_partially_evaluate() {
    // `f(a, x) = (a + a) + x` with `a` known folds the native `a + a` instruction through its payload rule.
    let r#type = ArrayType::scalar(DataType::F32);
    let mut builder = ProgramBuilder::<Array, ArrayFamily<Array>>::new();
    let a = builder.add_input(r#type.clone());
    let x = builder.add_input(r#type.clone());
    let doubled = builder.add_instruction(AddOperation::new(), Vec::new(), vec![a, a], None).unwrap()[0];
    let y = builder.add_instruction(AddOperation::new(), Vec::new(), vec![doubled, x], None).unwrap()[0];
    let program = builder.build::<Vec<Array>, Vec<Array>>(vec![y], vec![Placeholder; 2], vec![Placeholder]).unwrap();
    let evaluation = program
        .partially_evaluate(&[PartialValue::Known(Array::scalar(3f32).unwrap()), PartialValue::Unknown(r#type)])
        .unwrap();
    assert_eq!(
        evaluation.inputs(),
        &[PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(Array::scalar(6f32).unwrap())],
    );
    assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
    assert_eq!(
        evaluation.program().to_string(),
        indoc! {"
            lambda %0:f32[], %1:f32[] .
            let %2:f32[] = add %1 %0
            in (%2)"},
    );

    // Projected members take the family's canonical fold-or-residualize path, which evaluates them once every input
    // is known.
    let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrFamily<Array>>::new();
    let a = builder.add_input(ArrayIrType::from(ArrayType::scalar(DataType::F32)));
    let add = ArrayIrFamily::Array(ArrayOperation::from(AddOperation::new()));
    let doubled = builder.add_instruction(add, Vec::new(), vec![a, a], None).unwrap()[0];
    let program = builder
        .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![doubled],
            vec![Placeholder],
            vec![Placeholder],
        )
        .unwrap();
    let evaluation = program
        .partially_evaluate(&[PartialValue::Known(ArrayIrValue::Array(Array::scalar(3f32).unwrap()))])
        .unwrap();
    assert_eq!(
        evaluation.outputs(),
        &[PartialEvaluationOutput::Known(ArrayIrValue::Array(Array::scalar(6f32).unwrap()))]
    );
}

#[test]
fn test_operation_batch() {
    // Native variants batch through their payload rules over an eager parent context.
    let context = BatchingContext::new(EagerContext::<Array, UnaryFamily<Array>>::new(), 2);
    let operation = UnaryFamily::<Array>::from(TagOperation::new("x"));
    let input = ArrayBatch::new(Array::vector(vec![1f32, 2f32]).unwrap(), BatchAxis::new(0)).unwrap();
    assert_eq!(operation.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)), Ok(vec![input]));

    // Computational projected members batch through their member family's rules under the projected policy. Array
    // members need the whole array IR family surface for that, so this uses the production family.
    let extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
    let context = BatchingContext::<_, ArrayIrBatchingPolicy>::with_policy(
        EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
        extent.clone(),
    );
    let operation = ArrayIrOperation::<Array>::Array(ArrayOperation::from(NegOperation::new()));
    let input = ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![1f32, 2f32]).unwrap()), BatchAxis::new(0));
    assert_eq!(
        operation.batch(&context, &EmptyRegionDriver, &[input.unwrap()]),
        Ok(vec![
            ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![-1f32, -2f32]).unwrap()), BatchAxis::new(0))
                .unwrap()
        ]),
    );

    // Structural projected members bind their replicated inputs once through the parent context, so their member
    // family needs no batching rules of its own.
    #[derive(Clone, Debug, Operation)]
    #[ryft(type(ArrayIrType), constant(ArrayIrValue<Array>), members(structural(DimensionType)), dispatch(batching))]
    enum StructuralFamily {
        #[ryft(projected(DimensionType, structural))]
        Dimension(DimensionOperation<DimensionValue>),
    }

    let context = BatchingContext::<_, ArrayIrBatchingPolicy>::with_policy(
        EagerContext::<ArrayIrValue<Array>, StructuralFamily>::new(),
        extent,
    );
    let dimension = |extent| DimensionValue::constant(extent).unwrap();
    let add = DimensionAddOperation::new(dimension(3).r#type().as_ref(), dimension(4).r#type().as_ref()).unwrap();
    let operation = StructuralFamily::from(DimensionOperation::Add(add));
    let inputs = [
        ArrayIrBatch::replicated(ArrayIrValue::Dimension(dimension(3))),
        ArrayIrBatch::replicated(ArrayIrValue::Dimension(dimension(4))),
    ];
    assert_eq!(
        operation.batch(&context, &EmptyRegionDriver, &inputs),
        Ok(vec![ArrayIrBatch::replicated(ArrayIrValue::Dimension(dimension(7)))]),
    );
}

#[test]
fn test_operation_jvp() {
    // Native variants apply their payloads' forward mode rules, so a tag stages a tagged primal and passes its tangent
    // through unchanged.
    let r#type = ArrayType::scalar(DataType::F32);
    let context = TracingContext::<Array, UnaryFamily<Array>>::new();
    let input = DifferentiationDual::new(context.input(r#type.clone()), context.input(r#type)).unwrap();
    let operation = UnaryFamily::<Array>::from(TagOperation::new("x"));
    let outputs = operation
        .jvp(&DifferentiationContext::fused(context.clone()), &EmptyRegionDriver, &[input])
        .unwrap();
    let MaybeZero::Value(tangent) = outputs[0].tangent() else {
        panic!("a tag passes a tangent value through");
    };
    let atoms = vec![outputs[0].primal().atom_id().unwrap(), tangent.atom_id().unwrap()];
    let builder = context.builder().borrow().clone();
    let traced = builder.build::<Vec<Array>, Vec<Array>>(atoms, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
    assert_eq!(
        traced.to_string(),
        indoc! {"
            lambda %0:f32[], %1:f32[] .
            let %2:f32[] = tag [key=x] %0
            in (%2, %1)"},
    );

    // `jvp` and `jvp_for_transpose` each dispatch to the payload rule of the same name, including through a box.
    let context = DifferentiationContext::fused(TracingContext::<Array, ProbeFamily>::new());
    for (operation, id) in
        [(ProbeFamily::from(ProbeOperation::<0>::new()), 0), (ProbeFamily::from(ProbeOperation::<1>::new()), 1)]
    {
        assert!(matches!(
            operation.jvp(&context, &EmptyRegionDriver, &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!("probe {id} jvp"),
        ));
        assert!(matches!(
            operation.jvp_for_transpose(&context, &EmptyRegionDriver, &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!("probe {id} jvp_for_transpose"),
        ));
    }
}

#[test]
fn test_operation_jvp_members() {
    type Family = MixedFamily<Array>;

    // Computational mixed and projected members dispatch each forward mode rule to their parent-universe rule.
    let context = DifferentiationContext::fused(TracingContext::<ArrayIrValue<Array>, Family>::new());
    assert!(matches!(
        Family::from(InterleavedOperation).jvp_for_transpose(&context, &EmptyRegionDriver, &[]),
        Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
            if message == "mixed jvp_for_transpose",
    ));
    assert!(matches!(
        Family::Array(MixedMemberFamily::Interleaved(InterleavedOperation)).jvp_for_transpose(
            &context,
            &EmptyRegionDriver,
            &[],
        ),
        Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
            if message == "projected jvp_for_transpose",
    ));

    // A structural mixed payload stages its primal and one zero tangent per output. The array output's tangent is
    // staged in the computational member universe, while the first-class dimension output has a zero differential
    // space and therefore receives a structural zero tangent with no staged instruction.
    let (array_type, dimension_type) = mixed_types();
    let operation =
        Family::from(ConstructorOperation { array_type: array_type.clone(), dimension_type: dimension_type.clone() });
    let context = TracingContext::<ArrayIrValue<Array>, Family>::new();
    let duals = operation.jvp(&DifferentiationContext::fused(context.clone()), &EmptyRegionDriver, &[]).unwrap();
    assert_eq!(duals.len(), 2);
    assert_eq!(duals[0].primal().r#type().as_ref(), &ArrayIrType::from(array_type.clone()));
    assert_eq!(duals[1].primal().r#type().as_ref(), &ArrayIrType::from(dimension_type));
    let MaybeZero::Value(array_tangent) = duals[0].tangent() else {
        panic!("the array output of a structural mixed payload stages a member zero tangent: {duals:?}");
    };
    assert_eq!(array_tangent.r#type().as_ref(), &ArrayIrType::from(array_type.tangent().unwrap()));
    assert!(matches!(duals[1].tangent(), MaybeZero::Zero(_)));
    let names = context
        .builder()
        .borrow()
        .instructions()
        .iter()
        .map(|instruction| instruction.operation().name())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["constructor", "zero"]);
}

#[test]
fn test_operation_transpose() {
    // Native variants apply their payloads' transpose rules, so a tag passes its output cotangent to its input.
    let r#type = ArrayType::scalar(DataType::F32);
    let mut context = TranspositionContext::new(TracingContext::<Array, UnaryFamily<Array>>::new());
    let output_cotangent = context.input(r#type.clone());
    let inputs = [PartialValue::Unknown(r#type)];
    let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
    let outputs = [MaybeZero::Value(output_cotangent.clone())];
    let operation = UnaryFamily::<Array>::from(TagOperation::new("x"));
    operation.transpose(&mut context, &EmptyRegionDriver, &inputs, &outputs, &accumulators).unwrap();
    assert!(matches!(
        context.take_cotangents(&accumulators).unwrap().as_slice(),
        [MaybeZero::Value(cotangent)] if cotangent.atom_id() == output_cotangent.atom_id(),
    ));
}

#[test]
fn test_operation_transpose_members() {
    type Family = MixedFamily<Array>;

    // Transposing an interleaved mixed instruction delegates its array inputs, in input order, to the payload's
    // homogeneous rule and gives each interleaved dimension input a structural zero cotangent.
    let (array_type, dimension_type) = mixed_types();
    let array_type = ArrayIrType::from(array_type);
    let dimension_type = ArrayIrType::from(dimension_type);
    let context = TracingContext::<ArrayIrValue<Array>, Family>::new();
    let output_cotangent = context.input(array_type.clone());
    let inputs = [
        PartialValue::Unknown(array_type.clone()),
        PartialValue::Unknown(dimension_type.clone()),
        PartialValue::Unknown(array_type.clone()),
        PartialValue::Unknown(dimension_type.clone()),
    ];
    let mut transposition = TranspositionContext::new(context.clone());
    let accumulators = transposition.cotangent_accumulators(&inputs, &[]).unwrap();
    Family::from(InterleavedOperation)
        .transpose(
            &mut transposition,
            &EmptyRegionDriver,
            &inputs,
            &[MaybeZero::Value(output_cotangent.clone())],
            &accumulators,
        )
        .unwrap();
    let cotangents = transposition.take_cotangents(&accumulators).unwrap();
    let [MaybeZero::Value(first), MaybeZero::Zero(second), MaybeZero::Value(third), MaybeZero::Zero(fourth)] =
        cotangents.as_slice()
    else {
        panic!("interleaved mixed transposition must classify each input: {cotangents:?}");
    };
    assert_eq!(first.atom_id(), output_cotangent.atom_id());
    assert_eq!(third.atom_id(), output_cotangent.atom_id());
    let dimension_cotangent_type = dimension_type.cotangent().unwrap();
    assert_eq!((second, fourth), (&dimension_cotangent_type, &dimension_cotangent_type));

    // A computational projected member applies its parent-universe rule rather than the homogeneous one, so its
    // inputs receive no contributions.
    let inputs = [PartialValue::Unknown(array_type.clone()), PartialValue::Unknown(array_type.clone())];
    let mut transposition = TranspositionContext::new(context);
    let accumulators = transposition.cotangent_accumulators(&inputs, &[]).unwrap();
    Family::Array(MixedMemberFamily::Interleaved(InterleavedOperation))
        .transpose(
            &mut transposition,
            &EmptyRegionDriver,
            &inputs,
            &[MaybeZero::Value(output_cotangent)],
            &accumulators,
        )
        .unwrap();
    assert_eq!(
        transposition.take_cotangents(&accumulators).unwrap(),
        vec![MaybeZero::Zero(array_type.clone()), MaybeZero::Zero(array_type)],
    );
}

#[test]
fn test_operation_transpose_cotangent_accumulation() {
    static PROVIDED: AtomicUsize = AtomicUsize::new(0);

    /// Family whose addition is reachable only through a handwritten provider, because `skip_from` removes the
    /// generated conversion.
    #[derive(Clone, Debug, Operation)]
    #[ryft(type(ArrayType), constant(Array))]
    enum ProvidedFamily {
        #[ryft(skip_from)]
        Add(AddOperation<ArrayType>),
    }

    impl OperationProvider<ArrayType, AddOperation<ArrayType>> for ProvidedFamily {
        type Operation = Self;

        fn provide(_request: AddOperation<ArrayType>, input_types: &[&ArrayType]) -> Result<Self, ProgramError> {
            check_count!("input", input_types, 2, ProgramError);
            PROVIDED.fetch_add(1, Ordering::Relaxed);
            Ok(Self::Add(AddOperation::new()))
        }
    }

    // The first contribution is stored as is, and the second one is added to it through the family's provider.
    let mut context = TranspositionContext::new(TracingContext::<Array, ProvidedFamily>::new());
    let input_type = ArrayType::scalar(DataType::F32);
    let accumulator = context.cotangent_accumulator(input_type.clone(), true);
    let contribution = context.input(input_type);
    accumulator.accumulate(&mut context, MaybeZero::Value(contribution.clone())).unwrap();
    assert_eq!(PROVIDED.load(Ordering::Relaxed), 0);
    accumulator.accumulate(&mut context, MaybeZero::Value(contribution)).unwrap();
    assert_eq!(PROVIDED.load(Ordering::Relaxed), 1);
    let builder = context.builder().borrow().clone();
    let outputs = builder.instructions()[0].outputs().to_vec();
    let program = builder.build::<Array, Array>(outputs, Placeholder, Placeholder).unwrap();
    assert_eq!(
        program.to_string(),
        indoc! {"
            lambda %0:f32[] .
            let %1:f32[] = add %0 %0
            in (%1)"},
    );
    assert_eq!(program.interpret(Array::scalar(3f32).unwrap()), Ok(Array::scalar(6f32).unwrap()));
}

#[test]
fn test_operation_payload_projection() {
    type Family = ArrayIrFamily<Array>;

    // Native payloads that are `'static` independently of the family's parameters are projected and constructed
    // directly, while a payload that mentions a family parameter is neither.
    let add = ArrayFamily::<Array>::from(AddOperation::new());
    let constant = ArrayFamily::<Array>::from(ConstantOperation::new(Array::scalar(1f32).unwrap()));
    assert_eq!(add.projected_payload::<AddOperation<ArrayType>>(), Some(&AddOperation::new()));
    assert_eq!(add.projected_payload::<ZeroOperation<ArrayType>>(), None);
    assert!(constant.projected_payload::<ConstantOperation<Array>>().is_none());
    assert!(matches!(
        ArrayFamily::<Array>::from_payload(ErasedOperation::new(AddOperation::<ArrayType>::new())),
        Ok(ArrayFamily::Add(_)),
    ));
    let payload = ErasedOperation::new(ConstantOperation::new(Array::scalar(1f32).unwrap()));
    assert!(
        ArrayFamily::<Array>::from_payload(payload)
            .unwrap_err()
            .downcast::<ConstantOperation<Array>>()
            .is_ok()
    );

    // Projected members delegate to their member families in both directions, and a payload that no variant can hold
    // is returned unchanged.
    let operation = Family::Array(ArrayOperation::from(AddOperation::new()));
    assert_eq!(operation.projected_payload::<AddOperation<ArrayType>>(), Some(&AddOperation::new()));
    assert!(matches!(
        Family::from_payload(ErasedOperation::new(AddOperation::<ArrayType>::new())),
        Ok(Family::Array(ArrayOperation::Add(_))),
    ));
    let payload = ErasedOperation::new(ProbeOperation::<0>::new());
    assert!(Family::from_payload(payload).unwrap_err().downcast::<ProbeOperation<0>>().is_ok());

    // Mixed variants project into their payloads but are never constructed, so construction reaches the projected
    // member that holds the same payload type.
    let operation = MixedFamily::<Array>::from(InterleavedOperation);
    assert!(operation.projected_payload::<InterleavedOperation>().is_some());
    assert!(matches!(
        MixedFamily::<Array>::from_payload(ErasedOperation::new(InterleavedOperation)),
        Ok(MixedFamily::Array(MixedMemberFamily::Interleaved(_))),
    ));
}

#[test]
fn test_operation_collective() {
    /// Homogeneous family that marks a collective that it holds directly and one that it holds boxed, but not its
    /// `axis_index`.
    #[derive(Clone, Debug, Operation)]
    #[ryft(type(ArrayType), constant(Array))]
    enum CollectiveFamily {
        #[ryft(collective)]
        Vary(ParallelVaryOperation),
        #[ryft(collective)]
        Permute(Box<ParallelPermuteOperation>),
        AxisIndex(AxisIndexOperation),
        Add(AddOperation<ArrayType>),
    }

    // Marked variants delegate to their payloads, directly and through a box.
    let x = CollectiveAxes::from("x");
    let operation = CollectiveFamily::from(ParallelVaryOperation::new("x".to_string()));
    assert_eq!(operation.axes().as_ref(), &x);
    let operation = CollectiveFamily::from(ParallelPermuteOperation::new(["x", "y"], 4, vec![(0, 1)]));
    assert_eq!(operation.axes().as_ref(), &CollectiveAxes::from(["x", "y"]));

    // Unmarked variants report no axes, even when their payloads are collectives.
    let operation = CollectiveFamily::from(AxisIndexOperation::new("x".to_string()));
    assert_eq!(operation.axes().as_ref(), &CollectiveAxes::default());
    let operation = CollectiveFamily::from(AddOperation::new());
    assert_eq!(operation.axes().as_ref(), &CollectiveAxes::default());

    // A marked projected member delegates to its member family.
    let operation = ArrayIrFamily::<Array>::Array(ArrayOperation::from(ParallelVaryOperation::new("x".to_string())));
    assert_eq!(operation.axes().as_ref(), &x);
    let operation = ArrayIrFamily::<Array>::Array(ArrayOperation::from(AddOperation::new()));
    assert_eq!(operation.axes().as_ref(), &CollectiveAxes::default());
}

#[test]
fn test_operation_guard_provider() {
    // Both inputs share the identity `n`, which relates their extents, and the program tags the first one.
    let n = DimensionVariable::new("n", DimensionBounds::positive(Some(8)).unwrap());
    let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(n)]));
    let mut builder = ProgramBuilder::<Array, ArrayFamily<Array>>::new();
    let first = builder.add_input(vector_type.clone());
    builder.add_input(vector_type.clone());
    let output = builder.add_instruction(TagOperation::new("first"), Vec::new(), vec![first], None).unwrap()[0];
    let program = builder
        .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
        .unwrap();

    // Eager interpretation establishes the input relationships from concrete values, so it needs no guard and the
    // family is usable as an eager family without a handwritten provider.
    let two = Array::vector(vec![1f32, 2f32]).unwrap();
    assert_eq!(program.interpret(vec![two.clone(), Array::vector(vec![3f32, 4f32]).unwrap()]), Ok(vec![two.clone()]));

    // Replaying at a lifted `f32[2]` and a symbolic `f32[n]` leaves `n = 2` unproven, so replay requests a guard that
    // this family cannot represent and reports that instead of dropping the input check.
    let context = TracingContext::<Array, ArrayFamily<Array>>::new();
    let inputs = vec![context.lift(two).unwrap(), context.input(vector_type)];
    assert!(matches!(
        program.interpret_in_context(&context, inputs),
        Err(ProgramError::UnsupportedOperation { message })
            if message == "the input types of `region` leave identity relationships unproven, but the operation \
                           family cannot represent the guard required to preserve these input checks",
    ));
}

#[test]
fn test_operation_guard_provider_native_guard() {
    /// Family with a native guard variant, which provides guards through the blanket supporting provider over its
    /// generated `From<GuardOperation<ArrayType>>` conversion instead of the generated rejecting provider.
    #[derive(Clone, Debug, Operation)]
    #[ryft(type(ArrayType), constant(Array))]
    enum GuardedFamily {
        Tag(TagOperation<ArrayType>),
        Guard(GuardOperation<ArrayType>),
    }

    let n = DimensionVariable::new("n", DimensionBounds::positive(Some(8)).unwrap());
    let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(n)]));
    let mut builder = ProgramBuilder::<Array, GuardedFamily>::new();
    let first = builder.add_input(vector_type.clone());
    builder.add_input(vector_type.clone());
    let output = builder.add_instruction(TagOperation::new("first"), Vec::new(), vec![first], None).unwrap()[0];
    let program = builder
        .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
        .unwrap();

    // The same replay as for a guardless family stages the family's own guard ahead of the body, which then consumes
    // the checked values.
    let two = Array::vector(vec![1f32, 2f32]).unwrap();
    let context = TracingContext::<Array, GuardedFamily>::new();
    let inputs = vec![context.lift(two.clone()).unwrap(), context.input(vector_type)];
    let outputs = program.interpret_in_context(&context, inputs).unwrap();
    let output_atoms = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
    let traced = context
        .builder()
        .borrow()
        .clone()
        .build::<Vec<Array>, Vec<Array>>(output_atoms, vec![Placeholder], vec![Placeholder])
        .unwrap();
    assert_eq!(
        traced.to_string(),
        indoc! {r#"
            lambda %1:f32[n] .
            let %0:f32[2] = const [1.0, 2.0]
                %2:f32[2], %3:f32[2] = guard [boundary="region", declared_types=[f32[n], f32[n]]] %0 %1
                %4:f32[2] = tag [key=first] %2
            in (%4)"#},
    );
    assert_eq!(traced.interpret(vec![two.clone()]), Ok(vec![two]));
}

#[test]
fn test_operation_from_and_try_from() {
    type Family = ArrayFamily<Array>;

    // Owned conversions wrap payloads into their variants, and borrowed conversions recover them.
    let zero = ZeroOperation::new(ArrayType::scalar(DataType::F32));
    let tag = TagOperation::new("key");
    assert!(matches!(Family::from(zero.clone()), Family::Zero(payload) if payload == zero));
    assert_eq!(<&TagOperation<ArrayType>>::try_from(&Family::from(tag.clone())), Ok(&tag));
    assert_eq!(
        <&ProbeOperation<1>>::try_from(&ProbeFamily::from(ProbeOperation::<1>::new())),
        Ok(&ProbeOperation::new())
    );

    // A borrowed conversion of another variant names the stored operation and the expected payload type.
    assert!(matches!(
        <&ZeroOperation<ArrayType>>::try_from(&Family::from(tag)),
        Err(TypeError::Invalid { message })
            if message == "cannot project operation `tag` into a `ZeroOperation<ArrayType>` payload",
    ));

    // `skip_from` leaves room for a handwritten conversion that normalizes the payload, while keeping the borrowed one,
    // and a bare generic payload has neither conversion, so it cannot overlap the concrete variants' conversions.
    #[derive(Clone, Debug, Operation)]
    #[ryft(type(ArrayType), constant(Array))]
    enum ExtensibleFamily<E: Operation<Type = ArrayType>> {
        #[ryft(skip_from)]
        Tag(TagOperation<ArrayType>),
        Zero(ZeroOperation<ArrayType>),
        Extension(E),
    }

    impl<E: Operation<Type = ArrayType>> From<TagOperation<ArrayType>> for ExtensibleFamily<E> {
        fn from(_operation: TagOperation<ArrayType>) -> Self {
            Self::Tag(TagOperation::new("normalized"))
        }
    }

    let operation = ExtensibleFamily::<ZeroOperation<ArrayType>>::from(TagOperation::new("key"));
    assert_eq!(<&TagOperation<ArrayType>>::try_from(&operation), Ok(&TagOperation::new("normalized")));
    let operation = ExtensibleFamily::Extension(ZeroOperation::new(ArrayType::scalar(DataType::F32)));
    assert_eq!(operation.name(), "zero");
    assert!(matches!(
        <&ZeroOperation<ArrayType>>::try_from(&operation),
        Err(TypeError::Invalid { message })
            if message == "cannot project operation `zero` into a `ZeroOperation<ArrayType>` payload",
    ));
}

#[test]
fn test_errors() {
    let test_cases = trybuild::TestCases::new();
    test_cases.compile_fail("tests/operations/error_attributes.rs");
    test_cases.compile_fail("tests/operations/error_conflicting_guard_provider.rs");
    test_cases.compile_fail("tests/operations/error_enums.rs");
    test_cases.compile_fail("tests/operations/error_mismatched_payload_type.rs");
    test_cases.compile_fail("tests/operations/error_structural_batching_without_replicated_projection.rs");
    test_cases.compile_fail("tests/operations/error_types.rs");
    test_cases.compile_fail("tests/operations/error_variants.rs");
}
