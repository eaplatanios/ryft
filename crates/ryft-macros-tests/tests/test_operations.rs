//! Tests for the `#[derive(Operation)]` procedural macro and its optional transform dispatchers.
//!
//! These tests define local stand-in traits and types that mirror the shapes the derives emit against. That keeps the
//! macro tests focused on generated code rather than on the current `ryft-core` implementation details. The fixtures
//! and tests are grouped and ordered by the traits the derives generate: [`Operation`] together with its
//! [`InterpretableOperation`] and [`PartiallyEvaluatableOperation`] companions, then [`BatchableOperation`],
//! [`DifferentiableOperation`], and [`TransposableOperation`].

#![allow(private_interfaces, dead_code)]

use std::marker::PhantomData;

pub(crate) use self::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation,
};

/// Stand-in for `ryft_core::TypeIdentity`.
trait TypeIdentity: Clone {}

/// Identity used by stand-in types that carry no nominal metadata.
#[derive(Clone)]
struct NoIdentity;

impl TypeIdentity for NoIdentity {}

/// Stand-in for `ryft_core::TypeIdentityRenaming`.
struct TypeIdentityRenaming<I: TypeIdentity> {
    marker: PhantomData<I>,
}

impl<I: TypeIdentity> TypeIdentityRenaming<I> {
    /// Creates an empty stand-in identity renaming.
    fn new() -> Self {
        Self { marker: PhantomData }
    }
}

/// Stand-in for `ryft_core::Type`.
trait Type: Clone {
    type Identity: TypeIdentity;
}

/// Stand-in for `ryft_core::DifferentiableType`.
trait DifferentiableType: Type {}

/// Stand-in for `ryft_core::TypeError`. The stand-in keeps the diagnostic message so that generated conversion
/// diagnostics can be asserted exactly.
#[derive(Debug, PartialEq, Eq)]
struct TypeError {
    /// Diagnostic message describing the invalid type contract.
    message: String,
}

impl TypeError {
    /// Stand-in for `ryft_core::TypeError::invalid`.
    fn invalid<M: Into<String>>(message: M) -> Self {
        Self { message: message.into() }
    }
}

/// Stand-in for `ryft_core::ProgramError`.
#[derive(Debug, PartialEq, Eq)]
struct ProgramError;

/// Stand-in for `ryft_core::RegionInterface`.
struct RegionInterface<T: Type> {
    marker: PhantomData<T>,
}

impl<T: Type> RegionInterface<T> {
    /// Creates a stand-in region interface.
    fn new() -> Self {
        Self { marker: PhantomData }
    }
}

/// Stand-in for `ryft_core::OutputRegionProvenance`.
#[derive(Debug, PartialEq, Eq)]
struct OutputRegionProvenance {
    region_index: usize,
    output_index: usize,
}

/// Stand-in for `ryft_core::RegionDriver`.
trait RegionDriver<V: Value, O: Operation<Type = V::Type>> {
    fn region_count(&self) -> usize {
        0
    }
}

/// Stand-in for `ryft_core::InterpretationDriver`.
trait InterpretationDriver<C: Domain>: RegionDriver<C::Value, C::Operation> {}

/// Stand-in for `ryft_core::DifferentiationDriver`.
trait DifferentiationDriver<C: Context> {}

/// Stand-in for `ryft_core::TranspositionDriver`.
trait TranspositionDriver<V: Value, O: Operation<Type = V::Type>> {}

/// Empty region transform driver used when testing operation rules that do not access nested regions.
struct EmptyRegionDriver;

impl<V: Value, O: Operation<Type = V::Type>> RegionDriver<V, O> for EmptyRegionDriver {}

impl<C: Domain> InterpretationDriver<C> for EmptyRegionDriver {}

impl<C: Context> DifferentiationDriver<C> for EmptyRegionDriver {}

impl<V: Value, O: Operation<Type = V::Type>> TranspositionDriver<V, O> for EmptyRegionDriver {}

/// Stand-in for `ryft_core::DifferentiationError`, the error type the differentiation dispatchers return.
#[derive(Debug, PartialEq, Eq)]
struct DifferentiationError;

impl From<ProgramError> for DifferentiationError {
    fn from(_error: ProgramError) -> Self {
        DifferentiationError
    }
}

/// Stand-in for `ryft_core::BatchingError`, the error type the batching dispatchers return.
#[derive(Debug, PartialEq, Eq)]
struct BatchingError;

/// Stand-in for `ryft_core::Domain`.
trait Domain {
    type Type: Type;
    type Value: Value<Type = Self::Type>;
    type Constant: Value<Type = Self::Type>;
    type Operation: Operation<Type = Self::Type>;
}

/// Stand-in for `ryft_core::Context`.
trait Context: Domain {
    fn lift(&self, constant: Self::Constant) -> Result<Self::Value, ProgramError>;
}

/// Stand-in minimal operation family for value-only contexts.
#[derive(Clone, Debug)]
struct NoOperation<T: Type>(PhantomData<fn() -> T>);

impl<T: Type> Operation for NoOperation<T> {
    type Type = T;

    fn name(&self) -> &'static str {
        "no_operation"
    }

    fn infer_output_types(
        &self,
        _input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        Ok(Vec::new())
    }
}

/// Stand-in interpretation context that lifts constants by cloning them.
struct TestContext<V: Value, O: Operation<Type = V::Type> = NoOperation<<V as Value>::Type>> {
    marker: PhantomData<(V, O)>,
}

impl<V: Value, O: Operation<Type = V::Type>> Domain for TestContext<V, O> {
    type Type = V::Type;
    type Value = V;
    type Constant = V;
    type Operation = O;
}

impl<V: Value, O: Operation<Type = V::Type>> Context for TestContext<V, O> {
    fn lift(&self, constant: Self::Constant) -> Result<Self::Value, ProgramError> {
        Ok(constant)
    }
}

/// Stand-in for `ryft_core::Value`. Mirrors the real trait's associated `Type` descriptor, which the generated code
/// pins with `Value<Type = …>` equality bounds.
trait Value: Clone {
    type Type: Type;
}

/// Stand-in for `ryft_core::ValueProjection`.
trait ValueProjection<T: Type>: Value {
    type Projected: Value<Type = T>;

    fn from_projected(value: Self::Projected) -> Self;

    fn into_projected(self) -> Result<Self::Projected, TypeError>;
}

/// Stand-in for `ryft_core::Concretizable`.
trait Concretizable<V> {}

/// Stand-in for `ryft_core::Parameterized`.
trait Parameterized<V> {
    type To<T>;
    type ParameterStructure;
}

impl<V> Parameterized<V> for Vec<V> {
    type To<T> = Vec<T>;
    type ParameterStructure = ();
}

/// Stand-in for `ryft_core::Zero`.
trait Zero<V: Value> {}

impl<V: Value, O: Operation<Type = V::Type>> Zero<V> for TestContext<V, O> {}

/// Stand-in for `ryft_core::Constant`.
trait Constant<V: Value, Stored> {
    fn constant(&self, value: Stored) -> Result<V, ProgramError>;
}

impl<V: Value, O: Operation<Type = V::Type>, Stored: Clone> Constant<V, Stored> for TestContext<V, O>
where
    V: From<Stored>,
{
    fn constant(&self, value: Stored) -> Result<V, ProgramError> {
        Ok(V::from(value))
    }
}

/// Stand-in for `ryft_core::Effects`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Effects {
    Pure,
    Ordered,
}

/// Stand-in for `ryft_core::ReferenceOperationSemantics`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
enum ReferenceOperationSemantics {
    #[default]
    None,
    Read,
}

/// Stand-in for `ryft_core::RegionRole`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RegionRole {
    Computation,
    Rule,
}

/// Stand-in for `ryft_core::RegionSlot`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct RegionSlot {
    name: &'static str,
    role: RegionRole,
}

/// Stand-in for `ryft_core::Operation`.
trait Operation: Clone {
    type Type: Type;

    fn name(&self) -> &'static str;

    fn region_slots(&self) -> &'static [RegionSlot] {
        &[]
    }

    fn region_role(&self, index: usize) -> Option<RegionRole> {
        self.region_slots().get(index).map(|slot| slot.role)
    }

    fn infer_region_input_types(
        &self,
        _input_types: &[Self::Type],
        region_interfaces: &[RegionInterface<Self::Type>],
    ) -> Result<Vec<Option<Vec<Self::Type>>>, TypeError> {
        Ok(vec![None; region_interfaces.len()])
    }

    fn infer_output_types(
        &self,
        input_types: &[Self::Type],
        region_interfaces: &[RegionInterface<Self::Type>],
    ) -> Result<Vec<Self::Type>, TypeError>;

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        let _ = output_index;
        Vec::new()
    }

    fn is_zero(&self, output_index: usize) -> bool {
        let _ = output_index;
        false
    }

    fn reference_semantics(&self) -> std::borrow::Cow<'_, ReferenceOperationSemantics> {
        std::borrow::Cow::Owned(ReferenceOperationSemantics::None)
    }

    fn effects(&self) -> Effects {
        Effects::Pure
    }

    fn rename_type_identities(
        &self,
        _renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(self.clone())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let _ = indentation;
        formatter.write_str(self.name())
    }
}

/// Stand-in for `ryft_core::OperationProjection`.
trait OperationProjection<T: Type>: Operation + From<Self::Projected> {
    type Projected: Operation<Type = T>;
}

/// Stand-in for the operation-family capability used to construct typed structural zeros.
trait ZeroOperationProvider<T: Type>: Operation<Type = T> {}

impl<T: Type, O: Operation<Type = T>> ZeroOperationProvider<T> for O {}

/// Infers projected region input types using the same contract as `ryft_core`'s derive support helper.
fn infer_projected_operation_region_input_types<T: Type, U: Type, O: Operation<Type = T>>(
    operation: &O,
    input_types: &[U],
    region_interfaces: &[RegionInterface<U>],
) -> Result<Vec<Option<Vec<U>>>, TypeError>
where
    U: From<T>,
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
{
    let (input_types, region_interfaces) = project_operation_boundary(input_types, region_interfaces)?;
    Ok(operation
        .infer_region_input_types(&input_types, &region_interfaces)?
        .into_iter()
        .map(|types| types.map(|types| types.into_iter().map(U::from).collect()))
        .collect())
}

/// Infers projected output types using the same contract as `ryft_core`'s derive support helper.
fn infer_projected_operation_output_types<T: Type, U: Type, O: Operation<Type = T>>(
    operation: &O,
    input_types: &[U],
    region_interfaces: &[RegionInterface<U>],
) -> Result<Vec<U>, TypeError>
where
    U: From<T>,
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
{
    let (input_types, region_interfaces) = project_operation_boundary(input_types, region_interfaces)?;
    Ok(operation.infer_output_types(&input_types, &region_interfaces)?.into_iter().map(U::from).collect())
}

/// Projects a stand-in composite inference boundary to one member type.
fn project_operation_boundary<T: Type, U: Type>(
    input_types: &[U],
    region_interfaces: &[RegionInterface<U>],
) -> Result<(Vec<T>, Vec<RegionInterface<T>>), TypeError>
where
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
{
    Ok((
        input_types.iter().map(|r#type| <&T>::try_from(r#type).cloned()).collect::<Result<_, _>>()?,
        region_interfaces.iter().map(|_| RegionInterface::new()).collect(),
    ))
}

/// Stand-in for `ryft_core::InterpretableOperation`.
trait InterpretableOperation<C: Domain>: Operation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError>;
}

/// Interprets a stand-in projected operation and lifts its outputs into the composite value family.
fn interpret_projected_operation<C: Domain, T: Type, O, D>(
    _context: &C,
    operation: &O,
    driver: &D,
    inputs: &[C::Value],
) -> Result<Vec<C::Value>, ProgramError>
where
    C::Value: ValueProjection<T, Projected: Value<Type = T>>,
    O: Operation<Type = T> + InterpretableOperation<EagerContext<<C::Value as ValueProjection<T>>::Projected, O>>,
    D: InterpretationDriver<C>,
{
    if !operation.region_slots().is_empty() || driver.region_count() != 0 {
        return Err(ProgramError);
    }
    let context = EagerContext::<<C::Value as ValueProjection<T>>::Projected, O> { marker: PhantomData };
    let inputs = inputs
        .iter()
        .cloned()
        .map(<C::Value as ValueProjection<T>>::into_projected)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| ProgramError)?;
    Ok(operation
        .interpret(&context, &EmptyRegionDriver, &inputs)?
        .into_iter()
        .map(<C::Value as ValueProjection<T>>::from_projected)
        .collect())
}

/// Stand-in for `ryft_core::TracingContext`. Mirrors the real context's defaulted capture parameter and its
/// `StagingContext` membership at the capture-pinned form used by generated transform dispatch.
struct TracingContext<V: Value, O: Operation<Type = V::Type>, Capture = V> {
    marker: PhantomData<(V, O, Capture)>,
}

impl<V: Value, O: Operation<Type = V::Type>> Domain for TracingContext<V, O> {
    type Type = V::Type;
    type Value = Tracer<TracingContext<V, O>>;
    type Constant = V;
    type Operation = O;
}

impl<V: Value, O: Operation<Type = V::Type>> Context for TracingContext<V, O> {
    fn lift(&self, _constant: Self::Constant) -> Result<Self::Value, ProgramError> {
        Err(ProgramError)
    }
}

impl<V: Value, O: Operation<Type = V::Type>> StagingContext for TracingContext<V, O> {
    type Meta = ();
}

/// Stand-in for `ryft_core::Tracer`. Mirrors the real `Tracer`'s `Value` membership so it can be the value type of a
/// `PartialValue` input in the generated transpose signature, and its defaulted `Meta` parameter so the generated
/// batching dispatchers can name `Tracer<C, <C as StagingContext>::Meta>`.
struct Tracer<C, Meta = ()> {
    marker: PhantomData<(C, Meta)>,
}

impl<C, Meta> Clone for Tracer<C, Meta> {
    fn clone(&self) -> Self {
        Self { marker: PhantomData }
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Value for Tracer<TracingContext<V, O>> {
    type Type = V::Type;
}

/// Stand-in for `ryft_core::MaybeZero`. The manual trait implementations avoid bounding the value parameter, which
/// is instantiated at the `Debug`-less `Tracer` stand-in.
struct MaybeZero<V> {
    label: &'static str,
    marker: PhantomData<V>,
}

impl<V> Clone for MaybeZero<V> {
    fn clone(&self) -> Self {
        Self { label: self.label, marker: PhantomData }
    }
}

impl<V> std::fmt::Debug for MaybeZero<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.label)
    }
}

impl<V> PartialEq for MaybeZero<V> {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label
    }
}

impl<V> Eq for MaybeZero<V> {}

/// Stand-in for `ryft_core::TransposableOperation`.
trait TransposableOperation<V: Value, O: Operation<Type = V::Type>>: Operation<Type = V::Type> {
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError>;
}

/// Stand-in for `ryft_core::Program`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Program<V: Value, O: Operation<Type = V::Type>, Input, Output> {
    label: &'static str,
    constant: Option<V>,
    operation: Option<O>,
    marker: PhantomData<(V, O, Input, Output)>,
}

/// Stand-in for `ryft_core::RegionId`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct RegionId(usize);

/// Stand-in for `ryft_core::Instruction`. The stand-in [`Program`] never attaches regions to its single
/// instruction, so [`regions`](Self::regions) always returns an empty slice and interpretation takes its region-free
/// replay path.
struct Instruction<'operation, O> {
    operation: &'operation O,
}

impl<O> Instruction<'_, O> {
    fn operation(&self) -> &O {
        self.operation
    }

    fn regions(&self) -> &[RegionId] {
        &[]
    }
}

impl<Constant, O, Input, Output> Program<Constant, O, Input, Output>
where
    Constant: Value,
    O: Operation<Type = Constant::Type>,
{
    fn interpret_with<V, LiftConstantFn, InterpretInstructionFn>(
        &self,
        mut input: Vec<V>,
        mut lift_constant: LiftConstantFn,
        mut interpret_instruction: InterpretInstructionFn,
    ) -> Result<Vec<V>, ProgramError>
    where
        V: Value<Type = Constant::Type>,
        LiftConstantFn: FnMut(usize, &Constant) -> Result<V, ProgramError>,
        InterpretInstructionFn: FnMut(&Instruction<'_, O>, &[V]) -> Result<Vec<V>, ProgramError>,
    {
        if let Some(constant) = &self.constant {
            input.push(lift_constant(0, constant)?);
        }
        if let Some(operation) = &self.operation {
            interpret_instruction(&Instruction { operation }, &input)
        } else {
            Ok(input)
        }
    }
}

impl<T, V, O, Input, Output> Program<V, O, Input, Output>
where
    T: DifferentiableType,
    V: Value<Type = T>,
    O: TransposableOperation<V, O> + From<ZeroOperation<T>> + From<AddOperation>,
{
    fn transpose_with_respect_to(
        &self,
        input_indices: &[usize],
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        let _ = input_indices;
        Ok(Program { label: "program_transpose_with_respect_to", constant: None, operation: None, marker: PhantomData })
    }
}

/// Stand-in for `ryft_core::StagingContext`. Mirrors the real trait's `Meta` associated type, which the generated
/// batching dispatchers project when naming the staged flowing tracer type.
trait StagingContext: Context {
    type Meta;
}

impl<V: Value, O: Operation<Type = V::Type>> StagingContext for TestContext<V, O> {
    type Meta = ();
}

/// Stand-in for `ryft_core::DifferentiationDual`. Mirrors only what the generated forward-mode dispatcher references:
/// the generated `jvp` signature names the type over the context's value type, so a label field suffices to observe
/// payload dispatch.
struct DifferentiationDual<V> {
    label: &'static str,
    marker: PhantomData<V>,
}

impl<V> std::fmt::Debug for DifferentiationDual<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("DifferentiationDual").field("label", &self.label).finish()
    }
}

/// Stand-in for `ryft_core::DifferentiableOperation`.
trait DifferentiableOperation<C: Context>: Operation<Type = C::Type> {
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>;
}

/// Stand-in for `ryft_core::Linearization`.
struct Linearization<V: Value, O> {
    label: &'static str,
    marker: PhantomData<(V, O)>,
}

impl<T, V, O, Input, Output> Program<V, O, Input, Output>
where
    T: Type,
    V: Value<Type = T> + SpecialDifferentiableValue,
    O: Operation<Type = T> + From<ZeroOperation<T>>,
{
    /// Stand-in for `ryft_core::Program::linearize`. The `SpecialDifferentiableValue` bound on the value type stands
    /// in for the extra value capabilities a concrete differentiation implementation can require.
    fn linearize(&self) -> Result<Linearization<V, O>, DifferentiationError> {
        Ok(Linearization { label: "program_linearize", marker: PhantomData })
    }

    /// Stand-in for `ryft_core::Program::jvp`.
    fn jvp(&self) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        Ok(Program { label: "program_jvp", constant: None, operation: None, marker: PhantomData })
    }
}

/// Stand-ins for partial-evaluation machinery re-exported through the test crate's root facade.
mod partial {
    use super::{Context, Operation, PhantomData, Program, ProgramError, Value};

    /// Stand-in for `ryft_core::partial::PartialValue`.
    pub(crate) enum PartialValue<V: Value> {
        Known(V),
        Unknown(V::Type),
    }

    /// Stand-in for `ryft_core::partial::PartialEvaluationValue`.
    pub(crate) struct PartialEvaluationValue<V: Value> {
        marker: PhantomData<V>,
    }

    /// Stand-in for `ryft_core::partial::PartialEvaluationContext`.
    pub(crate) struct PartialEvaluationContext<C: Context> {
        context: C,
    }

    impl<C: Context> PartialEvaluationContext<C> {
        pub(crate) fn new(context: C) -> Self {
            Self { context }
        }
    }

    /// Stand-in for `ryft_core::RegionRef`.
    pub(crate) struct RegionRef<V: Value, O: Operation<Type = V::Type>> {
        marker: PhantomData<(V, O)>,
    }

    impl<V: Value, O: Operation<Type = V::Type>> RegionRef<V, O> {
        pub(crate) fn to_program(self) -> Program<V, O, Vec<V>, Vec<V>> {
            Program { label: "region", constant: None, operation: None, marker: PhantomData }
        }
    }

    impl<C: Context> PartialEvaluationContext<C> {
        pub(crate) fn fold_or_residualize<P: Into<C::Operation>>(
            &self,
            operation: P,
            regions: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
            inputs: &[PartialEvaluationValue<C::Value>],
        ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
            let _ = (operation.into(), regions, inputs);
            Ok(Vec::new())
        }
    }

    /// Stand-in for `ryft_core::partial::PartialEvaluationDriver`.
    pub(crate) trait PartialEvaluationDriver<C: Context> {
        fn regions(&self) -> Vec<RegionRef<C::Constant, C::Operation>>;
    }

    /// Stand-in for `ryft_core::partial::PartiallyEvaluatableOperation`.
    pub(crate) trait PartiallyEvaluatableOperation<C: Context>: Clone + Into<C::Operation> {
        fn partially_evaluate<D: PartialEvaluationDriver<C>>(
            &self,
            context: &PartialEvaluationContext<C>,
            _driver: &D,
            inputs: &[PartialEvaluationValue<C::Value>],
        ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
            let _ = (&context.context, inputs);
            Ok(Vec::new())
        }
    }
}

impl<C: Context> partial::PartialEvaluationDriver<C> for EmptyRegionDriver {
    fn regions(&self) -> Vec<partial::RegionRef<C::Constant, C::Operation>> {
        Vec::new()
    }
}

fn transposed<T: Type, V: Value<Type = T>, O: Operation<Type = T>>(
    label: &'static str,
) -> MaybeZero<Tracer<TracingContext<V, O>>> {
    MaybeZero { label, marker: PhantomData }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DataType;

impl Type for DataType {
    type Identity = NoIdentity;
}
impl DifferentiableType for DataType {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ArrayType;

impl Type for ArrayType {
    type Identity = NoIdentity;
}
impl DifferentiableType for ArrayType {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Factor(i64);

impl Value for Factor {
    type Type = ArrayType;
}

/// Data-type-universe counterpart of [`Factor`]. A value type pins exactly one type descriptor through the associated
/// `Type`, so the data-type test enums flow this type instead of reusing [`Factor`] across universes.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ScalarFactor(i64);

impl Value for ScalarFactor {
    type Type = DataType;
}

/// Alternate scalar flowing value used to verify that a derived operation enum's stored constant type need not be
/// the interpretation context's value type.
#[derive(Clone, Debug, PartialEq, Eq)]
struct InterpretedScalarFactor(i64);

impl Value for InterpretedScalarFactor {
    type Type = DataType;
}

impl From<ScalarFactor> for InterpretedScalarFactor {
    fn from(value: ScalarFactor) -> Self {
        Self(value.0)
    }
}

/// Test-only member type used to prove projected operation derivation is independent of any production type family.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ProjectedMemberType<const MEMBER: u8>;

impl<const MEMBER: u8> Type for ProjectedMemberType<MEMBER> {
    type Identity = NoIdentity;
}

/// Composite type containing two unrelated projected member kinds.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ProjectedProgramType {
    First(ProjectedMemberType<0>),
    Third(ProjectedMemberType<2>),
}

impl Type for ProjectedProgramType {
    type Identity = NoIdentity;
}

/// Test-only value belonging to one projected member kind.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ProjectedMemberValue<const MEMBER: u8>(i64);

impl<const MEMBER: u8> Value for ProjectedMemberValue<MEMBER> {
    type Type = ProjectedMemberType<MEMBER>;
}

/// Composite value containing the same two member kinds as [`ProjectedProgramType`].
#[derive(Clone, Debug, PartialEq, Eq)]
enum ProjectedProgramValue<A: Value<Type = ProjectedMemberType<0>> = ProjectedMemberValue<0>> {
    First(A),
    Third(ProjectedMemberValue<2>),
}

impl<A: Value<Type = ProjectedMemberType<0>>> Value for ProjectedProgramValue<A> {
    type Type = ProjectedProgramType;
}

macro_rules! impl_projected_test_member {
    // Generates the type and value projection vocabulary for one test-only composite member.
    ($member:literal, $variant:ident, $projected:ty) => {
        impl From<ProjectedMemberType<$member>> for ProjectedProgramType {
            fn from(r#type: ProjectedMemberType<$member>) -> Self {
                Self::$variant(r#type)
            }
        }

        impl<'t> TryFrom<&'t ProjectedProgramType> for &'t ProjectedMemberType<$member> {
            type Error = TypeError;

            fn try_from(r#type: &'t ProjectedProgramType) -> Result<Self, Self::Error> {
                match r#type {
                    ProjectedProgramType::$variant(r#type) => Ok(r#type),
                    _ => Err(TypeError::invalid("wrong projected member type")),
                }
            }
        }

        impl<A: Value<Type = ProjectedMemberType<0>>> ValueProjection<ProjectedMemberType<$member>>
            for ProjectedProgramValue<A>
        {
            type Projected = $projected;

            fn from_projected(value: Self::Projected) -> Self {
                Self::$variant(value)
            }

            fn into_projected(self) -> Result<Self::Projected, TypeError> {
                match self {
                    Self::$variant(value) => Ok(value),
                    _ => Err(TypeError::invalid("wrong projected member value")),
                }
            }
        }
    };
}

impl_projected_test_member!(0, First, A);
impl_projected_test_member!(2, Third, ProjectedMemberValue<2>);

#[derive(Clone, Debug, PartialEq, Eq)]
struct TranspositionFactor(i64);

impl Value for TranspositionFactor {
    type Type = ArrayType;
}

impl Concretizable<bool> for Factor {}

impl Concretizable<bool> for ScalarFactor {}

trait SpecialTransposableValue {}

impl SpecialTransposableValue for Factor {}

impl SpecialTransposableValue for ScalarFactor {}

impl SpecialTransposableValue for TranspositionFactor {}

/// Extra program-constant capability required by one payload's differentiation implementation.
trait SpecialDifferentiableValue {}

impl SpecialDifferentiableValue for Factor {}

/// Extra value capability required by a recursive payload's partial-evaluation implementation.
trait SpecialPartiallyEvaluatableValue {}

impl SpecialPartiallyEvaluatableValue for Factor {}

impl SpecialPartiallyEvaluatableValue for ScalarFactor {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ZeroOperation<T: Type> {
    r#type: T,
}

impl<T: Type> Operation for ZeroOperation<T> {
    type Type = T;

    fn name(&self) -> &'static str {
        "zero"
    }

    fn infer_output_types(
        &self,
        _input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        Ok(vec![self.r#type.clone()])
    }
}

impl<T: Type, C: Domain<Type = T>> InterpretableOperation<C> for ZeroOperation<T> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<T: Type, C: Context<Type = T>> partial::PartiallyEvaluatableOperation<C> for ZeroOperation<T> where
    C::Operation: From<ZeroOperation<T>>
{
}

impl<T: Type, C: Context<Type = T>> DifferentiableOperation<C> for ZeroOperation<T> {
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Ok(vec![DifferentiationDual { label: "zero", marker: PhantomData }])
    }
}

impl<T: Type, V: Value<Type = T>, O: Operation<Type = T>> TransposableOperation<V, O> for ZeroOperation<T> {
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        Ok(vec![transposed("zero")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AddOperation;

impl Operation for AddOperation {
    type Type = DataType;

    fn name(&self) -> &'static str {
        "add"
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<C: Domain<Type = DataType>> InterpretableOperation<C> for AddOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = DataType>> partial::PartiallyEvaluatableOperation<C> for AddOperation where
    C::Operation: From<AddOperation>
{
}

impl<V: Value<Type = DataType>, O: Operation<Type = DataType>> TransposableOperation<V, O> for AddOperation {
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        Ok(vec![transposed("add")])
    }
}

/// Stand-in for an operation that overrides the optional [`Operation`] metadata and rendering methods so generated
/// enum forwarding through every base operation method is observable.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PrintOperation;

impl Operation for PrintOperation {
    type Type = DataType;

    fn name(&self) -> &'static str {
        "print"
    }

    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot { name: "body", role: RegionRole::Rule }] }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<Option<Vec<DataType>>>, TypeError> {
        Ok(vec![Some(input_types.to_vec())])
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        Ok(input_types.to_vec())
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![OutputRegionProvenance { region_index: 0, output_index }]
    }

    fn is_zero(&self, output_index: usize) -> bool {
        output_index == 3
    }

    fn effects(&self) -> Effects {
        Effects::Ordered
    }

    fn reference_semantics(&self) -> std::borrow::Cow<'_, ReferenceOperationSemantics> {
        std::borrow::Cow::Owned(ReferenceOperationSemantics::Read)
    }

    fn rename_type_identities(
        &self,
        _renaming: &TypeIdentityRenaming<<DataType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        write!(formatter, "{:indentation$}rendered print", "")
    }
}

impl<C: Domain<Type = DataType>> InterpretableOperation<C> for PrintOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = DataType>> partial::PartiallyEvaluatableOperation<C> for PrintOperation where
    C::Operation: From<PrintOperation>
{
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FactorOperation<T: Type, V> {
    factor: V,
    marker: PhantomData<T>,
}

impl<T: Type, V: Clone> Operation for FactorOperation<T, V> {
    type Type = T;

    fn name(&self) -> &'static str {
        "factor"
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<T: Type, F: Clone, C: Domain<Type = T>> InterpretableOperation<C> for FactorOperation<T, F> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<T: Type, F: Clone, C: Context<Type = T>> partial::PartiallyEvaluatableOperation<C> for FactorOperation<T, F> where
    C::Operation: From<FactorOperation<T, F>>
{
}

impl<T: Type, V: Value<Type = T>, O: Operation<Type = T>, F: Clone> TransposableOperation<V, O>
    for FactorOperation<T, F>
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        Ok(vec![transposed("factor")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ConstantOperation<T: Type, V> {
    value: V,
    marker: PhantomData<T>,
}

impl<T: Type, V: Clone> Operation for ConstantOperation<T, V> {
    type Type = T;

    fn name(&self) -> &'static str {
        "constant"
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<T: Type, Constant: Clone, C: Domain<Type = T>> InterpretableOperation<C> for ConstantOperation<T, Constant> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<T: Type, Constant: Clone, C: Context<Type = T>> partial::PartiallyEvaluatableOperation<C>
    for ConstantOperation<T, Constant>
where
    C::Operation: From<ConstantOperation<T, Constant>>,
{
}

impl<T: Type, Constant: Clone, C: Context<Type = T>> DifferentiableOperation<C> for ConstantOperation<T, Constant> {
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Ok(vec![DifferentiationDual { label: "constant", marker: PhantomData }])
    }
}

/// Region-free operation family for one test-only projected member kind.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ProjectedMemberOperation<const MEMBER: u8>;

impl<const MEMBER: u8> Operation for ProjectedMemberOperation<MEMBER> {
    type Type = ProjectedMemberType<MEMBER>;

    fn name(&self) -> &'static str {
        "projected"
    }

    fn infer_output_types(
        &self,
        input_types: &[ProjectedMemberType<MEMBER>],
        _region_interfaces: &[RegionInterface<ProjectedMemberType<MEMBER>>],
    ) -> Result<Vec<ProjectedMemberType<MEMBER>>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<const MEMBER: u8>
    InterpretableOperation<EagerContext<ProjectedMemberValue<MEMBER>, ProjectedMemberOperation<MEMBER>>>
    for ProjectedMemberOperation<MEMBER>
{
    fn interpret<
        D: InterpretationDriver<EagerContext<ProjectedMemberValue<MEMBER>, ProjectedMemberOperation<MEMBER>>>,
    >(
        &self,
        _context: &EagerContext<ProjectedMemberValue<MEMBER>, ProjectedMemberOperation<MEMBER>>,
        _driver: &D,
        inputs: &[ProjectedMemberValue<MEMBER>],
    ) -> Result<Vec<ProjectedMemberValue<MEMBER>>, ProgramError> {
        Ok(inputs.iter().map(|value| ProjectedMemberValue(value.0 + i64::from(MEMBER))).collect())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(
    crate = "crate",
    type = ProjectedProgramType,
    constant = ProjectedProgramValue<A>,
)]
enum ProjectedProgramOperation<A: Value<Type = ProjectedMemberType<0>>> {
    #[ryft(projected(ProjectedMemberType<0>))]
    First(ProjectedMemberOperation<0>),

    #[ryft(projected(ProjectedMemberType<2>, structural))]
    Third(ProjectedMemberOperation<2>),

    Constant(ConstantOperation<ProjectedProgramType, ProjectedProgramValue<A>>),
}

#[test]
fn test_operation_generates_projected_member_dispatch() {
    use partial::PartiallyEvaluatableOperation as _;

    type Value = ProjectedProgramValue<ProjectedMemberValue<0>>;
    type Operation = ProjectedProgramOperation<ProjectedMemberValue<0>>;
    type Context = EagerContext<Value, Operation>;

    let first = Operation::from(ProjectedMemberOperation::<0>);
    let third = Operation::from(ProjectedMemberOperation::<2>);
    let first_type = ProjectedProgramType::First(ProjectedMemberType);
    let third_type = ProjectedProgramType::Third(ProjectedMemberType);

    // Base operation metadata and inference project both computational and structural members to their declared type
    // and lift results back into the composite family.
    assert_eq!(first.name(), "projected");
    assert_eq!(first.to_string(), "projected");
    assert_eq!(first.infer_output_types(std::slice::from_ref(&first_type), &[]), Ok(vec![first_type.clone()]));
    assert_eq!(
        first.infer_output_types(std::slice::from_ref(&third_type), &[]),
        Err(TypeError::invalid("wrong projected member type")),
    );
    assert_eq!(
        first.infer_region_input_types(std::slice::from_ref(&first_type), &[RegionInterface::new()]),
        Ok(vec![None]),
    );

    // Eager interpretation executes in the selected member universe and lifts the result into the composite value.
    assert_eq!(
        third.interpret(
            &Context { marker: PhantomData },
            &EmptyRegionDriver,
            &[ProjectedProgramValue::Third(ProjectedMemberValue(7))],
        ),
        Ok(vec![ProjectedProgramValue::Third(ProjectedMemberValue(9))]),
    );

    // Projected members use the outer operation's canonical fold-or-residualize path and therefore require no
    // projected payload partial-evaluation implementation.
    let partial_context = partial::PartialEvaluationContext::new(Context { marker: PhantomData });
    assert!(third.partially_evaluate(&partial_context, &EmptyRegionDriver, &[]).unwrap().is_empty());

    // Concrete member payload conversions retain their variant identity, and a wrong-payload projection reports the
    // canonical conversion diagnostic naming both the stored operation and the expected payload type.
    assert_eq!(<&ProjectedMemberOperation<0>>::try_from(&first), Ok(&ProjectedMemberOperation));
    assert_eq!(
        <&ProjectedMemberOperation<2>>::try_from(&first),
        Err(TypeError::invalid("cannot project operation 'projected' into a 'ProjectedMemberOperation<2>' payload")),
    );
}

impl<T: Type, V: Value<Type = T>, O: Operation<Type = T>, F: Clone> TransposableOperation<V, O>
    for ConstantOperation<T, F>
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        Ok(vec![transposed("constant")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CustomJvpOperation<T: Type, V> {
    tag: &'static str,
    marker: PhantomData<(T, V)>,
}

impl<T: Type, V: Clone> Operation for CustomJvpOperation<T, V> {
    type Type = T;

    fn name(&self) -> &'static str {
        "custom_jvp"
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<T: Type, Constant: Clone, C: Domain<Type = T>> InterpretableOperation<C> for CustomJvpOperation<T, Constant> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<T: Type, Constant: Clone, C: Context<Type = T>> partial::PartiallyEvaluatableOperation<C>
    for CustomJvpOperation<T, Constant>
where
    C::Operation: From<CustomJvpOperation<T, Constant>>,
{
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
enum DataOperation<V: Value<Type = DataType>> {
    Zero(ZeroOperation<DataType>),
    Add(AddOperation),
    Print(PrintOperation),
    Factor(FactorOperation<DataType, V>),
    CustomJvp(Box<CustomJvpOperation<DataType, V>>),
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
enum WhereBoundOperation<V>
where
    V: Value<Type = DataType>,
{
    Zero(ZeroOperation<DataType>),
    Factor(FactorOperation<DataType, V>),
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
#[ryft(dispatch(transposition))]
enum LinearScalarOperation<V: Value<Type = DataType>, C: Value<Type = DataType> = V> {
    Zero(ZeroOperation<DataType>),
    Constant(ConstantOperation<DataType, V>),
    Add(AddOperation),
    Factor(FactorOperation<DataType, C>),
}

#[test]
fn test_data_operation() {
    let zero = DataOperation::<ScalarFactor>::from(ZeroOperation { r#type: DataType });
    let add = DataOperation::<ScalarFactor>::from(AddOperation);
    let factor = DataOperation::<ScalarFactor>::from(FactorOperation { factor: ScalarFactor(7), marker: PhantomData });
    let custom_jvp = DataOperation::<ScalarFactor>::from(CustomJvpOperation { tag: "tag", marker: PhantomData });

    assert_eq!(zero.name(), "zero");
    assert_eq!(add.name(), "add");
    assert_eq!(factor.name(), "factor");
    assert_eq!(custom_jvp.name(), "custom_jvp");

    assert_eq!(add.infer_output_types(&[DataType], &[]), Ok(vec![DataType]));
    assert_eq!(zero.to_string(), "zero");
    assert_eq!(custom_jvp.to_string(), "custom_jvp");

    assert_eq!(<&ZeroOperation<DataType>>::try_from(&zero), Ok(&ZeroOperation { r#type: DataType }));
    assert_eq!(<&AddOperation>::try_from(&add), Ok(&AddOperation));
    assert_eq!(
        <&FactorOperation<DataType, ScalarFactor>>::try_from(&factor),
        Ok(&FactorOperation { factor: ScalarFactor(7), marker: PhantomData }),
    );
    assert_eq!(
        <&CustomJvpOperation<DataType, ScalarFactor>>::try_from(&custom_jvp),
        Ok(&CustomJvpOperation { tag: "tag", marker: PhantomData }),
    );
    assert_eq!(
        <&AddOperation>::try_from(&zero),
        Err(TypeError::invalid("cannot project operation 'zero' into a 'AddOperation' payload")),
    );
}

#[test]
fn test_operation_generates_operation_forwarding() {
    let add = DataOperation::<ScalarFactor>::from(AddOperation);
    let print = DataOperation::<ScalarFactor>::from(PrintOperation);

    assert_eq!(add.effects(), Effects::Pure);
    assert_eq!(print.effects(), Effects::Ordered);
    assert_eq!(print.region_slots(), &[RegionSlot { name: "body", role: RegionRole::Rule }]);
    assert_eq!(print.region_role(0), Some(RegionRole::Rule));
    assert_eq!(print.rename_type_identities(&TypeIdentityRenaming::new()), Ok(print.clone()));
    assert_eq!(print.infer_region_input_types(&[DataType], &[]), Ok(vec![Some(vec![DataType])]));
    assert_eq!(print.output_region_provenance(3), vec![OutputRegionProvenance { region_index: 0, output_index: 3 }],);
    assert!(!add.is_zero(0));
    assert!(print.is_zero(3));
    assert!(!print.is_zero(4));
    assert_eq!(add.reference_semantics().into_owned(), ReferenceOperationSemantics::None);
    assert_eq!(print.reference_semantics().into_owned(), ReferenceOperationSemantics::Read);
    assert_eq!(print.to_string(), "rendered print");
}

#[test]
fn test_operation_infers_value_type_from_where_clause() {
    let operation = WhereBoundOperation::<ScalarFactor>::from(ZeroOperation { r#type: DataType });

    assert_eq!(operation.name(), "zero");
    assert_eq!(operation.infer_output_types(&[], &[]), Ok(vec![DataType]));
}

#[derive(Clone, Debug, ryft::Operation)]
enum DefaultPathOperation<V: ryft::Value<Type = ryft::ArrayType>> {
    Zero(ryft::ZeroOperation<ryft::ArrayType>),
    Constant(ryft::ConstantOperation<V>),
}

#[derive(Clone, Debug, ryft::Operation)]
#[ryft(dispatch(transposition))]
enum DefaultPathLinearOperation<V: ryft::Value<Type = ryft::ArrayType>> {
    Zero(ryft::ZeroOperation<ryft::ArrayType>),
    Constant(ryft::ConstantOperation<V>),
}

#[test]
fn test_operation_default_crate_path_is_ryft() {
    let r#type = ryft::ArrayType::scalar(ryft::DataType::F64);
    let operation = DefaultPathOperation::<ryft::Array>::from(ryft::ZeroOperation::new(r#type.clone()));
    let linear_operation = DefaultPathLinearOperation::<ryft::Array>::from(ryft::ZeroOperation::new(r#type));
    assert_eq!(ryft::Operation::name(&operation), "zero");
    assert_eq!(ryft::Operation::name(&linear_operation), "zero");
}

/// Mixed-boundary fixtures for the declared-member machinery. Unlike the rest of this file, this module builds on the
/// real `ryft` member vocabulary instead of stand-ins, because the mixed contracts the derive emits against
/// ([`MemberOperation`](ryft::MemberOperation), the member zero constructor selected per output universe, and
/// [`transpose_mixed_operation`](ryft::transpose_mixed_operation)) are defined over real member universes, and a
/// stand-in could not pin how the real machinery classifies an interleaved operand list.
mod mixed_members {
    use ryft::arrays::Array;
    use ryft::{
        ArrayIrType, ArrayIrValue, ArrayType, Context, DataType, DifferentiableOperation, DifferentiableType,
        DifferentiationDriver, DifferentiationDual, Dimension, DimensionBounds, DimensionOperation, DimensionType,
        DimensionValue, DimensionVariable, EmptyRegionDriver, MaybeZero, MemberDifferentiableOperation,
        MemberInterpretableOperation, MemberOperation, Operation, PartialValue, ProgramError, RegionInterface, Shape,
        StagingContext, Tracer, TracingContext, TransposableOperation, TranspositionDriver, TypeError,
        TypeIdentityRenaming, Typed, Value, ZeroOperation, ZeroOperationProvider,
    };

    /// Member payload whose parent instruction interleaves its two array data operands with two first-class dimension
    /// operands. No production payload arranges its operands this way, so this fixture is what pins that the generated
    /// mixed dispatchers classify operands individually instead of splitting on the first dimension operand.
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
            if input_types.len() != 2 {
                return Err(TypeError::invalid("interleaved expects two array operands"));
            }
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
            // The parent boundary is `(array, dimension, array, dimension) -> array`, so the payload's own homogeneous
            // rule sees the two array operands and the dimension operands only select the result geometry.
            let arrays = input_types
                .iter()
                .filter_map(|r#type| <&ArrayType>::try_from(r#type).ok())
                .cloned()
                .collect::<Vec<_>>();
            Ok(self.infer_output_types(arrays.as_slice(), &[])?.into_iter().map(Into::into).collect())
        }

        fn rename_parent_type_identities(
            &self,
            _renaming: &TypeIdentityRenaming<DimensionVariable>,
        ) -> Result<Self, TypeError> {
            Ok(self.clone())
        }
    }

    impl<C: Context<Type = ArrayIrType>> MemberInterpretableOperation<C> for InterleavedOperation {
        fn interpret_in_parent<D: ryft::InterpretationDriver<C>>(
            &self,
            _context: &C,
            _driver: &D,
            inputs: &[C::Value],
        ) -> Result<Vec<C::Value>, ProgramError> {
            Ok(vec![inputs[0].clone()])
        }
    }

    impl<C: Context<Type = ArrayIrType, Operation: From<InterleavedOperation>>> MemberDifferentiableOperation<C>
        for InterleavedOperation
    {
        fn jvp_in_parent<D: DifferentiationDriver<C>>(
            &self,
            context: &C,
            _driver: &D,
            inputs: &[DifferentiationDual<C::Value>],
        ) -> Result<Vec<DifferentiationDual<C::Value>>, ryft::DifferentiationError> {
            // A linear payload pushes its tangents through the same mixed instruction, keeping the geometry operands
            // as primals.
            let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let primal = context.bind(self.clone(), Vec::new(), primals.as_slice())?.remove(0);
            Ok(vec![DifferentiationDual::new(primal, inputs[0].tangent().clone())?])
        }
    }

    impl<V: Value<Type = ArrayType>, O: Operation<Type = ArrayType>> TransposableOperation<V, O> for InterleavedOperation {
        fn transpose<D: TranspositionDriver<V, O>>(
            &self,
            _context: &mut TracingContext<V, O>,
            _driver: &D,
            inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
            outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ryft::DifferentiationError> {
            // The homogeneous rule sees exactly the two array data operands and forwards the output cotangent to each
            // one that is linear in the transposed program.
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(ProgramError::InvalidArgument {
                    message: "interleaved transposition expects two operands and one output".to_string(),
                }
                .into());
            }
            Ok(inputs
                .iter()
                .map(|input| match input {
                    PartialValue::Unknown(_) => Ok(outputs[0].clone()),
                    PartialValue::Known(_) => input.r#type().cotangent().map(MaybeZero::Zero),
                })
                .collect::<Result<Vec<_>, _>>()?)
        }
    }

    /// Structural member payload whose parent instruction produces one array output and one first-class dimension
    /// output, so its generated forward-mode arm must select a zero tangent universe per output instead of assuming
    /// that every output belongs to the family's computational member.
    #[derive(Clone, Debug)]
    struct MixedUniverseConstructorOperation {
        /// Array output type of the constructed instruction.
        r#type: ArrayType,

        /// First-class dimension output type of the constructed instruction.
        dimension_type: DimensionType,
    }

    impl Operation for MixedUniverseConstructorOperation {
        type Type = ArrayType;

        fn name(&self) -> &'static str {
            "mixed_universe_constructor"
        }

        fn infer_output_types(
            &self,
            _input_types: &[ArrayType],
            _region_interfaces: &[RegionInterface<ArrayType>],
        ) -> Result<Vec<ArrayType>, TypeError> {
            Ok(vec![self.r#type.clone()])
        }
    }

    impl MemberOperation<ArrayIrType> for MixedUniverseConstructorOperation {
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
            Ok(vec![self.r#type.clone().into(), self.dimension_type.clone().into()])
        }

        fn rename_parent_type_identities(
            &self,
            _renaming: &TypeIdentityRenaming<DimensionVariable>,
        ) -> Result<Self, TypeError> {
            Ok(self.clone())
        }
    }

    impl<C: Context<Type = ArrayIrType>> MemberInterpretableOperation<C> for MixedUniverseConstructorOperation {
        fn interpret_in_parent<D: ryft::InterpretationDriver<C>>(
            &self,
            _context: &C,
            _driver: &D,
            _inputs: &[C::Value],
        ) -> Result<Vec<C::Value>, ProgramError> {
            Err(ProgramError::UnsupportedOperation {
                message: "mixed_universe_constructor has no eager semantics".to_string(),
            })
        }
    }

    impl<V: Value<Type = ArrayType>, O: Operation<Type = ArrayType>> TransposableOperation<V, O>
        for MixedUniverseConstructorOperation
    {
        fn transpose<D: TranspositionDriver<V, O>>(
            &self,
            _context: &mut TracingContext<V, O>,
            _driver: &D,
            inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
            _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ryft::DifferentiationError> {
            inputs.iter().map(|input| input.r#type().cotangent().map(MaybeZero::Zero)).collect()
        }
    }

    /// Homogeneous array member family of [`MixedProgramOperation`], which is the family every projected boundary and
    /// every delegated mixed transpose rule runs in.
    #[derive(Clone, Debug, ryft::Operation)]
    #[ryft(dispatch(transposition))]
    enum MixedMemberOperation<V: Value<Type = ArrayType>> {
        /// Member zero constructor, which is also the tangent constructor the family's structural mixed arm stages.
        Zero(ZeroOperation<ArrayType>),

        /// Member view of the interleaved mixed payload, which owns its homogeneous transpose rule.
        Interleaved(InterleavedOperation),

        /// Member view of the structural mixed constructor.
        MixedUniverseConstructor(MixedUniverseConstructorOperation),

        /// Member constant, which is what ties this family to its flowing value type.
        Constant(ryft::ConstantOperation<V>),
    }

    /// Operation family with two declared member universes: computational arrays and structural first-class
    /// dimensions. Its mixed variants take their data universe from that declaration instead of naming it.
    #[derive(Clone, Debug, ryft::Operation)]
    #[ryft(type = ArrayIrType, constant = ArrayIrValue<A>)]
    #[ryft(members(ArrayType, structural(DimensionType)))]
    #[ryft(dispatch(differentiation, transposition))]
    enum MixedProgramOperation<A: Value<Type = ArrayType>> {
        /// Computational mixed payload whose parent instruction interleaves array and dimension operands.
        #[ryft(mixed)]
        Interleaved(InterleavedOperation),

        /// Structural mixed payload whose parent outputs span both declared member universes.
        #[ryft(mixed(structural))]
        MixedUniverseConstructor(MixedUniverseConstructorOperation),

        /// Homogeneous array member family, which is also this family's canonical array projection.
        #[ryft(projected(ArrayType))]
        Array(MixedMemberOperation<A>),

        /// Homogeneous first-class-dimension member family.
        #[ryft(projected(DimensionType, structural))]
        Dimension(DimensionOperation<DimensionValue>),
    }

    impl<A, C> MemberDifferentiableOperation<C> for MixedMemberOperation<A>
    where
        A: Value<Type = ArrayType>,
        C: Context<Type = ArrayIrType, Operation: From<MixedMemberOperation<A>>>,
    {
        fn jvp_in_parent<D: DifferentiationDriver<C>>(
            &self,
            context: &C,
            _driver: &D,
            inputs: &[DifferentiationDual<C::Value>],
        ) -> Result<Vec<DifferentiationDual<C::Value>>, ryft::DifferentiationError> {
            // The fixture's member family is only the projection target of this operation family, so its
            // parent-universe rule stages the member instruction and reports constant outputs.
            let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            context
                .bind(self.clone(), Vec::new(), primals.as_slice())?
                .into_iter()
                .map(DifferentiationDual::new_with_zero_tangent)
                .collect()
        }
    }

    impl<A: Value<Type = ArrayType>> From<ZeroOperation<ArrayType>> for MixedProgramOperation<A> {
        fn from(operation: ZeroOperation<ArrayType>) -> Self {
            Self::Array(MixedMemberOperation::Zero(operation))
        }
    }

    impl<A: Value<Type = ArrayType>> ZeroOperationProvider<ArrayIrType> for MixedProgramOperation<A> {
        fn zero_operation(r#type: ArrayIrType) -> Result<Self, ProgramError> {
            Ok(Self::from(ZeroOperation::new(<&ArrayType>::try_from(&r#type)?.clone())))
        }
    }

    /// Returns the fixture's array member type together with the first-class dimension member type its mixed
    /// instructions consume and produce. The array type is static because reconstructing a mixed instruction from type
    /// metadata alone requires that its geometry not live in runtime identity references.
    fn fixture_types() -> (ArrayType, DimensionType) {
        let dimension_type =
            DimensionType::new(DimensionVariable::new("items", DimensionBounds::new(1, Some(8)).unwrap()));
        (ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])), dimension_type)
    }

    #[test]
    fn test_operation_generates_declared_member_mixed_dispatch() {
        type Operation = MixedProgramOperation<Array>;

        // A bare `mixed` marker declares the same computational data universe the family declares, so base inference
        // still runs through the payload's parent boundary and accepts the interleaved operand arrangement.
        let (array_type, dimension_type) = fixture_types();
        let operation = Operation::from(InterleavedOperation);
        assert_eq!(operation.name(), "interleaved");
        assert_eq!(
            operation.infer_output_types(
                &[
                    array_type.clone().into(),
                    dimension_type.clone().into(),
                    array_type.clone().into(),
                    dimension_type.clone().into(),
                ],
                &[],
            ),
            Ok(vec![array_type.clone().into()]),
        );

        // Transposing that interleaved instruction delegates the array operands, in operand order, to the payload's
        // homogeneous rule and gives each interleaved dimension operand a structural zero cotangent.
        let mut context = TracingContext::<ArrayIrValue<Array>, Operation>::new();
        let output_cotangent = context.input(array_type.clone().into());
        let cotangents = operation
            .transpose(
                &mut context,
                &EmptyRegionDriver,
                &[
                    PartialValue::Unknown(array_type.clone().into()),
                    PartialValue::Unknown(dimension_type.clone().into()),
                    PartialValue::Unknown(array_type.clone().into()),
                    PartialValue::Unknown(dimension_type.clone().into()),
                ],
                &[MaybeZero::Value(output_cotangent.clone())],
            )
            .unwrap();
        let [
            MaybeZero::Value(first_cotangent),
            MaybeZero::Zero(second_cotangent_type),
            MaybeZero::Value(third_cotangent),
            MaybeZero::Zero(fourth_cotangent_type),
        ] = cotangents.as_slice()
        else {
            panic!("interleaved mixed transposition must classify each operand: {cotangents:?}");
        };
        assert_eq!(first_cotangent.atom_id(), output_cotangent.atom_id());
        assert_eq!(third_cotangent.atom_id(), output_cotangent.atom_id());
        let dimension_cotangent_type = ArrayIrType::from(dimension_type).cotangent().unwrap();
        assert_eq!(second_cotangent_type, &dimension_cotangent_type);
        assert_eq!(fourth_cotangent_type, &dimension_cotangent_type);
    }

    #[test]
    fn test_operation_generates_structural_mixed_tangents_per_output_universe() {
        type Operation = MixedProgramOperation<Array>;

        // A structural mixed payload stages its primal and one zero tangent per output. The array output's tangent is
        // staged in the computational member universe, while the first-class dimension output has a zero differential
        // space and therefore receives a structural zero tangent with no staged instruction.
        let (array_type, dimension_type) = fixture_types();
        let operation = Operation::from(MixedUniverseConstructorOperation {
            r#type: array_type.clone(),
            dimension_type: dimension_type.clone(),
        });
        let context = TracingContext::<ArrayIrValue<Array>, Operation>::new();
        let duals = operation.jvp(&context, &EmptyRegionDriver, &[]).unwrap();

        assert_eq!(duals.len(), 2);
        assert_eq!(duals[0].primal().r#type().as_ref(), &ArrayIrType::from(array_type.clone()));
        assert_eq!(duals[1].primal().r#type().as_ref(), &ArrayIrType::from(dimension_type.clone()));
        let MaybeZero::Value(array_tangent) = duals[0].tangent() else {
            panic!("an array output of a structural mixed payload stages a member zero tangent: {duals:?}");
        };
        assert_eq!(array_tangent.r#type().as_ref(), &ArrayIrType::from(array_type.tangent().unwrap()));
        assert!(matches!(duals[1].tangent(), MaybeZero::Zero(_)));

        // The staged program holds the primal instruction plus exactly one staged array zero tangent.
        let names = context
            .builder()
            .borrow()
            .instructions()
            .iter()
            .map(|instruction| instruction.operation().name())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["mixed_universe_constructor", "zero"]);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DotOperation;

impl Operation for DotOperation {
    type Type = ArrayType;

    fn name(&self) -> &'static str {
        "dot"
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for DotOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C> for DotOperation where
    C::Operation: From<DotOperation>
{
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum BackendPayload {}

impl Operation for BackendPayload {
    type Type = ArrayType;

    fn name(&self) -> &'static str {
        match *self {}
    }

    fn infer_output_types(
        &self,
        _input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        match *self {}
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for BackendPayload {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        match *self {}
    }
}

impl<C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C> for BackendPayload where
    C::Operation: From<BackendPayload>
{
}

impl<V: Value<Type = ArrayType>, O: Operation<Type = ArrayType>> TransposableOperation<V, O> for BackendPayload {
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        match *self {}
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SpecialOperation;

impl Operation for SpecialOperation {
    type Type = ArrayType;

    fn name(&self) -> &'static str {
        "special"
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for SpecialOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C> for SpecialOperation where
    C::Operation: From<SpecialOperation>
{
}

impl<C> DifferentiableOperation<C> for SpecialOperation
where
    C: Context<Type = ArrayType>,
    C::Constant: SpecialDifferentiableValue,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Ok(vec![DifferentiationDual { label: "special", marker: PhantomData }])
    }
}

impl<V, O> TransposableOperation<V, O> for SpecialOperation
where
    V: Value<Type = ArrayType> + SpecialTransposableValue,
    O: Operation<Type = ArrayType>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        Ok(vec![transposed("special")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
#[ryft(dispatch(transposition))]
enum SpecialLinearOperation<V: Value<Type = ArrayType>> {
    Special(SpecialOperation),
    Constant(ConstantOperation<ArrayType, V>),
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
#[ryft(dispatch(differentiation))]
enum DifferentiableArrayOperation<V: Value<Type = ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    Special(SpecialOperation),
    Constant(ConstantOperation<ArrayType, V>),
}

#[test]
fn test_operation_propagates_differentiation_payload_bounds() {
    type Operation = DifferentiableArrayOperation<Factor>;

    let context = TestContext::<Factor, Operation> { marker: PhantomData };
    let operation = Operation::from(SpecialOperation);
    let outputs = operation.jvp(&context, &EmptyRegionDriver, &[]).unwrap();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].label, "special");
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
enum InferredArrayOperation<V: Value<Type = ArrayType>, C: Value<Type = ArrayType> = V> {
    Zero(ZeroOperation<ArrayType>),
    Constant(ConstantOperation<ArrayType, V>),
    Factor(FactorOperation<ArrayType, C>),
}

#[test]
fn test_array_operation_type_inference() {
    type Operation = InferredArrayOperation<Factor>;

    let zero = Operation::from(ZeroOperation { r#type: ArrayType });
    let constant = Operation::from(ConstantOperation { value: Factor(5), marker: PhantomData });
    let factor = Operation::from(FactorOperation { factor: Factor(17), marker: PhantomData });

    assert_eq!(zero.name(), "zero");
    assert_eq!(constant.name(), "constant");
    assert_eq!(factor.name(), "factor");
    assert_eq!(zero.infer_output_types(&[], &[]), Ok(vec![ArrayType]));
    assert_eq!(
        <&FactorOperation<ArrayType, Factor>>::try_from(&factor),
        Ok(&FactorOperation { factor: Factor(17), marker: PhantomData }),
    );
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
enum ArrayOperation<V: Value<Type = ArrayType>, Backend = BackendPayload> {
    Zero(ZeroOperation<ArrayType>),
    Dot(DotOperation),
    Factor(FactorOperation<ArrayType, V>),
    Backend(Backend),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WhileOperation<T: Type, V, O> {
    marker: PhantomData<(T, V, O)>,
}

impl<T: Type, V: Clone, O: Clone> Operation for WhileOperation<T, V, O> {
    type Type = T;

    fn name(&self) -> &'static str {
        "while"
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<T: Type, W: Clone, O: Clone, C: Domain<Type = T>> InterpretableOperation<C> for WhileOperation<T, W, O> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<T: Type, W: Clone, O: Operation<Type = T>, C: Context<Type = T>> partial::PartiallyEvaluatableOperation<C>
    for WhileOperation<T, W, O>
where
    C::Operation: From<WhileOperation<T, W, O>>,
{
}

impl<T: Type, V: Value<Type = T>, O: Operation<Type = T>, W: Clone, P: Clone> TransposableOperation<V, O>
    for WhileOperation<T, W, P>
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        Ok(vec![transposed("while")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RecomputeOperation<O> {
    operation: O,
}

impl<O: Operation<Type = ArrayType>> Operation for RecomputeOperation<O> {
    type Type = ArrayType;

    fn name(&self) -> &'static str {
        self.operation.name()
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        self.operation.infer_output_types(input_types, region_interfaces)
    }
}

impl<O: Operation<Type = ArrayType>, C: Domain<Type = ArrayType>> InterpretableOperation<C> for RecomputeOperation<O> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<RecomputedOperation: Operation<Type = ArrayType>, C: Context<Type = ArrayType>>
    partial::PartiallyEvaluatableOperation<C> for RecomputeOperation<RecomputedOperation>
where
    C::Operation: From<RecomputeOperation<RecomputedOperation>>,
{
}

impl<V: Value<Type = ArrayType>, O: Operation<Type = ArrayType>, P: Operation<Type = ArrayType>>
    TransposableOperation<V, O> for RecomputeOperation<P>
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        Ok(vec![transposed("recompute")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CustomVjpCallOperation<T: Type, C, O, F> {
    marker: PhantomData<(T, C, O, F)>,
}

impl<T: Type, C: Clone, O: Clone, F: Clone> Operation for CustomVjpCallOperation<T, C, O, F> {
    type Type = T;

    fn name(&self) -> &'static str {
        "custom_vjp_call"
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<T: Type, Constant: Clone, O: Clone, F: Clone, C: Domain<Type = T>> InterpretableOperation<C>
    for CustomVjpCallOperation<T, Constant, O, F>
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<T: Type, Constant: Clone, CallOperation: Clone, F: Clone, C: Context<Type = T>>
    partial::PartiallyEvaluatableOperation<C> for CustomVjpCallOperation<T, Constant, CallOperation, F>
where
    C::Operation: From<CustomVjpCallOperation<T, Constant, CallOperation, F>>,
{
}

impl<T: Type, V: Value<Type = T>, O: Operation<Type = T>, C: Clone, P: Clone, F: Clone> TransposableOperation<V, O>
    for CustomVjpCallOperation<T, C, P, F>
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        Ok(vec![transposed("custom_vjp_call")])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
#[ryft(dispatch(transposition))]
enum LinearArrayOperation<
    V: Value<Type = ArrayType>,
    C: Value<Type = ArrayType>,
    Backend = BackendPayload,
    F: Value<Type = ArrayType> = V,
    P: Operation<Type = ArrayType> = ArrayOperation<C, Backend>,
> {
    Zero(ZeroOperation<ArrayType>),
    Factor(FactorOperation<ArrayType, F>),
    Recompute(RecomputeOperation<P>),
    While(Box<WhileOperation<ArrayType, V, Self>>),
    CustomVjpCall(Box<CustomVjpCallOperation<ArrayType, C, P, F>>),
    Backend(Backend),
}

#[test]
fn test_array_operation_generic_payload_conversion_skip() {
    let zero = ArrayOperation::<Factor>::from(ZeroOperation { r#type: ArrayType });
    let dot = ArrayOperation::<Factor>::from(DotOperation);
    let factor = ArrayOperation::<Factor>::from(FactorOperation { factor: Factor(11), marker: PhantomData });

    assert_eq!(zero.name(), "zero");
    assert_eq!(dot.name(), "dot");
    assert_eq!(factor.name(), "factor");
    assert_eq!(dot.infer_output_types(&[ArrayType], &[]), Ok(vec![ArrayType]));
    assert_eq!(dot.to_string(), "dot");

    assert_eq!(<&DotOperation>::try_from(&dot), Ok(&DotOperation));
    assert_eq!(
        <&FactorOperation<ArrayType, Factor>>::try_from(&factor),
        Ok(&FactorOperation { factor: Factor(11), marker: PhantomData }),
    );

    // If the derive generated `From<Backend>` for `Backend(Backend)`, it would overlap with `From<DotOperation>`
    // because `Backend` is unconstrained and could be `DotOperation`. This test target compiling proves that the
    // bare generic payload was skipped automatically.
}

#[test]
fn test_linear_array_operation_shape() {
    type Linear = LinearArrayOperation<Factor, Factor>;

    let zero = Linear::from(ZeroOperation { r#type: ArrayType });
    let factor = Linear::from(FactorOperation { factor: Factor(13), marker: PhantomData });
    let recompute = Linear::from(RecomputeOperation { operation: ArrayOperation::<Factor>::from(DotOperation) });
    let while_operation = Linear::from(WhileOperation::<ArrayType, Factor, Linear> { marker: PhantomData });
    let custom_vjp_call =
        Linear::from(CustomVjpCallOperation::<ArrayType, Factor, ArrayOperation<Factor, BackendPayload>, Factor> {
            marker: PhantomData,
        });

    assert_eq!(zero.name(), "zero");
    assert_eq!(factor.name(), "factor");
    assert_eq!(recompute.name(), "dot");
    assert_eq!(while_operation.name(), "while");
    assert_eq!(custom_vjp_call.name(), "custom_vjp_call");
    assert_eq!(while_operation.infer_output_types(&[ArrayType], &[]), Ok(vec![ArrayType]));
    assert_eq!(recompute.infer_output_types(&[ArrayType], &[]), Ok(vec![ArrayType]));

    assert_eq!(recompute, Linear::Recompute(RecomputeOperation { operation: ArrayOperation::from(DotOperation) }));
    assert_eq!(
        <&RecomputeOperation<ArrayOperation<Factor, BackendPayload>>>::try_from(&recompute),
        Ok(&RecomputeOperation { operation: ArrayOperation::from(DotOperation) }),
    );
    assert_eq!(
        <&WhileOperation<ArrayType, Factor, Linear>>::try_from(&while_operation),
        Ok(&WhileOperation { marker: PhantomData }),
    );
    assert_eq!(
        <&CustomVjpCallOperation<ArrayType, Factor, ArrayOperation<Factor, BackendPayload>, Factor>>::try_from(
            &custom_vjp_call
        ),
        Ok(&CustomVjpCallOperation { marker: PhantomData }),
    );
    assert_eq!(
        <&ZeroOperation<ArrayType>>::try_from(&while_operation),
        Err(TypeError::invalid("cannot project operation 'while' into a 'ZeroOperation<ArrayType>' payload")),
    );

    // `Backend(Backend)` is a bare generic payload, so its conversion is skipped automatically, while the
    // recompute wrapper and boxed payloads still expose conversions.
}

#[test]
fn test_transposable_operation_dispatches_to_payloads() {
    type Operation = LinearScalarOperation<ScalarFactor>;

    let operation = Operation::from(ZeroOperation { r#type: DataType });
    let mut context = TracingContext::<ScalarFactor, Operation> { marker: PhantomData };

    assert_eq!(operation.transpose(&mut context, &EmptyRegionDriver, &[], &[]).unwrap(), vec![transposed("zero")],);
}

#[test]
fn test_operation_generates_interpretation_forwarding() {
    let context = TestContext::<InterpretedScalarFactor> { marker: PhantomData };
    let operation = DataOperation::<ScalarFactor>::from(AddOperation);

    assert_eq!(
        operation.interpret(&context, &EmptyRegionDriver, &[InterpretedScalarFactor(1), InterpretedScalarFactor(2)],),
        Ok(vec![InterpretedScalarFactor(1), InterpretedScalarFactor(2)]),
    );
}

#[test]
fn test_operation_generates_direct_interpretation_dispatch() {
    type Operation = LinearScalarOperation<ScalarFactor>;

    let context = TestContext::<ScalarFactor> { marker: PhantomData };
    let operation = Operation::from(FactorOperation { factor: ScalarFactor(5), marker: PhantomData });

    assert_eq!(operation.interpret(&context, &EmptyRegionDriver, &[ScalarFactor(8)]), Ok(vec![ScalarFactor(8)]),);
}

/// Recursive payload whose partial-evaluation rule requires [`SpecialPartiallyEvaluatableValue`] on the flowing value
/// type, verifying that the generated per-payload predicate transports that requirement to the enum's use site.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PartialEvaluationRecursiveOperation<V, O> {
    marker: PhantomData<(V, O)>,
}

impl<V: Clone, O: Clone> Operation for PartialEvaluationRecursiveOperation<V, O> {
    type Type = ArrayType;

    fn name(&self) -> &'static str {
        "partial_evaluation_recursive"
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<W: Clone, O: Clone, C: Domain<Type = ArrayType>> InterpretableOperation<C>
    for PartialEvaluationRecursiveOperation<W, O>
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<W: Clone, O: Operation<Type = ArrayType>, C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C>
    for PartialEvaluationRecursiveOperation<W, O>
where
    C::Value: SpecialPartiallyEvaluatableValue,
    C::Operation: From<PartialEvaluationRecursiveOperation<W, O>>,
{
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
enum PartialEvaluationPayloadOperation<V: Value<Type = ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    Recursive(PartialEvaluationRecursiveOperation<V, Self>),
}

#[test]
fn test_operation_generates_partial_evaluation_dispatch() {
    fn assert_partially_evaluatable<C: Context, O: partial::PartiallyEvaluatableOperation<C>>() {}

    // The derive now forwards partial evaluation for every variant, so each enum must satisfy the per-operation
    // partial-evaluation trait at any known-side context pinned to its program-constant value type and to itself as
    // the residual operation family. This covers leaf payloads, the generic `Backend` payload, and the boxed
    // nested-program payloads.
    assert_partially_evaluatable::<TestContext<ScalarFactor, DataOperation<ScalarFactor>>, DataOperation<ScalarFactor>>(
    );
    assert_partially_evaluatable::<
        TestContext<ScalarFactor, LinearScalarOperation<ScalarFactor>>,
        LinearScalarOperation<ScalarFactor>,
    >();
    assert_partially_evaluatable::<TestContext<Factor, ArrayOperation<Factor>>, ArrayOperation<Factor>>();
    assert_partially_evaluatable::<
        TestContext<Factor, LinearArrayOperation<Factor, Factor>>,
        LinearArrayOperation<Factor, Factor>,
    >();
}

#[test]
fn test_operation_propagates_partial_evaluation_payload_bounds() {
    // The `Recursive` payload's partial-evaluation rule requires `SpecialPartiallyEvaluatableValue`. The generated
    // per-variant obligation transports that requirement without repeating it on the enum.
    use partial::PartiallyEvaluatableOperation as _;

    fn assert_partially_evaluatable<C: Context, O: partial::PartiallyEvaluatableOperation<C>>() {}
    assert_partially_evaluatable::<
        TestContext<Factor, PartialEvaluationPayloadOperation<Factor>>,
        PartialEvaluationPayloadOperation<Factor>,
    >();

    let context = TestContext::<Factor, PartialEvaluationPayloadOperation<Factor>> { marker: PhantomData };
    let context = partial::PartialEvaluationContext::new(context);
    let operation = PartialEvaluationPayloadOperation::<Factor>::from(ZeroOperation { r#type: ArrayType });
    let evaluation = operation.partially_evaluate(&context, &EmptyRegionDriver, &[]).unwrap();
    assert!(evaluation.is_empty());
}

/// Stand-in for `ryft_core::ArrayBatch`. A label suffices to observe payload dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ArrayBatch<V> {
    label: &'static str,
    marker: PhantomData<V>,
}

impl<V> ArrayBatch<V> {
    fn labeled(label: &'static str) -> Self {
        Self { label, marker: PhantomData }
    }
}

/// Stand-in for `ryft_core::BatchedOutputs`. These fixtures observe payload dispatch only, so the stand-in carries a
/// rule's batches without the real type's operation-local validation evidence.
struct BatchedOutputs<C: Context, P: BatchingPolicy<C>> {
    batches: Vec<P::Batch>,
    marker: PhantomData<fn() -> C>,
}

impl<C: Context, P: BatchingPolicy<C>> BatchedOutputs<C, P> {
    fn into_batches(self) -> Vec<P::Batch> {
        self.batches
    }
}

impl<C: Context, P: BatchingPolicy<C>> From<Vec<P::Batch>> for BatchedOutputs<C, P> {
    fn from(batches: Vec<P::Batch>) -> Self {
        Self { batches, marker: PhantomData }
    }
}

/// Stand-in for `ryft_core::BatchableOperation`. Every rule receives the active [`BatchingContext`] and its optional
/// instruction-scoped [`BatchingDriver`] while physical values remain owned by the parent context `C`.
trait BatchableOperation<C: Context, P: BatchingPolicy<C>>: Operation<Type = C::Type> {
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError>;
}

/// Stand-in for `ryft_core::EagerContext`. Mirrors the real context's `Context` membership so that a top-level
/// eager batch can be represented as `BatchingContext<EagerContext<...>, ArrayBatching>`.
struct EagerContext<V: Value, O: Operation<Type = V::Type>> {
    marker: PhantomData<(V, O)>,
}

impl<V: Value, O: Operation<Type = V::Type>> Domain for EagerContext<V, O> {
    type Type = V::Type;
    type Value = V;
    type Constant = V;
    type Operation = O;
}

impl<V: Value, O: Operation<Type = V::Type>> Context for EagerContext<V, O> {
    fn lift(&self, constant: Self::Constant) -> Result<Self::Value, ProgramError> {
        Ok(constant)
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Zero<V> for EagerContext<V, O> {}

impl<V: Value, O: Operation<Type = V::Type>, Stored: Clone> Constant<V, Stored> for EagerContext<V, O>
where
    V: From<Stored>,
{
    fn constant(&self, value: Stored) -> Result<V, ProgramError> {
        Ok(V::from(value))
    }
}

/// Stand-in for `ryft_core::BatchingContext`. Mirrors the real context's parent accessor and observable axis
/// metadata, which active rules (e.g., named-axis collectives) inspect.
struct BatchingContext<C, P> {
    parent: C,
    axis_name: Option<&'static str>,
    policy: PhantomData<P>,
}

impl<C, P> BatchingContext<C, P> {
    fn parent(&self) -> &C {
        &self.parent
    }

    fn axis_name(&self) -> Option<&'static str> {
        self.axis_name
    }
}

/// Stand-in for `ryft_core::BatchingDriver`.
trait BatchingDriver<C: Context, P: BatchingPolicy<C>> {}

impl<C: Context, P: BatchingPolicy<C>> BatchingDriver<C, P> for EmptyRegionDriver {}

/// Stand-in for `ryft_core::BatchingPolicy`.
trait BatchingPolicy<C: Context> {
    type Batch;
}

/// Stand-in for `ryft_core::ArrayBatchingPolicy`.
trait ArrayBatchingPolicy<C: Context<Type = ArrayType>> {}

/// Stand-in for `ryft_core::StaticArrayBatchingPolicy`.
#[derive(Copy, Clone, Debug)]
struct StaticArrayBatchingPolicy;

impl<C: Context<Type = ArrayType>> ArrayBatchingPolicy<C> for StaticArrayBatchingPolicy {}

/// Stand-in for `ryft_core::ArrayBatching`.
#[derive(Copy, Clone, Debug)]
struct ArrayBatching<M = StaticArrayBatchingPolicy>(PhantomData<fn() -> M>);

impl<C: Context<Type = ArrayType>, M: ArrayBatchingPolicy<C>> BatchingPolicy<C> for ArrayBatching<M> {
    type Batch = ArrayBatch<C::Value>;
}

/// Stand-in for `ryft_core::BatchAxis`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct BatchAxis(Option<usize>);

/// Stand-in for `ryft_core::ProgramBatchingOutputAxesPolicy`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ProgramBatchingOutputAxesPolicy {
    Natural,
}

impl<V, O> Program<V, O, Vec<V>, Vec<V>>
where
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType>,
{
    /// Stand-in for `ryft_core::Program::batched`.
    fn batched(
        &self,
        _axis_size: usize,
        input_batch_axes: &[BatchAxis],
        _output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<(Self, Vec<BatchAxis>), BatchingError> {
        Ok((self.clone(), input_batch_axes.to_vec()))
    }
}

/// Stand-in value capability required by payload batching rules, verifying per-variant predicate transport.
trait SpecialBatchValue {}

impl SpecialBatchValue for Factor {}

impl<C, Meta> SpecialBatchValue for Tracer<C, Meta> {}

/// Ordinary leaf rule: it neither needs the active frame nor any value capability, and its physical work runs
/// through the parent context (observed here through the parent-lifted constant in its output label).
impl<C: Context<Type = ArrayType>, M: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<M>>
    for ZeroOperation<ArrayType>
{
    fn batch<D: BatchingDriver<C, ArrayBatching<M>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<M>>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<M>>, BatchingError> {
        // Ordinary rules execute their lifted work through the parent context.
        let _ = context.parent();
        Ok(vec![ArrayBatch::labeled("zero")].into())
    }
}

impl<Constant: Clone, C: Context<Type = ArrayType>, M: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<M>>
    for ConstantOperation<ArrayType, Constant>
{
    fn batch<D: BatchingDriver<C, ArrayBatching<M>>>(
        &self,
        _context: &BatchingContext<C, ArrayBatching<M>>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<M>>, BatchingError> {
        Ok(vec![ArrayBatch::labeled("constant")].into())
    }
}

/// Batching rule requiring a value capability that the generated per-variant predicate transports to the owning
/// enum's use sites without the enum spelling it.
impl<C: Context<Type = ArrayType>, M: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<M>> for DotOperation
where
    C::Value: SpecialBatchValue,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<M>>>(
        &self,
        _context: &BatchingContext<C, ArrayBatching<M>>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<M>>, BatchingError> {
        Ok(vec![ArrayBatch::labeled("dot")].into())
    }
}

/// Stand-in for a named-axis collective: a rule whose semantics depend on the active frame's axis metadata, which
/// the fixed-context contract exposes to every rule without any variant-level marker.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CollectiveLikeOperation;

impl Operation for CollectiveLikeOperation {
    type Type = ArrayType;

    fn name(&self) -> &'static str {
        "collective_like"
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for CollectiveLikeOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C> for CollectiveLikeOperation where
    C::Operation: From<CollectiveLikeOperation>
{
}

impl<C: Context<Type = ArrayType>, M: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<M>>
    for CollectiveLikeOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatching<M>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<M>>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<M>>, BatchingError> {
        // The rule observes the active frame's axis metadata directly.
        Ok(vec![ArrayBatch::labeled(if context.axis_name().is_some() {
            "collective_like_named"
        } else {
            "collective_like_unnamed"
        })]
        .into())
    }
}

/// Stand-in recursive higher-order payload whose batching rule mirrors the leaf obligations the real control-flow
/// rules carry: an operation-shaped `From` conversion (discharged structurally from the closed enum), the parent
/// context's `Zero`, and value capabilities carried by its own batching implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BatchRecursiveOperation<V, O> {
    marker: PhantomData<(V, O)>,
}

impl<V: Clone, O: Clone> Operation for BatchRecursiveOperation<V, O> {
    type Type = ArrayType;

    fn name(&self) -> &'static str {
        "batch_recursive"
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        Ok(input_types.to_vec())
    }
}

impl<W: Clone, O: Clone, C: Domain<Type = ArrayType>> InterpretableOperation<C> for BatchRecursiveOperation<W, O> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(inputs.to_vec())
    }
}

impl<W: Clone, O: Operation<Type = ArrayType>, C: Context<Type = ArrayType>> partial::PartiallyEvaluatableOperation<C>
    for BatchRecursiveOperation<W, O>
where
    C::Operation: From<BatchRecursiveOperation<W, O>>,
{
}

impl<C, M> BatchableOperation<C, ArrayBatching<M>> for BatchRecursiveOperation<C::Constant, C::Operation>
where
    C: Context<Type = ArrayType> + Zero<C::Value>,
    C::Value: Concretizable<bool> + SpecialBatchValue,
    C::Operation: From<ZeroOperation<ArrayType>>,
    M: ArrayBatchingPolicy<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<M>>>(
        &self,
        _context: &BatchingContext<C, ArrayBatching<M>>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<M>>, BatchingError> {
        Ok(vec![ArrayBatch::labeled("batch_recursive")].into())
    }
}

#[derive(Clone, Debug, ryft::Operation)]
#[ryft(crate = "crate")]
#[ryft(dispatch(batching))]
enum BatchableArrayOperation<V: Value<Type = ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    Dot(DotOperation),
    Collective(CollectiveLikeOperation),
    Recursive(Box<BatchRecursiveOperation<V, Self>>),
}

#[test]
fn test_batchable_operation_dispatches_batching_to_payloads() {
    type Operation = BatchableArrayOperation<Factor>;
    type Staging = TestContext<Factor, Operation>;

    // Every arm receives the active batching context and flows the parent context's own value (`<Staging as
    // Domain>::Value`, here `Factor`).
    let context = BatchingContext::<Staging, ArrayBatching> {
        parent: TestContext { marker: PhantomData },
        axis_name: Some("batch"),
        policy: PhantomData,
    };

    let zero = Operation::from(ZeroOperation { r#type: ArrayType });
    assert_eq!(
        zero.batch(&context, &EmptyRegionDriver, &[]).unwrap().into_batches(),
        vec![ArrayBatch::labeled("zero")]
    );

    // The `Dot` rule requires `SpecialBatchValue` on the flowing value, transported to this use site by the
    // generated per-variant `BatchableOperation` predicate.
    let dot = Operation::from(DotOperation);
    assert_eq!(dot.batch(&context, &EmptyRegionDriver, &[]).unwrap().into_batches(), vec![ArrayBatch::labeled("dot")]);

    // The collective-like rule observes the active frame's axis metadata without any variant-level marker.
    let collective = Operation::from(CollectiveLikeOperation);
    assert_eq!(
        collective.batch(&context, &EmptyRegionDriver, &[]).unwrap().into_batches(),
        vec![ArrayBatch::labeled("collective_like_named")],
    );

    let recursive = Operation::from(BatchRecursiveOperation::<Factor, Operation> { marker: PhantomData });
    assert_eq!(
        recursive.batch(&context, &EmptyRegionDriver, &[]).unwrap().into_batches(),
        vec![ArrayBatch::labeled("batch_recursive")],
    );
}

#[test]
fn test_batchable_operation_dispatches_batching_over_eager_parents() {
    type Operation = BatchableArrayOperation<Factor>;

    // A top-level eager batch is represented by a `BatchingContext` over an eager parent, not by a separate eager
    // dispatch mechanism, and unnamed frames are observable to rules that inspect the axis metadata.
    let context = BatchingContext::<EagerContext<Factor, Operation>, ArrayBatching> {
        parent: EagerContext { marker: PhantomData },
        axis_name: None,
        policy: PhantomData,
    };

    let zero = Operation::from(ZeroOperation { r#type: ArrayType });
    assert_eq!(
        zero.batch(&context, &EmptyRegionDriver, &[]).unwrap().into_batches(),
        vec![ArrayBatch::labeled("zero")]
    );

    let collective = Operation::from(CollectiveLikeOperation);
    assert_eq!(
        collective.batch(&context, &EmptyRegionDriver, &[]).unwrap().into_batches(),
        vec![ArrayBatch::labeled("collective_like_unnamed")],
    );

    let recursive = Operation::from(BatchRecursiveOperation::<Factor, Operation> { marker: PhantomData });
    assert_eq!(
        recursive.batch(&context, &EmptyRegionDriver, &[]).unwrap().into_batches(),
        vec![ArrayBatch::labeled("batch_recursive")],
    );
}

#[derive(Clone, Debug, PartialEq, Eq, ryft::Operation)]
#[ryft(crate = "crate")]
#[ryft(dispatch(batching, differentiation, transposition))]
enum AllDispatcherOperation<V: Value<Type = ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    Constant(ConstantOperation<ArrayType, V>),
}

#[test]
fn test_operation_generates_all_selected_dispatchers() {
    type Operation = AllDispatcherOperation<Factor>;
    type Context = TestContext<Factor, Operation>;

    let operation = Operation::from(ZeroOperation { r#type: ArrayType });
    let context = Context { marker: PhantomData };
    let batching_context = BatchingContext::<_, ArrayBatching> {
        parent: Context { marker: PhantomData },
        axis_name: None,
        policy: PhantomData,
    };
    assert_eq!(
        operation.batch(&batching_context, &EmptyRegionDriver, &[]).unwrap().into_batches(),
        vec![ArrayBatch::labeled("zero")],
    );

    let differentiated = operation.jvp(&context, &EmptyRegionDriver, &[]).unwrap();
    assert_eq!(differentiated.len(), 1);
    assert_eq!(differentiated[0].label, "zero");

    let mut transposition_context = TracingContext::<Factor, Operation> { marker: PhantomData };
    assert_eq!(
        operation.transpose(&mut transposition_context, &EmptyRegionDriver, &[], &[]).unwrap(),
        vec![transposed("zero")],
    );
}

#[test]
fn test_errors() {
    let test_cases = trybuild::TestCases::new();
    test_cases.compile_fail("tests/operations/error_ambiguous_type.rs");
    test_cases.compile_fail("tests/operations/error_bad_variant.rs");
    test_cases.compile_fail("tests/operations/error_conflicting_variant_classes.rs");
    test_cases.compile_fail("tests/operations/error_duplicate_dispatch_attribute.rs");
    test_cases.compile_fail("tests/operations/error_duplicate_dispatcher.rs");
    test_cases.compile_fail("tests/operations/error_duplicate_variant_class.rs");
    test_cases.compile_fail("tests/operations/error_empty_dispatch.rs");
    test_cases.compile_fail("tests/operations/error_members_attribute.rs");
    test_cases.compile_fail("tests/operations/error_missing_type.rs");
    test_cases.compile_fail("tests/operations/error_missing_variant_member_type.rs");
    test_cases.compile_fail("tests/operations/error_misplaced_variant_class.rs");
    test_cases.compile_fail("tests/operations/error_mismatched_payload_type.rs");
    test_cases.compile_fail("tests/operations/error_multiple_operation_types.rs");
    test_cases.compile_fail("tests/operations/error_removed_structural_variant_class.rs");
    test_cases.compile_fail("tests/operations/error_type_attribute.rs");
    test_cases.compile_fail("tests/operations/error_undeclared_variant_member_type.rs");
    test_cases.compile_fail("tests/operations/error_unknown_dispatcher.rs");
    test_cases.compile_fail("tests/operations/error_unsupported_variant_class.rs");
}
