//! Downstream compile proof for the reference extension surface.
//!
//! Everything in this file is written from the position of a backend crate outside `ryft-core`: a reference universe
//! of its own, a [`ReferenceDischargePolicy`] selected through [`ReferenceDischargeableType`], per-operation
//! [`ReferenceDischargeableOperation`] rules, and the transform rules that carry the family through forward mode,
//! reverse mode, and batching, all reaching `ryft-core` only through its public API. Because an integration test is a
//! separate crate, the compiler itself enforces the property this file exists to establish, namely that a third-party
//! reference universe can be discharged and transformed without any private `ryft-core` item.
//!
//! The universe is deliberately non-accumulating at the discharge policy level, which makes it the standing proof of
//! the policy's per-access capability granularity: it implements [`ReferenceDischargePolicy`] and not
//! [`ReferenceAccumulationPolicy`](ryft_core::ReferenceAccumulationPolicy), and still discharges every program that
//! reads, writes, or swaps. Only its own `register.add_update` has no discharge arm, so a program containing it fails
//! to discharge at exactly that operation.
//!
//! The universe describes static halves and dynamically selected bits through folded access paths. Static halves
//! exercise disjoint-path analysis; bit accesses carry their index as an ordinary trailing input. Discharge closes
//! paths over destination values and binds the family's bit extraction/insertion operations. Forward and reverse mode
//! retain the same paths and primal index bindings on tangent and cotangent accesses. Lazy eager views keep the root
//! and bindings in [`ReferenceView`], without creating another value or reference identity.
//!
//! The transform legs are reached through the public entry points ([`differentiate_at`] for `jvp`, `vjp`, and
//! `value_and_gradient`, and [`batch`]) over a live register reference. The generic reference primitives
//! ([`ReferenceNewOperation`] and its siblings) are wrapped by the family and interpret eagerly through the value-level
//! capabilities implemented on [`RegisterValue`], and their generic differentiation, transposition, and batching rules
//! apply at the eager context and at the staged contexts that transforms instantiate. The family selects allocation,
//! accumulation, and freezing operations through [`ReferenceNewOperation`],
//! [`ReferenceAddUpdateOperation`], and [`ReferenceFreezeOperation`] over its register referent family.
//! Generic transposition can therefore allocate cotangent references and use [`ReferenceAddUpdate`] on core-owned
//! tracers without a downstream tracer implementation. `register.add_update` retains family-owned addition semantics;
//! the other reference primitives reuse their generic transform rules. The gradient convenience boundary also uses
//! allocation and freezing to manage internal cotangent references, returning ordinary register values in the original
//! parameter structure.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt::Display;

use indoc::indoc;
use pretty_assertions::assert_eq;

use ryft_core::{
    AddOperation, Array, ArrayIrType, ArrayIrValue, ArrayType, AtomId, BatchAxis, BatchAxisSpecification,
    BatchableOperation, BatchableReferenceTransform, BatchableType, BatchedOutputs, BatchingContext, BatchingDriver,
    BatchingEntrypointPolicy, BatchingError, BatchingPolicy, BoundReferenceTransform, BoundaryPreservingBatchedProgram,
    BroadcastOperation, CompareOperation, ConstantOperation, Context, ConvertElementTypeOperation,
    CotangentAccumulator, CotangentDestination, CotangentDestinationKind, CotangentSeed, DataType,
    DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationPolicy, DivOperation, Domain, EagerContext, EffectClass, EffectClasses,
    Effects, ExpOperation, ExternalReferenceBinding, InputRegionProvenance, InstructionId, InterpretableOperation,
    InterpretationDriver, MaybeZero, MemberDifferentiableOperation, MemberTransposableOperation, MulOperation,
    NegOperation, NoIdentity, NoReferenceTransform, OneLikeOperation, OneOperation, Operation, OperationFormatter,
    OperationProvider, OutputRegionProvenance, ParallelVaryOperation, Parameter, PartialValue,
    PartiallyEvaluatableOperation, Placeholder, Program, ProgramBatchingOutputAxesPolicy, ProgramBuilder, ProgramError,
    ProjectedContext, RecursiveBatchingPolicy, RecursiveReferenceDischargeDriver, ReduceOperation, Reference,
    ReferenceAccessDescriptor, ReferenceAccessMode, ReferenceAccessOperation, ReferenceAddUpdate,
    ReferenceAddUpdateOperation, ReferenceBoundary, ReferenceBoundaryError, ReferenceDischargeContext,
    ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeRegionBoundary,
    ReferenceDischargeRegionBoundaryInsertion, ReferenceDischargeResult, ReferenceDischargeTarget,
    ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceDischargeableType, ReferenceEffect,
    ReferenceFreeze, ReferenceFreezeOperation, ReferenceId, ReferenceMemberType, ReferenceNew, ReferenceNewOperation,
    ReferenceRead, ReferenceReadOperation, ReferenceSource, ReferenceSwap, ReferenceSwapOperation, ReferenceTransform,
    ReferenceTransformPath, ReferenceType, ReferenceView, ReferenceViewOverlap, ReferenceWrite,
    ReferenceWriteOperation, RegionId, RegionInterface, RegionRef, RegionSlot, ReshapeOperation, ReshardOperation,
    ResidualZeroProvider, SubOperation, Trace, Tracer, TracingContext, TransposableOperation, TransposeOperation,
    TranspositionContext, TranspositionDriver, Type, TypeError, Typed, Value, ValueId, ValueProjection, Zero,
    ZeroLikeOperation, ZeroOperation, batch, check_count, differentiate_at, discharge_reference_free_operation,
    infer_reference_view_type, jvp_projected_operation, transpose_projected_operation, validate_reference_boundary,
};

/// Destination universe of the downstream programs: the eager context over the register family, which is what a
/// concrete backend value family looks like from outside `ryft-core` and the execution domain every register value
/// names.
type RegisterDestination = EagerContext<RegisterValue, RegisterOperation>;

/// Discharge context over the downstream destination universe.
type RegisterDischargeContext = ReferenceDischargeContext<RegisterDestination, RegisterReferenceDischarge>;

/// Carrier flowing through downstream discharge.
type RegisterDischargeValue = ReferenceDischargeValue<RegisterDestination, RegisterReferenceDischarge>;

/// Staged register value inside a program under construction.
type RegisterTracer = Tracer<TracingContext<RegisterValue, RegisterOperation>>;

/// Referent type of the downstream universe: one 64-bit integer register.
#[derive(Clone, Debug, PartialEq)]
struct RegisterType;

impl Display for RegisterType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("register")
    }
}

impl Parameter for RegisterType {}

impl Type for RegisterType {
    type Identity = NoIdentity;
    type Refinements = ();

    fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }

    fn is_refined_by(&self, other: &Self) -> bool {
        self == other
    }

    fn is_scalar(&self) -> bool {
        true
    }

    fn is_complex(&self) -> bool {
        false
    }
}

/// Type universe of the downstream programs.
#[derive(Clone, Debug, PartialEq)]
enum RegisterIrType {
    Register(RegisterType),
    Reference(ReferenceType<RegisterType>),
}

impl Display for RegisterIrType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Register(r#type) => Display::fmt(r#type, formatter),
            Self::Reference(r#type) => Display::fmt(r#type, formatter),
        }
    }
}

impl Parameter for RegisterIrType {}

impl From<RegisterType> for RegisterIrType {
    fn from(r#type: RegisterType) -> Self {
        Self::Register(r#type)
    }
}

impl From<ReferenceType<RegisterType>> for RegisterIrType {
    fn from(r#type: ReferenceType<RegisterType>) -> Self {
        Self::Reference(r#type)
    }
}

impl<'t> TryFrom<&'t RegisterIrType> for &'t RegisterType {
    type Error = TypeError;

    fn try_from(r#type: &'t RegisterIrType) -> Result<Self, Self::Error> {
        match r#type {
            RegisterIrType::Register(r#type) => Ok(r#type),
            RegisterIrType::Reference(_) => Err(TypeError::invalid("expected register type but got reference type")),
        }
    }
}

impl<'t> TryFrom<&'t RegisterIrType> for &'t ReferenceType<RegisterType> {
    type Error = TypeError;

    fn try_from(r#type: &'t RegisterIrType) -> Result<Self, Self::Error> {
        match r#type {
            RegisterIrType::Reference(r#type) => Ok(r#type),
            RegisterIrType::Register(_) => Err(TypeError::invalid("expected reference type but got register type")),
        }
    }
}

impl Type for RegisterIrType {
    type Identity = NoIdentity;
    type Refinements = ();

    fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }

    fn is_refined_by(&self, other: &Self) -> bool {
        self == other
    }

    fn is_scalar(&self) -> bool {
        matches!(self, Self::Register(_))
    }

    fn is_complex(&self) -> bool {
        false
    }

    fn is_reference(&self) -> bool {
        matches!(self, Self::Reference(_))
    }
}

impl ReferenceMemberType for RegisterIrType {
    type Referent = RegisterType;

    fn referent(&self) -> Option<&RegisterType> {
        match self {
            Self::Register(_) => None,
            Self::Reference(r#type) => Some(r#type.referent()),
        }
    }

    fn as_referent(&self) -> Option<&RegisterType> {
        match self {
            Self::Register(r#type) => Some(r#type),
            Self::Reference(_) => None,
        }
    }
}

// A register is its own tangent and cotangent, and a register reference's tangent is a register reference: nothing in
// the universe is zero-space, so every leaf keeps a boundary slot under every transform.
impl DifferentiableType for RegisterIrType {
    fn is_zero_space(&self) -> bool {
        false
    }

    fn tangent(&self) -> Result<Self, DifferentiationError> {
        Ok(self.clone())
    }

    fn cotangent(&self) -> Result<Self, DifferentiationError> {
        Ok(self.clone())
    }
}

/// Value universe of the downstream programs.
#[derive(Clone, Debug, PartialEq)]
enum RegisterValue {
    /// One 64-bit integer register.
    Register(i64),

    /// Live handle to a complete register allocation.
    Reference(Reference<RegisterValue>),
}

impl RegisterValue {
    /// Returns the register this value holds, rejecting a reference.
    fn register(&self) -> Result<i64, ProgramError> {
        match self {
            Self::Register(value) => Ok(*value),
            Self::Reference(_) => Err(TypeError::invalid("expected a register value but got a reference").into()),
        }
    }

    /// Returns the live complete-register reference this value holds, rejecting a register value.
    fn reference(&self) -> Result<&Reference<RegisterValue>, ProgramError> {
        match self {
            Self::Reference(reference) => Ok(reference),
            Self::Register(_) => Err(TypeError::invalid("expected a register reference but got a register").into()),
        }
    }
}

/// Validates `index` as a bit position of a 64-bit register.
fn bit_index(index: i64) -> Result<u32, ProgramError> {
    u32::try_from(index).ok().filter(|index| *index < 64).ok_or_else(|| ProgramError::InvalidArgument {
        message: format!("bit index {index} is out of range for a 64-bit register"),
    })
}

/// Returns bit `index` of `register` as a register holding 0 or 1.
fn extract_bit(register: i64, index: i64) -> Result<i64, ProgramError> {
    Ok((register >> bit_index(index)?) & 1)
}

/// Returns `register` with bit `index` replaced by `bit`, which must be a register holding 0 or 1.
fn insert_bit(register: i64, bit: i64, index: i64) -> Result<i64, ProgramError> {
    let index = bit_index(index)?;
    if bit != 0 && bit != 1 {
        return Err(ProgramError::InvalidArgument {
            message: format!("a register bit holds 0 or 1 but {bit} was stored into one"),
        });
    }
    Ok((register & !(1 << index)) | (bit << index))
}

impl Display for RegisterValue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Register(value) => Display::fmt(value, formatter),
            Self::Reference(reference) => Display::fmt(reference, formatter),
        }
    }
}

impl Parameter for RegisterValue {}

impl Typed for RegisterValue {
    type Type = RegisterIrType;

    fn r#type(&self) -> Cow<'_, RegisterIrType> {
        Cow::Owned(match self {
            Self::Register(_) => RegisterIrType::Register(RegisterType),
            Self::Reference(_) => RegisterIrType::Reference(ReferenceType::new(RegisterType)),
        })
    }
}

impl Value for RegisterValue {
    type DispatchDomain = RegisterDestination;
    type ExecutionDomain = RegisterDestination;

    fn dispatch_domain(&self) -> RegisterDestination {
        EagerContext::new()
    }

    fn execution_domain(&self) -> RegisterDestination {
        EagerContext::new()
    }

    fn reference_id(&self) -> Option<ReferenceId> {
        match self {
            Self::Register(_) => None,
            Self::Reference(reference) => Some(reference.id()),
        }
    }
}

// The eager reference capabilities of the register universe. The generic reference primitives interpret through these
// at the eager context, so the family's interpretation delegates to the primitives for every access it wraps. A bit
// handle accesses its root through the root handle: a read extracts the bit, a write replaces it and preserves the
// other bits, and a swap or additive update is a read followed by a write, which is the eager form of the discharge
// policy's default.
impl ReferenceNew for RegisterValue {
    fn reference_new(&self) -> Result<Self, ProgramError> {
        self.register()?;
        Ok(Self::Reference(Reference::new(self.clone()).map_err(ProgramError::custom)?))
    }
}

impl ReferenceRead<RegisterTransform> for RegisterValue {
    fn read_through(&self, transforms: &[RegisterTransform], bindings: &[Self]) -> Result<Self, ProgramError> {
        if !transforms.is_empty() || !bindings.is_empty() {
            let path = ReferenceTransformPath::from_transforms(transforms, bindings)?;
            return RegisterReferenceDischarge::read(&RegisterDestination::new(), &self.read()?, &path);
        }
        self.reference()?.read().map_err(ProgramError::custom)
    }
}

impl ReferenceWrite<RegisterTransform> for RegisterValue {
    fn write_through(
        &self,
        replacement: &Self,
        transforms: &[RegisterTransform],
        bindings: &[Self],
    ) -> Result<(), ProgramError> {
        if !transforms.is_empty() || !bindings.is_empty() {
            let path = ReferenceTransformPath::from_transforms(transforms, bindings)?;
            let updated = RegisterReferenceDischarge::write(
                &RegisterDestination::new(),
                &self.read()?,
                replacement.clone(),
                &path,
            )?;
            return self.write(&updated);
        }
        replacement.register()?;
        self.reference()?.write(replacement.clone()).map_err(ProgramError::custom)
    }
}

impl ReferenceSwap<RegisterTransform> for RegisterValue {
    fn swap_through(
        &self,
        replacement: &Self,
        transforms: &[RegisterTransform],
        bindings: &[Self],
    ) -> Result<Self, ProgramError> {
        if !transforms.is_empty() || !bindings.is_empty() {
            let previous = self.read_through(transforms, bindings)?;
            self.write_through(replacement, transforms, bindings)?;
            return Ok(previous);
        }
        replacement.register()?;
        self.reference()?.swap(replacement.clone()).map_err(ProgramError::custom)
    }
}

impl ReferenceFreeze for RegisterValue {
    fn freeze(self) -> Result<Self, ProgramError> {
        self.reference()?.freeze().map_err(ProgramError::custom)
    }
}

// The eager context materializes register zeros; a reference has no zero, exactly as in the array universe.
impl Zero<RegisterValue> for RegisterDestination {
    fn zero(&self, r#type: &RegisterIrType) -> Result<RegisterValue, ProgramError> {
        match r#type {
            RegisterIrType::Register(_) => Ok(RegisterValue::Register(0)),
            RegisterIrType::Reference(r#type) => {
                Err(TypeError::invalid(format!("cannot materialize a zero for reference type `{type}`")).into())
            }
        }
    }
}

/// Reference discharge policy of the downstream universe.
#[derive(Copy, Clone, Debug)]
struct RegisterReferenceDischarge;

impl ReferenceDischargeableType for RegisterIrType {
    type Policy = RegisterReferenceDischarge;
}

/// Returns the destination value bound to one bit transform, rejecting the static half descriptions reserved for
/// analysis.
fn bit_coordinate<V>(bound_transform: &BoundReferenceTransform<RegisterTransform, V>) -> Result<&V, ProgramError> {
    match bound_transform.transform() {
        RegisterTransform::Bit => {
            check_count!("input", bound_transform.bindings(), 1, ProgramError);
            Ok(&bound_transform.bindings()[0])
        }
        RegisterTransform::Half(_) => Err(ProgramError::UnsupportedOperation {
            message: "static half descriptions support analysis only".to_string(),
        }),
    }
}

/// Returns `current` with the bit that `bound_transforms` select replaced by `replacement`, binding the family's bit
/// operations on `context`. Nested bit transforms recurse: the bit each non-final transform selects is extracted,
/// rewritten through the remaining transforms, and inserted back.
fn insert_bits<C: Context<Type = RegisterIrType, Operation: From<RegisterOperation>>>(
    context: &C,
    current: C::Value,
    replacement: C::Value,
    bound_transforms: &[BoundReferenceTransform<RegisterTransform, C::Value>],
) -> Result<C::Value, ProgramError> {
    let Some((bound_transform, rest)) = bound_transforms.split_first() else {
        return Ok(replacement);
    };
    let coordinate = bit_coordinate(bound_transform)?.clone();
    let selected = if rest.is_empty() {
        replacement
    } else {
        let selected =
            bind_register_output(context, RegisterOperation::BitExtract, &[current.clone(), coordinate.clone()])?;
        insert_bits(context, selected, replacement, rest)?
    };
    bind_register_output(context, RegisterOperation::BitInsert, &[current, selected, coordinate])
}

// The policy is generic over the destination context rather than pinned to `RegisterValue`, which is what lets one
// implementation serve an eager destination and a staging destination alike. Its alias is the transform path closed
// over destination values, so a bit transform carries the destination value of its index and the policy reads and
// writes through it by binding the family's bit operations on the destination, with no environment lookup. The policy
// declines accumulation entirely by not implementing `ReferenceAccumulationPolicy`.
impl<C: Context<Type = RegisterIrType, Operation: From<RegisterOperation>>> ReferenceDischargePolicy<C>
    for RegisterReferenceDischarge
{
    type Referent = RegisterType;
    type Transform = RegisterTransform;
    type Alias = ReferenceTransformPath<RegisterTransform, C::Value>;

    fn apply_transforms(
        _context: &C,
        alias: &Self::Alias,
        transforms: &[RegisterTransform],
        bindings: &[C::Value],
    ) -> Result<Self::Alias, ProgramError> {
        let suffix = ReferenceTransformPath::from_transforms(transforms, bindings)?;
        let mut composed = alias.clone();
        for bound_transform in suffix.bound_transforms() {
            composed = composed.with_bound_transform(*bound_transform.transform(), bound_transform.bindings().to_vec());
        }
        Ok(composed)
    }

    fn storage_alias(_referent: &RegisterType) -> ReferenceTransformPath<RegisterTransform, C::Value> {
        ReferenceTransformPath::root()
    }

    fn read(
        context: &C,
        current: &C::Value,
        alias: &ReferenceTransformPath<RegisterTransform, C::Value>,
    ) -> Result<C::Value, ProgramError> {
        let mut selected = current.clone();
        for bound_transform in alias.bound_transforms() {
            let coordinate = bit_coordinate(bound_transform)?.clone();
            selected = bind_register_output(context, RegisterOperation::BitExtract, &[selected, coordinate])?;
        }
        Ok(selected)
    }

    fn write(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &ReferenceTransformPath<RegisterTransform, C::Value>,
    ) -> Result<C::Value, ProgramError> {
        insert_bits(context, current.clone(), replacement, alias.bound_transforms())
    }
}

/// Operation family of the downstream universe. The reference accesses wrap the generic `ryft-core` primitives, so
/// their type inference, reference semantics, effects, and eager interpretation are the canonical ones; the additive
/// update is the family's own because the generic primitive requires an [`Operation`] implementation for
/// `AddOperation<RegisterType>` that only `ryft-core` can provide. Static half and dynamic bit descriptions are the
/// family's static and dynamic transforms (refer to the module documentation), and `register.bit_extract` and
/// `register.bit_insert` are the value-level bit operations through which the discharge policy reads and writes a bit
/// view.
#[derive(Clone, Debug)]
enum RegisterOperation {
    Negate,
    Add(AddOperation<RegisterIrType>),
    Zero(ZeroOperation<RegisterIrType>),
    One,
    ReferenceNew(ReferenceNewOperation<RegisterType, RegisterIrType>),
    Read(ReferenceReadOperation<RegisterType, RegisterIrType, RegisterTransform>),
    Write(ReferenceWriteOperation<RegisterType, RegisterIrType, RegisterTransform>),
    Swap(ReferenceSwapOperation<RegisterType, RegisterIrType, RegisterTransform>),
    AddUpdate(Vec<RegisterTransform>),
    Freeze(ReferenceFreezeOperation<RegisterType, RegisterIrType>),
    Call,
    BitExtract,
    BitInsert,
}

impl From<AddOperation<RegisterIrType>> for RegisterOperation {
    fn from(operation: AddOperation<RegisterIrType>) -> Self {
        Self::Add(operation)
    }
}

impl From<ZeroOperation<RegisterIrType>> for RegisterOperation {
    fn from(operation: ZeroOperation<RegisterIrType>) -> Self {
        Self::Zero(operation)
    }
}

// Register zeros need no runtime geometry, so this universe opts into the input-free residual defaults.
impl ResidualZeroProvider<RegisterIrType> for RegisterOperation {}

impl OperationProvider<RegisterIrType, OneOperation<RegisterIrType>> for RegisterOperation {
    type Operation = Self;

    fn provide(request: OneOperation<RegisterIrType>, input_types: &[&RegisterIrType]) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 0, ProgramError);
        <&RegisterType>::try_from(request.r#type())?;
        Ok(Self::One)
    }
}

impl Display for RegisterOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl From<ReferenceNewOperation<RegisterType, RegisterIrType>> for RegisterOperation {
    fn from(operation: ReferenceNewOperation<RegisterType, RegisterIrType>) -> Self {
        Self::ReferenceNew(operation)
    }
}

impl From<ReferenceReadOperation<RegisterType, RegisterIrType, RegisterTransform>> for RegisterOperation {
    fn from(operation: ReferenceReadOperation<RegisterType, RegisterIrType, RegisterTransform>) -> Self {
        Self::Read(operation)
    }
}

impl From<ReferenceWriteOperation<RegisterType, RegisterIrType, RegisterTransform>> for RegisterOperation {
    fn from(operation: ReferenceWriteOperation<RegisterType, RegisterIrType, RegisterTransform>) -> Self {
        Self::Write(operation)
    }
}

impl From<ReferenceSwapOperation<RegisterType, RegisterIrType, RegisterTransform>> for RegisterOperation {
    fn from(operation: ReferenceSwapOperation<RegisterType, RegisterIrType, RegisterTransform>) -> Self {
        Self::Swap(operation)
    }
}

impl From<ReferenceFreezeOperation<RegisterType, RegisterIrType>> for RegisterOperation {
    fn from(operation: ReferenceFreezeOperation<RegisterType, RegisterIrType>) -> Self {
        Self::Freeze(operation)
    }
}

impl OperationProvider<RegisterIrType, ReferenceNewOperation<RegisterType, RegisterIrType>> for RegisterOperation {
    type Operation = Self;

    fn provide(
        _request: ReferenceNewOperation<RegisterType, RegisterIrType>,
        input_types: &[&RegisterIrType],
    ) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        Ok(Self::ReferenceNew(ReferenceNewOperation::new()))
    }
}

impl OperationProvider<RegisterIrType, ReferenceFreezeOperation<RegisterType, RegisterIrType>> for RegisterOperation {
    type Operation = Self;

    fn provide(
        _request: ReferenceFreezeOperation<RegisterType, RegisterIrType>,
        input_types: &[&RegisterIrType],
    ) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        Ok(Self::Freeze(ReferenceFreezeOperation::new()))
    }
}

impl OperationProvider<RegisterIrType, ReferenceAddUpdateOperation<RegisterType, RegisterIrType, RegisterTransform>>
    for RegisterOperation
{
    type Operation = Self;

    fn provide(
        request: ReferenceAddUpdateOperation<RegisterType, RegisterIrType, RegisterTransform>,
        input_types: &[&RegisterIrType],
    ) -> Result<Self, ProgramError> {
        check_count!(
            "input",
            input_types,
            2 + request.transforms().iter().map(ReferenceTransform::binding_count).sum::<usize>(),
            ProgramError
        );
        let operation = Self::AddUpdate(request.transforms().to_vec());
        operation.infer_output_types(&input_types.iter().map(|r#type| (*r#type).clone()).collect::<Vec<_>>(), &[])?;
        Ok(operation)
    }
}

impl Operation for RegisterOperation {
    type Type = RegisterIrType;

    fn name(&self) -> &'static str {
        match self {
            Self::Negate => "register.negate",
            Self::Add(_) => "register.add",
            Self::Zero(_) => "register.zero",
            Self::One => "register.one",
            Self::ReferenceNew(operation) => operation.name(),
            Self::Read(operation) => operation.name(),
            Self::Write(operation) => operation.name(),
            Self::Swap(operation) => operation.name(),
            Self::AddUpdate(_) => "register.add_update",
            Self::Freeze(operation) => operation.name(),
            Self::Call => "register.call",
            Self::BitExtract => "register.bit_extract",
            Self::BitInsert => "register.bit_insert",
        }
    }

    fn region_slots(&self) -> &'static [RegionSlot] {
        match self {
            Self::Call => const { &[RegionSlot::computation("callee")] },
            _ => &[],
        }
    }

    fn input_region_provenance(&self, _region_index: usize, input_index: usize) -> InputRegionProvenance {
        if matches!(self, Self::Call) {
            InputRegionProvenance::Input { index: input_index }
        } else {
            InputRegionProvenance::None
        }
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        match self {
            Self::Call => vec![OutputRegionProvenance { region_index: 0, output_index }],
            _ => Vec::new(),
        }
    }

    fn allows_reference_access_through_region_input(&self, region_index: usize, mode: ReferenceAccessMode) -> bool {
        matches!(self, Self::Call) && region_index == 0 && !mode.is_consuming()
    }

    fn infer_output_types(
        &self,
        input_types: &[RegisterIrType],
        region_interfaces: &[RegionInterface<RegisterIrType>],
    ) -> Result<Vec<RegisterIrType>, TypeError> {
        let referent = || match input_types.first() {
            Some(RegisterIrType::Reference(reference)) => Ok(reference.referent().clone()),
            _ => Err(TypeError::invalid(format!("`{}` expects a reference operand", self.name()))),
        };
        match self {
            Self::Negate => {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![RegisterIrType::Register(RegisterType)])
            }
            Self::Add(_) => {
                check_count!("input", input_types, 2, TypeError);
                for r#type in input_types {
                    <&RegisterType>::try_from(r#type)?;
                }
                Ok(vec![RegisterIrType::Register(RegisterType)])
            }
            Self::Zero(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::One => {
                check_count!("input", input_types, 0, TypeError);
                check_count!("region", region_interfaces, 0, TypeError);
                Ok(vec![RegisterIrType::Register(RegisterType)])
            }
            Self::ReferenceNew(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Read(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Write(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Swap(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::AddUpdate(transforms) => {
                let bindings = transforms.iter().map(ReferenceTransform::binding_count).sum::<usize>();
                check_count!("input", input_types, 2 + bindings, TypeError);
                infer_reference_view_type(
                    &referent()?,
                    transforms,
                    &input_types[2..].iter().collect::<Vec<_>>(),
                    ReferenceAccessMode::Accumulate,
                )?;
                <&RegisterType>::try_from(&input_types[1])?;
                Ok(Vec::new())
            }
            Self::Freeze(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::BitExtract | Self::BitInsert => {
                check_count!("input", input_types, if matches!(self, Self::BitExtract) { 2 } else { 3 }, TypeError);
                for r#type in input_types {
                    <&RegisterType>::try_from(r#type)?;
                }
                Ok(vec![RegisterIrType::Register(RegisterType)])
            }
            Self::Call => match region_interfaces.first() {
                Some(interface) => Ok(interface.output_types().to_vec()),
                None => Err(TypeError::invalid("`register.call` expects one callee region")),
            },
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Read(operation) => operation.render(formatter, indentation),
            Self::Write(operation) => operation.render(formatter, indentation),
            Self::Swap(operation) => operation.render(formatter, indentation),
            Self::AddUpdate(transforms) if !transforms.is_empty() => {
                OperationFormatter::new(formatter, indentation, self.name())?
                    .bracketed(|operation| operation.field("transforms", format_args!("{transforms:?}")))
            }
            _ => OperationFormatter::new(formatter, indentation, self.name()).map(|_| ()),
        }
    }

    fn effects(&self) -> Cow<'_, Effects> {
        match self {
            Self::Negate | Self::Add(_) | Self::Zero(_) | Self::One | Self::BitExtract | Self::BitInsert => {
                Cow::Borrowed(Effects::empty())
            }
            Self::ReferenceNew(operation) => operation.effects(),
            Self::Read(operation) => operation.effects(),
            Self::Write(operation) => operation.effects(),
            Self::Swap(operation) => operation.effects(),
            Self::Freeze(operation) => operation.effects(),
            Self::AddUpdate(_) => Cow::Owned(
                Effects::new(
                    EffectClasses::NONE,
                    vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Accumulate }],
                )
                .unwrap(),
            ),
            // A structured operation declares no operation-local reference effects (its accesses are summarized
            // transitively from the region closure it attaches) but carries opaque ordered state of its own.
            Self::Call => Cow::Owned(Effects::explicit(EffectClasses::single(EffectClass::OrderedState))),
        }
    }
}

impl ReferenceAccessOperation for RegisterOperation {
    type Transform = RegisterTransform;

    fn base_input_count(&self) -> usize {
        match self {
            Self::Read(operation) => operation.base_input_count(),
            Self::Write(operation) => operation.base_input_count(),
            Self::Swap(operation) => operation.base_input_count(),
            Self::AddUpdate(_) => 2,
            Self::Freeze(_) => 1,
            _ => 0,
        }
    }

    fn reference_access_descriptor(
        &self,
        input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, RegisterTransform>> {
        match self {
            Self::Read(operation) => operation.reference_access_descriptor(input_index),
            Self::Write(operation) => operation.reference_access_descriptor(input_index),
            Self::Swap(operation) => operation.reference_access_descriptor(input_index),
            Self::AddUpdate(transforms) if input_index == 0 => Some(ReferenceAccessDescriptor::new(
                transforms,
                2..2 + transforms.iter().map(ReferenceTransform::binding_count).sum::<usize>(),
            )),
            Self::Freeze(_) if input_index == 0 => Some(ReferenceAccessDescriptor::new(&[], 1..1)),
            _ => None,
        }
    }

    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<RegisterTransform>,
    ) -> Result<Self, ProgramError> {
        match self {
            Self::Read(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::Read)
            }
            Self::Write(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::Write)
            }
            Self::Swap(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::Swap)
            }
            Self::AddUpdate(_) if input_index == 0 => Ok(Self::AddUpdate(transforms)),
            _ if self.reference_access_descriptor(input_index).is_some() && transforms.is_empty() => Ok(self.clone()),
            _ => Err(ProgramError::UnsupportedOperation {
                message: "register operation does not support replacing these access transforms".to_string(),
            }),
        }
    }
}

impl<C> ReferenceDischargeableOperation<C, RegisterReferenceDischarge> for RegisterOperation
where
    C: Context<Type = RegisterIrType, Operation: From<RegisterOperation>>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, RegisterReferenceDischarge>>(
        &self,
        context: &ReferenceDischargeContext<C, RegisterReferenceDischarge>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, RegisterReferenceDischarge>],
    ) -> Result<Vec<ReferenceDischargeValue<C, RegisterReferenceDischarge>>, ProgramError> {
        // Access arms see only discharged references: the dispatch path replays accesses to preserved references verbatim
        // before any rule runs, so only the allocation arm still distinguishes selected from preserved.
        match self {
            Self::Negate | Self::Add(_) | Self::Zero(_) | Self::One | Self::BitExtract | Self::BitInsert => {
                discharge_reference_free_operation(self, context, driver, inputs)
            }
            Self::ReferenceNew(_) => {
                check_count!("input", inputs, 1, ProgramError);
                let initial = inputs[0].try_as_value("an initial state")?.clone();
                if context.selects_internal(driver.source_instruction_id(), 0) {
                    return Ok(vec![context.bind_discharged(ReferenceType::new(RegisterType), initial)?.into()]);
                }
                let mut outputs = context.parent().bind(self.clone(), Vec::new(), std::slice::from_ref(&initial))?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![context.bind_preserved(ReferenceType::new(RegisterType), outputs.remove(0))?.into()])
            }
            Self::Read(operation) => operation.discharge_references(context, driver, inputs),
            Self::Write(operation) => operation.discharge_references(context, driver, inputs),
            Self::Swap(operation) => operation.discharge_references(context, driver, inputs),
            Self::Freeze(_) => {
                check_count!("input", inputs, 1, ProgramError);
                let reference = inputs[0].try_as_reference("a reference to freeze")?;
                Ok(vec![ReferenceDischargeValue::Value(context.consume(reference)?)])
            }
            // This policy deliberately does not implement accumulation.
            Self::AddUpdate(_) => Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` has no discharge rule in the register universe", self.name()),
            }),
            // The hand-rolled structured widening a backend-owned region operation performs: summarize the closure,
            // widen the boundary with the reached state, rebuild the region in isolation, validate the result
            // against the summary's predictions, and merge every published successor state back. This is the same
            // shape a future kernel-call rule needs, expressed purely through the public discharge surface.
            Self::Call => {
                let region = driver.region(0)?;
                check_count!("input", region.input_ids(), inputs.len(), ProgramError);
                let mut declared = Vec::with_capacity(inputs.len());
                for input in inputs {
                    declared.push(context.boundary_allocation(input)?);
                }
                let summary = context.region_summary(self, 0, region, declared.as_slice())?;
                if summary.output_allocations().iter().any(Option::is_some) {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{}` does not return references from its callee",
                        self.name(),
                    )));
                }
                let operand_allocations = declared.iter().copied().flatten().collect::<BTreeSet<_>>();
                let widening = context.boundary_widening(&summary, &operand_allocations)?;
                let entering = widening.entering().to_vec();
                let source_output_count = region.output_ids().len();

                let result = driver.rebuild_region(
                    context,
                    region,
                    &ReferenceDischargeRegionBoundary::new(
                        self,
                        0,
                        declared,
                        ReferenceDischargeRegionBoundaryInsertion::new(entering.clone(), inputs.len()),
                        [ReferenceDischargeRegionBoundaryInsertion::new(
                            widening.published().to_vec(),
                            source_output_count,
                        )
                        .into()],
                    ),
                )?;
                result.validate_predicted_mutations(widening.published(), self.name())?;
                result.validate_predicted_output_allocations(summary.output_allocations(), self.name())?;

                let mut operands = Vec::with_capacity(inputs.len() + entering.len());
                for input in inputs {
                    operands.push(context.boundary_value(input)?);
                }
                for allocation in &entering {
                    operands.push(context.discharged_state(*allocation)?);
                }
                let outputs = context.parent().bind(self.clone(), vec![result.into_program()], operands.as_slice())?;
                check_count!("output", outputs, source_output_count + widening.published().len(), ProgramError);

                let mut results = Vec::with_capacity(source_output_count);
                for (position, output) in outputs.into_iter().enumerate() {
                    if position < source_output_count {
                        results.push(ReferenceDischargeValue::Value(output));
                    } else {
                        let allocation = widening.published()[position - source_output_count];
                        context.merge_boundary_state(&summary, &widening, allocation, output)?;
                    }
                }
                Ok(results)
            }
        }
    }
}

/// Static half selector used to compare disjoint folded access paths.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum RegisterHalf {
    Low,
    High,
}

/// Transform description of the downstream universe.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum RegisterTransform {
    /// Static half of a register, used by analysis-only access fixtures.
    Half(RegisterHalf),

    /// One bit of a register, selected by the next dynamic binding in the access input list.
    Bit,
}

// A half is a static description while a bit consumes one dynamic index binding. Paths are compared transform by
// transform: two static halves are disjoint as soon as they differ, two bits are the same index iff their bindings are
// equal and may otherwise overlap, a bit and a half may overlap, and paths that agree on every shared transform are the
// same when they have the same length and otherwise one is a strict prefix that contains the other.
impl Display for RegisterTransform {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Half(RegisterHalf::Low) => formatter.write_str("half(low)"),
            Self::Half(RegisterHalf::High) => formatter.write_str("half(high)"),
            Self::Bit => formatter.write_str("bit"),
        }
    }
}

impl ReferenceTransform for RegisterTransform {
    type Type = RegisterIrType;
    type Referent = RegisterType;

    fn binding_count(&self) -> usize {
        usize::from(matches!(self, Self::Bit))
    }

    fn validate_bindings(&self, _input: &RegisterType, bindings: &[&RegisterIrType]) -> Result<(), TypeError> {
        check_count!("binding", bindings, self.binding_count(), TypeError);
        for binding in bindings {
            <&RegisterType>::try_from(*binding)?;
        }
        Ok(())
    }

    fn output_type(&self, input: &RegisterType) -> Result<RegisterType, TypeError> {
        Ok(input.clone())
    }

    fn overlap(
        _type: &RegisterIrType,
        lhs: &[BoundReferenceTransform<Self>],
        rhs: &[BoundReferenceTransform<Self>],
    ) -> ReferenceViewOverlap {
        for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
            match (lhs.transform(), rhs.transform()) {
                (Self::Half(lhs_half), Self::Half(rhs_half)) if lhs_half != rhs_half => {
                    return ReferenceViewOverlap::Disjoint;
                }
                (Self::Half(_), Self::Half(_)) => {}
                (Self::Bit, Self::Bit) if lhs == rhs => {}
                _ => return ReferenceViewOverlap::MayOverlap,
            }
        }
        if lhs.len() == rhs.len() { ReferenceViewOverlap::Same } else { ReferenceViewOverlap::MayOverlap }
    }
}

// Registers have no axes: replicated descriptions pass through unchanged, while mapped axes are unsupported.
impl BatchableReferenceTransform for RegisterTransform {
    fn batch(&self, _type: &RegisterIrType, batch_axis: BatchAxis) -> Result<(Self, BatchAxis), BatchingError> {
        if !batch_axis.is_replicated() {
            return Err(BatchingError::UnsupportedOperation {
                message: "a register transform cannot carry a mapped batch axis; registers have no axes".to_string(),
            });
        }
        Ok((*self, batch_axis))
    }
}

impl<C: Domain<Type = RegisterIrType, Value = RegisterValue>> InterpretableOperation<C> for RegisterOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[RegisterValue],
    ) -> Result<Vec<RegisterValue>, ProgramError> {
        match self {
            Self::Negate => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![RegisterValue::Register(-inputs[0].register()?)])
            }
            Self::Add(_) => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![RegisterValue::Register(inputs[0].register()? + inputs[1].register()?)])
            }
            Self::Zero(operation) => {
                check_count!("input", inputs, 0, ProgramError);
                match operation.r#type() {
                    RegisterIrType::Register(_) => Ok(vec![RegisterValue::Register(0)]),
                    r#type => {
                        Err(TypeError::invalid(format!("cannot materialize a zero for reference type `{type}`")).into())
                    }
                }
            }
            Self::One => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![RegisterValue::Register(1)])
            }
            // The wrapped primitives interpret through the eager capabilities of `RegisterValue`.
            Self::ReferenceNew(operation) => operation.interpret(context, driver, inputs),
            Self::Read(operation) => operation.interpret(context, driver, inputs),
            Self::Write(operation) => operation.interpret(context, driver, inputs),
            Self::Swap(operation) => operation.interpret(context, driver, inputs),
            Self::Freeze(operation) => operation.interpret(context, driver, inputs),
            Self::AddUpdate(transforms) => {
                check_count!(
                    "input",
                    inputs,
                    2 + transforms.iter().map(ReferenceTransform::binding_count).sum::<usize>(),
                    ProgramError
                );
                // Execute directly: rebinding the additive capability would dispatch back to this operation.
                let current = inputs[0].read_through(transforms, &inputs[2..])?.register()?;
                inputs[0].write_through(
                    &RegisterValue::Register(current + inputs[1].register()?),
                    transforms,
                    &inputs[2..],
                )?;
                Ok(Vec::new())
            }
            Self::Call => driver.interpret_region(context, 0, inputs.to_vec()),
            Self::BitExtract => {
                check_count!("input", inputs, 2, ProgramError);
                Ok(vec![RegisterValue::Register(extract_bit(inputs[0].register()?, inputs[1].register()?)?)])
            }
            Self::BitInsert => {
                check_count!("input", inputs, 3, ProgramError);
                let inserted = insert_bit(inputs[0].register()?, inputs[1].register()?, inputs[2].register()?)?;
                Ok(vec![RegisterValue::Register(inserted)])
            }
        }
    }
}

/// Binds a region-free single-output register operation and checks its output count.
fn bind_register_output<C: Context<Type = RegisterIrType, Operation: From<RegisterOperation>>>(
    context: &C,
    operation: RegisterOperation,
    inputs: &[C::Value],
) -> Result<C::Value, ProgramError> {
    let mut outputs = context.bind(operation, Vec::new(), inputs)?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

impl<C: Context<Type = RegisterIrType, Operation: From<RegisterOperation>>> PartiallyEvaluatableOperation<C>
    for RegisterOperation
{
}

// Generic reference primitives provide their own forward rules at every context. Only the family-owned numerical
// operations and additive update need rules here.
impl<C: Context<Type = RegisterIrType, Operation = RegisterOperation> + Zero<C::Value>> DifferentiableOperation<C>
    for RegisterOperation
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        match self {
            Self::Negate => {
                check_count!("input", inputs, 1, ProgramError);
                let primal = bind_register_output(context.primal(), self.clone(), &primals)?;
                let tangent = match inputs[0].tangent() {
                    MaybeZero::Value(tangent) => MaybeZero::Value(bind_register_output(
                        context.tangent(),
                        self.clone(),
                        std::slice::from_ref(tangent),
                    )?),
                    MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
                };
                Ok(vec![DifferentiationDual::new(primal, tangent)?])
            }
            Self::Add(_) => {
                check_count!("input", inputs, 2, ProgramError);
                let primal = bind_register_output(context.primal(), self.clone(), &primals)?;
                let tangent = match (inputs[0].tangent(), inputs[1].tangent()) {
                    (MaybeZero::Zero(r#type), MaybeZero::Zero(_)) => MaybeZero::Zero(r#type.clone()),
                    (MaybeZero::Value(tangent), MaybeZero::Zero(_))
                    | (MaybeZero::Zero(_), MaybeZero::Value(tangent)) => MaybeZero::Value(tangent.clone()),
                    (MaybeZero::Value(left), MaybeZero::Value(right)) => MaybeZero::Value(bind_register_output(
                        context.tangent(),
                        self.clone(),
                        &[left.clone(), right.clone()],
                    )?),
                };
                Ok(vec![DifferentiationDual::new(primal, tangent)?])
            }
            Self::Zero(_) | Self::One => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![DifferentiationDual::new_with_zero_tangent(bind_register_output(
                    context.primal(),
                    self.clone(),
                    &[],
                )?)?])
            }
            Self::ReferenceNew(operation) => operation.jvp(context, driver, inputs),
            Self::Read(operation) => operation.jvp(context, driver, inputs),
            Self::Freeze(operation) => operation.jvp(context, driver, inputs),
            Self::Write(operation) => operation.jvp(context, driver, inputs),
            Self::Swap(operation) => operation.jvp(context, driver, inputs),
            Self::AddUpdate(transforms) => {
                check_count!(
                    "input",
                    inputs,
                    2 + transforms.iter().map(ReferenceTransform::binding_count).sum::<usize>(),
                    ProgramError
                );
                if inputs[0].tangent().is_zero() && !inputs[1].tangent().is_zero() {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "`{}` writes a live tangent into a reference that carries no tangent; pass the reference \
                             as a differentiated input instead of capturing it",
                            self.name(),
                        ),
                    }
                    .into());
                }
                context.primal().bind(self.clone(), Vec::new(), &primals)?;
                if let (MaybeZero::Value(reference), MaybeZero::Value(tangent)) =
                    (inputs[0].tangent(), inputs[1].tangent())
                {
                    let mut tangent_inputs = vec![reference.clone(), tangent.clone()];
                    for binding in &primals[2..] {
                        tangent_inputs.push(context.primal_to_tangent(binding.clone())?);
                    }
                    context.tangent().bind(self.clone(), Vec::new(), &tangent_inputs)?;
                }
                Ok(Vec::new())
            }
            Self::Call | Self::BitExtract | Self::BitInsert => Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` has no forward-mode rule in the register universe", self.name()),
            }
            .into()),
        }
    }
}

// The transposition rules of the family over its own staged programs. State cotangents live in the accumulators the
// transposition context owns: a read or freeze accumulates its result cotangent into the root's cotangent reference,
// a write swaps a zero into it, a swap swaps the result cotangent into it, an additive update reads it, and the
// allocation freezes it into the initial value's cotangent.
impl TransposableOperation<RegisterValue, RegisterOperation> for RegisterOperation {
    fn transpose<D: TranspositionDriver<RegisterValue, RegisterOperation>>(
        &self,
        context: &mut TranspositionContext<RegisterValue, RegisterOperation>,
        driver: &D,
        inputs: &[PartialValue<RegisterTracer>],
        outputs: &[MaybeZero<RegisterTracer>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        match self {
            Self::Negate => {
                check_count!("input", inputs, 1, ProgramError);
                check_count!("output", outputs, 1, ProgramError);
                check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
                let contribution = match &outputs[0] {
                    MaybeZero::Value(cotangent) => MaybeZero::Value(bind_register_output(
                        &**context,
                        self.clone(),
                        std::slice::from_ref(cotangent),
                    )?),
                    MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
                };
                accumulators[0].accumulate(context, contribution)
            }
            Self::Add(_) => {
                check_count!("input", inputs, 2, ProgramError);
                check_count!("output", outputs, 1, ProgramError);
                check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
                accumulators[0].accumulate(context, outputs[0].clone())?;
                accumulators[1].accumulate(context, outputs[0].clone())
            }
            Self::Zero(_) | Self::One => {
                check_count!("input", inputs, 0, ProgramError);
                check_count!("output", outputs, 1, ProgramError);
                check_count!("accumulator", accumulators, 0, DifferentiationError);
                Ok(())
            }
            Self::ReferenceNew(operation) => operation.transpose(context, driver, inputs, outputs, accumulators),
            Self::Read(operation) => operation.transpose(context, driver, inputs, outputs, accumulators),
            Self::Freeze(operation) => operation.transpose(context, driver, inputs, outputs, accumulators),
            Self::Write(operation) => operation.transpose(context, driver, inputs, outputs, accumulators),
            Self::Swap(operation) => operation.transpose(context, driver, inputs, outputs, accumulators),
            Self::AddUpdate(transforms) => {
                check_count!(
                    "input",
                    inputs,
                    2 + transforms.iter().map(ReferenceTransform::binding_count).sum::<usize>(),
                    ProgramError
                );
                check_count!("output", outputs, 0, ProgramError);
                check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
                let update_cotangent = match context.cotangent_reference_if_allocated(driver, 0)? {
                    Some(accumulator) => {
                        let mut access_inputs = vec![accumulator];
                        for binding in &inputs[2..] {
                            access_inputs.push(binding.as_known().cloned().ok_or_else(|| {
                                ProgramError::UnsupportedOperation {
                                    message: "register transform bindings must be known during transposition".into(),
                                }
                            })?);
                        }
                        MaybeZero::Value(bind_register_output(
                            &**context,
                            Self::Read(ReferenceReadOperation::new().with_transforms(transforms.clone())),
                            &access_inputs,
                        )?)
                    }
                    None => MaybeZero::Zero(inputs[1].r#type().cotangent()?),
                };
                accumulators[1].accumulate(context, update_cotangent)
            }
            Self::Call | Self::BitExtract | Self::BitInsert => Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` has no transposition rule in the register universe", self.name()),
            }
            .into()),
        }
    }
}

// Registers have no axes, so the family batches replicated carriers only: every region-free operation runs once on the
// parent context over the packed values and its outputs stay replicated. A mapped carrier is rejected by name.
// Folded paths remain on the copied access, and their dynamic bindings retain their packed input positions.
impl<C: Context<Type = RegisterIrType, Operation: From<RegisterOperation>>, P: BatchingPolicy<C>>
    BatchableOperation<C, P> for RegisterOperation
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        _driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        match self {
            Self::Call => {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!("`{}` has no batching rule in the register universe", self.name()),
                });
            }
            _ => {}
        }
        let values = inputs
            .iter()
            .map(|input| match P::batch_axis(input).axis() {
                None => Ok(P::value(input).clone()),
                Some(_) => Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "`{}` cannot batch a mapped register carrier; registers have no axes",
                        self.name()
                    ),
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = context.parent().bind(self.clone(), Vec::new(), values.as_slice())?;
        Ok(outputs.into_iter().map(P::replicated).collect::<Vec<_>>().into())
    }
}

/// Replicated-only batching policy selected by [`RegisterIrType`] for the public [`batch`] entry point. The batch
/// carrier is the value itself because this policy never attaches a mapped axis.
#[derive(Copy, Clone, Debug)]
struct RegisterBatching;

impl BatchableType for RegisterIrType {
    type Policy = RegisterBatching;
}

impl<C: Context<Type = RegisterIrType>> BatchingPolicy<C> for RegisterBatching {
    type Batch = C::Value;
    type Extent = usize;
    type Evidence = ();
    type BatchedProgram = BoundaryPreservingBatchedProgram<C::Constant, C::Operation>;

    fn batch(value: C::Value, batch_axis: BatchAxis) -> Result<Self::Batch, BatchingError> {
        if !batch_axis.is_replicated() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "register values have no axes, so a value of type `{}` cannot be mapped",
                    value.r#type().as_ref(),
                ),
            });
        }
        Ok(value)
    }

    fn replicated(value: C::Value) -> Self::Batch {
        value
    }

    fn value(batch: &Self::Batch) -> &C::Value {
        batch
    }

    fn batch_axis(_batch: &Self::Batch) -> BatchAxis {
        BatchAxis::replicated()
    }

    fn unbatched_type(batch: &Self::Batch) -> Cow<'_, RegisterIrType> {
        batch.r#type()
    }

    fn adapt_batched_program<
        CollapseFn: Fn(
            &TracingContext<C::Constant, C::Operation>,
            Tracer<TracingContext<C::Constant, C::Operation>>,
            ryft_core::Axis,
        ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError>,
    >(
        program: Self::BatchedProgram,
        _required_output_axes: Option<&[BatchAxis]>,
        _collapse_fn: CollapseFn,
    ) -> Result<BoundaryPreservingBatchedProgram<C::Constant, C::Operation>, BatchingError> {
        // Every carrier is replicated, so a batched program already carries the source boundary.
        Ok(program)
    }
}

// The register universe batches region-free operations only; recursion into nested programs is left unsupported.
impl<C: Context<Type = RegisterIrType>> RecursiveBatchingPolicy<C> for RegisterBatching {
    fn batch_region(
        _context: &BatchingContext<C, Self>,
        _region: RegionRef<'_, C::Constant, C::Operation>,
        _inputs: Vec<Self::Batch>,
    ) -> Result<Vec<Self::Batch>, BatchingError> {
        Err(BatchingError::UnsupportedOperation {
            message: "the register universe batches region-free operations only".to_string(),
        })
    }

    fn batch_program(
        _context: &BatchingContext<C, Self>,
        _region: RegionRef<'_, C::Constant, C::Operation>,
        _input_axes: &[BatchAxis],
        _output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<Self::BatchedProgram, BatchingError> {
        Err(BatchingError::UnsupportedOperation {
            message: "the register universe batches region-free operations only".to_string(),
        })
    }
}

impl<C: Context<Type = RegisterIrType>> BatchingEntrypointPolicy<C> for RegisterBatching {
    fn pack_inputs(
        context: &C,
        inputs: Vec<C::Value>,
        input_batch_axes: Vec<BatchAxis>,
        batch_axis: BatchAxisSpecification<usize>,
    ) -> Result<(BatchingContext<C, Self>, Vec<Self::Batch>), BatchingError> {
        if inputs.len() != input_batch_axes.len() {
            return Err(
                ProgramError::InvalidInputCount { expected: inputs.len(), actual: input_batch_axes.len() }.into()
            );
        }
        // No register carries a mapped axis, so the extent can come only from the specification.
        let extent = *batch_axis.extent().ok_or(BatchingError::EmptyBatch)?;
        let inputs = inputs
            .into_iter()
            .zip(input_batch_axes)
            .map(|(input, input_batch_axis)| <Self as BatchingPolicy<C>>::batch(input, input_batch_axis))
            .collect::<Result<Vec<_>, _>>()?;
        let context =
            BatchingContext::with_policy(context.clone(), extent).with_axis_name(batch_axis.name().map(String::from));
        Ok((context, inputs))
    }

    fn materialize_output(
        _context: &BatchingContext<C, Self>,
        output: Self::Batch,
        output_batch_axis: BatchAxis,
    ) -> Result<C::Value, BatchingError> {
        if !output_batch_axis.is_replicated() {
            return Err(BatchingError::UnsupportedOperation {
                message: "register outputs stay replicated; registers have no axis to materialize a batch at"
                    .to_string(),
            });
        }
        Ok(output)
    }
}

#[test]
fn test_register_ir_type_referent() {
    let register_member = RegisterIrType::from(RegisterType);
    let reference_member = RegisterIrType::from(ReferenceType::new(RegisterType));

    // L1: `is_reference` agrees with `referent` on both member kinds.
    assert!(reference_member.is_reference());
    assert!(reference_member.referent().is_some());
    assert!(!register_member.is_reference());
    assert!(register_member.referent().is_none());

    // L2: embedding a reference type and projecting it back yields exactly the referent it wraps.
    assert_eq!(reference_member.referent(), Some(&RegisterType));

    // L4: the borrowed reference conversion succeeds exactly where the projection does and agrees with it.
    assert_eq!(<&ReferenceType<RegisterType>>::try_from(&reference_member), Ok(&ReferenceType::new(RegisterType)),);
    assert_eq!(<&ReferenceType<RegisterType>>::try_from(&reference_member).unwrap().referent(), &RegisterType);
    assert_eq!(
        <&ReferenceType<RegisterType>>::try_from(&register_member),
        Err(TypeError::invalid("expected reference type but got register type")),
    );

    // L5 concerns universes whose referent family is `NoReferent`; this universe's referent family is `RegisterType`,
    // which is inhabited, so the law does not constrain it.
}

#[test]
fn test_register_ir_type_as_referent() {
    let register_member = RegisterIrType::from(RegisterType);
    let reference_member = RegisterIrType::from(ReferenceType::new(RegisterType));

    // L3: embedding a referent and viewing it in the referent family yields it back, while the reference member is
    // not a referent.
    assert_eq!(register_member.as_referent(), Some(&RegisterType));
    assert_eq!(reference_member.as_referent(), None);

    // L4: the borrowed referent conversion succeeds exactly where the projection does and agrees with it.
    assert_eq!(<&RegisterType>::try_from(&register_member), Ok(&RegisterType));
    assert_eq!(
        <&RegisterType>::try_from(&reference_member),
        Err(TypeError::invalid("expected register type but got reference type")),
    );
}

#[test]
fn test_downstream_reference_universe_discharges_through_the_public_surface() {
    // `f(initial, replacement) = (replaced value, frozen final state)`, written entirely in a reference universe that
    // `ryft-core` knows nothing about.
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let initial = builder.add_input(RegisterIrType::Register(RegisterType));
    let replacement = builder.add_input(RegisterIrType::Register(RegisterType));
    let allocation = builder
        .add_instruction(RegisterOperation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![initial], None)
        .unwrap()[0];
    let replaced = builder
        .add_instruction(
            RegisterOperation::Swap(ReferenceSwapOperation::new()),
            Vec::new(),
            vec![allocation, replacement],
            None,
        )
        .unwrap()[0];
    let snapshot = builder
        .add_instruction(RegisterOperation::Read(ReferenceReadOperation::new()), Vec::new(), vec![allocation], None)
        .unwrap()[0];
    let frozen = builder
        .add_instruction(RegisterOperation::Freeze(ReferenceFreezeOperation::new()), Vec::new(), vec![allocation], None)
        .unwrap()[0];
    let program = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(
            vec![replaced, snapshot, frozen],
            vec![Placeholder; 2],
            vec![Placeholder; 3],
        )
        .unwrap();

    // Discharging through the region driver rewrites every reference primitive into explicit state threading, so the
    // downstream universe reaches the same outputs an eager reference execution would have produced.
    let context = RegisterDischargeContext::new(RegisterDestination::new());
    let regions = [program];
    let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
    let inputs = vec![
        RegisterDischargeValue::Value(RegisterValue::Register(4)),
        RegisterDischargeValue::Value(RegisterValue::Register(3)),
    ];
    assert_eq!(
        driver.inline_region(&context, 0, inputs),
        Ok(vec![
            RegisterDischargeValue::Value(RegisterValue::Register(4)),
            RegisterDischargeValue::Value(RegisterValue::Register(3)),
            RegisterDischargeValue::Value(RegisterValue::Register(3)),
        ]),
    );
    assert_eq!(context.live_allocation_ids(), Vec::new());
}

#[test]
fn test_downstream_reference_discharge_context_environment_accessors() {
    let context = RegisterDischargeContext::new(RegisterDestination::new());
    let bound = ReferenceDischargeValue::from(
        context.bind_discharged(ReferenceType::new(RegisterType), RegisterValue::Register(1)).unwrap(),
    );
    let allocation = bound.try_as_reference("a downstream allocation").unwrap().allocation_id();

    // These ID-based operations are the public seam custom structured transforms use to inspect, thread, and merge
    // discharged state without accessing the environment's private representation.
    assert_eq!(context.live_allocation_ids(), vec![allocation]);
    assert_eq!(context.is_allocation_discharged(allocation), Ok(true));
    assert_eq!(context.discharged_state(allocation), Ok(RegisterValue::Register(1)));
    assert_eq!(context.is_mutated(allocation), Ok(false));
    assert_eq!(context.allocation_reference(allocation).map(ReferenceDischargeValue::from), Ok(bound));
    assert_eq!(context.set_discharged_state(allocation, RegisterValue::Register(2), true), Ok(()));
    assert_eq!(context.discharged_state(allocation), Ok(RegisterValue::Register(2)));
    assert_eq!(context.is_mutated(allocation), Ok(true));
}

#[test]
fn test_downstream_reference_universe_discharges_into_a_staged_program() {
    // The same universe discharged against a staging destination, which is the shape production discharge uses: the
    // rewritten work is recorded into a destination program instead of executed.
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let initial = builder.add_input(RegisterIrType::Register(RegisterType));
    let replacement = builder.add_input(RegisterIrType::Register(RegisterType));
    let allocation = builder
        .add_instruction(RegisterOperation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![initial], None)
        .unwrap()[0];
    let replaced = builder
        .add_instruction(
            RegisterOperation::Swap(ReferenceSwapOperation::new()),
            Vec::new(),
            vec![allocation, replacement],
            None,
        )
        .unwrap()[0];
    let negated = builder.add_instruction(RegisterOperation::Negate, Vec::new(), vec![replaced], None).unwrap()[0];
    let frozen = builder
        .add_instruction(RegisterOperation::Freeze(ReferenceFreezeOperation::new()), Vec::new(), vec![allocation], None)
        .unwrap()[0];
    let source = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(
            vec![negated, frozen],
            vec![Placeholder; 2],
            vec![Placeholder; 2],
        )
        .unwrap();

    let discharge = |inputs: Vec<Tracer<TracingContext<RegisterValue, RegisterOperation>>>| {
        let context = ReferenceDischargeContext::new(inputs[0].context().clone());
        let carriers = inputs.into_iter().map(ReferenceDischargeValue::Value).collect::<Vec<_>>();
        let regions = [source];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let outputs = driver.inline_region(&context, 0, carriers)?;
        assert_eq!(context.live_allocation_ids(), Vec::new());
        outputs
            .iter()
            .map(|output| output.try_as_value("a discharged output").cloned())
            .collect::<Result<Vec<_>, _>>()
    };
    let (_, discharged): (_, Program<_, _, Vec<RegisterValue>, Vec<RegisterValue>>) =
        EagerContext::<RegisterValue, RegisterOperation>::trace(
            discharge,
            vec![RegisterIrType::Register(RegisterType); 2],
        )
        .unwrap();

    // Every reference primitive was normalized away, so the staged program threads the replacement directly and
    // records only the universe's one pure operation.
    assert_eq!(
        discharged.to_string(),
        indoc! {"
            lambda %0:register, %1:register .
            let %2:register = register.negate %0
            in (%2, %1)"},
    );
}

#[test]
fn test_downstream_program_level_discharge_threads_external_state_through_the_entry_boundary() {
    // `Program::discharge_references` is the program-level entry point, and it is universe-generic: this
    // exercises it over the downstream universe, so nothing about the array universe can be load-bearing for it.
    // `f(counter, other, replacement) = replaced`, where `counter` is written and `other` is only read.
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let counter = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let other = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let replacement = builder.add_input(RegisterIrType::Register(RegisterType));
    let observed = builder
        .add_instruction(RegisterOperation::Read(ReferenceReadOperation::new()), Vec::new(), vec![other], None)
        .unwrap()[0];
    let replaced = builder
        .add_instruction(
            RegisterOperation::Swap(ReferenceSwapOperation::new()),
            Vec::new(),
            vec![counter, replacement],
            None,
        )
        .unwrap()[0];
    let negated = builder.add_instruction(RegisterOperation::Negate, Vec::new(), vec![observed], None).unwrap()[0];
    let source = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(
            vec![replaced, negated],
            vec![Placeholder; 3],
            vec![Placeholder; 2],
        )
        .unwrap();

    // Each reference input keeps its boundary position and becomes a value input carrying the referent's lifted
    // type, the public outputs are exactly the source outputs, and only the written allocation appends a hidden final-state
    // output after them.
    let discharged = source.discharge_references(1).unwrap();
    assert_eq!(discharged.output_count(), 2);
    assert_eq!(
        discharged.external_reference_bindings(),
        &[
            ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, Some(2)),
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None),
        ],
    );
    assert_eq!(
        discharged.program().to_string(),
        indoc! {"
            lambda %0:register, %1:register, %2:register .
            let %3:register = register.negate %1
            in (%0, %3, %2)"},
    );

    // An external reference remains owned by the caller, so a program that consumes one is rejected by name.
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let external = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let frozen = builder
        .add_instruction(RegisterOperation::Freeze(ReferenceFreezeOperation::new()), Vec::new(), vec![external], None)
        .unwrap()[0];
    let source = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
        .unwrap();
    assert_eq!(
        source.discharge_references(0).unwrap_err(),
        ProgramError::MalformedProgram(
            "reference discharge consumed external input 0, whose state must remain owned by the caller".to_string(),
        ),
    );
}

#[test]
fn test_downstream_region_summary_exposes_exact_access_modes() {
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let reference = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let replacement = builder.add_input(RegisterIrType::Register(RegisterType));
    let read = builder
        .add_instruction(RegisterOperation::Read(ReferenceReadOperation::new()), Vec::new(), vec![reference], None)
        .unwrap()[0];
    builder
        .add_instruction(
            RegisterOperation::Write(ReferenceWriteOperation::new()),
            Vec::new(),
            vec![reference, replacement],
            None,
        )
        .unwrap();
    let swapped = builder
        .add_instruction(
            RegisterOperation::Swap(ReferenceSwapOperation::new()),
            Vec::new(),
            vec![reference, replacement],
            None,
        )
        .unwrap()[0];
    let region = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(
            vec![read, swapped],
            vec![Placeholder; 2],
            vec![Placeholder; 2],
        )
        .unwrap();

    let context = RegisterDischargeContext::new(RegisterDestination::new());
    let reference = ReferenceDischargeValue::from(
        context.bind_discharged(ReferenceType::new(RegisterType), RegisterValue::Register(1)).unwrap(),
    );
    let allocation = reference.try_as_reference("a downstream allocation").unwrap().allocation_id();
    let summary = context
        .region_summary(&RegisterOperation::Call, 0, region.entry_region_ref(), &[Some(allocation), None])
        .unwrap();

    assert_eq!(summary.accessed_allocations().collect::<Vec<_>>(), vec![allocation]);
    assert_eq!(
        summary.access_modes(allocation).collect::<Vec<_>>(),
        vec![ReferenceAccessMode::Read, ReferenceAccessMode::Write, ReferenceAccessMode::ReadWrite],
    );
    assert!(summary.access_modes(allocation).any(|mode| mode == ReferenceAccessMode::ReadWrite));
    assert!(!summary.access_modes(allocation).any(|mode| mode == ReferenceAccessMode::Accumulate));
    assert!(summary.is_mutated(allocation));
}

#[test]
fn test_downstream_partial_discharge_preserves_the_allocations_it_was_not_asked_to_discharge() {
    // Partial discharge is reachable from downstream position too, and the register universe declines accumulation
    // and views entirely, so this is the minimal shape a backend needs: `f(counter, buffer, replacement) = replaced`,
    // where `counter` is selected and `buffer` stays a reference the rewritten program still reads and writes.
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let counter = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let buffer = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let replacement = builder.add_input(RegisterIrType::Register(RegisterType));
    let observed = builder
        .add_instruction(RegisterOperation::Read(ReferenceReadOperation::new()), Vec::new(), vec![buffer], None)
        .unwrap()[0];
    let replaced = builder
        .add_instruction(
            RegisterOperation::Swap(ReferenceSwapOperation::new()),
            Vec::new(),
            vec![counter, observed],
            None,
        )
        .unwrap()[0];
    builder
        .add_instruction(
            RegisterOperation::Write(ReferenceWriteOperation::new()),
            Vec::new(),
            vec![buffer, replacement],
            None,
        )
        .unwrap();
    let source = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![replaced], vec![Placeholder; 3], vec![Placeholder])
        .unwrap();

    let targets = source.reference_discharge_targets(0).unwrap();
    assert_eq!(
        targets,
        vec![
            ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 }),
            ReferenceDischargeTarget::External(ReferenceSource::Input { index: 1 }),
        ],
    );
    let discharged = source.partially_discharge_references(0, &targets[..1]).unwrap();

    // The selected allocation became state at its own boundary position and publishes its final state as a hidden output;
    // the preserved reference kept its reference type and reports no binding, and both of its accesses replayed verbatim.
    assert_eq!(discharged.output_count(), 1);
    assert_eq!(
        discharged.external_reference_bindings(),
        &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
    );
    assert_eq!(
        discharged.program().to_string(),
        indoc! {"
            lambda %0:register, %1:ref<register>, %2:register .
            let %3:register = reference_read %1
                () = reference_write %1 %2
            in (%0, %3)"},
    );
}

#[test]
fn test_downstream_structured_rule_discharges_through_the_region_boundary_api() {
    // The hand-rolled `register.call` rule exercises the complete structured surface from a third-party position:
    // region summaries, state widening, boundary construction, isolated region rebuilding, prediction validation,
    // and successor-state merging. The same caller allocation enters at two declared positions, so the rebuilt region must
    // preserve the aliasing: the write through position 0 is observed by the read through position 1.
    let mut callee = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let first = callee.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let second = callee.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let replacement = callee.add_input(RegisterIrType::Register(RegisterType));
    callee
        .add_instruction(
            RegisterOperation::Write(ReferenceWriteOperation::new()),
            Vec::new(),
            vec![first, replacement],
            None,
        )
        .unwrap();
    let observed = callee
        .add_instruction(RegisterOperation::Read(ReferenceReadOperation::new()), Vec::new(), vec![second], None)
        .unwrap()[0];
    let callee = callee
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![observed], vec![Placeholder; 3], vec![Placeholder])
        .unwrap();

    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let reference = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let update = builder.add_input(RegisterIrType::Register(RegisterType));
    let region = builder.import_program(callee);
    let result = builder
        .add_instruction(RegisterOperation::Call, vec![region], vec![reference, reference, update], None)
        .unwrap()[0];
    let source = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![result], vec![Placeholder; 2], vec![Placeholder])
        .unwrap();

    let discharged = source.discharge_references(0).unwrap();
    assert_eq!(discharged.output_count(), 1);
    assert_eq!(
        discharged.external_reference_bindings(),
        &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
    );
    assert_eq!(
        discharged.program().to_string(),
        indoc! {"
            lambda %0:register, %1:register .
            let %2:register, %3:register = register.call %0 %0 %1 [
                callee={
                    lambda %0:register, %1:register, %2:register .
                    in (%2, %2)
                },
            ]
            in (%2, %3)"},
    );
}

#[test]
fn test_downstream_partial_targets_reach_an_internal_allocation_inside_a_structured_region() {
    // The allocation target sits inside the callee region, so whether it discharges is decided by the replay location
    // the downstream rule's driver hands to `selects_internal` inside the fork. An empty target list must preserve the
    // allocation inside the rebuilt region, and selecting the enumerated target must discharge it completely — which is
    // exactly the behavior a driver without a real `source_instruction_id()` would silently break.
    let mut callee = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let initial = callee.add_input(RegisterIrType::Register(RegisterType));
    let local = callee
        .add_instruction(RegisterOperation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![initial], None)
        .unwrap()[0];
    let frozen = callee
        .add_instruction(RegisterOperation::Freeze(ReferenceFreezeOperation::new()), Vec::new(), vec![local], None)
        .unwrap()[0];
    let callee = callee
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
        .unwrap();

    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let input = builder.add_input(RegisterIrType::Register(RegisterType));
    let region = builder.import_program(callee);
    let result = builder.add_instruction(RegisterOperation::Call, vec![region], vec![input], None).unwrap()[0];
    let source = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![result], vec![Placeholder], vec![Placeholder])
        .unwrap();

    let preserved = source.clone().partially_discharge_references(0, &[]).unwrap();
    assert_eq!(preserved.external_reference_bindings(), &[]);
    assert_eq!(
        preserved.program().to_string(),
        indoc! {"
            lambda %0:register .
            let %1:register = register.call %0 [
                callee={
                    lambda %0:register .
                    let %1:ref<register> = reference_new %0
                        %2:register = reference_freeze %1
                    in (%2)
                },
            ]
            in (%1)"},
    );

    let targets = source.reference_discharge_targets(0).unwrap();
    assert_eq!(targets.len(), 1);
    let full =
        ReferenceDischargeResult::try_from(source.partially_discharge_references(0, targets.as_slice()).unwrap())
            .unwrap();
    assert_eq!(
        full.program().to_string(),
        indoc! {"
            lambda %0:register .
            let %1:register = register.call %0 [
                callee={
                    lambda %0:register .
                    in (%0)
                },
            ]
            in (%1)"},
    );
}

#[test]
fn test_downstream_access_descriptors_record_distinct_paths() {
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let reference = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let low = builder
        .add_instruction(
            RegisterOperation::Read(
                ReferenceReadOperation::new().with_transforms(vec![RegisterTransform::Half(RegisterHalf::Low)]),
            ),
            Vec::new(),
            vec![reference],
            None,
        )
        .unwrap()[0];
    let high = builder
        .add_instruction(
            RegisterOperation::Read(
                ReferenceReadOperation::new().with_transforms(vec![RegisterTransform::Half(RegisterHalf::High)]),
            ),
            Vec::new(),
            vec![reference],
            None,
        )
        .unwrap()[0];
    let program = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![low, high], vec![Placeholder], vec![Placeholder; 2])
        .unwrap();
    let region = program.entry_region_ref();
    let analysis = region.reference_view_analysis(0).unwrap();
    assert_eq!(
        analysis.path(InstructionId::new(region.id(), 0), 0).cloned(),
        Some(ReferenceTransformPath::root().with_transform(RegisterTransform::Half(RegisterHalf::Low)))
    );
    assert_eq!(
        analysis.path(InstructionId::new(region.id(), 1), 0).cloned(),
        Some(ReferenceTransformPath::root().with_transform(RegisterTransform::Half(RegisterHalf::High)))
    );
    assert!(analysis.analysis().values().filter_map(|value| analysis.analysis().alias(value)).next().is_none());
}

/// `f(r, x) = { r.add_update(x); r.read() }` over the register universe, written against whatever context the input
/// values dispatch to so that one closure serves the forward-mode, reverse-mode, and batching tracers alike.
fn read_modify_write<V: Value<Type = RegisterIrType>>((reference, x): (V, V)) -> Result<V, ProgramError>
where
    V::DispatchDomain: Context<Type = RegisterIrType, Operation: From<RegisterOperation>>,
{
    let context = reference.dispatch_domain();
    context.bind(RegisterOperation::AddUpdate(Vec::new()), Vec::new(), &[reference.clone(), x])?;
    bind_register_output(&context, RegisterOperation::Read(ReferenceReadOperation::new()), &[reference])
}

/// Writes and reads one dynamic bit with the path stored on each access operation.
fn write_read_folded_bit<V: Value<Type = RegisterIrType>>(
    (reference, value, index): (V, V, V),
) -> Result<V, ProgramError>
where
    V::DispatchDomain: Context<Type = RegisterIrType, Operation: From<RegisterOperation>>,
{
    let context = reference.dispatch_domain();
    let transforms = vec![RegisterTransform::Bit];
    context.bind(
        RegisterOperation::Write(ReferenceWriteOperation::new().with_transforms(transforms.clone())),
        Vec::new(),
        &[reference.clone(), value, index.clone()],
    )?;
    bind_register_output(
        &context,
        RegisterOperation::Read(ReferenceReadOperation::new().with_transforms(transforms)),
        &[reference, index],
    )
}

#[test]
fn test_downstream_transform_overlap_and_batch() {
    // The family's overlap rule compares paths transform by transform: the two halves are disjoint, a path is the same
    // as itself, the complete root or a shorter prefix contains what it narrows to, two bits are the same index exactly
    // when their bindings agree and may otherwise overlap, and a bit may overlap with a half.
    let root = RegisterIrType::Reference(ReferenceType::new(RegisterType));
    let value = |atom: usize| ValueId::new(RegionId::new(0), AtomId::new(atom));
    let empty = ReferenceTransformPath::<RegisterTransform>::root();
    let low = empty.clone().with_transform(RegisterTransform::Half(RegisterHalf::Low));
    let high = empty.clone().with_transform(RegisterTransform::Half(RegisterHalf::High));
    let low_high = low.clone().with_transform(RegisterTransform::Half(RegisterHalf::High));
    let bit_of_1 = empty.clone().with_bound_transform(RegisterTransform::Bit, vec![value(1)]);
    let bit_of_2 = empty.clone().with_bound_transform(RegisterTransform::Bit, vec![value(2)]);
    assert_eq!(low.overlap(&high, &root), ReferenceViewOverlap::Disjoint);
    assert_eq!(low.overlap(&low, &root), ReferenceViewOverlap::Same);
    assert_eq!(empty.overlap(&empty, &root), ReferenceViewOverlap::Same);
    assert_eq!(empty.overlap(&low, &root), ReferenceViewOverlap::MayOverlap);
    assert_eq!(low_high.overlap(&low, &root), ReferenceViewOverlap::MayOverlap);
    assert_eq!(low_high.overlap(&high, &root), ReferenceViewOverlap::Disjoint);
    assert_eq!(bit_of_1.overlap(&bit_of_1, &root), ReferenceViewOverlap::Same);
    assert_eq!(bit_of_1.overlap(&bit_of_2, &root), ReferenceViewOverlap::MayOverlap);
    assert_eq!(bit_of_1.overlap(&low, &root), ReferenceViewOverlap::MayOverlap);
    assert_eq!(empty.overlap(&bit_of_1, &root), ReferenceViewOverlap::MayOverlap);
    assert_eq!(
        low.with_bound_transform(RegisterTransform::Bit, vec![value(1)]).overlap(&bit_of_1, &root),
        ReferenceViewOverlap::MayOverlap
    );

    // Registers have no axes, so a description batches only replicated sources and passes through unchanged.
    assert_eq!(
        RegisterTransform::Half(RegisterHalf::Low).batch(&root, BatchAxis::replicated()),
        Ok((RegisterTransform::Half(RegisterHalf::Low), BatchAxis::replicated()))
    );
    assert_eq!(
        RegisterTransform::Bit.batch(&root, BatchAxis::replicated()),
        Ok((RegisterTransform::Bit, BatchAxis::replicated()))
    );
    assert!(matches!(
        RegisterTransform::Half(RegisterHalf::High).batch(&root, BatchAxis::new(0)),
        Err(BatchingError::UnsupportedOperation { message })
            if message == "a register transform cannot carry a mapped batch axis; registers have no axes",
    ));
}

#[test]
fn test_downstream_dynamic_access_analysis_closes_index_bindings() {
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let reference = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let first_index = builder.add_input(RegisterIrType::Register(RegisterType));
    let second_index = builder.add_input(RegisterIrType::Register(RegisterType));
    for index in [first_index, first_index, second_index] {
        builder
            .add_instruction(
                RegisterOperation::Read(ReferenceReadOperation::new().with_transforms(vec![RegisterTransform::Bit])),
                Vec::new(),
                vec![reference, index],
                None,
            )
            .unwrap();
    }
    let program = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
        .unwrap();
    let region = program.entry_region_ref();
    let analysis = region.reference_view_analysis(0).unwrap();
    let first = analysis.path(InstructionId::new(region.id(), 0), 0).cloned().unwrap();
    let repeated = analysis.path(InstructionId::new(region.id(), 1), 0).cloned().unwrap();
    let second = analysis.path(InstructionId::new(region.id(), 2), 0).cloned().unwrap();
    assert_eq!(
        first,
        ReferenceTransformPath::root()
            .with_bound_transform(RegisterTransform::Bit, vec![ValueId::new(region.id(), first_index)])
    );
    assert_eq!(repeated, first);
    assert_eq!(
        second,
        ReferenceTransformPath::root()
            .with_bound_transform(RegisterTransform::Bit, vec![ValueId::new(region.id(), second_index)])
    );
    assert_eq!(
        RegisterTransform::overlap(
            &RegisterIrType::Register(RegisterType),
            first.bound_transforms(),
            second.bound_transforms()
        ),
        ReferenceViewOverlap::MayOverlap
    );
}

#[test]
fn test_downstream_reference_access_analysis_does_not_require_batching() {
    /// Downstream bit description intentionally lacking a batching implementation.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct UnbatchedBit;

    impl Display for UnbatchedBit {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("bit")
        }
    }

    impl ReferenceTransform for UnbatchedBit {
        type Type = RegisterIrType;
        type Referent = RegisterType;

        fn binding_count(&self) -> usize {
            1
        }

        fn validate_bindings(&self, input: &RegisterType, bindings: &[&RegisterIrType]) -> Result<(), TypeError> {
            RegisterTransform::Bit.validate_bindings(input, bindings)
        }

        fn output_type(&self, input: &RegisterType) -> Result<RegisterType, TypeError> {
            Ok(input.clone())
        }

        fn overlap(
            _type: &RegisterIrType,
            lhs: &[BoundReferenceTransform<Self>],
            rhs: &[BoundReferenceTransform<Self>],
        ) -> ReferenceViewOverlap {
            if lhs == rhs { ReferenceViewOverlap::Same } else { ReferenceViewOverlap::MayOverlap }
        }
    }

    let mut builder =
        ProgramBuilder::<RegisterValue, ReferenceReadOperation<RegisterType, RegisterIrType, UnbatchedBit>>::new();
    let root = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let index = builder.add_input(RegisterIrType::Register(RegisterType));
    builder
        .add_instruction(
            ReferenceReadOperation::new().with_transforms(vec![UnbatchedBit]),
            Vec::new(),
            vec![root, index],
            None,
        )
        .unwrap();
    let program = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
        .unwrap();
    let region = program.entry_region_ref();
    let analysis = region.reference_view_analysis(0).unwrap();
    assert_eq!(
        analysis.path(InstructionId::new(region.id(), 0), 0).cloned(),
        Some(ReferenceTransformPath::root().with_bound_transform(UnbatchedBit, vec![ValueId::new(region.id(), index)]))
    );
}

#[test]
fn test_downstream_reference_access_analysis_binds_explicit_region_inputs() {
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let reference = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let first_index = builder.add_input(RegisterIrType::Register(RegisterType));
    let second_index = builder.add_input(RegisterIrType::Register(RegisterType));
    for index in [first_index, second_index] {
        builder
            .add_instruction(
                RegisterOperation::Read(ReferenceReadOperation::new().with_transforms(vec![RegisterTransform::Bit])),
                Vec::new(),
                vec![reference, index],
                None,
            )
            .unwrap();
    }
    let body = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
        .unwrap();
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let inputs = body.input_types().iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
    let region = builder.import_region(body.entry_region_ref());
    builder.add_instruction(RegisterOperation::Call, vec![region], inputs, None).unwrap();
    let program = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
        .unwrap();
    let entry = program.entry_region_ref();
    let analysis = entry.reference_view_analysis(0).unwrap();
    assert_eq!(
        analysis.path(InstructionId::new(region, 0), 0).cloned(),
        Some(
            ReferenceTransformPath::root()
                .with_bound_transform(RegisterTransform::Bit, vec![ValueId::new(region, first_index)])
        )
    );
    assert_eq!(
        analysis.path(InstructionId::new(region, 1), 0).cloned(),
        Some(
            ReferenceTransformPath::root()
                .with_bound_transform(RegisterTransform::Bit, vec![ValueId::new(region, second_index)])
        )
    );
}

#[test]
fn test_downstream_dynamic_view_discharges_through_a_value_bound_alias() {
    // `f(initial, i, x) = { r = new(initial); b = bit(r, i); old = swap(b, x); (old, read(b), freeze(r)) }`.
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let initial = builder.add_input(RegisterIrType::Register(RegisterType));
    let index = builder.add_input(RegisterIrType::Register(RegisterType));
    let replacement = builder.add_input(RegisterIrType::Register(RegisterType));
    let allocation = builder
        .add_instruction(RegisterOperation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![initial], None)
        .unwrap()[0];
    let previous = builder
        .add_instruction(
            RegisterOperation::Swap(ReferenceSwapOperation::new().with_transforms(vec![RegisterTransform::Bit])),
            Vec::new(),
            vec![allocation, replacement, index],
            None,
        )
        .unwrap()[0];
    let observed = builder
        .add_instruction(
            RegisterOperation::Read(ReferenceReadOperation::new().with_transforms(vec![RegisterTransform::Bit])),
            Vec::new(),
            vec![allocation, index],
            None,
        )
        .unwrap()[0];
    let frozen = builder
        .add_instruction(RegisterOperation::Freeze(ReferenceFreezeOperation::new()), Vec::new(), vec![allocation], None)
        .unwrap()[0];
    let source = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(
            vec![previous, observed, frozen],
            vec![Placeholder; 3],
            vec![Placeholder; 3],
        )
        .unwrap();

    // Eager execution accesses the bit through the transform path: `r = 0b101`, bit 1 was `0`, becomes `1`, and the
    // register ends at `0b111`. Discharge into the eager destination reaches the same values through the value-bound
    // alias, whose bit transform the policy reads and writes with the family's bit operations.
    let inputs = vec![RegisterValue::Register(5), RegisterValue::Register(1), RegisterValue::Register(1)];
    let expected = vec![RegisterValue::Register(0), RegisterValue::Register(1), RegisterValue::Register(7)];
    assert_eq!(source.interpret(inputs.clone()), Ok(expected.clone()));
    let context = RegisterDischargeContext::new(RegisterDestination::new());
    let regions = [source.clone()];
    let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
    let carriers = inputs.into_iter().map(RegisterDischargeValue::Value).collect::<Vec<_>>();
    assert_eq!(
        driver.inline_region(&context, 0, carriers),
        Ok(expected.into_iter().map(RegisterDischargeValue::Value).collect::<Vec<_>>()),
    );
    assert_eq!(context.live_allocation_ids(), Vec::new());

    // Against a staging destination the alias stages the same bit operations: the swap is a read followed by a write
    // of the bit (the policy's default), and the read after it extracts the bit of the updated state.
    let discharge = |inputs: Vec<Tracer<TracingContext<RegisterValue, RegisterOperation>>>| {
        let context = ReferenceDischargeContext::new(inputs[0].context().clone());
        let carriers = inputs.into_iter().map(ReferenceDischargeValue::Value).collect::<Vec<_>>();
        let regions = [source.clone()];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let outputs = driver.inline_region(&context, 0, carriers)?;
        outputs
            .iter()
            .map(|output| output.try_as_value("a discharged output").cloned())
            .collect::<Result<Vec<_>, _>>()
    };
    let (_, discharged): (_, Program<_, _, Vec<RegisterValue>, Vec<RegisterValue>>) =
        EagerContext::<RegisterValue, RegisterOperation>::trace(
            discharge,
            vec![RegisterIrType::Register(RegisterType); 3],
        )
        .unwrap();
    assert_eq!(
        discharged.to_string(),
        indoc! {"
            lambda %0:register, %1:register, %2:register .
            let %3:register = register.bit_extract %0 %1
                %4:register = register.bit_insert %0 %2 %1
                %5:register = register.bit_extract %4 %1
            in (%3, %5, %4)"},
    );

    // Preserving the allocation keeps the same root and bindings on the copied access operations.
    let preserved = source.partially_discharge_references(0, &[]).unwrap();
    assert_eq!(
        preserved.program().to_string(),
        indoc! {"
            lambda %0:register, %1:register, %2:register .
            let %3:ref<register> = reference_new %0
                %4:register = reference_swap [transforms=[bit]] %3 %2 %1
                %5:register = reference_read [transforms=[bit]] %3 %1
                %6:register = reference_freeze %3
            in (%4, %5, %6)"},
    );
}

#[test]
fn test_downstream_reference_add_update_stages_the_family_operation() {
    // The generic value capability selects the downstream operation for a core-owned tracer, while eager replay
    // executes the register operation's own addition. Neither path requires a canonical array accumulation operation.
    let (_, program): (_, Program<_, _, (RegisterValue, RegisterValue), RegisterValue>) = RegisterDestination::trace(
        |(reference, update): (RegisterTracer, RegisterTracer)| {
            reference.add_update(&update)?;
            Ok(update)
        },
        (RegisterIrType::Reference(ReferenceType::new(RegisterType)), RegisterIrType::Register(RegisterType)),
    )
    .unwrap();
    assert_eq!(
        program.to_string(),
        indoc! {"
            lambda %0:ref<register>, %1:register .
            let () = register.add_update %0 %1
            in (%1)"},
    );
    let reference = Reference::new(RegisterValue::Register(4)).unwrap();
    assert_eq!(
        program.interpret((RegisterValue::Reference(reference.clone()), RegisterValue::Register(3))),
        Ok(RegisterValue::Register(3)),
    );
    assert_eq!(reference.read(), Ok(RegisterValue::Register(7)));
}

#[test]
fn test_downstream_reference_universe_jvp_through_the_public_boundary() {
    // Forward mode pairs the live register reference with the caller's tangent reference and mutates both in program
    // order: `r = 1 + 3` and `ṫ = 5 + 2`, with the read returning the updated contents of each.
    let reference = Reference::new(RegisterValue::Register(1)).unwrap();
    let tangent_reference = Reference::new(RegisterValue::Register(5)).unwrap();
    assert_eq!(
        differentiate_at((RegisterValue::Reference(reference.clone()), RegisterValue::Register(3)))
            .jvp((RegisterValue::Reference(tangent_reference.clone()), RegisterValue::Register(2)), read_modify_write,),
        Ok((RegisterValue::Register(4), RegisterValue::Register(7))),
    );
    assert_eq!(reference.read(), Ok(RegisterValue::Register(4)));
    assert_eq!(tangent_reference.read(), Ok(RegisterValue::Register(7)));

    // The canonical boundary validator runs over the register values: a tangent reference aliasing the primal one is
    // rejected before anything is mutated.
    let reference = Reference::new(RegisterValue::Register(1)).unwrap();
    assert!(matches!(
        differentiate_at((RegisterValue::Reference(reference.clone()), RegisterValue::Register(3)))
            .jvp((RegisterValue::Reference(reference.clone()), RegisterValue::Register(2)), read_modify_write,),
        Err(DifferentiationError::Program(ProgramError::InvalidArgument { .. })),
    ));
    assert_eq!(reference.read(), Ok(RegisterValue::Register(1)));
}

#[test]
fn test_downstream_reference_universe_vjp_through_the_public_boundary() {
    // Reverse mode linearizes the closure over the register family, transposes the linear program under the caller's
    // destinations, and replays the transposed program eagerly through the family's own interpretation. The
    // destination holds the cotangent of the reference's post-execution state on entry (`10`) and the cotangent of its
    // pre-execution state on return: the read accumulates `ȳ = 2` into it and the update's cotangent reads it back, so
    // `x̄ = 12` and the destination ends at `12`.
    let reference = Reference::new(RegisterValue::Register(1)).unwrap();
    let (value, pullback) = differentiate_at((RegisterValue::Reference(reference.clone()), RegisterValue::Register(3)))
        .vjp(read_modify_write)
        .unwrap();
    assert_eq!(value, RegisterValue::Register(4));
    assert_eq!(reference.read(), Ok(RegisterValue::Register(4)));
    let destination = Reference::new(RegisterValue::Register(10)).unwrap();
    assert_eq!(
        pullback.apply_with_destinations(
            CotangentSeed::Value(RegisterValue::Register(2)),
            (
                CotangentDestination::Reference(RegisterValue::Reference(destination.clone())),
                CotangentDestination::Return
            ),
        ),
        Ok((None, Some(RegisterValue::Register(12)))),
    );
    assert_eq!(destination.read(), Ok(RegisterValue::Register(12)));

    // Ignoring the initial reference cotangent still allocates state for the cotangent of the stored value.
    assert_eq!(
        pullback.apply_with_destinations(
            CotangentSeed::Value(RegisterValue::Register(2)),
            (CotangentDestination::Ignore, CotangentDestination::Return),
        ),
        Ok((None, Some(RegisterValue::Register(2)))),
    );
}

#[test]
fn test_downstream_reference_universe_vjp_with_a_local_allocation() {
    let (value, pullback) = differentiate_at(RegisterValue::Register(3))
        .vjp(|initial| {
            let context = initial.context();
            let reference = bind_register_output(
                context,
                RegisterOperation::ReferenceNew(ReferenceNewOperation::new()),
                std::slice::from_ref(&initial),
            )?;
            bind_register_output(context, RegisterOperation::Freeze(ReferenceFreezeOperation::new()), &[reference])
        })
        .unwrap();
    assert_eq!(value, RegisterValue::Register(3));
    assert_eq!(pullback.apply(RegisterValue::Register(7)), Ok(RegisterValue::Register(7)));
}

#[test]
fn test_downstream_reference_operation_providers_support_value_only_composite_families() {
    // A downstream family can use the core composite type without supporting reference operations. Owning the
    // providers on the operation family lets it opt into ordinary gradients without an orphan-rule conflict.
    /// Ordinary array primitives needed by elementwise differentiation and its shape alignment rules.
    #[derive(Clone, Debug, ryft_macros::Operation)]
    #[ryft(crate = "ryft_core", type = ArrayType, constant = Array, dispatch(differentiation, transposition))]
    enum ValueOnlyArrayOperation {
        Constant(ConstantOperation<Array>),
        Zero(ZeroOperation<ArrayType>),
        One(OneOperation<ArrayType>),
        ZeroLike(ZeroLikeOperation<ArrayType>),
        OneLike(OneLikeOperation<ArrayType>),
        Neg(NegOperation<ArrayType>),
        Add(AddOperation<ArrayType>),
        Sub(SubOperation<ArrayType>),
        Exp(ExpOperation<ArrayType>),
        Mul(MulOperation<ArrayType>),
        ConvertElementType(ConvertElementTypeOperation<ArrayType>),
        Broadcast(BroadcastOperation),
        Transpose(TransposeOperation),
        Reshape(ReshapeOperation),
        Reduce(ReduceOperation),
        Reshard(ReshardOperation),
        Compare(CompareOperation<ArrayType>),
        Div(DivOperation<ArrayType>),
    }

    impl OperationProvider<ArrayType, ParallelVaryOperation> for ValueOnlyArrayOperation {
        type Operation = Self;

        fn provide(_request: ParallelVaryOperation, input_types: &[&ArrayType]) -> Result<Self, ProgramError> {
            check_count!("input", input_types, 1, ProgramError);
            Err(ProgramError::UnsupportedOperation {
                message: "value-only test operation family cannot align manual variation".to_string(),
            })
        }
    }

    /// A downstream composite family containing only constants and ordinary array operations.
    #[derive(Clone, Debug, ryft_macros::Operation)]
    #[ryft(
        crate = "ryft_core",
        type = ArrayIrType,
        constant = ArrayIrValue<Array>,
        members(ArrayType),
        dispatch(differentiation, transposition),
    )]
    enum ValueOnlyOperation {
        Constant(ConstantOperation<ArrayIrValue<Array>>),
        Zero(ZeroOperation<ArrayIrType>),
        One(OneOperation<ArrayIrType>),
        #[ryft(projected(ArrayType))]
        Array(ValueOnlyArrayOperation),
    }

    impl ReferenceAccessOperation for ValueOnlyOperation {
        type Transform = NoReferenceTransform<ArrayType, ArrayIrType>;

        fn base_input_count(&self) -> usize {
            0
        }

        fn reference_access_descriptor(
            &self,
            _input_index: usize,
        ) -> Option<ReferenceAccessDescriptor<'_, Self::Transform>> {
            None
        }

        fn with_reference_access_transforms(
            &self,
            _input_index: usize,
            _transforms: Vec<Self::Transform>,
        ) -> Result<Self, ProgramError> {
            Err(ProgramError::UnsupportedOperation {
                message: "value-only operations have no reference accesses".to_string(),
            })
        }
    }

    // The member rules stay entirely within ordinary arrays, so the standard projection adapters suffice.
    impl<C> MemberDifferentiableOperation<C> for ValueOnlyArrayOperation
    where
        C: Context<
                Type = ArrayIrType,
                Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
                Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
                Operation = ValueOnlyOperation,
            >,
        Self: DifferentiableOperation<ProjectedContext<C, ArrayType>>,
    {
        fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
            &self,
            context: &DifferentiationContext<C, P>,
            _driver: &D,
            inputs: &[DifferentiationDual<C::Value>],
        ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
            jvp_projected_operation(context, self, inputs)
        }
    }

    impl MemberTransposableOperation<ArrayIrValue<Array>, ValueOnlyOperation> for ValueOnlyArrayOperation {
        fn transpose_in_parent<D: TranspositionDriver<ArrayIrValue<Array>, ValueOnlyOperation>>(
            &self,
            context: &mut TranspositionContext<ArrayIrValue<Array>, ValueOnlyOperation>,
            _driver: &D,
            inputs: &[PartialValue<Tracer<TracingContext<ArrayIrValue<Array>, ValueOnlyOperation>>>],
            outputs: &[MaybeZero<Tracer<TracingContext<ArrayIrValue<Array>, ValueOnlyOperation>>>],
            accumulators: &[CotangentAccumulator],
        ) -> Result<(), DifferentiationError> {
            transpose_projected_operation(context, self, inputs, outputs, accumulators)
        }
    }

    // This fixture uses static shapes, whose residual zeros need no runtime dimensions.
    impl ResidualZeroProvider<ArrayIrType> for ValueOnlyOperation {}

    impl From<AddOperation<ArrayIrType>> for ValueOnlyOperation {
        fn from(_operation: AddOperation<ArrayIrType>) -> Self {
            Self::Array(ValueOnlyArrayOperation::Add(AddOperation::new()))
        }
    }

    // The family converts from no reference payload, so the array-IR blankets do not apply and it states its own
    // answer for the array referent family.
    impl OperationProvider<ArrayIrType, ReferenceNewOperation<ArrayType, ArrayIrType>> for ValueOnlyOperation {
        type Operation = Self;

        fn provide(
            _request: ReferenceNewOperation<ArrayType, ArrayIrType>,
            input_types: &[&ArrayIrType],
        ) -> Result<Self, ProgramError> {
            check_count!("input", input_types, 1, ProgramError);
            Err(ProgramError::UnsupportedOperation {
                message: "this operation family does not support reference allocation".to_string(),
            })
        }
    }

    impl OperationProvider<ArrayIrType, ReferenceAddUpdateOperation<ArrayType, ArrayIrType>> for ValueOnlyOperation {
        type Operation = Self;

        fn provide(
            _request: ReferenceAddUpdateOperation<ArrayType, ArrayIrType>,
            input_types: &[&ArrayIrType],
        ) -> Result<Self, ProgramError> {
            check_count!("input", input_types, 2, ProgramError);
            Err(ProgramError::UnsupportedOperation {
                message: "this operation family does not support reference accumulation".to_string(),
            })
        }
    }

    impl OperationProvider<ArrayIrType, ReferenceFreezeOperation<ArrayType, ArrayIrType>> for ValueOnlyOperation {
        type Operation = Self;

        fn provide(
            _request: ReferenceFreezeOperation<ArrayType, ArrayIrType>,
            input_types: &[&ArrayIrType],
        ) -> Result<Self, ProgramError> {
            check_count!("input", input_types, 1, ProgramError);
            Err(ProgramError::UnsupportedOperation {
                message: "this operation family does not support reference freezing".to_string(),
            })
        }
    }

    let scalar_type = ArrayType::scalar(DataType::F32);
    let primal = ArrayIrValue::Array(Array::from_elements(scalar_type.clone(), &[3.0_f32]).unwrap());
    let context = EagerContext::<ArrayIrValue<Array>, ValueOnlyOperation>::new();
    assert_eq!(
        differentiate_at(primal).in_context(&context).value_and_gradient(|value| {
            let operation = ValueOnlyOperation::Array(ValueOnlyArrayOperation::Add(AddOperation::new()));
            let outputs = value.context().bind(operation, Vec::new(), &[value.clone(), value.clone()])?;
            Ok::<_, ProgramError>(outputs.into_iter().next().unwrap())
        }),
        Ok((
            ArrayIrValue::Array(Array::from_elements(scalar_type.clone(), &[6.0_f32]).unwrap()),
            ArrayIrValue::Array(Array::from_elements(scalar_type, &[2.0_f32]).unwrap()),
        )),
    );

    // Unsupported constructors are fallible; ordinary gradients do not call them for value-only inputs.
    assert!(matches!(
        ValueOnlyOperation::provide(ReferenceNewOperation::new(), &[&ArrayIrType::Array(ArrayType::scalar(DataType::F32))]),
        Err(ProgramError::UnsupportedOperation { message, .. })
            if message == "this operation family does not support reference allocation",
    ));
    assert!(matches!(
        ValueOnlyOperation::provide(ReferenceAddUpdateOperation::new(), &[&ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))), &ArrayIrType::Array(ArrayType::scalar(DataType::F32))]),
        Err(ProgramError::UnsupportedOperation { message, .. })
            if message == "this operation family does not support reference accumulation",
    ));
    assert!(matches!(
        ValueOnlyOperation::provide(ReferenceFreezeOperation::new(), &[&ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)))]),
        Err(ProgramError::UnsupportedOperation { message, .. })
            if message == "this operation family does not support reference freezing",
    ));
}

#[test]
fn test_downstream_reference_universe_value_and_gradient() {
    // The composite parameter is one leaf regardless of its member variant. A reference input therefore reconstructs
    // as an ordinary register cotangent, beside the ordinary input's cotangent, while primal mutation remains visible.
    let reference = Reference::new(RegisterValue::Register(1)).unwrap();
    assert_eq!(
        differentiate_at((RegisterValue::Reference(reference.clone()), RegisterValue::Register(3)))
            .value_and_gradient(read_modify_write),
        Ok((RegisterValue::Register(4), (RegisterValue::Register(1), RegisterValue::Register(1)))),
    );
    assert_eq!(reference.read(), Ok(RegisterValue::Register(4)));
}

#[test]
fn test_downstream_reference_universe_batch_through_the_public_boundary() {
    // Registers have no axes, so the family's batching policy is replicated-only: the closure runs once over the
    // packed values with an explicit extent and every output stays replicated, while a mapped input is rejected by
    // the policy before any rule runs.
    let reference = Reference::new(RegisterValue::Register(1)).unwrap();
    assert_eq!(
        batch(
            read_modify_write,
            (RegisterValue::Reference(reference.clone()), RegisterValue::Register(3)),
            BatchAxis::replicated(),
            BatchAxis::replicated(),
            BatchAxisSpecification::with_extent(4),
        ),
        Ok(RegisterValue::Register(4)),
    );
    assert_eq!(reference.read(), Ok(RegisterValue::Register(4)));
    assert!(matches!(
        batch(
            read_modify_write,
            (RegisterValue::Reference(reference.clone()), RegisterValue::Register(3)),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            BatchAxis::replicated(),
            BatchAxisSpecification::with_extent(4),
        ),
        Err(BatchingError::UnsupportedOperation { message })
            if message == "register values have no axes, so a value of type `register` cannot be mapped",
    ));
    assert_eq!(reference.read(), Ok(RegisterValue::Register(4)));
}

#[test]
fn test_downstream_folded_dynamic_view_jvp() {
    let reference = Reference::new(RegisterValue::Register(1)).unwrap();
    let tangent = Reference::new(RegisterValue::Register(8)).unwrap();
    assert_eq!(
        differentiate_at((
            RegisterValue::Reference(reference.clone()),
            RegisterValue::Register(1),
            RegisterValue::Register(2),
        ))
        .jvp(
            (RegisterValue::Reference(tangent.clone()), RegisterValue::Register(1), RegisterValue::Register(0),),
            write_read_folded_bit,
        ),
        Ok((RegisterValue::Register(1), RegisterValue::Register(1))),
    );
    assert_eq!(reference.read(), Ok(RegisterValue::Register(5)));
    assert_eq!(tangent.read(), Ok(RegisterValue::Register(12)));
}

#[test]
fn test_downstream_folded_dynamic_view_vjp() {
    let reference = Reference::new(RegisterValue::Register(1)).unwrap();
    let (value, pullback) = differentiate_at((
        RegisterValue::Reference(reference.clone()),
        RegisterValue::Register(1),
        RegisterValue::Register(2),
    ))
    .vjp(write_read_folded_bit)
    .unwrap();
    assert_eq!(value, RegisterValue::Register(1));
    assert_eq!(reference.read(), Ok(RegisterValue::Register(5)));
    let transposed = pullback
        .linear_program()
        .transpose_with_respect_to(
            &[0, 1, 2],
            &[CotangentDestinationKind::Reference, CotangentDestinationKind::Return, CotangentDestinationKind::Return],
        )
        .unwrap();
    assert_eq!(
        transposed
            .instructions()
            .iter()
            .map(|instruction| instruction.operation().name())
            .collect::<Vec<_>>(),
        vec!["register.add_update", "register.zero", "reference_swap", "register.zero"],
    );
    assert_eq!(
        transposed.instructions()[0].operation().reference_access_descriptor(0).unwrap().transforms(),
        &[RegisterTransform::Bit]
    );
    assert_eq!(
        transposed.instructions()[2].operation().reference_access_descriptor(0).unwrap().transforms(),
        &[RegisterTransform::Bit]
    );
    let destination = Reference::new(RegisterValue::Register(8)).unwrap();
    assert_eq!(
        pullback.apply_with_destinations(
            CotangentSeed::Value(RegisterValue::Register(1)),
            (
                CotangentDestination::Reference(RegisterValue::Reference(destination.clone())),
                CotangentDestination::Return,
                CotangentDestination::Return,
            ),
        ),
        Ok((None, Some(RegisterValue::Register(1)), Some(RegisterValue::Register(0)))),
    );
    assert_eq!(destination.read(), Ok(RegisterValue::Register(8)));
}

#[test]
fn test_downstream_folded_dynamic_view_batch() {
    // Replicated batching retains the dynamic binding beside the root on each folded access.
    let reference = Reference::new(RegisterValue::Register(1)).unwrap();
    assert_eq!(
        batch(
            write_read_folded_bit,
            (RegisterValue::Reference(reference.clone()), RegisterValue::Register(1), RegisterValue::Register(2)),
            BatchAxis::replicated(),
            BatchAxis::replicated(),
            BatchAxisSpecification::with_extent(4),
        ),
        Ok(RegisterValue::Register(1)),
    );
    assert_eq!(reference.read(), Ok(RegisterValue::Register(5)));
}

#[test]
fn test_downstream_value_reports_reference_identity_for_live_handles() {
    // A register value holds no allocation, while a live handle reports the identity of the allocation it denotes, so
    // the canonical boundary validator accepts distinct positions and rejects one allocation bound twice.
    let reference = Reference::new(RegisterValue::Register(3)).unwrap();
    assert_eq!(RegisterValue::Register(3).reference_id(), None);
    assert_eq!(RegisterValue::Reference(reference.clone()).reference_id(), Some(reference.id()));
    assert_eq!(
        validate_reference_boundary(
            [RegisterValue::Register(1), RegisterValue::Reference(reference.clone())].iter(),
            std::iter::empty(),
        ),
        Ok(()),
    );
    assert!(matches!(
        validate_reference_boundary(
            [RegisterValue::Reference(reference.clone()), RegisterValue::Reference(reference)].iter(),
            std::iter::empty(),
        ),
        Err(ReferenceBoundaryError::Aliased { .. }),
    ));
}

#[test]
fn test_downstream_reference_boundary_accepts_owned_positions() {
    /// Backend-defined port name, without `Copy`, `Default`, or `Display` requirements.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Port(String);

    let context = RegisterDestination::new();
    let source = RegisterValue::Reference(Reference::new(RegisterValue::Register(3)).unwrap());
    let cache = RegisterValue::Reference(Reference::new(RegisterValue::Register(5)).unwrap());
    let result = RegisterValue::Reference(Reference::new(RegisterValue::Register(7)).unwrap());
    let source_port = Port("source".to_string());
    let cache_port = Port("cache".to_string());
    let result_port = Port("result".to_string());
    let boundary =
        ReferenceBoundary::new(&context, [(source_port.clone(), &source), (cache_port.clone(), &cache)]).unwrap();

    assert_eq!(boundary.validate(&context, [(result_port.clone(), &result)]), Ok(()));
    assert_eq!(
        ReferenceBoundary::new(&context, [(source_port.clone(), &source), (cache_port.clone(), &source)]).unwrap_err(),
        ReferenceBoundaryError::Aliased { position: cache_port.clone(), other: source_port.clone() },
    );
    assert_eq!(
        boundary.validate(&context, [(result_port.clone(), &source)]),
        Err(ReferenceBoundaryError::AliasedRetained { position: result_port.clone(), other: source_port }),
    );
    assert_eq!(
        boundary.validate(&context, [(result_port.clone(), &result), (cache_port.clone(), &result)]),
        Err(ReferenceBoundaryError::Aliased { position: cache_port, other: result_port.clone() }),
    );
    // Validation checks later arguments without adding them to the retained boundary.
    assert_eq!(boundary.validate(&context, [(result_port, &result)]), Ok(()));
}

#[test]
fn test_downstream_lazy_bit_view_accesses_one_bit_of_its_root() {
    let root = Reference::new(RegisterValue::Register(5)).unwrap();
    let viewed =
        ReferenceView::<_, RegisterTransform, RegisterValue>::new(RegisterValue::Reference(root.clone())).unwrap();
    let bit = viewed
        .clone()
        .with_bound_transform(RegisterTransform::Bit, vec![RegisterValue::Register(1)])
        .unwrap();
    assert_eq!(bit.root().reference_id(), Some(root.id()));
    assert_eq!(bit.read(), Ok(RegisterValue::Register(0)));
    assert_eq!(bit.write(&RegisterValue::Register(1)), Ok(()));
    assert_eq!(root.read(), Ok(RegisterValue::Register(7)));
    assert_eq!(bit.swap(&RegisterValue::Register(0)), Ok(RegisterValue::Register(1)));
    assert_eq!(root.read(), Ok(RegisterValue::Register(5)));
    assert_eq!(bit.add_update(&RegisterValue::Register(1)), Ok(()));
    assert_eq!(root.read(), Ok(RegisterValue::Register(7)));
    assert!(matches!(bit.write(&RegisterValue::Register(2)), Err(ProgramError::InvalidArgument { message })
        if message == "a register bit holds 0 or 1 but 2 was stored into one"));
    let invalid = viewed.with_bound_transform(RegisterTransform::Bit, vec![RegisterValue::Register(64)]).unwrap();
    assert!(matches!(invalid.read(), Err(ProgramError::InvalidArgument { message })
        if message == "bit index 64 is out of range for a 64-bit register"));
    assert_eq!(root.read(), Ok(RegisterValue::Register(7)));
}

/// A downstream operation family that keeps the existing array type universe while owning its transform metadata.
mod custom_array_transforms {
    use pretty_assertions::assert_eq;

    use super::*;
    use ryft_core::{
        ArrayIrOperation, ArrayReferenceDischarge, ArrayReferenceTransform, ArrayReferenceTransformIndex,
        ArrayReferenceTransformOperation, ArrayReferenceTransformPath, DynamicSliceOperation,
        DynamicUpdateSliceOperation, ReferenceAccumulationPolicy, SliceOperation, UpdateSliceOperation,
    };

    type ArrayValue = ArrayIrValue<Array>;
    type CoreOperation = ArrayIrOperation<Array>;
    type ArrayContext = TracingContext<ArrayValue, CustomOperation>;
    type ArrayTracer = Tracer<ArrayContext>;

    /// Distinct downstream metadata over the unmodified core array type universe.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct CustomTransform(ArrayReferenceTransform);

    impl Display for CustomTransform {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            Display::fmt(&self.0, formatter)
        }
    }

    impl ReferenceTransform for CustomTransform {
        type Type = ArrayIrType;
        type Referent = ArrayType;
        fn binding_count(&self) -> usize {
            self.0.binding_count()
        }
        fn validate_bindings(&self, input: &ArrayType, bindings: &[&ArrayIrType]) -> Result<(), TypeError> {
            self.0.validate_bindings(input, bindings)
        }
        fn output_type(&self, input: &ArrayType) -> Result<ArrayType, TypeError> {
            self.0.output_type(input)
        }
        fn overlap(
            _type: &ArrayIrType,
            lhs: &[BoundReferenceTransform<Self>],
            rhs: &[BoundReferenceTransform<Self>],
        ) -> ReferenceViewOverlap {
            if lhs == rhs { ReferenceViewOverlap::Same } else { ReferenceViewOverlap::MayOverlap }
        }
    }

    impl BatchableReferenceTransform for CustomTransform {
        fn batch(&self, r#type: &ArrayIrType, axis: BatchAxis) -> Result<(Self, BatchAxis), BatchingError> {
            self.0.batch(r#type, axis).map(|(transform, axis)| (Self(transform), axis))
        }
    }

    /// Core pure operations and allocation lifecycle with downstream-owned accesses.
    #[derive(Clone, Debug)]
    enum CustomOperation {
        Core(CoreOperation),
        Read(ReferenceReadOperation<ArrayType, ArrayIrType, CustomTransform>),
        AddUpdate(ReferenceAddUpdateOperation<ArrayType, ArrayIrType, CustomTransform>),
    }

    impl CustomOperation {
        /// Converts metadata only at the eager execution boundary; staged accesses retain the downstream transform
        /// type.
        fn builtin(&self) -> CoreOperation {
            match self {
                Self::Core(operation) => operation.clone(),
                Self::Read(operation) => CoreOperation::ReferenceRead(
                    ReferenceReadOperation::new()
                        .with_transforms(operation.transforms().iter().map(|transform| transform.0.clone()).collect()),
                ),
                Self::AddUpdate(operation) => CoreOperation::ReferenceAddUpdate(
                    ReferenceAddUpdateOperation::new()
                        .with_transforms(operation.transforms().iter().map(|transform| transform.0.clone()).collect()),
                ),
            }
        }
    }

    impl From<ReferenceReadOperation<ArrayType, ArrayIrType, CustomTransform>> for CustomOperation {
        fn from(operation: ReferenceReadOperation<ArrayType, ArrayIrType, CustomTransform>) -> Self {
            Self::Read(operation)
        }
    }
    impl From<ReferenceAddUpdateOperation<ArrayType, ArrayIrType, CustomTransform>> for CustomOperation {
        fn from(operation: ReferenceAddUpdateOperation<ArrayType, ArrayIrType, CustomTransform>) -> Self {
            Self::AddUpdate(operation)
        }
    }
    impl From<ReferenceNewOperation<ArrayType, ArrayIrType>> for CustomOperation {
        fn from(operation: ReferenceNewOperation<ArrayType, ArrayIrType>) -> Self {
            Self::Core(operation.into())
        }
    }
    impl From<ReferenceFreezeOperation<ArrayType, ArrayIrType>> for CustomOperation {
        fn from(operation: ReferenceFreezeOperation<ArrayType, ArrayIrType>) -> Self {
            Self::Core(operation.into())
        }
    }
    impl From<AddOperation<ArrayIrType>> for CustomOperation {
        fn from(operation: AddOperation<ArrayIrType>) -> Self {
            Self::Core(operation.into())
        }
    }
    impl OperationProvider<ArrayIrType, ZeroOperation<ArrayIrType>> for CustomOperation {
        type Operation = Self;
        fn provide(request: ZeroOperation<ArrayIrType>, inputs: &[&ArrayIrType]) -> Result<Self, ProgramError> {
            CoreOperation::provide(request, inputs).map(Self::Core)
        }
    }
    impl ResidualZeroProvider<ArrayIrType> for CustomOperation {}

    impl Operation for CustomOperation {
        type Type = ArrayIrType;
        fn name(&self) -> &'static str {
            match self {
                Self::Core(operation) => operation.name(),
                Self::Read(operation) => operation.name(),
                Self::AddUpdate(operation) => operation.name(),
            }
        }
        fn infer_output_types(
            &self,
            inputs: &[ArrayIrType],
            regions: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            match self {
                Self::Core(operation) => operation.infer_output_types(inputs, regions),
                Self::Read(operation) => operation.infer_output_types(inputs, regions),
                Self::AddUpdate(operation) => operation.infer_output_types(inputs, regions),
            }
        }
        fn effects(&self) -> Cow<'_, Effects> {
            match self {
                Self::Core(operation) => operation.effects(),
                Self::Read(operation) => operation.effects(),
                Self::AddUpdate(operation) => operation.effects(),
            }
        }
        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            match self {
                Self::Core(operation) => operation.render(formatter, indentation),
                Self::Read(operation) => operation.render(formatter, indentation),
                Self::AddUpdate(operation) => operation.render(formatter, indentation),
            }
        }
    }
    impl Display for CustomOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.render(formatter, 0)
        }
    }
    impl ReferenceAccessOperation for CustomOperation {
        type Transform = CustomTransform;
        fn base_input_count(&self) -> usize {
            match self {
                Self::Core(operation) => operation.base_input_count(),
                Self::Read(operation) => operation.base_input_count(),
                Self::AddUpdate(operation) => operation.base_input_count(),
            }
        }
        fn reference_access_descriptor(&self, input: usize) -> Option<ReferenceAccessDescriptor<'_, CustomTransform>> {
            match self {
                Self::Read(operation) => operation.reference_access_descriptor(input),
                Self::AddUpdate(operation) => operation.reference_access_descriptor(input),
                Self::Core(CoreOperation::ReferenceFreeze(_)) if input == 0 => {
                    Some(ReferenceAccessDescriptor::new(&[], 1..1))
                }
                _ => None,
            }
        }
        fn with_reference_access_transforms(
            &self,
            input: usize,
            transforms: Vec<CustomTransform>,
        ) -> Result<Self, ProgramError> {
            match self {
                Self::Read(operation) => operation.with_reference_access_transforms(input, transforms).map(Self::Read),
                Self::AddUpdate(operation) => {
                    operation.with_reference_access_transforms(input, transforms).map(Self::AddUpdate)
                }
                _ if self.reference_access_descriptor(input).is_some() && transforms.is_empty() => Ok(self.clone()),
                _ => Err(ProgramError::UnsupportedOperation {
                    message: "custom operation cannot replace these transforms".to_owned(),
                }),
            }
        }
    }
    impl ArrayReferenceTransformOperation for CustomOperation {
        fn from_reference_reshape(operation: ReshapeOperation) -> Self {
            Self::Core(CoreOperation::from_reference_reshape(operation))
        }
        fn from_reference_slice(operation: SliceOperation) -> Self {
            Self::Core(CoreOperation::from_reference_slice(operation))
        }
        fn from_reference_update_slice(operation: UpdateSliceOperation) -> Self {
            Self::Core(CoreOperation::from_reference_update_slice(operation))
        }
        fn from_reference_dynamic_slice(operation: DynamicSliceOperation) -> Self {
            Self::Core(CoreOperation::from_reference_dynamic_slice(operation))
        }
        fn from_reference_dynamic_update_slice(operation: DynamicUpdateSliceOperation) -> Self {
            Self::Core(CoreOperation::from_reference_dynamic_update_slice(operation))
        }
    }
    impl<C: Domain<Type = ArrayIrType>> InterpretableOperation<C> for CustomOperation
    where
        CoreOperation: InterpretableOperation<C>,
    {
        fn interpret<D: InterpretationDriver<C>>(
            &self,
            context: &C,
            driver: &D,
            inputs: &[C::Value],
        ) -> Result<Vec<C::Value>, ProgramError> {
            self.builtin().interpret(context, driver, inputs)
        }
    }
    impl<C: Context<Type = ArrayIrType>> PartiallyEvaluatableOperation<C> for CustomOperation where
        C: Context<Operation = CustomOperation>
    {
    }

    /// An explicit downstream policy, distinct from ArrayIrType's canonical policy.
    #[derive(Copy, Clone, Debug)]
    struct CustomPolicy;
    impl<C: Context<Type = ArrayIrType, Operation = CustomOperation>> ReferenceDischargePolicy<C> for CustomPolicy {
        type Referent = ArrayType;
        type Transform = CustomTransform;
        type Alias = ArrayReferenceTransformPath<C::Value>;
        fn apply_transforms(
            context: &C,
            alias: &Self::Alias,
            transforms: &[CustomTransform],
            bindings: &[C::Value],
        ) -> Result<Self::Alias, ProgramError> {
            ArrayReferenceDischarge::apply_transforms(
                context,
                alias,
                &transforms.iter().map(|transform| transform.0.clone()).collect::<Vec<_>>(),
                bindings,
            )
        }
        fn storage_alias(_referent: &ArrayType) -> Self::Alias {
            ArrayReferenceTransformPath::root()
        }
        fn read(context: &C, current: &C::Value, alias: &Self::Alias) -> Result<C::Value, ProgramError> {
            ArrayReferenceDischarge::read(context, current, alias)
        }
        fn write(
            context: &C,
            current: &C::Value,
            replacement: C::Value,
            alias: &Self::Alias,
        ) -> Result<C::Value, ProgramError> {
            ArrayReferenceDischarge::write(context, current, replacement, alias)
        }
    }
    impl<C: Context<Type = ArrayIrType, Operation = CustomOperation>> ReferenceAccumulationPolicy<C> for CustomPolicy {
        fn accumulate(
            context: &C,
            current: &C::Value,
            update: C::Value,
            alias: &Self::Alias,
        ) -> Result<C::Value, ProgramError> {
            ArrayReferenceDischarge::accumulate(context, current, update, alias)
        }
    }
    impl<C: Context<Type = ArrayIrType, Operation = CustomOperation>> ReferenceDischargeableOperation<C, CustomPolicy>
        for CustomOperation
    {
        fn discharge_references<D: ReferenceDischargeDriver<C, CustomPolicy>>(
            &self,
            context: &ReferenceDischargeContext<C, CustomPolicy>,
            driver: &D,
            inputs: &[ReferenceDischargeValue<C, CustomPolicy>],
        ) -> Result<Vec<ReferenceDischargeValue<C, CustomPolicy>>, ProgramError> {
            match self {
                Self::Read(operation) => operation.discharge_references(context, driver, inputs),
                Self::AddUpdate(operation) => operation.discharge_references(context, driver, inputs),
                Self::Core(CoreOperation::ReferenceNew(operation)) => {
                    operation.discharge_references(context, driver, inputs)
                }
                Self::Core(CoreOperation::ReferenceFreeze(operation)) => {
                    operation.discharge_references(context, driver, inputs)
                }
                Self::Core(_) => discharge_reference_free_operation(self, context, driver, inputs),
            }
        }
    }
    impl<C: Context<Type = ArrayIrType, Operation = CustomOperation> + Zero<C::Value>> DifferentiableOperation<C>
        for CustomOperation
    {
        fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
            &self,
            context: &DifferentiationContext<C, P>,
            driver: &D,
            inputs: &[DifferentiationDual<C::Value>],
        ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
            match self {
                Self::Read(operation) => operation.jvp(context, driver, inputs),
                Self::AddUpdate(operation) => operation.jvp(context, driver, inputs),
                Self::Core(CoreOperation::ReferenceNew(operation)) => operation.jvp(context, driver, inputs),
                Self::Core(CoreOperation::ReferenceFreeze(operation)) => operation.jvp(context, driver, inputs),
                _ => Err(ProgramError::UnsupportedOperation {
                    message: "custom fixture only differentiates reference primitives".to_owned(),
                }
                .into()),
            }
        }
    }
    impl TransposableOperation<ArrayValue, CustomOperation> for CustomOperation {
        fn transpose<D: TranspositionDriver<ArrayValue, CustomOperation>>(
            &self,
            context: &mut TranspositionContext<ArrayValue, CustomOperation>,
            driver: &D,
            inputs: &[PartialValue<ArrayTracer>],
            outputs: &[MaybeZero<ArrayTracer>],
            accumulators: &[CotangentAccumulator],
        ) -> Result<(), DifferentiationError> {
            match self {
                Self::Read(operation) => operation.transpose(context, driver, inputs, outputs, accumulators),
                Self::AddUpdate(operation) => operation.transpose(context, driver, inputs, outputs, accumulators),
                Self::Core(CoreOperation::ReferenceNew(operation)) => {
                    operation.transpose(context, driver, inputs, outputs, accumulators)
                }
                _ => Err(ProgramError::UnsupportedOperation {
                    message: "custom fixture only transposes allocation and accesses".to_owned(),
                }
                .into()),
            }
        }
    }
    impl<C: Context<Type = ArrayIrType, Operation = CustomOperation>, P: BatchingPolicy<C>> BatchableOperation<C, P>
        for CustomOperation
    {
        fn batch<D: BatchingDriver<C, P>>(
            &self,
            context: &BatchingContext<C, P>,
            driver: &D,
            inputs: &[P::Batch],
        ) -> Result<BatchedOutputs<C, P>, BatchingError> {
            match self {
                Self::Read(operation) => operation.batch(context, driver, inputs),
                Self::AddUpdate(operation) => operation.batch(context, driver, inputs),
                _ => Err(ProgramError::UnsupportedOperation {
                    message: "custom fixture only batches accesses".to_owned(),
                }
                .into()),
            }
        }
    }

    /// Builds one dynamic read after a local allocation; index metadata belongs to the downstream family.
    fn source() -> Program<ArrayValue, CustomOperation, Vec<ArrayValue>, Vec<ArrayValue>> {
        let mut builder = ProgramBuilder::<ArrayValue, CustomOperation>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
        let index = builder.add_constant(ArrayValue::Array(Array::scalar(-1i32).unwrap()));
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let output = builder
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(vec![CustomTransform(ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Dynamic,
                })]),
                Vec::new(),
                vec![root, index],
                None,
            )
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_downstream_array_view_discharge() {
        let source = source();
        let input = ArrayValue::Array(Array::vector(vec![2f32, 3., 5.]).unwrap());
        let expected = vec![ArrayValue::Array(Array::scalar(5f32).unwrap())];
        assert_eq!(source.clone().interpret(vec![input.clone()]), Ok(expected.clone()));
        let discharged = source.discharge_references_with_policy::<CustomPolicy>(0).unwrap();
        assert_eq!(discharged.program().interpret(vec![input]), Ok(expected));
    }

    #[test]
    fn test_downstream_array_view_jvp() {
        let program = source().jvp_with_respect_to(&[0]).unwrap();
        let inputs = vec![
            ArrayValue::Array(Array::vector(vec![2f32, 3., 5.]).unwrap()),
            ArrayValue::Array(Array::vector(vec![7f32, 11., 13.]).unwrap()),
        ];
        let expected =
            vec![ArrayValue::Array(Array::scalar(5f32).unwrap()), ArrayValue::Array(Array::scalar(13f32).unwrap())];
        assert_eq!(program.clone().interpret(inputs.clone()), Ok(expected.clone()));
        let discharged = program.into_flat_program().discharge_references_with_policy::<CustomPolicy>(0).unwrap();
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_downstream_array_view_transpose() {
        let program = source().transpose_with_respect_to(&[0], &[]).unwrap();
        let access = program
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                CustomOperation::AddUpdate(operation) => Some((operation, instruction)),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            access.0.transforms(),
            &[CustomTransform(ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Dynamic
            })]
        );
        assert_eq!(access.1.inputs().len(), 3);
        let inputs = vec![ArrayValue::Array(Array::scalar(7f32).unwrap())];
        let expected = vec![ArrayValue::Array(Array::vector(vec![0f32, 0., 7.]).unwrap())];
        assert_eq!(program.clone().interpret(inputs.clone()), Ok(expected.clone()));
        let discharged = program.into_flat_program().discharge_references_with_policy::<CustomPolicy>(0).unwrap();
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
    }
    #[test]
    fn test_downstream_array_view_batching() {
        use ryft_core::{
            ArrayIrBatch, ArrayIrBatchingPolicy, DimensionBounds, DimensionType, DimensionValue, EmptyRegionDriver,
            StagingContext,
        };
        let (_, program) = ArrayContext::trace(
            |(root, index): (ArrayTracer, ArrayTracer)| {
                let parent = root.context().clone();
                let extent = parent.constant(ArrayValue::Dimension(DimensionValue::new(
                    DimensionType::new("batch", DimensionBounds::unbounded()),
                    2,
                )?));
                let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent, extent);
                let root = ArrayIrBatch::new(root, BatchAxis::new(0))?;
                let index = ArrayIrBatch::replicated(index);
                let operation =
                    CustomOperation::Read(ReferenceReadOperation::new().with_transforms(vec![CustomTransform(
                        ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
                    )]));
                let (outputs, _) = operation.batch(&context, &EmptyRegionDriver, &[root, index])?.into_parts();
                assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
                Ok(outputs[0].value().clone())
            },
            (
                ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3]))),
                ArrayIrType::Array(ArrayType::scalar(DataType::I32)),
            ),
        )
        .unwrap();
        let operation = program.instructions()[0].operation();
        assert_eq!(
            operation.reference_access_descriptor(0).unwrap().transforms(),
            &[CustomTransform(ArrayReferenceTransform::Index {
                axis: 1,
                index: ArrayReferenceTransformIndex::Dynamic
            },)]
        );
        let array = ArrayValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap());
        let index = ArrayValue::Array(Array::scalar(-1i32).unwrap());
        let expected = ArrayValue::Array(Array::vector(vec![3f32, 6.]).unwrap());
        assert_eq!(program.clone().interpret((array.reference_new().unwrap(), index.clone())), Ok(expected.clone()));
        let discharged = program.into_flat_program().discharge_references_with_policy::<CustomPolicy>(0).unwrap();
        assert_eq!(discharged.program().interpret(vec![array, index]), Ok(vec![expected]));
    }
}
