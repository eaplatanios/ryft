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
//! The universe has two views, which together exercise the generic view contract ([`ReferenceView`] and
//! [`ReferenceViewOperation`]) from downstream position. `register.halves` is a static two-output view whose outputs
//! carry two distinct descriptions; it has no discharge rule, no eager interpretation, and no transform rules, so it
//! pins the analysis side of the contract only. `register.bit` is a dynamic single-output view of one bit of a
//! register, described through the operand that carries the bit index ([`ViewSymbol::Operand`]): the analysis closes
//! the description over that operand, the discharge alias is the view path closed over destination values
//! (`ReferenceViewPath<RegisterView, C::Value>`) through which the policy reads and writes by binding the family's own
//! bit operations on the destination, forward mode reapplies the view to the tangent reference with the primal index,
//! reverse mode reaches the viewed cotangent reference through [`TranspositionContext`], which resolves the bound
//! index to its transposed-program value, and batching goes through the shared [`batch_reference_view_operation`]
//! rule. Eagerly, a bit view is a [`RegisterValue::BitReference`] handle over the root reference; a bit of a bit has no
//! eager handle, so nested bit views are reachable only through staged programs and their discharge.
//!
//! The transform legs are reached through the public entry points ([`differentiate_at`] for `jvp` and `vjp`, and
//! [`batch`]) over a live register reference. The generic reference primitives ([`ReferenceNewOperation`] and its
//! siblings) are wrapped by the family and interpret eagerly through the value-level capabilities implemented on
//! [`RegisterValue`], and their generic differentiation, transposition, and batching rules apply at the eager context
//! and at the staged contexts that transforms instantiate. The family supplies its allocation and accumulation
//! operations through [`ReferenceNewOperationProvider`] and [`ReferenceAddUpdateOperationProvider`], so generic
//! transposition can allocate cotangent references without any downstream implementation for a core-owned tracer.
//! `register.add_update` retains family-owned addition semantics; the other reference primitives reuse their generic
//! transform rules.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt::Display;

use indoc::indoc;
use pretty_assertions::assert_eq;

use ryft_core::macros::check_count;
use ryft_core::{
    AddOperation, AtomId, BatchAxis, BatchAxisSpecification, BatchableOperation, BatchableType, BatchedOutputs,
    BatchingContext, BatchingDriver, BatchingEntrypointPolicy, BatchingError, BatchingPolicy,
    BoundaryPreservingBatchedProgram, Context, CotangentDestination, CotangentDestinationKind, CotangentSeed,
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    Domain, EagerContext, EffectClass, EffectClasses, Effects, ExternalReferenceBinding, InputRegionProvenance,
    InstructionId, InterpretableOperation, InterpretationDriver, MaybeZero, NoIdentity, Operation,
    OutputRegionProvenance, Parameter, PartialValue, PartiallyEvaluatableOperation, Placeholder, Program,
    ProgramBatchingOutputAxesPolicy, ProgramBuilder, ProgramError, RecursiveBatchingPolicy,
    RecursiveReferenceDischargeDriver, Reference, ReferenceAccessMode, ReferenceAddUpdate,
    ReferenceAddUpdateOperationProvider, ReferenceAlias, ReferenceAliasEdge, ReferenceAliasKind, ReferenceAliasOrigin,
    ReferenceBoundaryError, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy,
    ReferenceDischargeRegionBoundary, ReferenceDischargeRegionStateInsertion, ReferenceDischargeResult,
    ReferenceDischargeTarget, ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceDischargeableType,
    ReferenceEffect, ReferenceFreeze, ReferenceFreezeOperation, ReferenceId, ReferenceNew, ReferenceNewOperation,
    ReferenceNewOperationProvider, ReferenceRead, ReferenceReadOperation, ReferenceSource, ReferenceSwap,
    ReferenceSwapOperation, ReferenceType, ReferenceView, ReferenceViewOperation, ReferenceViewPath, ReferenceViewStep,
    ReferenceViewValidationError, ReferenceWrite, ReferenceWriteOperation, RegionId, RegionInterface, RegionRef,
    RegionSlot, Trace, Tracer, TracingContext, TransposableOperation, TranspositionContext, TranspositionDriver, Type,
    TypeError, Typed, Value, ValueId, ViewOverlap, ViewSymbol, ViewSymbolBinding, Zero, ZeroOperation, batch,
    batch_reference_view_operation, differentiate_at, discharge_reference_free_operation, validate_reference_boundary,
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

    fn referent(&self) -> Option<Self> {
        match self {
            Self::Register(_) => None,
            Self::Reference(r#type) => Some(Self::Register(r#type.referent().clone())),
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

    /// Live handle to bit `index` of the register allocation `root`, which is the eager form of a `register.bit` view:
    /// it reads and writes that bit through the root handle and reports the root's identity. A bit of a bit has no
    /// eager handle.
    BitReference { root: Reference<RegisterValue>, index: i64 },
}

impl RegisterValue {
    /// Returns the register this value holds, rejecting a reference.
    fn register(&self) -> Result<i64, ProgramError> {
        match self {
            Self::Register(value) => Ok(*value),
            Self::Reference(_) | Self::BitReference { .. } => {
                Err(TypeError::invalid("expected a register value but got a reference").into())
            }
        }
    }

    /// Returns the live complete-register reference this value holds, rejecting a register and a bit handle.
    fn reference(&self) -> Result<&Reference<RegisterValue>, ProgramError> {
        match self {
            Self::Reference(reference) => Ok(reference),
            Self::Register(_) => Err(TypeError::invalid("expected a register reference but got a register").into()),
            Self::BitReference { .. } => {
                Err(TypeError::invalid("expected a complete register reference but got a bit handle").into())
            }
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
            Self::BitReference { root, index } => write!(formatter, "{root}[bit {index}]"),
        }
    }
}

impl Parameter for RegisterValue {}

impl Typed for RegisterValue {
    type Type = RegisterIrType;

    fn r#type(&self) -> Cow<'_, RegisterIrType> {
        Cow::Owned(match self {
            Self::Register(_) => RegisterIrType::Register(RegisterType),
            Self::Reference(_) | Self::BitReference { .. } => {
                RegisterIrType::Reference(ReferenceType::new(RegisterType))
            }
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
            Self::BitReference { root, .. } => Some(root.id()),
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

impl ReferenceRead for RegisterValue {
    fn read(&self) -> Result<Self, ProgramError> {
        match self {
            Self::BitReference { root, index } => {
                let register = root.read().map_err(ProgramError::custom)?.register()?;
                Ok(Self::Register(extract_bit(register, *index)?))
            }
            _ => self.reference()?.read().map_err(ProgramError::custom),
        }
    }
}

impl ReferenceWrite for RegisterValue {
    fn write(&self, replacement: &Self) -> Result<(), ProgramError> {
        let stored = replacement.register()?;
        match self {
            Self::BitReference { root, index } => {
                let register = root.read().map_err(ProgramError::custom)?.register()?;
                root.write(Self::Register(insert_bit(register, stored, *index)?)).map_err(ProgramError::custom)
            }
            _ => self.reference()?.write(replacement.clone()).map_err(ProgramError::custom),
        }
    }
}

impl ReferenceSwap for RegisterValue {
    fn swap(&self, replacement: &Self) -> Result<Self, ProgramError> {
        match self {
            Self::BitReference { .. } => {
                let previous = self.read()?;
                self.write(replacement)?;
                Ok(previous)
            }
            _ => {
                replacement.register()?;
                self.reference()?.swap(replacement.clone()).map_err(ProgramError::custom)
            }
        }
    }
}

impl ReferenceAddUpdate for RegisterValue {
    fn add_update(&self, update: &Self) -> Result<(), ProgramError> {
        let current = self.read()?.register()?;
        self.write(&Self::Register(current + update.register()?))
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

/// Returns the destination value of the bit index that the `register.bit` step `step` of a discharge alias is bound
/// to, rejecting a half step: `register.halves` has no discharge rule, so a half step never reaches the policy.
fn bit_coordinate<V>(step: &ReferenceViewStep<RegisterView, V>) -> Result<&V, ProgramError> {
    match step.view() {
        RegisterView::Bit(_) => {
            check_count!("input", step.bindings(), 1, ProgramError);
            Ok(&step.bindings()[0])
        }
        RegisterView::Half(_) => Err(ProgramError::UnsupportedOperation {
            message: "`register.halves` has no discharge rule in the register universe, so a half step never reaches \
                      its discharge policy"
                .to_string(),
        }),
    }
}

/// Returns `current` with the bit that `steps` select replaced by `replacement`, binding the family's bit operations
/// on `context`. Nested bit steps recurse: the bit each non-final step selects is extracted, rewritten through the
/// remaining steps, and inserted back.
fn insert_bits<C: Context<Type = RegisterIrType, Operation: From<RegisterOperation>>>(
    context: &C,
    current: C::Value,
    replacement: C::Value,
    steps: &[ReferenceViewStep<RegisterView, C::Value>],
) -> Result<C::Value, ProgramError> {
    let Some((step, rest)) = steps.split_first() else {
        return Ok(replacement);
    };
    let coordinate = bit_coordinate(step)?.clone();
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
// implementation serve an eager destination and a staging destination alike. Its alias is the view path closed over
// destination values, so a bit step carries the destination value of its index and the policy reads and writes through
// it by binding the family's bit operations on the destination, with no environment lookup. The policy declines
// accumulation entirely by not implementing `ReferenceAccumulationPolicy`.
impl<C: Context<Type = RegisterIrType, Operation: From<RegisterOperation>>> ReferenceDischargePolicy<C>
    for RegisterReferenceDischarge
{
    type Referent = RegisterType;
    type Alias = ReferenceViewPath<RegisterView, C::Value>;

    fn storage_alias(_referent: &RegisterType) -> ReferenceViewPath<RegisterView, C::Value> {
        ReferenceViewPath::root()
    }

    fn read(
        context: &C,
        current: &C::Value,
        alias: &ReferenceViewPath<RegisterView, C::Value>,
    ) -> Result<C::Value, ProgramError> {
        let mut selected = current.clone();
        for step in alias.steps() {
            let coordinate = bit_coordinate(step)?.clone();
            selected = bind_register_output(context, RegisterOperation::BitExtract, &[selected, coordinate])?;
        }
        Ok(selected)
    }

    fn write(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &ReferenceViewPath<RegisterView, C::Value>,
    ) -> Result<C::Value, ProgramError> {
        insert_bits(context, current.clone(), replacement, alias.steps())
    }
}

/// Operation family of the downstream universe. The reference accesses wrap the generic `ryft-core` primitives, so
/// their type inference, reference semantics, effects, and eager interpretation are the canonical ones; the additive
/// update is the family's own because the generic primitive requires an [`Operation`] implementation for
/// `AddOperation<RegisterType>` that only `ryft-core` can provide. `register.halves` and `register.bit` are the
/// family's static and dynamic views (refer to the module documentation), and `register.bit_extract` and
/// `register.bit_insert` are the value-level bit operations through which the discharge policy reads and writes a bit
/// view.
#[derive(Clone, Debug)]
enum RegisterOperation {
    Negate,
    Add(AddOperation<RegisterIrType>),
    Zero(ZeroOperation<RegisterIrType>),
    ReferenceNew(ReferenceNewOperation<RegisterType, RegisterIrType>),
    Read(ReferenceReadOperation<RegisterType, RegisterIrType>),
    Write(ReferenceWriteOperation<RegisterType, RegisterIrType>),
    Swap(ReferenceSwapOperation<RegisterType, RegisterIrType>),
    AddUpdate,
    Freeze(ReferenceFreezeOperation<RegisterType, RegisterIrType>),
    Call,
    Halves,
    Bit,
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

impl From<ReferenceReadOperation<RegisterType, RegisterIrType>> for RegisterOperation {
    fn from(operation: ReferenceReadOperation<RegisterType, RegisterIrType>) -> Self {
        Self::Read(operation)
    }
}

impl From<ReferenceWriteOperation<RegisterType, RegisterIrType>> for RegisterOperation {
    fn from(operation: ReferenceWriteOperation<RegisterType, RegisterIrType>) -> Self {
        Self::Write(operation)
    }
}

impl From<ReferenceSwapOperation<RegisterType, RegisterIrType>> for RegisterOperation {
    fn from(operation: ReferenceSwapOperation<RegisterType, RegisterIrType>) -> Self {
        Self::Swap(operation)
    }
}

impl From<ReferenceFreezeOperation<RegisterType, RegisterIrType>> for RegisterOperation {
    fn from(operation: ReferenceFreezeOperation<RegisterType, RegisterIrType>) -> Self {
        Self::Freeze(operation)
    }
}

impl ReferenceNewOperationProvider<RegisterIrType> for RegisterOperation {
    fn reference_new_operation() -> Self {
        Self::ReferenceNew(ReferenceNewOperation::new())
    }
}

impl ReferenceAddUpdateOperationProvider<RegisterIrType> for RegisterOperation {
    fn reference_add_update_operation() -> Result<Self, ProgramError> {
        Ok(Self::AddUpdate)
    }
}

impl Operation for RegisterOperation {
    type Type = RegisterIrType;

    fn name(&self) -> &'static str {
        match self {
            Self::Negate => "register.negate",
            Self::Add(_) => "register.add",
            Self::Zero(_) => "register.zero",
            Self::ReferenceNew(operation) => operation.name(),
            Self::Read(operation) => operation.name(),
            Self::Write(operation) => operation.name(),
            Self::Swap(operation) => operation.name(),
            Self::AddUpdate => "register.add_update",
            Self::Freeze(operation) => operation.name(),
            Self::Call => "register.call",
            Self::Halves => "register.halves",
            Self::Bit => "register.bit",
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

    fn input_region_provenance(&self, _region_index: usize, input_index: usize) -> Option<InputRegionProvenance> {
        matches!(self, Self::Call).then_some(InputRegionProvenance::Forwarded { input_index })
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
            Self::ReferenceNew(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Read(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Write(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Swap(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::AddUpdate => {
                check_count!("input", input_types, 2, TypeError);
                referent()?;
                <&RegisterType>::try_from(&input_types[1])?;
                Ok(Vec::new())
            }
            Self::Freeze(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Halves => {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![RegisterIrType::Reference(ReferenceType::new(referent()?)); 2])
            }
            // A bit is modeled as a register holding 0 or 1, so the referent type of the view is the referent type of
            // the viewed reference.
            Self::Bit => {
                check_count!("input", input_types, 2, TypeError);
                let referent = referent()?;
                <&RegisterType>::try_from(&input_types[1])?;
                Ok(vec![RegisterIrType::Reference(ReferenceType::new(referent))])
            }
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

    fn effects(&self) -> Cow<'_, Effects> {
        match self {
            Self::Negate | Self::Add(_) | Self::Zero(_) | Self::BitExtract | Self::BitInsert => {
                Cow::Borrowed(Effects::empty())
            }
            Self::ReferenceNew(operation) => operation.effects(),
            Self::Read(operation) => operation.effects(),
            Self::Write(operation) => operation.effects(),
            Self::Swap(operation) => operation.effects(),
            Self::Freeze(operation) => operation.effects(),
            Self::AddUpdate => Cow::Owned(
                Effects::new(
                    EffectClasses::NONE,
                    vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Accumulate }],
                    Vec::new(),
                )
                .unwrap(),
            ),
            // A structured operation declares no operation-local reference effects (its accesses are summarized
            // transitively from the region closure it attaches) but carries opaque ordered state of its own.
            Self::Call => Cow::Owned(Effects::explicit(EffectClasses::single(EffectClass::OrderedState))),
            // Both halves are narrowing views of the one operand.
            Self::Halves => Cow::Owned(
                Effects::new(
                    EffectClasses::NONE,
                    Vec::new(),
                    vec![
                        ReferenceAlias::new(0, 0, ReferenceAliasKind::View),
                        ReferenceAlias::new(1, 0, ReferenceAliasKind::View),
                    ],
                )
                .unwrap(),
            ),
            // The bit is a narrowing view of the reference operand; the index operand is a coordinate, not a reference.
            Self::Bit => Cow::Owned(
                Effects::new(
                    EffectClasses::NONE,
                    Vec::new(),
                    vec![ReferenceAlias::new(0, 0, ReferenceAliasKind::View)],
                )
                .unwrap(),
            ),
        }
    }
}

/// Static half selector of `register.halves`: which half of a register one of its outputs selects.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum RegisterHalf {
    Low,
    High,
}

/// View description of the downstream universe.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum RegisterView {
    /// Static half of a register, selected by `register.halves`.
    Half(RegisterHalf),

    /// One bit of a register, selected by `register.bit` at the index the symbol names: [`ViewSymbol::Operand`] of the
    /// index operand for an instruction output, or [`ViewSymbol::Iteration`] for a hypothetical boundary view.
    Bit(ViewSymbol),
}

// A half is a static description while a bit depends on the one coordinate its symbol names. Registers have no axes,
// so a replicated batch axis passes through either description unchanged and a mapped one is rejected. Paths are
// compared step by step: two static halves are disjoint as soon as they differ, two bits are the same coordinate iff
// their bindings are equal and may otherwise overlap, a bit and a half may overlap, and paths that agree on every
// shared step are the same when they have the same length and otherwise one is a strict prefix that contains the other.
impl ReferenceView for RegisterView {
    type Type = RegisterIrType;

    fn symbols(&self) -> Vec<ViewSymbol> {
        match self {
            Self::Half(_) => Vec::new(),
            Self::Bit(symbol) => vec![*symbol],
        }
    }

    fn batch(&self, _source: &RegisterIrType, batch_axis: BatchAxis) -> Result<(Self, BatchAxis), BatchingError> {
        if !batch_axis.is_replicated() {
            return Err(BatchingError::UnsupportedOperation {
                message: "a register view cannot carry a mapped batch axis; registers have no axes".to_string(),
            });
        }
        Ok((*self, batch_axis))
    }

    fn overlap(_root: &RegisterIrType, a: &[ReferenceViewStep<Self>], b: &[ReferenceViewStep<Self>]) -> ViewOverlap {
        for (a, b) in a.iter().zip(b.iter()) {
            match (a.view(), b.view()) {
                (Self::Half(a_half), Self::Half(b_half)) if a_half != b_half => return ViewOverlap::Disjoint,
                (Self::Half(_), Self::Half(_)) => {}
                (Self::Bit(_), Self::Bit(_)) if a == b => {}
                _ => return ViewOverlap::MayOverlap,
            }
        }
        if a.len() == b.len() { ViewOverlap::Same } else { ViewOverlap::MayOverlap }
    }
}

// The view contract from downstream position: one owned description per view output, a type-level check that only
// requires both ends to be register references (a half or a bit of a register is still a register), and reapplication
// that stages the describing operation over another source, keeping the described half or supplying the bit index.
impl ReferenceViewOperation for RegisterOperation {
    type View = RegisterView;

    fn reference_view(&self, output_index: usize) -> Option<RegisterView> {
        match (self, output_index) {
            (Self::Halves, 0) => Some(RegisterView::Half(RegisterHalf::Low)),
            (Self::Halves, 1) => Some(RegisterView::Half(RegisterHalf::High)),
            (Self::Bit, 0) => Some(RegisterView::Bit(ViewSymbol::Operand(1))),
            _ => None,
        }
    }

    fn validate_view(
        _view: &RegisterView,
        source: &RegisterIrType,
        output: &RegisterIrType,
    ) -> Result<(), ReferenceViewValidationError> {
        for r#type in [source, output] {
            if !r#type.is_reference() {
                return Err(ReferenceViewValidationError::InvalidComposition {
                    message: format!("expected a register reference but got `{type}`"),
                });
            }
        }
        Ok(())
    }

    fn reapply_view<C: Context<Type = RegisterIrType, Operation = Self>>(
        context: &C,
        view: &RegisterView,
        source: C::Value,
        symbols: &[C::Value],
    ) -> Result<C::Value, ProgramError> {
        match view {
            RegisterView::Half(half) => {
                check_count!("input", symbols, 0, ProgramError);
                let mut outputs = context.bind(Self::Halves, Vec::new(), std::slice::from_ref(&source))?;
                check_count!("output", outputs, 2, ProgramError);
                Ok(outputs.swap_remove(match half {
                    RegisterHalf::Low => 0,
                    RegisterHalf::High => 1,
                }))
            }
            RegisterView::Bit(ViewSymbol::Operand(_)) => {
                check_count!("input", symbols, 1, ProgramError);
                let mut outputs = context.bind(Self::Bit, Vec::new(), &[source, symbols[0].clone()])?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(outputs.remove(0))
            }
            RegisterView::Bit(ViewSymbol::Iteration) => Err(ProgramError::UnsupportedOperation {
                message: "a register bit indexed by the iteration counter is created by its region-carrying operation \
                          and cannot be reapplied"
                    .to_string(),
            }),
        }
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
            // The wrapped primitives interpret through the eager capabilities of `RegisterValue`.
            Self::ReferenceNew(operation) => operation.interpret(context, driver, inputs),
            Self::Read(operation) => operation.interpret(context, driver, inputs),
            Self::Write(operation) => operation.interpret(context, driver, inputs),
            Self::Swap(operation) => operation.interpret(context, driver, inputs),
            Self::Freeze(operation) => operation.interpret(context, driver, inputs),
            Self::AddUpdate => {
                check_count!("input", inputs, 2, ProgramError);
                inputs[0].add_update(&inputs[1])?;
                Ok(Vec::new())
            }
            Self::Call => driver.interpret_region(context, 0, inputs.to_vec()),
            Self::Halves => Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` has no eager interpretation in the register universe", self.name()),
            }),
            // The eager form of the bit view is a bit handle over the complete root reference; the index is validated
            // when the view is created so that every access through the handle is in range.
            Self::Bit => {
                check_count!("input", inputs, 2, ProgramError);
                let root = inputs[0].reference()?.clone();
                let index = inputs[1].register()?;
                bit_index(index)?;
                Ok(vec![RegisterValue::BitReference { root, index }])
            }
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
            Self::Negate | Self::Add(_) | Self::Zero(_) | Self::BitExtract | Self::BitInsert => {
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
            Self::Read(_) => {
                check_count!("input", inputs, 1, ProgramError);
                let reference = inputs[0].try_as_reference("a reference to read")?;
                Ok(vec![ReferenceDischargeValue::Value(context.read(reference)?)])
            }
            Self::Write(_) => {
                check_count!("input", inputs, 2, ProgramError);
                let reference = inputs[0].try_as_reference("a reference to write")?;
                let replacement = inputs[1].try_as_value("a replacement value")?.clone();
                context.write(reference, replacement)?;
                Ok(Vec::new())
            }
            Self::Swap(_) => {
                check_count!("input", inputs, 2, ProgramError);
                let reference = inputs[0].try_as_reference("a reference to replace")?;
                let replacement = inputs[1].try_as_value("a replacement value")?.clone();
                Ok(vec![ReferenceDischargeValue::Value(context.swap(reference, replacement)?)])
            }
            Self::Freeze(_) => {
                check_count!("input", inputs, 1, ProgramError);
                let reference = inputs[0].try_as_reference("a reference to freeze")?;
                Ok(vec![ReferenceDischargeValue::Value(context.consume(reference)?)])
            }
            // The bit view composes its step onto the operand's alias, closed over the destination value of its index,
            // exactly as the array family's view rules do; on a preserved allocation the view replays verbatim over the
            // parent destination reference.
            Self::Bit => {
                check_count!("input", inputs, 2, ProgramError);
                let reference = inputs[0].try_as_reference("a reference to view")?;
                let index = inputs[1].try_as_value("a bit index")?.clone();
                let alias = reference.alias().with_step(RegisterView::Bit(ViewSymbol::Operand(1)), vec![index.clone()]);
                let viewed = context.alias_reference(reference, alias, ReferenceType::new(RegisterType), |parent| {
                    bind_register_output(context.parent(), Self::Bit, &[parent.clone(), index])
                })?;
                Ok(vec![viewed.into()])
            }
            // The non-accumulating discharge policy has no accumulation capability for the additive update, and the
            // static two-output view has no discharge rule, so both are rejected by name.
            Self::AddUpdate | Self::Halves => Err(ProgramError::UnsupportedOperation {
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
                    declared.push(context.operand_allocation(input, self.name())?);
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
                    0,
                    &ReferenceDischargeRegionBoundary::new(
                        self,
                        0,
                        declared,
                        ReferenceDischargeRegionStateInsertion::new(entering.clone(), inputs.len()),
                        ReferenceDischargeRegionStateInsertion::new(widening.published().to_vec(), source_output_count),
                    ),
                )?;
                result.validate_predicted_mutations(widening.published(), self.name())?;
                result.validate_predicted_output_allocations(summary.output_allocations(), self.name())?;

                let mut operands = Vec::with_capacity(inputs.len() + entering.len());
                for input in inputs {
                    operands.push(context.operand_value(input)?);
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
// operations, additive update, and view need rules here.
impl<C: Context<Type = RegisterIrType, Operation = RegisterOperation> + Zero<C::Value>> DifferentiableOperation<C>
    for RegisterOperation
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        match self {
            Self::Negate => {
                check_count!("input", inputs, 1, ProgramError);
                let primal = bind_register_output(context, self.clone(), &primals)?;
                let tangent = match inputs[0].tangent() {
                    MaybeZero::Value(tangent) => {
                        MaybeZero::Value(bind_register_output(context, self.clone(), std::slice::from_ref(tangent))?)
                    }
                    MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
                };
                Ok(vec![DifferentiationDual::new(primal, tangent)?])
            }
            Self::Add(_) => {
                check_count!("input", inputs, 2, ProgramError);
                let primal = bind_register_output(context, self.clone(), &primals)?;
                let tangent = match (inputs[0].tangent(), inputs[1].tangent()) {
                    (MaybeZero::Zero(r#type), MaybeZero::Zero(_)) => MaybeZero::Zero(r#type.clone()),
                    (MaybeZero::Value(tangent), MaybeZero::Zero(_))
                    | (MaybeZero::Zero(_), MaybeZero::Value(tangent)) => MaybeZero::Value(tangent.clone()),
                    (MaybeZero::Value(left), MaybeZero::Value(right)) => {
                        MaybeZero::Value(bind_register_output(context, self.clone(), &[left.clone(), right.clone()])?)
                    }
                };
                Ok(vec![DifferentiationDual::new(primal, tangent)?])
            }
            Self::Zero(_) => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![DifferentiationDual::new_with_zero_tangent(bind_register_output(context, self.clone(), &[])?)?])
            }
            Self::ReferenceNew(operation) => operation.jvp(context, driver, inputs),
            Self::Read(operation) => operation.jvp(context, driver, inputs),
            Self::Freeze(operation) => operation.jvp(context, driver, inputs),
            Self::Write(operation) => operation.jvp(context, driver, inputs),
            Self::Swap(operation) => operation.jvp(context, driver, inputs),
            Self::AddUpdate => {
                check_count!("input", inputs, 2, ProgramError);
                if inputs[0].tangent().is_zero() && !inputs[1].tangent().is_zero() {
                    return Err(DifferentiationError::PlumbingReferenceTangent { operation: self.name() });
                }
                context.bind(self.clone(), Vec::new(), &primals)?;
                if let (MaybeZero::Value(reference), MaybeZero::Value(tangent)) =
                    (inputs[0].tangent(), inputs[1].tangent())
                {
                    context.bind(self.clone(), Vec::new(), &[reference.clone(), tangent.clone()])?;
                }
                Ok(Vec::new())
            }
            // The tangent of a bit view is the same bit of the tangent reference, selected by the primal index (the
            // index is a coordinate, so its tangent is dropped); a plumbing reference yields a plumbing view.
            Self::Bit => {
                check_count!("input", inputs, 2, ProgramError);
                let primal = bind_register_output(context, self.clone(), &primals)?;
                let tangent = match inputs[0].tangent() {
                    MaybeZero::Value(reference) => MaybeZero::Value(bind_register_output(
                        context,
                        self.clone(),
                        &[reference.clone(), primals[1].clone()],
                    )?),
                    MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().into_owned()),
                };
                Ok(vec![DifferentiationDual::new(primal, tangent)?])
            }
            Self::Call | Self::Halves | Self::BitExtract | Self::BitInsert => Err(ProgramError::UnsupportedOperation {
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
        context: &mut TranspositionContext<'_, RegisterValue, RegisterOperation>,
        driver: &D,
        inputs: &[PartialValue<RegisterTracer>],
        outputs: &[MaybeZero<RegisterTracer>],
    ) -> Result<Vec<MaybeZero<RegisterTracer>>, DifferentiationError> {
        match self {
            Self::Negate => {
                check_count!("input", inputs, 1, ProgramError);
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![match &outputs[0] {
                    MaybeZero::Value(cotangent) => MaybeZero::Value(bind_register_output(
                        &**context,
                        self.clone(),
                        std::slice::from_ref(cotangent),
                    )?),
                    MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
                }])
            }
            Self::Add(_) => {
                check_count!("input", inputs, 2, ProgramError);
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![outputs[0].clone(), outputs[0].clone()])
            }
            Self::Zero(_) => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(Vec::new())
            }
            Self::ReferenceNew(operation) => operation.transpose(context, driver, inputs, outputs),
            Self::Read(operation) => operation.transpose(context, driver, inputs, outputs),
            Self::Freeze(operation) => operation.transpose(context, driver, inputs, outputs),
            Self::Write(operation) => operation.transpose(context, driver, inputs, outputs),
            Self::Swap(operation) => operation.transpose(context, driver, inputs, outputs),
            Self::AddUpdate => {
                check_count!("input", inputs, 2, ProgramError);
                let update_cotangent = match context.cotangent_reference_if_allocated(0)? {
                    Some(accumulator) => MaybeZero::Value(bind_register_output(
                        &**context,
                        Self::Read(ReferenceReadOperation::new()),
                        &[accumulator],
                    )?),
                    None => MaybeZero::Zero(inputs[1].r#type().cotangent()?),
                };
                Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent()?), update_cotangent])
            }
            // A view has no cotangent of its own: the accesses through it reach the same bit of the root's cotangent
            // reference through the transposition context, which reapplies the view over the root's accumulator with
            // the transposed value of the index.
            Self::Bit => {
                check_count!("input", inputs, 2, ProgramError);
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![
                    MaybeZero::Zero(inputs[0].r#type().cotangent()?),
                    MaybeZero::Zero(inputs[1].r#type().cotangent()?),
                ])
            }
            Self::Call | Self::Halves | Self::BitExtract | Self::BitInsert => Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` has no transposition rule in the register universe", self.name()),
            }
            .into()),
        }
    }
}

// Registers have no axes, so the family batches replicated carriers only: every region-free operation runs once on the
// parent context over the packed values and its outputs stay replicated. A mapped carrier is rejected by name. The bit
// view goes through the shared view rule instead, which moves the source's (replicated) batch axis through the
// description and binds the batched view on the parent context with the packed index.
impl<
    C: Context<Type = RegisterIrType, Operation: ReferenceViewOperation + From<RegisterOperation>>,
    P: BatchingPolicy<C>,
> BatchableOperation<C, P> for RegisterOperation
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        _driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        match self {
            Self::Call | Self::Halves => {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!("`{}` has no batching rule in the register universe", self.name()),
                });
            }
            Self::Bit => return batch_reference_view_operation(self, context, inputs),
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
    fn prepare_inputs(
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
                reference_write %1 %2
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
    // The allocation target sits inside the callee region, so whether it discharges is decided by the replay coordinate
    // the downstream rule's driver hands to `selects_internal` inside the fork. An empty target list must preserve
    // the allocation inside the rebuilt region, and selecting the enumerated target must discharge it completely —
    // which is exactly the behavior a driver without a real `source_instruction_id()` would silently break.
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
fn test_downstream_view_operation_records_output_indices_and_distinct_paths() {
    // `f(register) = (read(low half), read(high half))`: one two-output view whose outputs are two distinct views of
    // the same root, exercised through the generic static view contract from downstream position.
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let reference = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let halves = builder
        .add_instruction(RegisterOperation::Halves, Vec::new(), vec![reference], None)
        .unwrap()
        .to_vec();
    let low = builder
        .add_instruction(RegisterOperation::Read(ReferenceReadOperation::new()), Vec::new(), vec![halves[0]], None)
        .unwrap()[0];
    let high = builder
        .add_instruction(RegisterOperation::Read(ReferenceReadOperation::new()), Vec::new(), vec![halves[1]], None)
        .unwrap()[0];
    let program = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![low, high], vec![Placeholder], vec![Placeholder; 2])
        .unwrap();

    // The contract itself: exactly the two view outputs are described, each by its own half.
    assert_eq!(RegisterOperation::Halves.reference_view(0), Some(RegisterView::Half(RegisterHalf::Low)));
    assert_eq!(RegisterOperation::Halves.reference_view(1), Some(RegisterView::Half(RegisterHalf::High)));
    assert_eq!(RegisterOperation::Halves.reference_view(2), None);
    assert_eq!(RegisterOperation::Read(ReferenceReadOperation::new()).reference_view(0), None);

    // The overlay records which output defines each alias edge and asks the operation for exactly that description, so
    // the two paths differ while the root keeps the empty path and the read outputs have none.
    let analysis = program.entry_region_ref().reference_view_analysis(0).unwrap();
    let value = |atom: usize| ValueId::new(RegionId::new(0), AtomId::new(atom));
    let halves_instruction = InstructionId::new(RegionId::new(0), 0);
    assert_eq!(
        analysis.analysis().alias(value(1)),
        Some(ReferenceAliasEdge::new(
            halves_instruction,
            ReferenceAliasOrigin::Output(0),
            value(0),
            ReferenceAliasKind::View,
            true,
        )),
    );
    assert_eq!(
        analysis.analysis().alias(value(2)),
        Some(ReferenceAliasEdge::new(
            halves_instruction,
            ReferenceAliasOrigin::Output(1),
            value(0),
            ReferenceAliasKind::View,
            true,
        )),
    );
    assert_eq!(analysis.path(value(0)), Some(&ReferenceViewPath::root()));
    assert_eq!(
        analysis.path(value(1)),
        Some(&ReferenceViewPath::root().with_view(RegisterView::Half(RegisterHalf::Low)))
    );
    assert_eq!(
        analysis.path(value(2)),
        Some(&ReferenceViewPath::root().with_view(RegisterView::Half(RegisterHalf::High)))
    );
    assert_eq!(analysis.path(value(3)), None);
    assert_eq!(analysis.path(value(4)), None);

    // Reapplication stages the view over another register reference and keeps the described half, which the traced
    // program then reads.
    let (_, reapplied): (_, Program<_, _, Vec<RegisterValue>, Vec<RegisterValue>>) =
        EagerContext::<RegisterValue, RegisterOperation>::trace(
            |inputs: Vec<Tracer<TracingContext<RegisterValue, RegisterOperation>>>| {
                let context = inputs[0].context().clone();
                let high = RegisterOperation::reapply_view(
                    &context,
                    &RegisterView::Half(RegisterHalf::High),
                    inputs[0].clone(),
                    &[],
                )?;
                context.bind(
                    RegisterOperation::Read(ReferenceReadOperation::new()),
                    Vec::new(),
                    std::slice::from_ref(&high),
                )
            },
            vec![RegisterIrType::Reference(ReferenceType::new(RegisterType))],
        )
        .unwrap();
    assert_eq!(
        reapplied.to_string(),
        indoc! {"
            lambda %0:ref<register> .
            let %1:ref<register>, %2:ref<register> = register.halves %0
                %3:register = reference_read %2
            in (%3)"},
    );
}
/// `f(r, x) = { r.add_update(x); r.read() }` over the register universe, written against whatever context the input
/// values dispatch to so that one closure serves the forward-mode, reverse-mode, and batching tracers alike.
fn read_modify_write<V: Value<Type = RegisterIrType>>((reference, x): (V, V)) -> Result<V, ProgramError>
where
    V::DispatchDomain: Context<Type = RegisterIrType, Operation: From<RegisterOperation>>,
{
    let context = reference.dispatch_domain();
    context.bind(RegisterOperation::AddUpdate, Vec::new(), &[reference.clone(), x])?;
    bind_register_output(&context, RegisterOperation::Read(ReferenceReadOperation::new()), &[reference])
}

/// `f(r, x, i) = { b = bit(r, i); b.write(x); b.read() }` over the register universe: a write and a read through the
/// dynamic bit view, written against whatever context the input values dispatch to (refer to [`read_modify_write`]).
fn write_read_bit<V: Value<Type = RegisterIrType>>((reference, x, index): (V, V, V)) -> Result<V, ProgramError>
where
    V::DispatchDomain: Context<Type = RegisterIrType, Operation: From<RegisterOperation>>,
{
    let context = reference.dispatch_domain();
    let bit = bind_register_output(&context, RegisterOperation::Bit, &[reference, index])?;
    context.bind(RegisterOperation::Write(ReferenceWriteOperation::new()), Vec::new(), &[bit.clone(), x])?;
    bind_register_output(&context, RegisterOperation::Read(ReferenceReadOperation::new()), &[bit])
}

#[test]
fn test_downstream_view_description_overlap_and_batch() {
    // The family's overlap rule compares paths step by step: the two halves are disjoint, a path is the same as
    // itself, the complete root or a shorter prefix contains what it narrows to, two bits are the same coordinate
    // exactly when their bindings agree and may otherwise overlap, and a bit may overlap with a half.
    let root = RegisterIrType::Reference(ReferenceType::new(RegisterType));
    let value = |atom: usize| ValueId::new(RegionId::new(0), AtomId::new(atom));
    let empty = ReferenceViewPath::<RegisterView>::root();
    let low = empty.with_view(RegisterView::Half(RegisterHalf::Low));
    let high = empty.with_view(RegisterView::Half(RegisterHalf::High));
    let low_high = low.with_view(RegisterView::Half(RegisterHalf::High));
    let bit_of_1 = empty.with_step(RegisterView::Bit(ViewSymbol::Operand(1)), vec![ViewSymbolBinding::Value(value(1))]);
    let bit_of_2 = empty.with_step(RegisterView::Bit(ViewSymbol::Operand(1)), vec![ViewSymbolBinding::Value(value(2))]);
    assert_eq!(low.overlap(&high, &root), ViewOverlap::Disjoint);
    assert_eq!(low.overlap(&low, &root), ViewOverlap::Same);
    assert_eq!(empty.overlap(&empty, &root), ViewOverlap::Same);
    assert_eq!(empty.overlap(&low, &root), ViewOverlap::MayOverlap);
    assert_eq!(low_high.overlap(&low, &root), ViewOverlap::MayOverlap);
    assert_eq!(low_high.overlap(&high, &root), ViewOverlap::Disjoint);
    assert_eq!(bit_of_1.overlap(&bit_of_1, &root), ViewOverlap::Same);
    assert_eq!(bit_of_1.overlap(&bit_of_2, &root), ViewOverlap::MayOverlap);
    assert_eq!(bit_of_1.overlap(&low, &root), ViewOverlap::MayOverlap);
    assert_eq!(empty.overlap(&bit_of_1, &root), ViewOverlap::MayOverlap);
    assert_eq!(
        low.with_step(RegisterView::Bit(ViewSymbol::Operand(1)), vec![ViewSymbolBinding::Value(value(1))])
            .overlap(&bit_of_1, &root),
        ViewOverlap::MayOverlap
    );

    // The analysis-level query resolves both values to their roots first: `f(register) = (read(low), read(high))`
    // has one root, so its two halves are disjoint, each half may overlap with the root, and a non-reference value has
    // no answer.
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let reference = builder.add_input(root.clone());
    let halves = builder
        .add_instruction(RegisterOperation::Halves, Vec::new(), vec![reference], None)
        .unwrap()
        .to_vec();
    let low_read = builder
        .add_instruction(RegisterOperation::Read(ReferenceReadOperation::new()), Vec::new(), vec![halves[0]], None)
        .unwrap()[0];
    let high_read = builder
        .add_instruction(RegisterOperation::Read(ReferenceReadOperation::new()), Vec::new(), vec![halves[1]], None)
        .unwrap()[0];
    let program = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(
            vec![low_read, high_read],
            vec![Placeholder],
            vec![Placeholder; 2],
        )
        .unwrap();
    let region = program.entry_region_ref();
    let analysis = region.reference_view_analysis(0).unwrap();
    assert_eq!(analysis.overlap(region, value(1), value(2)), Some(ViewOverlap::Disjoint));
    assert_eq!(analysis.overlap(region, value(0), value(1)), Some(ViewOverlap::MayOverlap));
    assert_eq!(analysis.overlap(region, value(2), value(2)), Some(ViewOverlap::Same));
    assert_eq!(analysis.overlap(region, value(1), value(3)), None);

    // Registers have no axes, so a description batches only replicated sources and passes through unchanged, symbols
    // included.
    assert_eq!(
        RegisterView::Half(RegisterHalf::Low).batch(&root, BatchAxis::replicated()),
        Ok((RegisterView::Half(RegisterHalf::Low), BatchAxis::replicated()))
    );
    assert_eq!(
        RegisterView::Bit(ViewSymbol::Operand(1)).batch(&root, BatchAxis::replicated()),
        Ok((RegisterView::Bit(ViewSymbol::Operand(1)), BatchAxis::replicated()))
    );
    assert!(matches!(
        RegisterView::Half(RegisterHalf::High).batch(&root, BatchAxis::new(0)),
        Err(BatchingError::UnsupportedOperation { message })
            if message == "a register view cannot carry a mapped batch axis; registers have no axes",
    ));
}

#[test]
fn test_downstream_dynamic_view_analysis_closes_the_index_operand() {
    // `f(r, i, j) = read(bit(r, i))` with two more views alongside: a second bit at the same index operand, a bit at
    // another index operand, and the two static halves.
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let reference = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let i = builder.add_input(RegisterIrType::Register(RegisterType));
    let j = builder.add_input(RegisterIrType::Register(RegisterType));
    let bit_i = builder.add_instruction(RegisterOperation::Bit, Vec::new(), vec![reference, i], None).unwrap()[0];
    builder.add_instruction(RegisterOperation::Bit, Vec::new(), vec![reference, i], None).unwrap();
    builder.add_instruction(RegisterOperation::Bit, Vec::new(), vec![reference, j], None).unwrap();
    builder.add_instruction(RegisterOperation::Halves, Vec::new(), vec![reference], None).unwrap();
    let observed = builder
        .add_instruction(RegisterOperation::Read(ReferenceReadOperation::new()), Vec::new(), vec![bit_i], None)
        .unwrap()[0];
    let program = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![observed], vec![Placeholder; 3], vec![Placeholder])
        .unwrap();

    // The dynamic description names the index operand symbolically and reports it as its one symbol.
    assert_eq!(RegisterOperation::Bit.reference_view(0), Some(RegisterView::Bit(ViewSymbol::Operand(1))));
    assert_eq!(RegisterOperation::Bit.reference_view(1), None);
    assert_eq!(RegisterView::Bit(ViewSymbol::Operand(1)).symbols(), vec![ViewSymbol::Operand(1)]);
    assert_eq!(RegisterView::Half(RegisterHalf::Low).symbols(), Vec::new());

    // The overlay closes the symbol over the index operand of the instruction that created each view, so the two bits
    // at the same operand share one path, the bit at the other operand has a different path, and the overlap query
    // decides all three outcomes from the closed paths alone.
    let region = program.entry_region_ref();
    let analysis = region.reference_view_analysis(0).unwrap();
    let value = |atom: usize| ValueId::new(RegionId::new(0), AtomId::new(atom));
    assert_eq!(
        analysis.analysis().alias(value(3)),
        Some(ReferenceAliasEdge::new(
            InstructionId::new(RegionId::new(0), 0),
            ReferenceAliasOrigin::Output(0),
            value(0),
            ReferenceAliasKind::View,
            true,
        )),
    );
    let bit_path = |index: usize| {
        ReferenceViewPath::root()
            .with_step(RegisterView::Bit(ViewSymbol::Operand(1)), vec![ViewSymbolBinding::Value(value(index))])
    };
    assert_eq!(analysis.path(value(3)), Some(&bit_path(1)));
    assert_eq!(analysis.path(value(4)), Some(&bit_path(1)));
    assert_eq!(analysis.path(value(5)), Some(&bit_path(2)));
    assert_eq!(analysis.overlap(region, value(3), value(4)), Some(ViewOverlap::Same));
    assert_eq!(analysis.overlap(region, value(3), value(5)), Some(ViewOverlap::MayOverlap));
    assert_eq!(analysis.overlap(region, value(3), value(6)), Some(ViewOverlap::MayOverlap));
    assert_eq!(analysis.overlap(region, value(6), value(7)), Some(ViewOverlap::Disjoint));
    assert_eq!(analysis.overlap(region, value(0), value(3)), Some(ViewOverlap::MayOverlap));
}

#[test]
fn test_downstream_dynamic_view_reapplies_with_its_index() {
    // Reapplication binds the view over another source with the supplied index value, here eagerly into a bit handle,
    // and rejects a symbol count that disagrees with the description or a description bound to the iteration counter.
    let context = RegisterDestination::new();
    let reference = Reference::new(RegisterValue::Register(6)).unwrap();
    let source = RegisterValue::Reference(reference.clone());
    let bit = RegisterView::Bit(ViewSymbol::Operand(1));
    let reapplied =
        RegisterOperation::reapply_view(&context, &bit, source.clone(), &[RegisterValue::Register(1)]).unwrap();
    assert_eq!(reapplied, RegisterValue::BitReference { root: reference.clone(), index: 1 });
    assert_eq!(reapplied.read(), Ok(RegisterValue::Register(1)));
    assert_eq!(
        RegisterOperation::reapply_view(&context, &bit, source.clone(), &[]),
        Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
    );
    assert_eq!(
        RegisterOperation::reapply_view(
            &context,
            &RegisterView::Half(RegisterHalf::Low),
            source.clone(),
            &[RegisterValue::Register(1)],
        ),
        Err(ProgramError::InvalidInputCount { expected: 0, actual: 1 }),
    );
    assert!(matches!(
        RegisterOperation::reapply_view(
            &context,
            &RegisterView::Bit(ViewSymbol::Iteration),
            source,
            &[RegisterValue::Register(1)],
        ),
        Err(ProgramError::UnsupportedOperation { message })
            if message == "a register bit indexed by the iteration counter is created by its region-carrying operation \
                and cannot be reapplied",
    ));
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
    let bit = builder.add_instruction(RegisterOperation::Bit, Vec::new(), vec![allocation, index], None).unwrap()[0];
    let previous = builder
        .add_instruction(
            RegisterOperation::Swap(ReferenceSwapOperation::new()),
            Vec::new(),
            vec![bit, replacement],
            None,
        )
        .unwrap()[0];
    let observed = builder
        .add_instruction(RegisterOperation::Read(ReferenceReadOperation::new()), Vec::new(), vec![bit], None)
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

    // Eager execution goes through the bit handle: `r = 0b101`, bit 1 was `0`, becomes `1`, and the register ends at
    // `0b111`. Discharge into the eager destination reaches the same values through the value-bound alias, whose bit
    // step the policy reads and writes with the family's bit operations.
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

    // When the allocation is not selected, the view replays verbatim over the preserved reference and the accesses
    // through it replay as well.
    let preserved = source.partially_discharge_references(0, &[]).unwrap();
    assert_eq!(
        preserved.program().to_string(),
        indoc! {"
            lambda %0:register, %1:register, %2:register .
            let %3:ref<register> = reference_new %0
                %4:ref<register> = register.bit %3 %1
                %5:register = reference_swap %4 %2
                %6:register = reference_read %4
                %7:register = reference_freeze %3
            in (%5, %6, %7)"},
    );
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
fn test_downstream_dynamic_view_jvp_reapplies_the_view_to_the_tangent_reference() {
    // Forward mode views the tangent reference at the primal index: the primal writes bit 2 of `r = 0b001`, the tangent
    // writes bit 2 of `ṫ = 0b1000`, and each read returns its own bit. The index tangent is a coordinate tangent and is
    // dropped.
    let reference = Reference::new(RegisterValue::Register(1)).unwrap();
    let tangent_reference = Reference::new(RegisterValue::Register(8)).unwrap();
    assert_eq!(
        differentiate_at((
            RegisterValue::Reference(reference.clone()),
            RegisterValue::Register(1),
            RegisterValue::Register(2)
        ))
        .jvp(
            (
                RegisterValue::Reference(tangent_reference.clone()),
                RegisterValue::Register(1),
                RegisterValue::Register(0)
            ),
            write_read_bit,
        ),
        Ok((RegisterValue::Register(1), RegisterValue::Register(1))),
    );
    assert_eq!(reference.read(), Ok(RegisterValue::Register(5)));
    assert_eq!(tangent_reference.read(), Ok(RegisterValue::Register(12)));
}

#[test]
fn test_downstream_dynamic_view_vjp_resolves_the_index_of_the_viewed_cotangent_reference() {
    // Reverse mode reaches the cotangent of the bit through the transposition context: the linear program views the
    // tangent reference at the residual index, so the transposed program views the destination at that index, which
    // the context resolves to the transposed-program value of the index operand. The read accumulates `ȳ = 1` into
    // bit 2 of the destination (`0b1000 ↦ 0b1100`), the write's transpose swaps a zero back out of it (`x̄ = 1`, the
    // destination returns to `0b1000`), and the index receives a zero cotangent.
    let reference = Reference::new(RegisterValue::Register(1)).unwrap();
    let (value, pullback) = differentiate_at((
        RegisterValue::Reference(reference.clone()),
        RegisterValue::Register(1),
        RegisterValue::Register(2),
    ))
    .vjp(write_read_bit)
    .unwrap();
    assert_eq!(value, RegisterValue::Register(1));
    assert_eq!(reference.read(), Ok(RegisterValue::Register(5)));

    // The transposed program makes the resolution visible: the cotangent destination is viewed at the residual index
    // before the read's accumulation and the write's swap act on that view, and it is returned by identity.
    let transposed = pullback
        .linear_program()
        .transpose_with_destinations(
            &[0, 1, 2],
            &[CotangentDestinationKind::Reference, CotangentDestinationKind::Return, CotangentDestinationKind::Return],
        )
        .unwrap();
    assert_eq!(
        transposed.to_string(),
        indoc! {"
            lambda %0:register, %1:ref<register>, %2:register .
            let %3:ref<register> = register.bit %1 %2
                register.add_update %3 %0
                %4:register = register.zero
                %5:register = reference_swap %3 %4
                %6:register = register.zero
            in (%1, %5, %6)"},
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
fn test_downstream_dynamic_view_batches_through_the_shared_view_rule() {
    // The shared view rule binds the batched view on the eager parent with the packed (replicated) index, so the
    // closure runs once over the packed values exactly as the replicated-only family rule does for the other
    // operations.
    let reference = Reference::new(RegisterValue::Register(1)).unwrap();
    assert_eq!(
        batch(
            write_read_bit,
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
fn test_downstream_bit_reference_handle_accesses_one_bit_of_its_root() {
    // The eager form of a bit view is a handle over the root: it has the root's reference type and identity, reads and
    // writes bit `index` of the root while preserving the other bits, swaps and accumulates through a read and a write,
    // and rejects a value other than 0 or 1, an out-of-range index, and consumption (a bit is not a complete handle).
    let root = Reference::new(RegisterValue::Register(5)).unwrap();
    let bit = RegisterValue::BitReference { root: root.clone(), index: 1 };
    assert_eq!(bit.r#type().into_owned(), RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    assert_eq!(bit.reference_id(), Some(root.id()));
    assert_eq!(bit.to_string(), format!("{root}[bit 1]"));
    assert_eq!(bit.read(), Ok(RegisterValue::Register(0)));
    assert_eq!(bit.write(&RegisterValue::Register(1)), Ok(()));
    assert_eq!(root.read(), Ok(RegisterValue::Register(7)));
    assert_eq!(bit.swap(&RegisterValue::Register(0)), Ok(RegisterValue::Register(1)));
    assert_eq!(root.read(), Ok(RegisterValue::Register(5)));
    assert_eq!(bit.add_update(&RegisterValue::Register(1)), Ok(()));
    assert_eq!(root.read(), Ok(RegisterValue::Register(7)));
    assert!(matches!(
        bit.write(&RegisterValue::Register(2)),
        Err(ProgramError::InvalidArgument { message })
            if message == "a register bit holds 0 or 1 but 2 was stored into one",
    ));
    assert!(matches!(
        RegisterValue::BitReference { root: root.clone(), index: 64 }.read(),
        Err(ProgramError::InvalidArgument { message })
            if message == "bit index 64 is out of range for a 64-bit register",
    ));
    assert!(matches!(bit.freeze(), Err(ProgramError::Type(_))));
    assert_eq!(root.read(), Ok(RegisterValue::Register(7)));
}
