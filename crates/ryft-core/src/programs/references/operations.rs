//! Generic reference primitive operations and their value-level capabilities.
//!
//! This module owns reference allocation, immutable reads, write-only replacement, swapping, ordered additive
//! updates, and consuming finalization. Each payload is parameterized by the referent type `T` and the enclosing
//! program type universe `U`; the ordinary [`From`] and borrowed [`TryFrom`] conversion seam is the complete
//! relationship between those types. Reference views remain value-family-owned because their coordinate
//! transformations are not generic.

// TODO(eaplatanios): Review from here onwards.

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::LazyLock;

use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiationDriver, DifferentiationDual, DifferentiationError,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_transposable_operation};
use crate::operations::AddOperation;
use crate::parameters::Parameter;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::effects::{Effect, Effects};
use crate::programs::operations::Operation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};

use super::discharge::{
    ReferenceAccumulationPolicy, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy,
    ReferenceDischargeValue, ReferenceDischargeableOperation,
};
use super::semantics::{ReferenceAccessMode, ReferenceInput, ReferenceOperationSemantics, ReferenceOutput};
use super::types::ReferenceType;

/// Canonical operation name for [`ReferenceNewOperation`].
pub const REFERENCE_NEW_OPERATION_NAME: &str = "reference_new";

/// Canonical operation name for [`ReferenceReadOperation`].
pub const REFERENCE_READ_OPERATION_NAME: &str = "reference_read";

/// Canonical operation name for [`ReferenceWriteOperation`].
pub const REFERENCE_WRITE_OPERATION_NAME: &str = "reference_write";

/// Canonical operation name for [`ReferenceSwapOperation`].
pub const REFERENCE_SWAP_OPERATION_NAME: &str = "reference_swap";

/// Canonical operation name for [`ReferenceAddUpdateOperation`].
pub const REFERENCE_ADD_UPDATE_OPERATION_NAME: &str = "reference_add_update";

/// Canonical operation name for [`ReferenceFreezeOperation`].
pub const REFERENCE_FREEZE_OPERATION_NAME: &str = "reference_freeze";

/// Creates a new reference initialized from this value.
pub trait ReferenceNew<Output = Self>: Sized {
    /// Creates an independent reference whose initial state is this value.
    fn reference_new(&self) -> Result<Output, ProgramError>;
}

/// Reads an immutable snapshot from a reference value.
pub trait ReferenceRead<Output = Self>: Sized {
    /// Returns the reference's current value as an immutable snapshot.
    fn read(&self) -> Result<Output, ProgramError>;
}

/// Replaces the value stored by a reference without observing the previous value.
pub trait ReferenceWrite<Replacement = Self>: Sized {
    /// Installs `replacement` in program order.
    fn write(&self, replacement: &Replacement) -> Result<(), ProgramError>;
}

/// Replaces the value stored by a reference in program order and returns its previous immutable snapshot.
pub trait ReferenceSwap<Replacement = Self, Output = Replacement>: Sized {
    /// Installs `replacement` in program order and returns the previously stored value.
    fn swap(&self, replacement: &Replacement) -> Result<Output, ProgramError>;
}

/// Adds an update into the value stored by a reference in program order.
pub trait ReferenceAddUpdate<Update = Self>: Sized {
    /// Adds `update` to the stored value in program order.
    fn add_update(&self, update: &Update) -> Result<(), ProgramError>;
}

/// Consumes a reference, returning its final value and invalidating its complete alias family.
pub trait ReferenceFreeze<Output = Self>: Sized {
    /// Returns the final stored value and invalidates this reference and all aliases.
    ///
    /// The handle is taken by value, because consumption is linear: after this call the reference denotes nothing.
    /// Passing it by value makes the common single-handle misuse — freezing and then reading through the same
    /// binding — a compile error rather than a runtime one. Aliases obtained by cloning the handle are a different
    /// case and remain a dynamic failure, because the type system cannot see them: an eager alias fails at its next
    /// access against the shared reference state, and a staged alias fails while tracing, because every clone of one
    /// [`Tracer`](crate::Tracer) names the same staged atom. Freezing through a shared borrow is therefore an
    /// explicit clone-then-freeze, which reads as the deliberate act it is.
    ///
    /// ```compile_fail
    /// use ryft_core::{Array, ArrayIrValue, ReferenceFreeze, ReferenceNew, ReferenceRead};
    ///
    /// let root = ArrayIrValue::Array(Array::scalar(1.0_f32)).reference_new()?;
    /// let frozen = root.freeze()?;
    /// // The handle was consumed, so reading it again does not compile.
    /// let stale = root.read()?;
    /// # Ok::<(), ryft_core::ProgramError>(())
    /// ```
    ///
    /// ```
    /// use ryft_core::{Array, ArrayIrValue, ReferenceFreeze, ReferenceNew, ReferenceError, ReferenceRead};
    ///
    /// // A clone is a separate handle onto the same reference allocation, so misuse is caught dynamically instead.
    /// let root = ArrayIrValue::Array(Array::scalar(1.0_f32)).reference_new()?;
    /// let alias = root.clone();
    /// assert_eq!(root.freeze()?, ArrayIrValue::Array(Array::scalar(1.0_f32)));
    /// assert_eq!(
    ///     alias.read().unwrap_err().downcast_custom::<ReferenceError>(),
    ///     Some(&ReferenceError::Frozen),
    /// );
    /// # Ok::<(), ryft_core::ProgramError>(())
    /// ```
    fn freeze(self) -> Result<Output, ProgramError>;
}

// Reference semantics descriptors are constant per operation type. Sharing them through `LazyLock` statics lets the
// per-instruction program analysis read them through `Cow::Borrowed` without allocating.
static REFERENCE_NEW_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> =
    LazyLock::new(|| ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceOutput::Root { output_index: 0 }]));

static REFERENCE_READ_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(vec![ReferenceInput::new(0, ReferenceAccessMode::Read)], Vec::new())
});

static REFERENCE_WRITE_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(vec![ReferenceInput::new(0, ReferenceAccessMode::Write)], Vec::new())
});

static REFERENCE_SWAP_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(vec![ReferenceInput::new(0, ReferenceAccessMode::ReadWrite)], Vec::new())
});

static REFERENCE_ADD_UPDATE_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(vec![ReferenceInput::new(0, ReferenceAccessMode::Accumulate)], Vec::new())
});

static REFERENCE_FREEZE_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(vec![ReferenceInput::new(0, ReferenceAccessMode::Consume)], Vec::new())
});

macro_rules! define_reference_primitive_payload {
    // Defines one type-indexed zero-sized payload without deriving unnecessary bounds on its phantom type parameters.
    ($(#[$documentation:meta])* $operation:ident) => {
        $(#[$documentation])*
        pub struct $operation<T: Type, U: Type>(PhantomData<fn() -> (T, U)>);

        impl<T: Type, U: Type> $operation<T, U> {
            #[doc = concat!("Creates a new [`", stringify!($operation), "`].")]
            #[inline]
            pub const fn new() -> Self {
                Self(PhantomData)
            }
        }

        impl<T: Type, U: Type> Copy for $operation<T, U> {}

        impl<T: Type, U: Type> Clone for $operation<T, U> {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<T: Type, U: Type> Debug for $operation<T, U> {
            #[inline]
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(stringify!($operation))
            }
        }

        impl<T: Type, U: Type> Default for $operation<T, U> {
            #[inline]
            fn default() -> Self {
                Self::new()
            }
        }

        impl<T: Type, U: Type> PartialEq for $operation<T, U> {
            #[inline]
            fn eq(&self, _other: &Self) -> bool {
                true
            }
        }

        impl<T: Type, U: Type> Eq for $operation<T, U> {}

        impl<T: Type, U: Type> Hash for $operation<T, U> {
            #[inline]
            fn hash<H: Hasher>(&self, _state: &mut H) {}
        }

        impl<T: Type, U: Type> Parameter for $operation<T, U> {}
    };
}

define_reference_primitive_payload!(
    /// Allocates a reference root for a referent of type `T` in the enclosing type universe `U`.
    ReferenceNewOperation
);

define_reference_primitive_payload!(
    /// Reads the current referent snapshot from a reference in the enclosing type universe `U`.
    ReferenceReadOperation
);

define_reference_primitive_payload!(
    /// Replaces a reference's stored value with an exactly matching referent without observing the old value.
    ReferenceWriteOperation
);

define_reference_primitive_payload!(
    /// Replaces a reference's stored value with an exactly matching referent and returns the old value.
    ReferenceSwapOperation
);

define_reference_primitive_payload!(
    /// Applies an ordered additive update whose result must retain the reference's exact referent type.
    ReferenceAddUpdateOperation
);

define_reference_primitive_payload!(
    /// Consumes a root reference, returning its final referent and invalidating its complete alias family.
    ReferenceFreezeOperation
);

macro_rules! impl_reference_primitive_display {
    // Renders one payload using its canonical operation name without requiring its conversion seam.
    ($operation:ident, $name:ident) => {
        impl<T: Type, U: Type> Display for $operation<T, U> {
            #[inline]
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str($name)
            }
        }
    };
}

impl_reference_primitive_display!(ReferenceNewOperation, REFERENCE_NEW_OPERATION_NAME);
impl_reference_primitive_display!(ReferenceReadOperation, REFERENCE_READ_OPERATION_NAME);
impl_reference_primitive_display!(ReferenceWriteOperation, REFERENCE_WRITE_OPERATION_NAME);
impl_reference_primitive_display!(ReferenceSwapOperation, REFERENCE_SWAP_OPERATION_NAME);
impl_reference_primitive_display!(ReferenceAddUpdateOperation, REFERENCE_ADD_UPDATE_OPERATION_NAME);
impl_reference_primitive_display!(ReferenceFreezeOperation, REFERENCE_FREEZE_OPERATION_NAME);

impl<T, U> Operation for ReferenceNewOperation<T, U>
where
    T: Type,
    U: Type + From<ReferenceType<T>>,
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_NEW_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let referent = <&T>::try_from(&input_types[0])?;
        if referent.is_reference() {
            return Err(TypeError::invalid(format!(
                "`{REFERENCE_NEW_OPERATION_NAME}` cannot allocate a reference whose referent type `{referent}` is \
                 itself a reference",
            )));
        }
        Ok(vec![ReferenceType::new(referent.clone()).into()])
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_NEW_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<T, U> Operation for ReferenceReadOperation<T, U>
where
    T: Type,
    U: Type + From<T>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_READ_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<T>>::try_from(&input_types[0])?;
        Ok(vec![reference.referent().clone().into()])
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_READ_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<T, U> Operation for ReferenceWriteOperation<T, U>
where
    T: Type,
    U: Type,
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_WRITE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<T>>::try_from(&input_types[0])?;
        let replacement = <&T>::try_from(&input_types[1])?;
        if replacement != reference.referent() {
            return Err(TypeError::invalid(format!(
                "`{REFERENCE_WRITE_OPERATION_NAME}` replacement type `{replacement}` must exactly match reference \
                 referent type `{}`",
                reference.referent(),
            )));
        }
        Ok(Vec::new())
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_WRITE_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<T, U> Operation for ReferenceSwapOperation<T, U>
where
    T: Type,
    U: Type + From<T>,
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_SWAP_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<T>>::try_from(&input_types[0])?;
        let replacement = <&T>::try_from(&input_types[1])?;
        if replacement != reference.referent() {
            return Err(TypeError::invalid(format!(
                "`{REFERENCE_SWAP_OPERATION_NAME}` replacement type `{replacement}` must exactly match reference \
                 referent type `{}`",
                reference.referent(),
            )));
        }
        Ok(vec![reference.referent().clone().into()])
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_SWAP_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<T, U> Operation for ReferenceAddUpdateOperation<T, U>
where
    T: Type,
    U: Type,
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
    AddOperation<T>: Operation<Type = T>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_ADD_UPDATE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<T>>::try_from(&input_types[0])?;
        let update = <&T>::try_from(&input_types[1])?;
        let addition_results =
            AddOperation::<T>::new().infer_output_types(&[reference.referent().clone(), update.clone()], &[])?;
        check_count!("output", addition_results, 1, TypeError);
        let addition_result = &addition_results[0];
        if addition_result != reference.referent() {
            return Err(TypeError::invalid(format!(
                "`{REFERENCE_ADD_UPDATE_OPERATION_NAME}` addition result type `{addition_result}` must exactly match \
                 reference referent type `{}`",
                reference.referent(),
            )));
        }
        Ok(Vec::new())
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_ADD_UPDATE_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<T, U> Operation for ReferenceFreezeOperation<T, U>
where
    T: Type,
    U: Type + From<T>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_FREEZE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<T>>::try_from(&input_types[0])?;
        Ok(vec![reference.referent().clone().into()])
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_FREEZE_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceNewOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceNewOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceNew<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].reference_new()?])
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceReadOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceReadOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceRead<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].read()?])
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceWriteOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceWriteOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceWrite<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        inputs[0].write(&inputs[1])?;
        Ok(Vec::new())
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceSwapOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceSwapOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceSwap<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].swap(&inputs[1])?])
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceAddUpdateOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceAddUpdateOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceAddUpdate<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        inputs[0].add_update(&inputs[1])?;
        Ok(Vec::new())
    }
}

impl<T, U, C> InterpretableOperation<C> for ReferenceFreezeOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceFreezeOperation<T, U>: Operation<Type = U>,
    C: Domain<Type = U, Value: ReferenceFreeze<C::Value>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);

        // Interpretation replays an already-built instruction, so the operand is borrowed from the environment rather
        // than owned, and cloning it is the faithful replay: a clone names the same root, so consuming it invalidates
        // the whole alias family exactly as the source program asked. The linearity the value-level capability
        // enforces is not weakened by the clone, because it was never this layer's to enforce: a staged handle is
        // held to it while the program is traced, and an eager clone shares the holder that reports the misuse.
        Ok(vec![inputs[0].clone().freeze()?])
    }
}

/// Re-derives one reference primitive's own type inference over the carriers it received, so that a rewrite acts only
/// on operands the operation itself accepts.
///
/// A [`Program`](crate::Program) built through a [`ProgramBuilder`](crate::ProgramBuilder) already ran this inference
/// when the instruction was added, but a rule invoked outside a checked program replay has no such guarantee, and the
/// rules that relate two operands to each other cannot recover that relationship from the carriers alone. Only those
/// rules call this: an allocation derives its own output type instead, and a read or a freeze relates no operands, so
/// re-deriving would restate the projection the rule already performs.
///
/// # Parameters
///
///   - `operation`: Reference primitive whose inference is re-derived.
///   - `inputs`: Carriers supplied as this application's operands, in operation-defined order.
fn validate_operand_types<U: Type, C, P, O>(
    operation: &O,
    inputs: &[ReferenceDischargeValue<C, P>],
) -> Result<(), ProgramError>
where
    C: Domain<Type = U>,
    P: ReferenceDischargePolicy<C>,
    O: Operation<Type = U>,
{
    let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    operation.infer_output_types(input_types.as_slice(), &[])?;
    Ok(())
}

// Every reference primitive owns its own discharge rewrite, and each one is expressed purely through the discharge
// context's root services and the policy's alias mechanics, so these six rules serve every reference universe. None
// of them names the referent type parameter `T`: an allocation reads its fresh root's reference type back out of its
// own inferred output type through the policy's projection, and every access reads the type off the flowing handle.
//
// Accesses to the roots partial discharge *preserves* never reach these rules: the dispatch path replays them
// verbatim through `discharge_preserved_access` before rule dispatch, so each access rule below rewrites only
// discharged roots. The rules still bind their destination by `Context` rather than by `Domain` because the
// allocation rule and the shared replay path require the conversion seam into the destination's own operation
// family.
impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceNewOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceNewOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceNewOperation<T, U>>>,
    P: ReferenceDischargePolicy<C>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let initial = inputs[0].expect_ordinary("an initial reference state")?.clone();

        // The allocation's reference type is exactly the one this operation's own inference derives from the
        // initializer, so the rewrite never re-derives a referent that the type system already settled.
        let output_types = self.infer_output_types(&[initial.r#type().into_owned()], &[])?;
        check_count!("output", output_types, 1, ProgramError);
        let r#type = P::project_reference_type(&output_types[0]).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "`{REFERENCE_NEW_OPERATION_NAME}` inferred the non-reference output type `{}`",
                output_types[0],
            ))
        })?;
        if context.selects_allocation(driver.instruction(), 0) {
            return Ok(vec![context.allocate_discharged(r#type, initial)?]);
        }

        // An unselected allocation site survives, so the allocation is replayed and the root it binds is the
        // destination reference that replay produced.
        let mut outputs = context.parent().bind(*self, Vec::new(), std::slice::from_ref(&initial))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![context.bind_preserved(r#type, outputs.remove(0))?])
    }
}

impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceReadOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceReadOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceReadOperation<T, U>>>,
    P: ReferenceDischargePolicy<C>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let reference = inputs[0].expect_reference("a reference to read")?;
        Ok(vec![ReferenceDischargeValue::Ordinary(context.read(reference)?)])
    }
}

impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceWriteOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceWriteOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceWriteOperation<T, U>>>,
    P: ReferenceDischargePolicy<C>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let reference = inputs[0].expect_reference("a reference to write")?;
        let replacement = inputs[1].expect_ordinary("a replacement value")?.clone();
        validate_operand_types(self, inputs)?;
        context.write(reference, replacement)?;
        Ok(Vec::new())
    }
}

impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceSwapOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceSwapOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceSwapOperation<T, U>>>,
    P: ReferenceDischargePolicy<C>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let reference = inputs[0].expect_reference("a reference to replace")?;
        let replacement = inputs[1].expect_ordinary("a replacement value")?.clone();

        // The replacement must carry exactly the handle's referent. A universe whose write mechanics only require the
        // replacement to fit inside the selected coordinates would otherwise perform a silent partial write, so the
        // rule re-derives the operand relationship its own inference already states.
        validate_operand_types(self, inputs)?;
        Ok(vec![ReferenceDischargeValue::Ordinary(context.replace(reference, replacement)?)])
    }
}

// Accumulation is the one rule that needs more than the base policy, so it is also the only one that names
// `ReferenceAccumulationPolicy`. A universe that cannot accumulate therefore fails to discharge exactly the programs
// that contain this operation, and keeps discharging every program that only reads and replaces. The requirement holds
// even where a root is preserved and the accumulation is only replayed, because whether a program is dischargeable at
// all must not depend on which sites the caller happened to select.
impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceAddUpdateOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceAddUpdateOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceAddUpdateOperation<T, U>>>,
    P: ReferenceAccumulationPolicy<C>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let reference = inputs[0].expect_reference("a reference to accumulate into")?;
        let update = inputs[1].expect_ordinary("an update value")?.clone();

        // The sum of the handle's referent and the update must itself be the handle's referent, which is exactly what
        // this operation's own inference states and what a universe's addition alone does not guarantee.
        validate_operand_types(self, inputs)?;
        context.accumulate(reference, update)?;
        Ok(Vec::new())
    }
}

impl<T, U, C, P> ReferenceDischargeableOperation<C, P> for ReferenceFreezeOperation<T, U>
where
    T: Type,
    U: Type,
    ReferenceFreezeOperation<T, U>: Operation<Type = U>,
    C: Context<Type = U, Operation: From<ReferenceFreezeOperation<T, U>>>,
    P: ReferenceDischargePolicy<C>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let reference = inputs[0].expect_reference("a reference to freeze")?;
        Ok(vec![ReferenceDischargeValue::Ordinary(context.consume(reference)?)])
    }
}

macro_rules! impl_unsupported_reference_transforms {
    // Installs the same conservative transform rejections for one unresolved generic reference primitive.
    ($operation:ident) => {
        impl<T, U, C> PartiallyEvaluatableOperation<C> for $operation<T, U>
        where
            T: Type,
            U: Type,
            C: Context<Type = U, Operation: From<$operation<T, U>>>,
        {
        }

        impl<T, U, C, P> BatchableOperation<C, P> for $operation<T, U>
        where
            T: Type,
            U: Type,
            $operation<T, U>: Operation<Type = U>,
            C: Context<Type = U, Operation: From<$operation<T, U>>>,
            P: BatchingPolicy<C>,
        {
            fn batch<D: BatchingDriver<C, P>>(
                &self,
                _context: &BatchingContext<C, P>,
                _driver: &D,
                _inputs: &[P::Batch],
            ) -> Result<BatchedOutputs<C, P>, BatchingError> {
                Err(BatchingError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before batching", self.name()),
                })
            }
        }

        impl<T, U, C> DifferentiableOperation<C> for $operation<T, U>
        where
            T: Type,
            U: Type,
            $operation<T, U>: Operation<Type = U>,
            C: Context<Type = U, Operation: From<$operation<T, U>>>,
        {
            fn jvp<D: DifferentiationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                _inputs: &[DifferentiationDual<C::Value>],
            ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
                Err(ProgramError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before differentiation", self.name()),
                }
                .into())
            }
        }
    };
}

impl_unsupported_reference_transforms!(ReferenceNewOperation);
impl_unsupported_reference_transforms!(ReferenceReadOperation);
impl_unsupported_reference_transforms!(ReferenceWriteOperation);
impl_unsupported_reference_transforms!(ReferenceSwapOperation);
impl_unsupported_reference_transforms!(ReferenceAddUpdateOperation);
impl_unsupported_reference_transforms!(ReferenceFreezeOperation);

impl_non_transposable_operation!(
    <T, U> ReferenceNewOperation<T, U>
    where
        T: Type,
        U: Type,
);
impl_non_transposable_operation!(
    <T, U> ReferenceReadOperation<T, U>
    where
        T: Type,
        U: Type,
);
impl_non_transposable_operation!(
    <T, U> ReferenceWriteOperation<T, U>
    where
        T: Type,
        U: Type,
);
impl_non_transposable_operation!(
    <T, U> ReferenceSwapOperation<T, U>
    where
        T: Type,
        U: Type,
);
impl_non_transposable_operation!(
    <T, U> ReferenceAddUpdateOperation<T, U>
    where
        T: Type,
        U: Type,
);
impl_non_transposable_operation!(
    <T, U> ReferenceFreezeOperation<T, U>
    where
        T: Type,
        U: Type,
);

#[cfg(test)]
mod tests {
    use std::fmt::Display;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::operations::Add;
    use crate::parameters::{Parameter, Parameterized, Placeholder};
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::identities::{TypeIdentity, TypeIdentityPosition, TypeIdentityRenaming};
    use crate::programs::references::discharge::{ReferenceDischargeReference, discharge_reference_free_operation};
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::Type;
    use crate::programs::values::Value;

    use super::*;

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    struct TestIdentity(u8);

    impl Display for TestIdentity {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "i{}", self.0)
        }
    }

    impl TypeIdentity for TestIdentity {
        fn fresh(&self) -> Self {
            Self(self.0.wrapping_add(128))
        }
    }

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    struct TestReferent {
        identity: TestIdentity,
        precision: u8,
    }

    impl TestReferent {
        const fn new(identity: u8, precision: u8) -> Self {
            Self { identity: TestIdentity(identity), precision }
        }
    }

    impl Display for TestReferent {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "value<{},p{}>", self.identity, self.precision)
        }
    }

    impl Parameter for TestReferent {}

    impl Type for TestReferent {
        type Identity = TestIdentity;
        type Refinements = ();

        fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
            std::iter::once((TypeIdentityPosition::Definition, &self.identity))
        }

        fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
            Ok(Self { identity: renaming.rename(&self.identity), precision: self.precision })
        }

        fn is_compatible_with(&self, other: &Self) -> bool {
            self.identity == other.identity && self.precision <= other.precision
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self.identity == other.identity && self.precision <= other.precision
        }

        fn is_scalar(&self) -> bool {
            true
        }

        fn is_complex(&self) -> bool {
            false
        }
    }

    impl Operation for AddOperation<TestReferent> {
        type Type = TestReferent;

        fn name(&self) -> &'static str {
            "test_add"
        }

        fn infer_output_types(
            &self,
            input_types: &[TestReferent],
            region_interfaces: &[RegionInterface<TestReferent>],
        ) -> Result<Vec<TestReferent>, TypeError> {
            check_count!("input", input_types, 2, TypeError);
            check_count!("region", region_interfaces, 0, TypeError);
            Ok(vec![TestReferent {
                identity: input_types[0].identity,
                precision: input_types.iter().map(|r#type| r#type.precision).max().unwrap(),
            }])
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    enum TestType {
        Value(TestReferent),
        Reference(ReferenceType<TestReferent>),
    }

    impl Display for TestType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Value(r#type) => Display::fmt(r#type, formatter),
                Self::Reference(r#type) => Display::fmt(r#type, formatter),
            }
        }
    }

    impl Parameter for TestType {}

    impl From<TestReferent> for TestType {
        fn from(r#type: TestReferent) -> Self {
            Self::Value(r#type)
        }
    }

    impl From<ReferenceType<TestReferent>> for TestType {
        fn from(r#type: ReferenceType<TestReferent>) -> Self {
            Self::Reference(r#type)
        }
    }

    impl<'t> TryFrom<&'t TestType> for &'t TestReferent {
        type Error = TypeError;

        fn try_from(r#type: &'t TestType) -> Result<Self, Self::Error> {
            match r#type {
                TestType::Value(r#type) => Ok(r#type),
                TestType::Reference(_) => Err(TypeError::invalid("expected value type but got reference type")),
            }
        }
    }

    impl<'t> TryFrom<&'t TestType> for &'t ReferenceType<TestReferent> {
        type Error = TypeError;

        fn try_from(r#type: &'t TestType) -> Result<Self, Self::Error> {
            match r#type {
                TestType::Reference(r#type) => Ok(r#type),
                TestType::Value(_) => Err(TypeError::invalid("expected reference type but got value type")),
            }
        }
    }

    impl Type for TestType {
        type Identity = TestIdentity;
        type Refinements = ();

        fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
            match self {
                Self::Value(r#type) => r#type.identities().collect::<Vec<_>>(),
                Self::Reference(r#type) => r#type.identities().collect::<Vec<_>>(),
            }
            .into_iter()
        }

        fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
            Ok(match self {
                Self::Value(r#type) => Self::Value(r#type.rename_identities(renaming)?),
                Self::Reference(r#type) => Self::Reference(r#type.rename_identities(renaming)?),
            })
        }

        fn is_compatible_with(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Value(left), Self::Value(right)) => left.is_compatible_with(right),
                (Self::Reference(left), Self::Reference(right)) => left.is_compatible_with(right),
                _ => false,
            }
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Value(left), Self::Value(right)) => left.is_refined_by(right),
                (Self::Reference(left), Self::Reference(right)) => left.is_refined_by(right),
                _ => false,
            }
        }

        fn is_scalar(&self) -> bool {
            matches!(self, Self::Value(r#type) if r#type.is_scalar())
        }

        fn is_complex(&self) -> bool {
            matches!(self, Self::Value(r#type) if r#type.is_complex())
        }

        fn is_reference(&self) -> bool {
            matches!(self, Self::Reference(_))
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    enum NestedTestType {
        Reference(ReferenceType<TestReferent>),
        Nested(ReferenceType<ReferenceType<TestReferent>>),
    }

    impl Display for NestedTestType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Reference(r#type) => Display::fmt(r#type, formatter),
                Self::Nested(r#type) => Display::fmt(r#type, formatter),
            }
        }
    }

    impl Parameter for NestedTestType {}

    impl From<ReferenceType<ReferenceType<TestReferent>>> for NestedTestType {
        fn from(r#type: ReferenceType<ReferenceType<TestReferent>>) -> Self {
            Self::Nested(r#type)
        }
    }

    impl<'t> TryFrom<&'t NestedTestType> for &'t ReferenceType<TestReferent> {
        type Error = TypeError;

        fn try_from(r#type: &'t NestedTestType) -> Result<Self, Self::Error> {
            match r#type {
                NestedTestType::Reference(r#type) => Ok(r#type),
                NestedTestType::Nested(_) => {
                    Err(TypeError::invalid("expected reference type but got nested reference type"))
                }
            }
        }
    }

    impl Type for NestedTestType {
        type Identity = TestIdentity;
        type Refinements = ();

        fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
            match self {
                Self::Reference(r#type) => r#type.identities().collect::<Vec<_>>(),
                Self::Nested(r#type) => r#type.identities().collect::<Vec<_>>(),
            }
            .into_iter()
        }

        fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
            Ok(match self {
                Self::Reference(r#type) => Self::Reference(r#type.rename_identities(renaming)?),
                Self::Nested(r#type) => Self::Nested(r#type.rename_identities(renaming)?),
            })
        }

        fn is_compatible_with(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Reference(left), Self::Reference(right)) => left.is_compatible_with(right),
                (Self::Nested(left), Self::Nested(right)) => left.is_compatible_with(right),
                _ => false,
            }
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Reference(left), Self::Reference(right)) => left.is_refined_by(right),
                (Self::Nested(left), Self::Nested(right)) => left.is_refined_by(right),
                _ => false,
            }
        }

        fn is_scalar(&self) -> bool {
            false
        }

        fn is_complex(&self) -> bool {
            false
        }

        fn is_reference(&self) -> bool {
            true
        }
    }

    macro_rules! define_partial_test_universe {
        // Defines one non-array universe whose conversion implementations are selected independently below.
        ($name:ident) => {
            #[derive(Clone, Debug, PartialEq, ryft_macros::Parameter)]
            enum $name {
                Value(TestReferent),
                Reference(ReferenceType<TestReferent>),
            }

            impl Display for $name {
                fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    match self {
                        Self::Value(r#type) => Display::fmt(r#type, formatter),
                        Self::Reference(r#type) => Display::fmt(r#type, formatter),
                    }
                }
            }

            impl Type for $name {
                type Identity = TestIdentity;
                type Refinements = ();

                fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
                    match self {
                        Self::Value(r#type) => r#type.identities().collect::<Vec<_>>(),
                        Self::Reference(r#type) => r#type.identities().collect::<Vec<_>>(),
                    }
                    .into_iter()
                }

                fn rename_identities(
                    &self,
                    renaming: &TypeIdentityRenaming<Self::Identity>,
                ) -> Result<Self, TypeError> {
                    Ok(match self {
                        Self::Value(r#type) => Self::Value(r#type.rename_identities(renaming)?),
                        Self::Reference(r#type) => Self::Reference(r#type.rename_identities(renaming)?),
                    })
                }

                fn is_compatible_with(&self, other: &Self) -> bool {
                    match (self, other) {
                        (Self::Value(left), Self::Value(right)) => left.is_compatible_with(right),
                        (Self::Reference(left), Self::Reference(right)) => left.is_compatible_with(right),
                        _ => false,
                    }
                }

                fn is_refined_by(&self, other: &Self) -> bool {
                    match (self, other) {
                        (Self::Value(left), Self::Value(right)) => left.is_refined_by(right),
                        (Self::Reference(left), Self::Reference(right)) => left.is_refined_by(right),
                        _ => false,
                    }
                }

                fn is_scalar(&self) -> bool {
                    matches!(self, Self::Value(r#type) if r#type.is_scalar())
                }

                fn is_complex(&self) -> bool {
                    matches!(self, Self::Value(r#type) if r#type.is_complex())
                }

                fn is_reference(&self) -> bool {
                    matches!(self, Self::Reference(_))
                }
            }
        };
    }

    define_partial_test_universe!(NewUniverse);
    define_partial_test_universe!(ReadFreezeUniverse);
    define_partial_test_universe!(WriteUniverse);
    define_partial_test_universe!(SwapUniverse);
    define_partial_test_universe!(AddUpdateUniverse);

    impl From<ReferenceType<TestReferent>> for NewUniverse {
        fn from(r#type: ReferenceType<TestReferent>) -> Self {
            Self::Reference(r#type)
        }
    }

    impl<'t> TryFrom<&'t NewUniverse> for &'t TestReferent {
        type Error = TypeError;

        fn try_from(r#type: &'t NewUniverse) -> Result<Self, Self::Error> {
            match r#type {
                NewUniverse::Value(r#type) => Ok(r#type),
                NewUniverse::Reference(_) => Err(TypeError::invalid("expected value type but got reference type")),
            }
        }
    }

    impl From<TestReferent> for ReadFreezeUniverse {
        fn from(r#type: TestReferent) -> Self {
            Self::Value(r#type)
        }
    }

    impl<'t> TryFrom<&'t ReadFreezeUniverse> for &'t ReferenceType<TestReferent> {
        type Error = TypeError;

        fn try_from(r#type: &'t ReadFreezeUniverse) -> Result<Self, Self::Error> {
            match r#type {
                ReadFreezeUniverse::Reference(r#type) => Ok(r#type),
                ReadFreezeUniverse::Value(_) => Err(TypeError::invalid("expected reference type but got value type")),
            }
        }
    }

    impl<'t> TryFrom<&'t WriteUniverse> for &'t TestReferent {
        type Error = TypeError;

        fn try_from(r#type: &'t WriteUniverse) -> Result<Self, Self::Error> {
            match r#type {
                WriteUniverse::Value(r#type) => Ok(r#type),
                WriteUniverse::Reference(_) => Err(TypeError::invalid("expected value type but got reference type")),
            }
        }
    }

    impl<'t> TryFrom<&'t WriteUniverse> for &'t ReferenceType<TestReferent> {
        type Error = TypeError;

        fn try_from(r#type: &'t WriteUniverse) -> Result<Self, Self::Error> {
            match r#type {
                WriteUniverse::Reference(r#type) => Ok(r#type),
                WriteUniverse::Value(_) => Err(TypeError::invalid("expected reference type but got value type")),
            }
        }
    }

    impl From<TestReferent> for SwapUniverse {
        fn from(r#type: TestReferent) -> Self {
            Self::Value(r#type)
        }
    }

    impl<'t> TryFrom<&'t SwapUniverse> for &'t TestReferent {
        type Error = TypeError;

        fn try_from(r#type: &'t SwapUniverse) -> Result<Self, Self::Error> {
            match r#type {
                SwapUniverse::Value(r#type) => Ok(r#type),
                SwapUniverse::Reference(_) => Err(TypeError::invalid("expected value type but got reference type")),
            }
        }
    }

    impl<'t> TryFrom<&'t SwapUniverse> for &'t ReferenceType<TestReferent> {
        type Error = TypeError;

        fn try_from(r#type: &'t SwapUniverse) -> Result<Self, Self::Error> {
            match r#type {
                SwapUniverse::Reference(r#type) => Ok(r#type),
                SwapUniverse::Value(_) => Err(TypeError::invalid("expected reference type but got value type")),
            }
        }
    }

    impl<'t> TryFrom<&'t AddUpdateUniverse> for &'t TestReferent {
        type Error = TypeError;

        fn try_from(r#type: &'t AddUpdateUniverse) -> Result<Self, Self::Error> {
            match r#type {
                AddUpdateUniverse::Value(r#type) => Ok(r#type),
                AddUpdateUniverse::Reference(_) => {
                    Err(TypeError::invalid("expected value type but got reference type"))
                }
            }
        }
    }

    impl<'t> TryFrom<&'t AddUpdateUniverse> for &'t ReferenceType<TestReferent> {
        type Error = TypeError;

        fn try_from(r#type: &'t AddUpdateUniverse) -> Result<Self, Self::Error> {
            match r#type {
                AddUpdateUniverse::Reference(r#type) => Ok(r#type),
                AddUpdateUniverse::Value(_) => Err(TypeError::invalid("expected reference type but got value type")),
            }
        }
    }

    type New = ReferenceNewOperation<TestReferent, TestType>;
    type Read = ReferenceReadOperation<TestReferent, TestType>;
    type Write = ReferenceWriteOperation<TestReferent, TestType>;
    type Swap = ReferenceSwapOperation<TestReferent, TestType>;
    type AddUpdate = ReferenceAddUpdateOperation<TestReferent, TestType>;
    type Freeze = ReferenceFreezeOperation<TestReferent, TestType>;

    // The fixtures below give the reference-primitive discharge rules a destination to write into. The universe is
    // deliberately view-less, so that these tests isolate the rules themselves: composed views are the policy's
    // concern and are covered where a universe with real view mechanics lives.
    //
    // Two destinations are named because the rules serve two kinds of root. A discharged root's rewrite reaches only
    // ordinary values, so the eager destination executes it and the tests read the results directly. A preserved root
    // is a *reference* of the destination universe, which an eager value of this fixture cannot be, so the preserved
    // replay is exercised against the staging destination and read back as the program it recorded.

    /// Destination universe of the discharge-rule tests.
    type TestDestination = EagerContext<TestValue, TestOperation>;

    /// Discharge context over the discharge-rule test destination.
    type TestDischargeContext = ReferenceDischargeContext<TestDestination, TestReferenceDischarge>;

    /// Carrier flowing through the discharge-rule tests.
    type TestDischargeValue = ReferenceDischargeValue<TestDestination, TestReferenceDischarge>;

    /// Operation family of the discharge-rule test destinations.
    ///
    /// It carries the six reference primitives, because replaying an access to a preserved root binds the access
    /// itself into the destination, and one addition, because that is how this universe's accumulation policy reaches
    /// its sum. A real family reaches the same shape through a dispatch derive.
    #[derive(Copy, Clone, Debug)]
    enum TestOperation {
        Add,
        New(New),
        Read(Read),
        Write(Write),
        Swap(Swap),
        AddUpdate(AddUpdate),
        Freeze(Freeze),
    }

    impl Display for TestOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation for TestOperation {
        type Type = TestType;

        fn name(&self) -> &'static str {
            match self {
                Self::Add => "test.add",
                Self::New(operation) => operation.name(),
                Self::Read(operation) => operation.name(),
                Self::Write(operation) => operation.name(),
                Self::Swap(operation) => operation.name(),
                Self::AddUpdate(operation) => operation.name(),
                Self::Freeze(operation) => operation.name(),
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[TestType],
            region_interfaces: &[RegionInterface<TestType>],
        ) -> Result<Vec<TestType>, TypeError> {
            match self {
                Self::Add => {
                    check_count!("input", input_types, 2, TypeError);
                    if input_types[0] != input_types[1] {
                        return Err(TypeError::invalid(format!(
                            "`test.add` cannot add `{}` to `{}`",
                            input_types[1], input_types[0],
                        )));
                    }
                    Ok(vec![input_types[0].clone()])
                }
                Self::New(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::Read(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::Write(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::Swap(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::AddUpdate(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::Freeze(operation) => operation.infer_output_types(input_types, region_interfaces),
            }
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            match self {
                Self::Add => Cow::Borrowed(ReferenceOperationSemantics::empty()),
                Self::New(operation) => operation.reference_semantics(),
                Self::Read(operation) => operation.reference_semantics(),
                Self::Write(operation) => operation.reference_semantics(),
                Self::Swap(operation) => operation.reference_semantics(),
                Self::AddUpdate(operation) => operation.reference_semantics(),
                Self::Freeze(operation) => operation.reference_semantics(),
            }
        }

        fn effects(&self) -> Effects {
            match self {
                Self::Add => Effects::PURE,
                Self::New(operation) => operation.effects(),
                Self::Read(operation) => operation.effects(),
                Self::Write(operation) => operation.effects(),
                Self::Swap(operation) => operation.effects(),
                Self::AddUpdate(operation) => operation.effects(),
                Self::Freeze(operation) => operation.effects(),
            }
        }
    }

    // Only the addition is executable: a reference primitive reaches an eager destination exclusively as the replay of
    // an access to a preserved root, and a preserved root lives in the staging destination, which records rather than
    // executes.
    impl<C: Domain<Type = TestType, Value = TestValue>> InterpretableOperation<C> for TestOperation {
        fn interpret<D: InterpretationDriver<C>>(
            &self,
            _context: &C,
            _driver: &D,
            inputs: &[TestValue],
        ) -> Result<Vec<TestValue>, ProgramError> {
            match self {
                Self::Add => {
                    check_count!("input", inputs, 2, ProgramError);
                    Ok(vec![inputs[0].add(&inputs[1])?])
                }
                _ => Err(ProgramError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before interpretation", self.name()),
                }),
            }
        }
    }

    macro_rules! impl_test_operation_from_reference_primitive {
        // Lifts one reference primitive into the test destination family, which is the conversion seam a rule spends
        // when it replays an access to a preserved root.
        ($variant:ident, $payload:ident) => {
            impl From<$payload> for TestOperation {
                fn from(operation: $payload) -> Self {
                    Self::$variant(operation)
                }
            }
        };
    }

    impl_test_operation_from_reference_primitive!(New, New);
    impl_test_operation_from_reference_primitive!(Read, Read);
    impl_test_operation_from_reference_primitive!(Write, Write);
    impl_test_operation_from_reference_primitive!(Swap, Swap);
    impl_test_operation_from_reference_primitive!(AddUpdate, AddUpdate);
    impl_test_operation_from_reference_primitive!(Freeze, Freeze);

    // The family delegates each variant to the primitive rule that owns it, which is exactly what a dispatch derive
    // generates, and is what lets the program-level entry point drive these rules over this universe.
    impl<C, P> ReferenceDischargeableOperation<C, P> for TestOperation
    where
        C: Context<Type = TestType, Operation = TestOperation>,
        P: ReferenceAccumulationPolicy<C>,
    {
        fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
            &self,
            context: &ReferenceDischargeContext<C, P>,
            driver: &D,
            inputs: &[ReferenceDischargeValue<C, P>],
        ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
            match self {
                Self::Add => discharge_reference_free_operation(self, context, driver, inputs),
                Self::New(operation) => operation.discharge_references(context, driver, inputs),
                Self::Read(operation) => operation.discharge_references(context, driver, inputs),
                Self::Write(operation) => operation.discharge_references(context, driver, inputs),
                Self::Swap(operation) => operation.discharge_references(context, driver, inputs),
                Self::AddUpdate(operation) => operation.discharge_references(context, driver, inputs),
                Self::Freeze(operation) => operation.discharge_references(context, driver, inputs),
            }
        }
    }

    /// Destination value of the discharge-rule tests: one integer payload carrying its own referent type.
    #[derive(Copy, Clone, Debug, PartialEq)]
    struct TestValue {
        referent: TestReferent,
        payload: i64,
    }

    impl TestValue {
        const fn new(referent: TestReferent, payload: i64) -> Self {
            Self { referent, payload }
        }
    }

    impl Display for TestValue {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}:{}", self.payload, self.referent)
        }
    }

    impl Parameter for TestValue {}

    impl Typed for TestValue {
        type Type = TestType;

        fn r#type(&self) -> Cow<'_, TestType> {
            Cow::Owned(TestType::Value(self.referent))
        }
    }

    impl Value for TestValue {
        type DispatchDomain = EagerContext<Self>;
        type ExecutionDomain = EagerContext<Self>;

        fn dispatch_domain(&self) -> Self::DispatchDomain {
            EagerContext::new()
        }

        fn execution_domain(&self) -> Self::ExecutionDomain {
            EagerContext::new()
        }
    }

    impl Add for TestValue {
        fn add(&self, rhs: &Self) -> Result<Self, ProgramError> {
            if self.referent != rhs.referent {
                return Err(ProgramError::MalformedProgram(format!(
                    "cannot add `{}` to `{}`",
                    rhs.referent, self.referent,
                )));
            }
            Ok(Self::new(self.referent, self.payload + rhs.payload))
        }
    }

    /// View chain of the discharge-rule test universe, which has no interior structure to select.
    #[derive(Copy, Clone, Debug, PartialEq)]
    struct TestAlias;

    impl Parameter for TestAlias {}

    /// Reference discharge policy of the discharge-rule test universe.
    #[derive(Copy, Clone, Debug)]
    struct TestReferenceDischarge;

    // The policy leaves the destination value generic, which is what lets one implementation serve both the eager and
    // the staging destination: a view-less universe needs no destination capability at all to read or replace, and it
    // reaches its sum by binding this family's addition rather than by requiring value-level arithmetic.
    impl<C: Context<Type = TestType, Operation: From<TestOperation>>> ReferenceDischargePolicy<C>
        for TestReferenceDischarge
    {
        type Referent = TestReferent;
        type Alias = TestAlias;

        fn root_alias(_referent: &TestReferent) -> TestAlias {
            TestAlias
        }

        fn lift_reference_type(r#type: ReferenceType<TestReferent>) -> TestType {
            TestType::Reference(r#type)
        }

        fn lift_referent_type(referent: TestReferent) -> TestType {
            TestType::Value(referent)
        }

        fn project_reference_type(r#type: &TestType) -> Option<ReferenceType<TestReferent>> {
            match r#type {
                TestType::Reference(reference) => Some(reference.clone()),
                TestType::Value(_) => None,
            }
        }

        fn read(_context: &C, current: &C::Value, _alias: &TestAlias) -> Result<C::Value, ProgramError> {
            Ok(current.clone())
        }

        fn write(
            _context: &C,
            _current: &C::Value,
            replacement: C::Value,
            _alias: &TestAlias,
        ) -> Result<C::Value, ProgramError> {
            Ok(replacement)
        }

        fn replace(
            _context: &C,
            current: &C::Value,
            replacement: C::Value,
            _alias: &TestAlias,
        ) -> Result<(C::Value, C::Value), ProgramError> {
            Ok((current.clone(), replacement))
        }
    }

    impl<C: Context<Type = TestType, Operation: From<TestOperation>>> ReferenceAccumulationPolicy<C>
        for TestReferenceDischarge
    {
        fn accumulate(
            context: &C,
            current: &C::Value,
            update: C::Value,
            _alias: &TestAlias,
        ) -> Result<C::Value, ProgramError> {
            let mut outputs = context.bind(TestOperation::Add, Vec::new(), &[current.clone(), update])?;
            check_count!("output", outputs, 1, ProgramError);
            Ok(outputs.remove(0))
        }
    }

    /// Reference policy that deliberately supports write discharge without supporting accumulation.
    #[derive(Copy, Clone, Debug)]
    struct WriteOnlyReferenceDischarge;

    impl<C: Context<Type = TestType, Operation: From<TestOperation>>> ReferenceDischargePolicy<C>
        for WriteOnlyReferenceDischarge
    {
        type Referent = TestReferent;
        type Alias = TestAlias;

        fn root_alias(_referent: &TestReferent) -> TestAlias {
            TestAlias
        }

        fn lift_reference_type(r#type: ReferenceType<TestReferent>) -> TestType {
            TestType::Reference(r#type)
        }

        fn lift_referent_type(referent: TestReferent) -> TestType {
            TestType::Value(referent)
        }

        fn project_reference_type(r#type: &TestType) -> Option<ReferenceType<TestReferent>> {
            match r#type {
                TestType::Reference(reference) => Some(reference.clone()),
                TestType::Value(_) => None,
            }
        }

        fn read(_context: &C, current: &C::Value, _alias: &TestAlias) -> Result<C::Value, ProgramError> {
            Ok(current.clone())
        }

        fn write(
            _context: &C,
            _current: &C::Value,
            replacement: C::Value,
            _alias: &TestAlias,
        ) -> Result<C::Value, ProgramError> {
            Ok(replacement)
        }

        fn replace(
            _context: &C,
            _current: &C::Value,
            _replacement: C::Value,
            _alias: &TestAlias,
        ) -> Result<(C::Value, C::Value), ProgramError> {
            Err(ProgramError::MalformedProgram("write-only discharge policy must not replace".to_string()))
        }
    }

    /// Handle to one live root in the discharge-rule test universe.
    type TestDischargeReference = ReferenceDischargeReference<TestDestination, TestReferenceDischarge>;

    /// Referent every discharge-rule test allocates its root over.
    const REFERENT: TestReferent = TestReferent::new(7, 16);

    // Allocates one root holding `payload` through the allocation rule and returns the discharge context together
    // with the handle denoting that root.
    fn allocated_root(payload: i64) -> (TestDischargeContext, TestDischargeReference) {
        let context = TestDischargeContext::new(TestDestination::new());
        let initial = ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, payload));
        let allocated = New::new().discharge_references(&context, &EmptyRegionDriver, &[initial]).unwrap();
        let reference = allocated[0].expect_reference("the allocated root").unwrap().clone();
        (context, reference)
    }

    #[test]
    fn test_generic_reference_primitives_accept_minimal_conversion_universes() {
        let referent = TestReferent::new(7, 16);

        assert_eq!(
            ReferenceNewOperation::<TestReferent, NewUniverse>::new()
                .infer_output_types(&[NewUniverse::Value(referent)], &[]),
            Ok(vec![NewUniverse::Reference(ReferenceType::new(referent))]),
        );

        let reference = ReadFreezeUniverse::Reference(ReferenceType::new(referent));
        assert_eq!(
            ReferenceReadOperation::<TestReferent, ReadFreezeUniverse>::new()
                .infer_output_types(std::slice::from_ref(&reference), &[]),
            Ok(vec![ReadFreezeUniverse::Value(referent)]),
        );
        assert_eq!(
            ReferenceFreezeOperation::<TestReferent, ReadFreezeUniverse>::new()
                .infer_output_types(std::slice::from_ref(&reference), &[]),
            Ok(vec![ReadFreezeUniverse::Value(referent)]),
        );

        assert_eq!(
            ReferenceWriteOperation::<TestReferent, WriteUniverse>::new().infer_output_types(
                &[WriteUniverse::Reference(ReferenceType::new(referent)), WriteUniverse::Value(referent),],
                &[],
            ),
            Ok(Vec::new()),
        );

        assert_eq!(
            ReferenceSwapOperation::<TestReferent, SwapUniverse>::new().infer_output_types(
                &[SwapUniverse::Reference(ReferenceType::new(referent)), SwapUniverse::Value(referent)],
                &[],
            ),
            Ok(vec![SwapUniverse::Value(referent)]),
        );

        assert_eq!(
            ReferenceAddUpdateOperation::<TestReferent, AddUpdateUniverse>::new().infer_output_types(
                &[AddUpdateUniverse::Reference(ReferenceType::new(referent)), AddUpdateUniverse::Value(referent)],
                &[],
            ),
            Ok(Vec::new()),
        );
    }

    #[test]
    fn test_generic_reference_primitive_type_inference() {
        let referent = TestReferent::new(7, 16);
        let promoted_refinement = TestReferent::new(7, 32);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));

        assert_eq!(
            referent.identities().collect::<Vec<_>>(),
            vec![(TypeIdentityPosition::Definition, &referent.identity)],
        );
        assert!(referent.is_refined_by(&promoted_refinement));

        assert_eq!(New::new().infer_output_types(std::slice::from_ref(&value), &[]), Ok(vec![reference.clone()]));
        assert_eq!(Read::new().infer_output_types(std::slice::from_ref(&reference), &[]), Ok(vec![value.clone()]));
        assert_eq!(Write::new().infer_output_types(&[reference.clone(), value.clone()], &[]), Ok(Vec::new()));
        assert_eq!(Swap::new().infer_output_types(&[reference.clone(), value.clone()], &[]), Ok(vec![value.clone()]));
        assert_eq!(AddUpdate::new().infer_output_types(&[reference.clone(), value.clone()], &[]), Ok(Vec::new()));
        assert_eq!(Freeze::new().infer_output_types(std::slice::from_ref(&reference), &[]), Ok(vec![value]));

        assert_eq!(
            Write::new().infer_output_types(&[reference.clone(), TestType::Value(promoted_refinement)], &[]),
            Err(TypeError::invalid(
                "`reference_write` replacement type `value<i7,p32>` must exactly match reference referent type \
                 `value<i7,p16>`",
            )),
        );
        assert_eq!(
            Swap::new().infer_output_types(&[reference.clone(), TestType::Value(promoted_refinement)], &[]),
            Err(TypeError::invalid(
                "`reference_swap` replacement type `value<i7,p32>` must exactly match reference referent type \
                 `value<i7,p16>`",
            )),
        );
        assert_eq!(
            AddUpdate::new().infer_output_types(&[reference, TestType::Value(promoted_refinement)], &[]),
            Err(TypeError::invalid(
                "`reference_add_update` addition result type `value<i7,p32>` must exactly match reference referent \
                 type `value<i7,p16>`",
            )),
        );
    }

    #[test]
    fn test_reference_new_rejects_nested_referent_types() {
        let referent = ReferenceType::new(TestReferent::new(7, 16));
        assert_eq!(
            ReferenceNewOperation::<ReferenceType<TestReferent>, NestedTestType>::new()
                .infer_output_types(&[NestedTestType::Reference(referent.clone())], &[]),
            Err(TypeError::invalid(format!(
                "`reference_new` cannot allocate a reference whose referent type `{referent}` is itself a reference",
            ))),
        );
    }

    #[test]
    fn test_generic_reference_primitive_arity_and_member_projection_errors() {
        let value = TestType::Value(TestReferent::new(7, 16));
        let reference = TestType::Reference(ReferenceType::new(TestReferent::new(7, 16)));
        let region = RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE);

        assert_eq!(New::new().infer_output_types(&[], &[]), Err(TypeError::invalid("expected 1 input but got 0")));
        assert_eq!(
            New::new().infer_output_types(std::slice::from_ref(&value), std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
        assert_eq!(
            Read::new().infer_output_types(std::slice::from_ref(&value), &[]),
            Err(TypeError::invalid("expected reference type but got value type")),
        );
        assert_eq!(
            New::new().infer_output_types(std::slice::from_ref(&reference), &[]),
            Err(TypeError::invalid("expected value type but got reference type")),
        );
        assert_eq!(
            Write::new().infer_output_types(std::slice::from_ref(&reference), &[]),
            Err(TypeError::invalid("expected 2 inputs but got 1")),
        );
        assert_eq!(
            Write::new().infer_output_types(&[value.clone(), value.clone()], &[]),
            Err(TypeError::invalid("expected reference type but got value type")),
        );
        assert_eq!(
            Write::new().infer_output_types(&[reference.clone(), reference.clone()], &[]),
            Err(TypeError::invalid("expected value type but got reference type")),
        );
        assert_eq!(
            Write::new().infer_output_types(&[reference.clone(), value], std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
        assert_eq!(
            Swap::new().infer_output_types(std::slice::from_ref(&reference), &[]),
            Err(TypeError::invalid("expected 2 inputs but got 1")),
        );
    }

    #[test]
    fn test_generic_reference_primitive_contracts() {
        fn assert_parameter_roundtrip<P>(parameter: P)
        where
            P: Copy + Debug + PartialEq + Parameter,
        {
            let structure = <P as Parameterized<P>>::parameter_structure(&parameter);
            assert_eq!(<P as Parameterized<P>>::from_parameters(structure, [parameter]), Ok(parameter));
        }

        assert_parameter_roundtrip(New::new());
        assert_parameter_roundtrip(Read::new());
        assert_parameter_roundtrip(Write::new());
        assert_parameter_roundtrip(Swap::new());
        assert_parameter_roundtrip(AddUpdate::new());
        assert_parameter_roundtrip(Freeze::new());

        assert_eq!(New::new(), New::default());
        assert_eq!(format!("{:?}", New::new()), "ReferenceNewOperation");
        assert_eq!(New::new().to_string(), REFERENCE_NEW_OPERATION_NAME);
        assert_eq!(Read::new().to_string(), REFERENCE_READ_OPERATION_NAME);
        assert_eq!(Write::new().to_string(), REFERENCE_WRITE_OPERATION_NAME);
        assert_eq!(Swap::new().to_string(), REFERENCE_SWAP_OPERATION_NAME);
        assert_eq!(AddUpdate::new().to_string(), REFERENCE_ADD_UPDATE_OPERATION_NAME);
        assert_eq!(Freeze::new().to_string(), REFERENCE_FREEZE_OPERATION_NAME);

        assert_eq!(New::new().effects(), Effects::single(Effect::OrderedState));
        assert_eq!(Read::new().effects(), Effects::single(Effect::OrderedState));
        assert_eq!(Write::new().effects(), Effects::single(Effect::OrderedState));
        assert_eq!(Swap::new().effects(), Effects::single(Effect::OrderedState));
        assert_eq!(AddUpdate::new().effects(), Effects::single(Effect::OrderedState));
        assert_eq!(Freeze::new().effects(), Effects::single(Effect::OrderedState));

        assert_eq!(New::new().reference_semantics().outputs(), &[ReferenceOutput::Root { output_index: 0 }],);
        assert_eq!(New::new().reference_semantics().inputs(), &[]);
        assert_eq!(Read::new().reference_semantics().outputs(), &[]);
        assert_eq!(Read::new().reference_semantics().inputs(), &[ReferenceInput::new(0, ReferenceAccessMode::Read)],);
        assert_eq!(Write::new().reference_semantics().inputs(), &[ReferenceInput::new(0, ReferenceAccessMode::Write)],);
        assert_eq!(
            Swap::new().reference_semantics().inputs(),
            &[ReferenceInput::new(0, ReferenceAccessMode::ReadWrite)],
        );
        assert_eq!(
            AddUpdate::new().reference_semantics().inputs(),
            &[ReferenceInput::new(0, ReferenceAccessMode::Accumulate)],
        );
        assert_eq!(
            Freeze::new().reference_semantics().inputs(),
            &[ReferenceInput::new(0, ReferenceAccessMode::Consume)],
        );
    }

    /// Policy whose projection deliberately disagrees with the reference primitives' own inference, which is the only
    /// way an allocation can infer a type the policy does not recognize as a reference.
    #[derive(Copy, Clone, Debug)]
    struct NonProjectingReferenceDischarge;

    impl<C: Context<Type = TestType, Operation: From<TestOperation>>> ReferenceDischargePolicy<C>
        for NonProjectingReferenceDischarge
    {
        type Referent = TestReferent;
        type Alias = TestAlias;

        fn root_alias(_referent: &TestReferent) -> TestAlias {
            TestAlias
        }

        fn lift_reference_type(r#type: ReferenceType<TestReferent>) -> TestType {
            TestType::Reference(r#type)
        }

        fn lift_referent_type(referent: TestReferent) -> TestType {
            TestType::Value(referent)
        }

        fn project_reference_type(_type: &TestType) -> Option<ReferenceType<TestReferent>> {
            None
        }

        fn read(_context: &C, current: &C::Value, _alias: &TestAlias) -> Result<C::Value, ProgramError> {
            Ok(current.clone())
        }

        fn write(
            _context: &C,
            _current: &C::Value,
            replacement: C::Value,
            _alias: &TestAlias,
        ) -> Result<C::Value, ProgramError> {
            Ok(replacement)
        }

        fn replace(
            _context: &C,
            current: &C::Value,
            replacement: C::Value,
            _alias: &TestAlias,
        ) -> Result<(C::Value, C::Value), ProgramError> {
            Ok((current.clone(), replacement))
        }
    }

    #[test]
    fn test_reference_new_operation_reference_discharge() {
        // Allocation binds a fresh discharged root whose entering state is the initializer and whose reference type is
        // the one this operation's own inference derives, exposed through the identity alias of an unviewed root.
        let (context, reference) = allocated_root(4);
        assert_eq!(context.live_roots(), vec![reference.root()]);
        assert_eq!(reference.r#type(), &ReferenceType::new(REFERENT));
        assert_eq!(reference.alias(), &TestAlias);
        assert_eq!(reference.preserved(), None);
        assert_eq!(context.discharged_state(reference.root()), Ok(TestValue::new(REFERENT, 4)));
        assert_eq!(context.is_mutated(reference.root()), Ok(false));

        // A reference operand is not an initial state, and the diagnostic says which operand the rule expected.
        let context = TestDischargeContext::new(TestDestination::new());
        let handle = ReferenceDischargeValue::Reference(reference);
        assert_eq!(
            New::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&handle)),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected an initial reference state but received {handle}",
            ))),
        );

        // The rule reads its fresh root's reference type back out of its own inferred output type, so a policy whose
        // projection disagrees with that inference cannot silently allocate an unclassifiable root.
        let disagreeing =
            ReferenceDischargeContext::<TestDestination, NonProjectingReferenceDischarge>::new(TestDestination::new());
        let initial = ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, 4));
        assert_eq!(
            New::new().discharge_references(&disagreeing, &EmptyRegionDriver, std::slice::from_ref(&initial)),
            Err(ProgramError::MalformedProgram(
                "`reference_new` inferred the non-reference output type `ref<value<i7,p16>>`".to_string(),
            )),
        );
    }

    #[test]
    fn test_reference_read_operation_reference_discharge() {
        // A read observes the root's current state without changing it, so the root stays unmutated.
        let (context, reference) = allocated_root(4);
        let handle = ReferenceDischargeValue::Reference(reference.clone());
        assert_eq!(
            Read::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&handle)),
            Ok(vec![ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, 4))]),
        );
        assert_eq!(context.is_mutated(reference.root()), Ok(false));

        // An ordinary operand denotes no root, so the rule reports what it expected instead of reading a value.
        let pure: TestDischargeValue = ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, 4));
        assert_eq!(
            Read::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&pure)),
            Err(ProgramError::MalformedProgram(
                "reference discharge expected a reference to read but received an ordinary value".to_string(),
            )),
        );
    }

    #[test]
    fn test_reference_write_operation_reference_discharge() {
        // A policy with no accumulation capability replaces state through `write`, produces no old-value
        // result, and marks the root mutated. Its `replace` path is an error, making accidental swap dispatch visible.
        let context =
            ReferenceDischargeContext::<TestDestination, WriteOnlyReferenceDischarge>::new(TestDestination::new());
        let initial = TestValue::new(REFERENT, 4);
        let allocated = context.allocate_discharged(ReferenceType::new(REFERENT), initial).unwrap();
        let reference = allocated.expect_reference("the allocated root").unwrap().clone();
        let inputs = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, 9)),
        ];
        assert_eq!(Write::new().discharge_references(&context, &EmptyRegionDriver, inputs.as_slice()), Ok(Vec::new()),);
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 9)));
        assert_eq!(context.is_mutated(reference.root()), Ok(true));

        // Exact operand inference runs before mutation, so a rejected replacement leaves the root unchanged.
        let invalid = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Ordinary(TestValue::new(TestReferent::new(7, 32), 1)),
        ];
        assert_eq!(
            Write::new().discharge_references(&context, &EmptyRegionDriver, invalid.as_slice()),
            Err(TypeError::invalid(
                "`reference_write` replacement type `value<i7,p32>` must exactly match reference referent type \
                 `value<i7,p16>`",
            )
            .into()),
        );
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 9)));
    }

    #[test]
    fn test_reference_swap_operation_reference_discharge() {
        // A replacement returns the previous state and commits the successor, which marks the root mutated.
        let (context, reference) = allocated_root(4);
        let inputs = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, 9)),
        ];
        assert_eq!(
            Swap::new().discharge_references(&context, &EmptyRegionDriver, inputs.as_slice()),
            Ok(vec![ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, 4))]),
        );
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 9)));
        assert_eq!(context.is_mutated(reference.root()), Ok(true));

        // The replacement itself must be an ordinary value rather than a second reference handle.
        let handles =
            vec![ReferenceDischargeValue::Reference(reference.clone()), ReferenceDischargeValue::Reference(reference)];
        assert_eq!(
            Swap::new().discharge_references(&context, &EmptyRegionDriver, handles.as_slice()),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected a replacement value but received {}",
                handles[1],
            ))),
        );
    }

    #[test]
    fn test_reference_add_update_operation_reference_discharge() {
        // An accumulation produces no result and replaces the current state with its sum with the update.
        let (context, reference) = allocated_root(4);
        let inputs = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, 9)),
        ];
        assert_eq!(
            AddUpdate::new().discharge_references(&context, &EmptyRegionDriver, inputs.as_slice()),
            Ok(Vec::new()),
        );
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 13)));
        assert_eq!(context.is_mutated(reference.root()), Ok(true));

        // An update whose sum with the referent would not itself be the referent is rejected by this operation's own
        // inference before the universe accumulates anything, so the root keeps its previous state.
        let promoted = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Ordinary(TestValue::new(TestReferent::new(7, 32), 1)),
        ];
        assert_eq!(
            AddUpdate::new().discharge_references(&context, &EmptyRegionDriver, promoted.as_slice()),
            Err(TypeError::invalid(
                "`reference_add_update` addition result type `value<i7,p32>` must exactly match reference referent \
                 type `value<i7,p16>`",
            )
            .into()),
        );
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 13)));
    }

    #[test]
    fn test_reference_freeze_operation_reference_discharge() {
        // A freeze yields the root's final state and unbinds the root, so every later access is a use-after-consume.
        let (context, reference) = allocated_root(4);
        let handle = ReferenceDischargeValue::Reference(reference.clone());
        assert_eq!(
            Freeze::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&handle)),
            Ok(vec![ReferenceDischargeValue::Ordinary(TestValue::new(REFERENT, 4))]),
        );
        assert_eq!(context.live_roots(), Vec::new());
        assert_eq!(
            Freeze::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&handle)),
            Err(ProgramError::MalformedProgram(format!("reference discharge accessed consumed {}", reference.root()))),
        );
    }

    #[test]
    fn test_reference_primitive_discharge_replays_accesses_to_a_preserved_root() {
        // A root that partial discharge preserved survives in the destination as an ordinary reference, so the
        // dispatch path replays every access verbatim over the handle's destination value instead of acting on
        // threaded state, and the access rules themselves never run. The rewritten program therefore performs the
        // same reference operations the source did, in the same order, and the consumed root contributes no binding.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(TestType::Reference(ReferenceType::new(REFERENT)));
        let update = builder.add_input(TestType::Value(REFERENT));
        let observed = builder
            .add_instruction(TestOperation::Read(Read::new()), Vec::new(), vec![reference], None)
            .unwrap()[0];
        builder
            .add_instruction(TestOperation::Write(Write::new()), Vec::new(), vec![reference, update], None)
            .unwrap();
        let previous = builder
            .add_instruction(TestOperation::Swap(Swap::new()), Vec::new(), vec![reference, update], None)
            .unwrap()[0];
        builder
            .add_instruction(TestOperation::AddUpdate(AddUpdate::new()), Vec::new(), vec![reference, update], None)
            .unwrap();
        let frozen = builder
            .add_instruction(TestOperation::Freeze(Freeze::new()), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![observed, previous, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();

        let preserved = source.partially_discharge_references_with_policy::<TestReferenceDischarge>(0, &[]).unwrap();
        assert_eq!(preserved.public_output_count(), 3);
        assert_eq!(preserved.external_states(), &[]);
        assert_eq!(
            preserved.program().to_string(),
            indoc! {"
                lambda %0:ref<value<i7,p16>>, %1:value<i7,p16> .
                let %2:value<i7,p16> = reference_read %0
                    reference_write %0 %1
                    %3:value<i7,p16> = reference_swap %0 %1
                    reference_add_update %0 %1
                    %4:value<i7,p16> = reference_freeze %0
                in (%2, %3, %4)"},
        );
    }

    #[test]
    fn test_reference_primitive_discharge_preserves_an_unselected_allocation() {
        // The allocation rule consults its own replay position against the selection, so an unselected allocation
        // site is replayed rather than turned into threaded state and the root it binds survives in the destination.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(TestType::Value(REFERENT));
        let update = builder.add_input(TestType::Value(REFERENT));
        let root = builder.add_instruction(TestOperation::New(New::new()), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(TestOperation::AddUpdate(AddUpdate::new()), Vec::new(), vec![root, update], None)
            .unwrap();
        let frozen =
            builder.add_instruction(TestOperation::Freeze(Freeze::new()), Vec::new(), vec![root], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let preserved =
            source.clone().partially_discharge_references_with_policy::<TestReferenceDischarge>(0, &[]).unwrap();
        assert_eq!(preserved.public_output_count(), 1);
        assert_eq!(preserved.external_states(), &[]);
        assert_eq!(
            preserved.program().to_string(),
            indoc! {"
                lambda %0:value<i7,p16>, %1:value<i7,p16> .
                let %2:ref<value<i7,p16>> = reference_new %0
                    reference_add_update %2 %1
                    %3:value<i7,p16> = reference_freeze %2
                in (%3)"},
        );

        // Selecting that same site is the everything-selected case, so it must agree with full discharge exactly.
        let sites = source.reference_discharge_sites(0).unwrap();
        let selected = source
            .clone()
            .partially_discharge_references_with_policy::<TestReferenceDischarge>(0, sites.as_slice())
            .unwrap()
            .try_into_full()
            .unwrap();
        let full = source.discharge_references_with_policy::<TestReferenceDischarge>(0).unwrap();
        assert_eq!(selected.program().to_string(), full.program().to_string());
        assert_eq!(
            full.program().to_string(),
            indoc! {"
                lambda %0:value<i7,p16>, %1:value<i7,p16> .
                let %2:value<i7,p16> = test.add %0 %1
                in (%2)"},
        );
    }
}
