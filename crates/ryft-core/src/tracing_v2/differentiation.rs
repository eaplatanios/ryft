use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::Display;
use std::rc::Rc;

use half::{bf16, f16};
use ryft_macros::Parameter;
use thiserror::Error;

use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, Parameterized};
use crate::tracing::engines::{Engine, ScalarEngine, Tracer, TracingContext, TracingEngine};
use crate::tracing::transposition::LinearOperation as LinearOperationTrait;
use crate::tracing::{Atom, AtomId, Instruction, Program, ProgramBuilder, Traceable, TracingError};
use crate::tracing_v2::operations::constants::Zero;
use crate::tracing_v2::operations::{SupportsAdd, SupportsNeg, SupportsScale, SupportsZero};
use crate::tracing_v2::{LinearScalarOperation, ScalarOperation};
use crate::types::{ArrayType, Type, Typed};

/// Errors emitted by the differentiation helpers in [`crate::tracing_v2`].
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DifferentiationError {
    /// Reverse-mode gradient was requested for a function with an invalid number of output leaves.
    #[error("gradient output must have exactly {expected} leaf but got {got}")]
    InvalidGradientOutputLeafCount { expected: usize, got: usize },

    /// Reverse-mode gradient was requested for a non-scalar array output.
    #[error("gradient output must be a rank-0 scalar array but got {output_type}")]
    NonScalarGradientOutput { output_type: ArrayType },

    /// Traced forward-mode differentiation was invoked without any staged input leaves.
    #[error("traced jvp requires at least one input leaf to recover the tracing context")]
    MissingTracedJvpInputLeaves,

    /// Traced reverse-mode differentiation was invoked without any staged input leaves.
    #[error("traced reverse-mode requires at least one input leaf to recover the tracing context")]
    MissingTracedReverseModeInputLeaves,

    /// Traced rematerialization was invoked without any staged input leaves.
    #[error("traced rematerialize requires at least one input leaf to recover the tracing context")]
    MissingTracedRematerializeInputLeaves,

    /// Linear rematerialization replay was invoked without any tangent leaves.
    #[error("linear rematerialize replay requires at least one tangent leaf to recover the tracing context")]
    MissingLinearRematerializeReplayTangentLeaves,

    /// Linear rematerialization transpose was invoked without any output cotangent leaves.
    #[error(
        "linear rematerialize transpose requires at least one output cotangent leaf to recover the tracing context"
    )]
    MissingLinearRematerializeTransposeCotangentLeaves,

    /// Dense Jacobian materialization produced an unexpected number of rows.
    #[error("invalid Jacobian row count; expected {expected} but got {got}")]
    InvalidJacobianRowCount { expected: usize, got: usize },

    /// Dense Jacobian materialization produced a row with an unexpected width.
    #[error("invalid Jacobian row width; expected {expected} but got {got}")]
    InvalidJacobianRowWidth { expected: usize, got: usize },

    /// Dense Jacobian materialization produced an unexpected number of columns.
    #[error("invalid Jacobian column count; expected {expected} but got {got}")]
    InvalidJacobianColumnCount { expected: usize, got: usize },

    /// Dense Jacobian materialization produced a column with an unexpected height.
    #[error("invalid Jacobian column height; expected {expected} but got {got}")]
    InvalidJacobianColumnHeight { expected: usize, got: usize },
}

/// Value-level contract for leaves that participate in automatic differentiation over `T`.
///
/// The associated [`Tangent`](Self::Tangent) type makes the tangent representation explicit even
/// though today's staged linear-program IR still requires `Tangent = Self` at the transform
/// boundary. Code paths that need to synthesize zero tangents or unit gradient seeds from abstract
/// type metadata add [`Zero`](crate::tracing_v2::operations::constants::Zero) and
/// [`One`](crate::tracing_v2::operations::constants::One) bounds at those synthesis sites instead
/// of requiring every tangent representation to support metadata-only construction.
pub trait Differentiable<T: Type>: Traceable<T> {
    /// Tangent and cotangent leaf type associated with this primal leaf.
    type Tangent: Traceable<T>;
}

impl<'engine, E> Differentiable<E::Type> for Tracer<'engine, E>
where
    E: TracingEngine + ?Sized,
    E::Value: Differentiable<E::Type>,
{
    type Tangent = Self;
}

/// Concrete state threaded through forward-mode JVP rules.
///
/// [`JvpContext`] owns the active linear-program builder where tangent ops are staged. It is the
/// forward-mode counterpart of
/// [`TranspositionContext`](crate::tracing::transposition::TranspositionContext): JVP rules call
/// [`apply_operation`](Self::apply_operation) to stage tangent ops on the active builder.
#[doc(hidden)]
pub struct JvpContext<'a, E: DifferentiableEngine + ?Sized> {
    /// [`DifferentiableEngine`] borrowed by this [`JvpContext`] for type-driven value synthesis and operation
    /// selection.
    pub engine: &'a E,

    /// [`ProgramBuilder`] that owns the staged linear [`Program`](crate::tracing::Program) that is currently being
    /// traced.
    pub builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::LinearOperation>>>,
}

impl<'a, E: DifferentiableEngine + ?Sized> JvpContext<'a, E> {
    /// Creates a JVP context that stages into `builder`.
    #[doc(hidden)]
    pub fn new(engine: &'a E, builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::LinearOperation>>>) -> Self {
        Self { engine, builder }
    }

    /// Stages one operation in the currently active linear program.
    pub fn apply_operation(
        &self,
        inputs: &[AtomId],
        operation: E::LinearOperation,
        output_count: usize,
    ) -> Result<Vec<AtomId>, TracingError> {
        let mut builder_borrow = self.builder.borrow_mut();
        let input_types =
            inputs.iter().map(|atom| builder_borrow.atoms[atom.index].r#type().into_owned()).collect::<Vec<_>>();
        let output_types = operation.infer_output_types(&input_types)?;
        if output_types.len() != output_count {
            return Err(TracingError::InvalidOutputCount { expected: output_count, got: output_types.len() });
        }
        let outputs = output_types.into_iter().map(|r#type| builder_borrow.add_variable(r#type)).collect::<Vec<_>>();
        builder_borrow
            .instructions
            .push(Instruction { operation, inputs: inputs.to_vec(), outputs: outputs.clone() });
        Ok(outputs)
    }

    /// Stages a constant tangent on the active linear builder.
    pub fn add_constant(&self, value: E::Value) -> AtomId {
        self.builder.borrow_mut().add_constant(value)
    }
}

/// Forward-mode tracer carrying both a primal and a tangent.
///
/// [`JvpTracer`] is to forward-mode AD what [`Tracer`](crate::tracing::engines::Tracer) is to ordinary
/// staging: it is the leaf wrapper that primitive operations see when a function is being evaluated
/// in JVP mode. The `primal` field carries the usual runtime value, while the `tangent` field
/// carries the directional derivative information flowing alongside it.
///
/// The type parameters have no bounds on the struct itself so that `JvpTracer` can appear in
/// signatures without eagerly propagating all tangent requirements. `tracing_v2` uses `T = AtomId`
/// for the rule-based JVP path threaded through [`JvpContext`], where rules manipulate symbolic
/// tangent atoms directly.
#[derive(Clone, Debug, Parameter)]
pub struct JvpTracer<V, T> {
    /// The primal value.
    pub primal: V,

    /// The tangent value associated with the primal.
    pub tangent: T,
}

impl<Ty: Type, V: Typed<Ty>, T> Typed<Ty> for JvpTracer<V, T> {
    #[inline]
    fn r#type(&self) -> Cow<'_, Ty> {
        <V as Typed<Ty>>::r#type(&self.primal)
    }
}

impl<V: Display, T> Display for JvpTracer<V, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.primal, formatter)
    }
}

impl<Ty: Type, V: Traceable<Ty>, T: Clone + Parameter> Traceable<Ty> for JvpTracer<V, T> {}

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<T, V, O, Input, Output>
{
    /// Converts this staged primal [`Program`] into a staged pushforward linear map.
    ///
    /// This is the reusable IR-level form of forward-mode differentiation. Instead of evaluating
    /// the JVP immediately, it builds a staged [`Program`] over linear operations that can be
    /// replayed later on arbitrary tangent inputs at the same primal point.
    ///
    /// # Parameters
    ///
    ///   - `engine`: Differentiable engine that supplies the linear operation carrier and primitive
    ///     JVP rules.
    ///   - `input_primals`: Concrete primal values aligned with this program's input atoms.
    pub fn linearize<E: DifferentiableEngine<Type = T, Value = V> + ?Sized>(
        &self,
        engine: &E,
        input_primals: Vec<V>,
    ) -> Result<Program<T, V, E::LinearOperation, Input, Output>, TracingError>
    where
        V: Differentiable<T, Tangent = V> + Zero<T>,
        O: DifferentiableOperation<E>,
    {
        fn tangent_for_atom<T, V, LinearOperation>(
            primal_values: &[Option<V>],
            builder: &Rc<RefCell<ProgramBuilder<T, V, LinearOperation>>>,
            tangents: &mut [Option<AtomId>],
            atom_id: AtomId,
        ) -> Result<AtomId, TracingError>
        where
            T: Type,
            V: Differentiable<T, Tangent = V> + Zero<T>,
            LinearOperation: Clone + Operation<T>,
        {
            if let Some(atom) = tangents[atom_id.index] {
                return Ok(atom);
            }
            let primal = primal_values[atom_id.index].as_ref().ok_or(TracingError::UnboundAtomId { id: atom_id })?;
            let atom = builder.borrow_mut().add_constant(<V as Zero<T>>::zero(primal.r#type().as_ref())?);
            tangents[atom_id.index] = Some(atom);
            Ok(atom)
        }

        if input_primals.len() != self.input_ids.len() {
            return Err(TracingError::InvalidInputCount { expected: self.input_ids.len(), got: input_primals.len() });
        }
        let builder = Rc::new(RefCell::new(ProgramBuilder::<T, V, E::LinearOperation>::new()));
        let mut primals: Vec<Option<V>> = vec![None; self.atoms.len()];
        let mut tangents: Vec<Option<AtomId>> = vec![None; self.atoms.len()];
        for (input_atom, input_primal) in self.input_ids.iter().copied().zip(input_primals.into_iter()) {
            let tangent_atom = builder.borrow_mut().add_input(input_primal.r#type().into_owned());
            tangents[input_atom.index] = Some(tangent_atom);
            primals[input_atom.index] = Some(input_primal);
        }
        for (atom_index, atom) in self.atoms.iter().enumerate() {
            let atom_id = AtomId { index: atom_index };
            if let Atom::Constant(value) = atom {
                primals[atom_id.index] = Some(value.clone());
            }
        }

        let mut context = JvpContext::new(engine, builder.clone());
        for instruction in &self.instructions {
            let input_duals = instruction
                .inputs
                .iter()
                .copied()
                .map(|input_atom| {
                    Ok(JvpTracer {
                        primal: primals[input_atom.index]
                            .clone()
                            .ok_or(TracingError::UnboundAtomId { id: input_atom })?,
                        tangent: tangent_for_atom::<T, V, E::LinearOperation>(
                            primals.as_slice(),
                            &builder,
                            tangents.as_mut_slice(),
                            input_atom,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>, TracingError>>()?;
            let output_duals = instruction.operation.jvp(&mut context, input_duals.as_slice())?;
            if output_duals.len() != instruction.outputs.len() {
                return Err(TracingError::InvalidOutputCount {
                    expected: instruction.outputs.len(),
                    got: output_duals.len(),
                });
            }
            for (output_atom, output_dual) in instruction.outputs.iter().copied().zip(output_duals.into_iter()) {
                primals[output_atom.index] = Some(output_dual.primal);
                tangents[output_atom.index] = Some(output_dual.tangent);
            }
        }

        let output_tangents = self
            .output_ids
            .iter()
            .copied()
            .map(|output_atom| {
                tangent_for_atom::<T, V, E::LinearOperation>(
                    primals.as_slice(),
                    &builder,
                    tangents.as_mut_slice(),
                    output_atom,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        drop(context);
        drop(tangents);
        let builder = match Rc::try_unwrap(builder) {
            Ok(builder) => builder.into_inner(),
            Err(_) => {
                return Err(TracingError::EscapedProgramBuilder);
            }
        };
        builder
            .build(output_tangents, self.input_structure.clone(), self.output_structure.clone())?
            .simplified()
    }
}

/// Operation-level contract for forward-mode Jacobian-Vector Product (JVP) staging.
///
/// A [`DifferentiableOperation`] is keyed by the [`DifferentiableEngine`] that supplies the value,
/// type, and linear-operation families used while differentiating. Implementors consume
/// [`JvpTracer`] inputs, each carrying a primal value and a tangent atom in the active linear
/// builder, and return traced primal/tangent outputs.
///
/// Primitive rules usually stage tangent operations through [`JvpContext::apply_operation`].
/// Higher-order rules use [`JvpContext::engine`] to recurse into nested programs with the same
/// engine.
pub trait DifferentiableOperation<E: DifferentiableEngine + ?Sized>: Operation<E::Type> {
    /// Applies this operation's forward-mode Jacobian-Vector Product (JVP) rule.
    ///
    /// The returned vector must be aligned with this operation's outputs and must carry both the
    /// primal output values and the staged tangent atoms for those outputs.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active JVP context used to stage tangent operations and access the
    ///     differentiable engine.
    ///   - `inputs`: Traced inputs aligned with this operation's inputs.
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError>;
}

/// Optional extension for tracing engines that support differentiation inside an active trace.
///
/// Plain tracing engines do not need to choose any linear carrier. This trait is the additional
/// contract required when a [`TracingContext`](crate::tracing::engines::TracingContext) itself needs to act
/// as a differentiable engine: tangent and cotangent programs then operate on
/// [`Tracer`] values, so the underlying tracing engine must select a linear operation carrier for
/// those traced leaves.
pub trait DifferentiableTracingEngine: TracingEngine {
    /// Linear operation carrier used for tangent and cotangent programs over traced values.
    type LinearOperation<'engine>: Clone
        + LinearOperationTrait<Self::Type, Tracer<'engine, Self>, Self::LinearOperation<'engine>>
        + SupportsAdd<Self::Type, Tracer<'engine, Self>>
        + SupportsNeg<Self::Type, Tracer<'engine, Self>>
        + SupportsScale<Self::Type, Tracer<'engine, Self>>
        + SupportsZero<Self::Type, Tracer<'engine, Self>>
    where
        Self: 'engine;
}

/// Extension of [`Engine`] for backends that support automatic differentiation.
///
/// Engines that only need ordinary tracing implement [`TracingEngine`] without this extension. AD
/// transforms such as [`grad`](crate::tracing_v2::grad), [`jvp`](crate::tracing_v2::jvp), and
/// [`vjp`](crate::tracing_v2::vjp) require this trait so non-differentiable backends do not need to
/// define fake tangent carriers.
///
/// Differentiated closures are traced through [`DifferentiableOperationTracingEngine`], whose
/// [`TracingEngine::Operation`] is [`DifferentiableEngine::DifferentiableOperation`]. That keeps
/// ordinary tracing free to use a wider operation carrier while making differentiation reject
/// unsupported operations at type-check time when the differentiation carrier omits them.
pub trait DifferentiableEngine: Engine {
    /// Staged operation type selected by this engine for tracing differentiable primal programs.
    type DifferentiableOperation: Clone + InterpretableOperation<Self::Type, Self::Value>;

    /// Linear staged operation type selected by this engine for tangent and cotangent programs.
    ///
    /// Linear programs produced by [`jvp_program`](crate::tracing_v2::jvp_program),
    /// [`vjp`](crate::tracing_v2::vjp), and related transforms store this carrier.
    type LinearOperation: Clone
        + LinearOperationTrait<Self::Type, Self::Value, Self::LinearOperation>
        + SupportsAdd<Self::Type, Self::Value>
        + SupportsNeg<Self::Type, Self::Value>
        + SupportsScale<Self::Type, Self::Value>;
}

/// Transparent tracing view used while tracing differentiable primal programs.
///
/// Automatic-differentiation transforms need to stage the user's primal closure with
/// [`DifferentiableEngine::DifferentiableOperation`] rather than the ordinary
/// [`TracingEngine::Operation`] selected by the backend. Those carriers may intentionally differ:
/// an engine can support a broad ordinary tracing universe while exposing a narrower
/// differentiable carrier whose variants all have differentiation rules. This adapter is the small
/// bridge between those two contracts.
///
/// [`DifferentiableOperationTracingEngine::new`] reborrows an `E: DifferentiableEngine` as a
/// [`TracingEngine`] without allocation or ownership. AD entry points construct this view at trace
/// boundaries such as [`jvp_program`](crate::tracing_v2::jvp_program),
/// [`vjp`](crate::tracing_v2::vjp), and [`grad`](crate::tracing_v2::grad), pass it immediately to
/// ordinary tracing helpers, and keep backend implementations centered on their real engine type.
/// User-facing ordinary tracing should keep using the backend's own [`TracingEngine`]
/// implementation; traced tangent and cotangent programs are selected separately through
/// [`DifferentiableTracingEngine`].
///
/// This type is public today because the public AD closure bounds still mention
/// `Tracer<'engine, DifferentiableOperationTracingEngine<E>>`. Once those APIs hide the concrete
/// active tracer carrier, this adapter can become a `pub(crate)` implementation detail.
#[repr(transparent)]
pub struct DifferentiableOperationTracingEngine<E: DifferentiableEngine + ?Sized> {
    /// Engine viewed through its differentiable operation carrier.
    engine: E,
}

impl<E: DifferentiableEngine + ?Sized> DifferentiableOperationTracingEngine<E> {
    /// Reborrows `engine` as a differentiable operation tracing view.
    #[inline]
    pub const fn new(engine: &E) -> &Self {
        // SAFETY: `DifferentiableOperationTracingEngine<E>` is `repr(transparent)` over `E` and adds no
        // fields, so references to `E` and references to this view have identical layout.
        unsafe { &*(std::ptr::from_ref(engine) as *const Self) }
    }

    /// Returns the wrapped engine.
    #[inline]
    pub const fn inner(&self) -> &E {
        // SAFETY: `DifferentiableOperationTracingEngine<E>` is `repr(transparent)` over `E` and adds no
        // fields, so references to this view and references to `E` have identical layout.
        unsafe { &*(std::ptr::from_ref(self) as *const E) }
    }
}

impl<E: DifferentiableEngine + ?Sized> std::fmt::Debug for DifferentiableOperationTracingEngine<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("DifferentiableOperationTracingEngine").finish_non_exhaustive()
    }
}

impl<E: DifferentiableEngine + ?Sized> Engine for DifferentiableOperationTracingEngine<E> {
    type Type = E::Type;
    type Value = E::Value;

    #[inline]
    fn zero(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError> {
        self.inner().zero(r#type)
    }

    #[inline]
    fn one(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError> {
        self.inner().one(r#type)
    }
}

impl<E: DifferentiableEngine + ?Sized> TracingEngine for DifferentiableOperationTracingEngine<E> {
    type Operation = E::DifferentiableOperation;
}

impl<E: DifferentiableEngine + ?Sized> DifferentiableEngine for DifferentiableOperationTracingEngine<E> {
    type DifferentiableOperation = E::DifferentiableOperation;
    type LinearOperation = E::LinearOperation;
}

impl<'engine, E> DifferentiableEngine for TracingContext<'engine, E>
where
    E: DifferentiableTracingEngine + ?Sized,
    E::Value: Differentiable<E::Type, Tangent = E::Value>,
    E::Operation: SupportsAdd<E::Type, E::Value>,
    crate::tracing_v2::operations::AddOperation: InterpretableOperation<E::Type, Tracer<'engine, E>>,
{
    type DifferentiableOperation = crate::tracing_v2::operations::AddOperation;
    type LinearOperation = E::LinearOperation<'engine>;
}

macro_rules! impl_differentiable_engine_for_scalar {
    ($ty:ty) => {
        impl DifferentiableEngine for ScalarEngine<$ty> {
            type DifferentiableOperation = ScalarOperation<$ty>;
            type LinearOperation = LinearScalarOperation<$ty>;
        }

        impl DifferentiableTracingEngine for ScalarEngine<$ty> {
            type LinearOperation<'engine>
                = LinearScalarOperation<Tracer<'engine, Self>>
            where
                Self: 'engine;
        }
    };
}

impl_differentiable_engine_for_scalar!(bf16);
impl_differentiable_engine_for_scalar!(f16);
impl_differentiable_engine_for_scalar!(f32);
impl_differentiable_engine_for_scalar!(f64);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use crate::tracing::engines::ScalarEngine;
    use crate::tracing_v2::jvp;

    use super::DifferentiableEngine;

    #[test]
    fn test_scalar_engine_half_and_float_engines_are_differentiable() {
        let _: Option<<ScalarEngine<bf16> as DifferentiableEngine>::DifferentiableOperation> = None;
        let _: Option<<ScalarEngine<f16> as DifferentiableEngine>::DifferentiableOperation> = None;
        let _: Option<<ScalarEngine<f32> as DifferentiableEngine>::DifferentiableOperation> = None;
        let _: Option<<ScalarEngine<f64> as DifferentiableEngine>::DifferentiableOperation> = None;
    }

    #[test]
    fn test_scalar_engine_half_engines_run_jvp() {
        let bf16_engine = ScalarEngine::<bf16>::new();
        assert_eq!(
            jvp(&bf16_engine, |x| x.clone() + x, bf16::from_f32(3.0), bf16::ONE),
            Ok((bf16::from_f32(6.0), bf16::from_f32(2.0)))
        );

        let f16_engine = ScalarEngine::<f16>::new();
        assert_eq!(
            jvp(&f16_engine, |x| x.clone() + x, f16::from_f32(3.0), f16::ONE),
            Ok((f16::from_f32(6.0), f16::from_f32(2.0)))
        );
    }
}
