use std::borrow::Cow;
use std::cell::RefCell;
use std::convert::Infallible;
use std::fmt::Display;
use std::rc::Rc;

use half::{bf16, f16};
use ryft_macros::Parameter;
use thiserror::Error;

use crate::macros::check_count;
use crate::operations::arithmetic::{AddOperation, SupportsAdd};
use crate::operations::constants::{One, SupportsZero, SupportsZeroLike, Zero, ZeroLike};
use crate::operations::scalars::{LinearScalarOperation, ScalarOperation};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily};
use crate::tracing::engines::{Engine, ScalarEngine, Tracer, TracingContext, TracingEngine};
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{Atom, AtomId, Instruction, Program, ProgramBuilder, Traceable, TracingError};
use crate::tracing_v2::operations::{SupportsNeg, SupportsScale};
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

// `Tangent<T, Infallible>` is the zero-only tangent representation described in the
// `Tangent` docs: `NonZero(Infallible)` cannot be constructed, but the generic enum still
// requires its payload type to satisfy the ordinary trace leaf contracts. These impls are vacuous
// because there is no `Infallible` value to inspect or print.
impl<T: Type> Typed<T> for Infallible {
    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        match *self {}
    }
}

impl Parameter for Infallible {}

impl<T: Type> Traceable<T> for Infallible {}

/// Tangent leaf that can represent either a concrete tangent payload or a symbolic zero.
///
/// Engines that need tangent programs containing both concrete tangent leaves and symbolic zero leaves can use this as
/// their tangent value type. Fully zero tangent spaces use `Tangent<T, Infallible>`, where
/// [`Tangent::NonZero`] is statically unconstructible. `NonZero` means "not the symbolic zero branch"; its payload may
/// still be a concrete value whose numeric contents are all zero. Operation semantics stay centralized in the linear
/// operation interpreters: the enum itself only stores the representation and deliberately does not implement arithmetic
/// or array operation traits.
#[derive(Clone, Debug, PartialEq, Parameter)]
pub enum Tangent<T: Type, V: Traceable<T>> {
    /// Symbolic zero with abstract type metadata and no concrete payload.
    Zero(T),

    /// Concrete tangent payload that is not represented by the symbolic zero branch.
    NonZero(V),
}

impl<T: Type, V: Traceable<T>> Tangent<T, V> {
    /// Creates a symbolic zero tangent carrying the provided abstract type metadata.
    #[inline]
    pub fn zero(r#type: T) -> Self {
        Self::Zero(r#type)
    }

    /// Wraps a concrete tangent payload in the non-symbolic-zero branch.
    #[inline]
    pub fn non_zero(value: V) -> Self {
        Self::NonZero(value)
    }

    /// Returns `true` when this tangent is represented as a symbolic zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Zero(_))
    }

    /// Returns the concrete tangent payload when this tangent is in the non-symbolic-zero branch.
    #[inline]
    pub fn as_non_zero(&self) -> Option<&V> {
        match self {
            Self::Zero(_) => None,
            Self::NonZero(value) => Some(value),
        }
    }
}

impl<T: Type, V: Traceable<T>> Display for Tangent<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero(r#type) => write!(formatter, "zero_tangent[{type}]", type = r#type),
            Self::NonZero(value) => Display::fmt(value, formatter),
        }
    }
}

impl<T: Type, V: Traceable<T>> Typed<T> for Tangent<T, V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        match self {
            Self::Zero(r#type) => Cow::Borrowed(r#type),
            Self::NonZero(value) => value.r#type(),
        }
    }
}

impl<T: Parameter + Type, V: Traceable<T>> Traceable<T> for Tangent<T, V> {}

impl<T: Type, V: Traceable<T>> Zero<T> for Tangent<T, V> {
    #[inline]
    fn zero(r#type: &T) -> Result<Self, TracingError> {
        Ok(Self::Zero(r#type.clone()))
    }
}

impl<T: Type, V: Traceable<T> + One<T>> One<T> for Tangent<T, V> {
    #[inline]
    fn one(r#type: &T) -> Result<Self, TracingError> {
        Ok(Self::NonZero(V::one(r#type)?))
    }
}

impl<T: Type, V: Traceable<T>> ZeroLike for Tangent<T, V> {
    #[inline]
    fn zero_like(&self) -> Self {
        Self::Zero(self.r#type().into_owned())
    }
}

/// Value-level contract for leaves that participate in automatic differentiation over `T`.
///
/// The associated [`Tangent`](Self::Tangent) type makes the tangent representation explicit. [`tangent_type`] returns
/// the canonical zero tangent/exemplar for this concrete primal value, carrying whatever metadata the tangent
/// representation needs. Transform code uses this hook when it must synthesize disconnected tangents from primal
/// values instead of guessing from abstract [`Type`] metadata alone.
pub trait Differentiable<T: Type>: Traceable<T> {
    /// Tangent and cotangent leaf type associated with this primal leaf.
    type Tangent: Traceable<T>;

    /// Returns the canonical zero tangent/exemplar associated with this primal value.
    fn tangent_type(&self) -> Result<Self::Tangent, TracingError>;
}

impl<'engine, E: TracingEngine<Value: Differentiable<E::Type>>> Differentiable<E::Type> for Tracer<'engine, E> {
    type Tangent = Self;

    #[inline]
    fn tangent_type(&self) -> Result<Self::Tangent, TracingError> {
        self.context.zero(self.r#type().as_ref())
    }
}

/// Extension of [`Engine`] for backends that support automatic differentiation.
///
/// Engines that only need ordinary tracing implement [`TracingEngine`] without this extension. AD
/// transforms such as [`DifferentiableEngine::jvp`], [`DifferentiableEngine::grad`], and
/// [`vjp`](crate::tracing_v2::vjp) require this trait so non-differentiable backends do not need to
/// define fake tangent carriers.
///
/// Differentiated closures are traced through [`DifferentiableOperationTracingEngine`], whose
/// [`TracingEngine::OperationCarrier`] is [`DifferentiableEngine::DifferentiableOperationCarrier`]. That keeps
/// ordinary tracing free to use a wider operation carrier while making differentiation reject
/// unsupported operations at type-check time when the differentiation carrier omits them.
pub trait DifferentiableEngine: Engine
where
    Self::LinearEngine: TracingEngine<Type = Self::Type, Value = Self::Tangent>,
    <Self::LinearEngine as TracingEngine>::OperationCarrier: Clone
        + InterpretableOperation<Self::Type, Self::Tangent>
        + LinearOperation<Self::Type, Self::Tangent, <Self::LinearEngine as TracingEngine>::OperationCarrier>
        + SupportsZero<Self::Type, Self::Tangent>
        + SupportsAdd<Self::Type, Self::Tangent>,
{
    /// Tangent and cotangent leaf type selected by this differentiable engine.
    type Tangent: Traceable<Self::Type>;

    /// Tracing engine selected by this differentiable engine for tangent and cotangent programs.
    type LinearEngine;

    /// Operation carrier selected by this engine for tracing differentiable primal programs.
    ///
    /// This carrier may be narrower than the ordinary [`TracingEngine::OperationCarrier`]. Every
    /// operation it stores must be interpretable for primal execution and must provide a
    /// [`DifferentiableOperation`] rule for linearization.
    type DifferentiableOperationCarrier: Clone
        + InterpretableOperation<Self::Type, Self::Value>
        + DifferentiableOperation<Self>;

    /// Returns the linearizable engine used for tangent and cotangent programs.
    fn linear_engine(&self) -> &Self::LinearEngine;

    /// Evaluates `function` on `primals` and propagates the supplied tangent values forward.
    ///
    /// The returned pair is `(primal_output, tangent_output)`. This is the canonical user-facing forward-mode
    /// Jacobian-Vector Product (JVP) entry point for differentiable engines.
    #[allow(private_bounds, private_interfaces)]
    fn jvp<
        'engine,
        F: FnOnce(D::FunctionInput) -> D::FunctionOutput,
        Input: Parameterized<D, ParameterStructure: std::fmt::Debug + PartialEq>,
        Output: Parameterized<D>,
        D: crate::tracing_v2::forward::JvpDispatch<'engine, Self, Input, Output, Marker>,
        Marker,
    >(
        &'engine self,
        function: F,
        primals: Input,
        tangents: Input::To<D::Tangent>,
    ) -> Result<(Output, Output::To<D::Tangent>), TracingError>
    where
        Input::Family: ParameterizedFamily<D::Tangent>,
        Output::Family: ParameterizedFamily<D::Tangent>,
    {
        crate::tracing_v2::forward::jvp_at(self, function, primals, tangents)
    }

    /// Computes the reverse-mode gradient of a scalar-output function.
    ///
    /// This is the canonical user-facing reverse-mode entry point for differentiable engines. The function must return
    /// exactly one rank-0 scalar array leaf.
    #[allow(private_bounds, private_interfaces)]
    fn grad<
        'engine,
        F,
        Input: Parameterized<Leaf, ParameterStructure: std::fmt::Debug + PartialEq>,
        Leaf: crate::tracing_v2::linear::ValueAndGradDispatch<Self, Input, Marker>,
        Marker,
    >(
        &'engine self,
        function: F,
        primals: Input,
    ) -> Result<<Leaf as crate::tracing_v2::linear::ValueAndGradDispatch<Self, Input, Marker>>::Gradient, TracingError>
    where
        F: FnOnce(
            <Leaf as crate::tracing_v2::linear::ValueAndGradDispatch<Self, Input, Marker>>::FunctionInput<'engine>,
        )
            -> <Leaf as crate::tracing_v2::linear::ValueAndGradDispatch<Self, Input, Marker>>::FunctionOutput<
            'engine,
        >,
    {
        crate::tracing_v2::linear::Grad::new(function).evaluate(self, primals)
    }

    /// Materializes a dense Jacobian using forward-mode differentiation.
    ///
    /// The returned matrix preserves the input and output parameter structures in its metadata while storing entries in
    /// row-major dense coordinate order.
    #[allow(private_bounds)]
    fn jacfwd<'engine, F, Input, Output, V>(
        &'engine self,
        function: F,
        primals: Input,
    ) -> Result<
        crate::tracing_v2::linear::DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>,
        TracingError,
    >
    where
        Self: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
        V: crate::tracing_v2::linear::CoordinateValue + Differentiable<ArrayType, Tangent = Self::Tangent>,
        Self::Tangent: crate::tracing_v2::linear::CoordinateValue<Coordinate = V::Coordinate>,
        Input: Parameterized<V, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
        Output: Parameterized<V, To<V> = Output, ParameterStructure: PartialEq>,
        Input::Family: ParameterizedFamily<Self::Tangent>
            + ParameterizedFamily<crate::tracing_v2::batching::ReferenceBatch<Self::Tangent>>
            + ParameterizedFamily<Tracer<'engine, DifferentiableOperationTracingEngine<Self>>>,
        Output::Family: ParameterizedFamily<Self::Tangent>
            + ParameterizedFamily<crate::tracing_v2::batching::ReferenceBatch<Self::Tangent>>
            + ParameterizedFamily<Tracer<'engine, DifferentiableOperationTracingEngine<Self>>>,
        Output::To<Tracer<'engine, DifferentiableOperationTracingEngine<Self>>>:
            Parameterized<Tracer<'engine, DifferentiableOperationTracingEngine<Self>>, To<V> = Output>,
        F: FnOnce(
            Input::To<Tracer<'engine, DifferentiableOperationTracingEngine<Self>>>,
        )
            -> Result<Output::To<Tracer<'engine, DifferentiableOperationTracingEngine<Self>>>, TracingError>,
    {
        crate::tracing_v2::linear::JacFwd::new(function).evaluate::<Self, Input, Output, V>(self, primals)
    }

    /// Materializes a dense Hessian of a scalar-output function.
    ///
    /// Hessian evaluation is expressed internally as a forward-mode dense Jacobian over a reverse-mode gradient
    /// transform.
    #[allow(private_bounds)]
    fn hessian<'engine, F, Input, V>(
        &'engine self,
        function: F,
        primals: Input,
    ) -> Result<
        crate::tracing_v2::linear::DenseJacobian<V::Coordinate, Input::ParameterStructure, Input::ParameterStructure>,
        TracingError,
    >
    where
        Self: DifferentiableEngine<Type = ArrayType, Value = V>
            + DifferentiableTracingEngine<Type = ArrayType, Value = V>
            + 'static,
        V: crate::tracing_v2::linear::CoordinateValue + Differentiable<ArrayType, Tangent = Self::Tangent>,
        Self::Tangent: crate::tracing_v2::linear::CoordinateValue<Coordinate = V::Coordinate>,
        Input: Parameterized<V, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
        Input::Family: ParameterizedFamily<Self::Tangent>
            + ParameterizedFamily<crate::tracing_v2::batching::ReferenceBatch<Self::Tangent>>
            + ParameterizedFamily<Tracer<'engine, Self>>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<V>
            + ParameterizedFamily<Tracer<'engine, DifferentiableOperationTracingEngine<Self>>>,
        Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, Self>> = Input::To<Tracer<'engine, Self>>>,
        Input::To<Tracer<'engine, Self>>:
            Parameterized<Tracer<'engine, Self>, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
        <Input::To<Tracer<'engine, Self>> as Parameterized<Tracer<'engine, Self>>>::To<ArrayType>:
            Parameterized<ArrayType, To<Tracer<'engine, Self>> = Input::To<Tracer<'engine, Self>>>,
        Input::To<Tracer<'engine, DifferentiableOperationTracingEngine<Self>>>:
            Parameterized<Tracer<'engine, DifferentiableOperationTracingEngine<Self>>, To<V> = Input>,
        F: FnOnce(Input::To<Tracer<'engine, Self>>) -> Tracer<'engine, Self>,
        Self::OperationCarrier: Clone
            + InterpretableOperation<ArrayType, V>
            + DifferentiableOperation<TracingContext<'engine, Self>>
            + DifferentiableOperation<Self>
            + SupportsZeroLike<ArrayType, V>
            + SupportsAdd<ArrayType, V>
            + 'static,
        AddOperation: InterpretableOperation<ArrayType, Tracer<'engine, Self>>,
    {
        crate::tracing_v2::linear::Hessian::new(function).evaluate(self, primals)
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
pub trait DifferentiableOperation<E: DifferentiableEngine>: Operation<E::Type> {
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

/// Concrete state threaded through forward-mode JVP rules.
///
/// [`JvpContext`] owns the active linear-program builder where tangent ops are staged. It is the
/// forward-mode counterpart of
/// [`TranspositionContext`](crate::tracing::transposition::TranspositionContext): JVP rules call
/// [`apply_operation`](Self::apply_operation) to stage tangent ops on the active builder.
#[doc(hidden)]
pub struct JvpContext<'a, E: DifferentiableEngine> {
    /// Differentiable engine borrowed by this [`JvpContext`] for primal semantics and linear-engine selection.
    pub engine: &'a E,

    /// [`ProgramBuilder`] that owns the staged linear [`Program`](crate::tracing::Program) that is currently being
    /// traced.
    pub builder: Rc<RefCell<ProgramBuilder<E::Type, E::Tangent, <E::LinearEngine as TracingEngine>::OperationCarrier>>>,
}

impl<'a, E: DifferentiableEngine> JvpContext<'a, E> {
    /// Creates a JVP context that stages into `builder`.
    #[doc(hidden)]
    pub fn new(
        engine: &'a E,
        builder: Rc<
            RefCell<
                ProgramBuilder<
                    E::Type,
                    E::Tangent,
                    <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier,
                >,
            >,
        >,
    ) -> Self {
        Self { engine, builder }
    }

    /// Stages one operation in the currently active linear program.
    pub fn stage(
        &self,
        operation: <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier,
        inputs: &[AtomId],
    ) -> Result<Vec<AtomId>, TracingError> {
        let mut builder_borrow = self.builder.borrow_mut();
        let input_types =
            inputs.iter().map(|atom| builder_borrow.atoms[atom.index].r#type().into_owned()).collect::<Vec<_>>();
        let output_types = operation.infer_output_types(&input_types)?;
        let outputs = output_types.into_iter().map(|r#type| builder_borrow.add_variable(r#type)).collect::<Vec<_>>();
        builder_borrow
            .instructions
            .push(Instruction { operation, inputs: inputs.to_vec(), outputs: outputs.clone() });
        Ok(outputs)
    }

    /// Stages a constant tangent on the active linear builder.
    pub fn add_constant(&self, value: E::Tangent) -> AtomId {
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

impl<Ty: Type, V: Traceable<Ty>, T: Clone + std::fmt::Debug + Parameter> Traceable<Ty> for JvpTracer<V, T> {}

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
    ///   - `engine`: Linearizing engine that supplies the linear operation carrier and primitive
    ///     JVP rules.
    ///   - `input_primals`: Concrete primal values aligned with this program's input atoms.
    pub fn linearize<E: DifferentiableEngine<Type = T, Value = V>>(
        &self,
        engine: &E,
        input_primals: Vec<V>,
    ) -> Result<
        Program<
            T,
            E::Tangent,
            <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier,
            Input::To<E::Tangent>,
            Output::To<E::Tangent>,
        >,
        TracingError,
    >
    where
        V: Differentiable<T, Tangent = E::Tangent>,
        Input::Family: ParameterizedFamily<E::Tangent>,
        Output::Family: ParameterizedFamily<E::Tangent>,
        O: DifferentiableOperation<E>,
    {
        fn tangent_for_atom<T, V, LinearOperationCarrier>(
            primal_values: &[Option<V>],
            builder: &Rc<RefCell<ProgramBuilder<T, V::Tangent, LinearOperationCarrier>>>,
            tangents: &mut [Option<AtomId>],
            atom_id: AtomId,
        ) -> Result<AtomId, TracingError>
        where
            T: Type,
            V: Differentiable<T>,
            LinearOperationCarrier: Clone + Operation<T>,
        {
            if let Some(atom) = tangents[atom_id.index] {
                return Ok(atom);
            }
            let primal = primal_values[atom_id.index].as_ref().ok_or(TracingError::UnboundAtomId { id: atom_id })?;
            let atom = builder.borrow_mut().add_constant(primal.tangent_type()?);
            tangents[atom_id.index] = Some(atom);
            Ok(atom)
        }

        check_count!("input", input_primals, self.input_ids.len(), TracingError);
        let builder = Rc::new(RefCell::new(ProgramBuilder::<
            T,
            E::Tangent,
            <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier,
        >::new()));
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
                        tangent: tangent_for_atom::<
                            T,
                            V,
                            <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier,
                        >(
                            primals.as_slice(), &builder, tangents.as_mut_slice(), input_atom
                        )?,
                    })
                })
                .collect::<Result<Vec<_>, TracingError>>()?;
            let output_duals = instruction.operation.jvp(&mut context, input_duals.as_slice())?;
            check_count!("output", output_duals, instruction.outputs.len(), TracingError);
            for (output_atom, output_dual) in instruction.outputs.iter().copied().zip(output_duals.into_iter()) {
                primals[output_atom.index] = Some(output_dual.primal);
                tangents[output_atom.index] = Some(output_dual.tangent);
            }
        }

        let output_tangents =
            self.output_ids
                .iter()
                .copied()
                .map(|output_atom| {
                    tangent_for_atom::<
                        T,
                        V,
                        <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier,
                    >(primals.as_slice(), &builder, tangents.as_mut_slice(), output_atom)
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

/// Optional extension for tracing engines that support differentiation inside an active trace.
///
/// Plain tracing engines do not need to choose any linear carrier. This trait is the additional
/// contract required when a [`TracingContext`](crate::tracing::engines::TracingContext) itself needs to act
/// as a differentiable engine: tangent and cotangent programs then operate on
/// [`Tracer`] values, so the underlying tracing engine must select a linear operation carrier for
/// those traced leaves.
pub trait DifferentiableTracingEngine: TracingEngine {
    /// Linear operation carrier selected for tangent and cotangent programs over traced values.
    ///
    /// This carrier is stored in nested linear programs whose leaves are [`Tracer`] values from an
    /// active outer trace.
    type LinearOperationCarrier<'engine>: Clone
        + InterpretableOperation<Self::Type, Tracer<'engine, Self>>
        + LinearOperation<Self::Type, Tracer<'engine, Self>, Self::LinearOperationCarrier<'engine>>
        + SupportsZero<Self::Type, Tracer<'engine, Self>>
        + SupportsNeg<Self::Type, Tracer<'engine, Self>>
        + SupportsAdd<Self::Type, Tracer<'engine, Self>>
        + SupportsScale<Self::Type, Tracer<'engine, Self>>
    where
        Self: 'engine;
}

/// Transparent tracing view used while tracing differentiable primal programs.
///
/// Automatic-differentiation transforms need to stage the user's primal closure with
/// [`DifferentiableEngine::DifferentiableOperationCarrier`] rather than the ordinary
/// [`TracingEngine::OperationCarrier`] selected by the backend. Those carriers may intentionally differ:
/// an engine can support a broad ordinary tracing universe while exposing a narrower
/// differentiable carrier whose variants all have differentiation rules under the real engine. This adapter
/// only selects that carrier while tracing; the resulting program is still linearized with the wrapped engine.
///
/// [`DifferentiableOperationTracingEngine::new`] reborrows an `E: DifferentiableEngine` as a
/// [`TracingEngine`] without allocation or ownership. AD entry points construct this view at trace
/// boundaries such as [`linearize`](crate::tracing_v2::linearize),
/// [`vjp`](crate::tracing_v2::vjp), and [`DifferentiableEngine::grad`], pass it immediately to
/// ordinary tracing helpers, and keep backend implementations centered on their real engine type.
/// User-facing ordinary tracing should keep using the backend's own [`TracingEngine`]
/// implementation; traced tangent and cotangent programs are selected separately through
/// [`DifferentiableTracingEngine`].
///
/// This type is public today because the public AD closure bounds still mention
/// `Tracer<'engine, DifferentiableOperationTracingEngine<E>>`. Once those APIs hide the concrete
/// active tracer carrier, this adapter can become a `pub(crate)` implementation detail.
#[repr(transparent)]
pub struct DifferentiableOperationTracingEngine<E: DifferentiableEngine> {
    /// Engine viewed through its differentiable operation carrier.
    engine: E,
}

impl<E: DifferentiableEngine> DifferentiableOperationTracingEngine<E> {
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

impl<E: DifferentiableEngine> std::fmt::Debug for DifferentiableOperationTracingEngine<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("DifferentiableOperationTracingEngine").finish_non_exhaustive()
    }
}

impl<E: DifferentiableEngine> Engine for DifferentiableOperationTracingEngine<E> {
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

impl<E: DifferentiableEngine> TracingEngine for DifferentiableOperationTracingEngine<E> {
    type OperationCarrier = E::DifferentiableOperationCarrier;
}

impl<'engine, E> TracingEngine for TracingContext<'engine, E>
where
    E: DifferentiableTracingEngine,
{
    type OperationCarrier = E::LinearOperationCarrier<'engine>;
}

impl<'engine, E> DifferentiableEngine for TracingContext<'engine, E>
where
    E: DifferentiableTracingEngine,
    E::Value: Differentiable<E::Type>,
    E::OperationCarrier: SupportsAdd<E::Type, E::Value>,
    AddOperation: InterpretableOperation<E::Type, Tracer<'engine, E>>,
{
    type Tangent = Tracer<'engine, E>;
    type LinearEngine = Self;
    type DifferentiableOperationCarrier = AddOperation;

    #[inline]
    fn linear_engine(&self) -> &Self::LinearEngine {
        self
    }
}

macro_rules! impl_differentiable_engine_for_scalar {
    ($ty:ty) => {
        impl DifferentiableEngine for ScalarEngine<$ty> {
            type Tangent = $ty;
            type LinearEngine = crate::tracing::engines::LinearScalarEngine<$ty>;
            type DifferentiableOperationCarrier = ScalarOperation<$ty>;

            #[inline]
            fn linear_engine(&self) -> &Self::LinearEngine {
                &self.linear_engine
            }
        }

        impl DifferentiableTracingEngine for ScalarEngine<$ty> {
            type LinearOperationCarrier<'engine>
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
    use std::convert::Infallible;

    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use crate::operations::constants::{One, Zero, ZeroLike};
    use crate::tracing::engines::ScalarEngine;
    use crate::types::{ArrayType, DataType, Shape, Size, Typed};

    use super::{DifferentiableEngine, Tangent};

    #[test]
    fn test_tangent_value_carries_symbolic_zero_or_non_zero_tangent() {
        let zero = Tangent::<DataType, f64>::zero(DataType::F64);
        let non_zero = Tangent::<DataType, f64>::non_zero(2.5);

        assert!(zero.is_zero());
        assert_eq!(zero.as_non_zero(), None);
        assert_eq!(zero.r#type().into_owned(), DataType::F64);
        assert_eq!(zero.to_string(), "zero_tangent[f64]");
        assert_eq!(<Tangent<DataType, f64> as Zero<DataType>>::zero(&DataType::F64), Ok(zero.clone()));
        assert_eq!(non_zero.as_non_zero(), Some(&2.5));
        assert_eq!(non_zero.r#type().into_owned(), DataType::F64);
        assert_eq!(non_zero.to_string(), "2.5");
        assert_eq!(<Tangent<DataType, f64> as One<DataType>>::one(&DataType::F64), Ok(Tangent::non_zero(1.0)));
        assert_eq!(non_zero.zero_like(), zero);

        let zero_only = Tangent::<DataType, Infallible>::zero(DataType::I32);
        assert_eq!(zero_only.r#type().into_owned(), DataType::I32);
        assert_eq!(zero_only.to_string(), "zero_tangent[i32]");
        assert_eq!(<Tangent<DataType, Infallible> as Zero<DataType>>::zero(&DataType::I32), Ok(zero_only.clone()));
        assert_eq!(zero_only.zero_like(), zero_only);

        let array_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(2)]), None, None).unwrap();
        let array_tangent = Tangent::<ArrayType, Infallible>::zero(array_type.clone());
        assert_eq!(array_tangent.r#type().into_owned(), array_type);
    }

    #[test]
    fn test_scalar_engine_half_and_float_engines_are_differentiable() {
        let _: Option<<ScalarEngine<bf16> as DifferentiableEngine>::DifferentiableOperationCarrier> = None;
        let _: Option<<ScalarEngine<f16> as DifferentiableEngine>::DifferentiableOperationCarrier> = None;
        let _: Option<<ScalarEngine<f32> as DifferentiableEngine>::DifferentiableOperationCarrier> = None;
        let _: Option<<ScalarEngine<f64> as DifferentiableEngine>::DifferentiableOperationCarrier> = None;
    }

    #[test]
    fn test_scalar_engine_half_engines_run_jvp() {
        let bf16_engine = ScalarEngine::<bf16>::new();
        assert_eq!(
            bf16_engine.jvp(|x| x.clone() + x, bf16::from_f32(3.0), bf16::ONE),
            Ok((bf16::from_f32(6.0), bf16::from_f32(2.0)))
        );

        let f16_engine = ScalarEngine::<f16>::new();
        assert_eq!(
            f16_engine.jvp(|x| x.clone() + x, f16::from_f32(3.0), f16::ONE),
            Ok((f16::from_f32(6.0), f16::from_f32(2.0)))
        );
    }
}
