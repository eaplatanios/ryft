//! Rematerialization (i.e., gradient checkpointing), which trades computation for memory under differentiation. A
//! rematerialized function computes what its body computes, but differentiation saves only the values of the body that
//! its [`ResidualPolicy`] selects and recomputes the others from the saved values when the derivative computation
//! needs them, instead of keeping every value of the body alive until then. This is the analogue of
//! [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html) (also known as `jax.remat`).
//!
//! [`rematerialize`] creates a rematerialized function from a closure, [`Rematerialize::with_policy`] selects which
//! values it saves (refer to the [`policies`] module for the built-in policies), and [`Rematerialize::call`] stages one
//! call of the function as a [`RematerializeOperation`] whose body is the traced closure. Nothing is derived when the
//! function is called: the transforms derive the derivatives of the body when they need them.
//!
//! # Examples
//!
//! ```rust
//! # use ryft_core::differentiation::rematerialization::{DotsSaveable, rematerialize};
//! # use ryft_core::{Array, ArrayOperation, DomainTracer, EagerContext, ProgramError, Sin};
//! # fn main() -> Result<(), ProgramError> {
//! type Context = EagerContext<Array, ArrayOperation<Array>>;
//! type Tracer = DomainTracer<Context>;
//!
//! // The closure annotates its tracer input, which determines the context that the function is traced in.
//! let function = rematerialize(|x: Tracer| Ok(x.sin()?)).with_policy(DotsSaveable);
//! assert_eq!(function.call_in_context(&Context::new(), Array::scalar(0.5f64)?)?, Array::scalar(0.5f64.sin())?);
//! # Ok(())
//! # }
//! ```

// TODO(eaplatanios): Review this module.

pub mod policies;

use std::any::{Any, TypeId};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, LazyLock, Mutex};

use crate::axes::NamedAxes;
use crate::contexts::Context;
use crate::operations::RematerializeOperation;
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::partial::{ResidualPolicy, ResidualPolicyReference};
use crate::programs::{ProgramError, Type, TypeError, Typed, Value};
use crate::tracing::{DomainTracer, DomainTracingContext};

pub use policies::{
    DOTS_SAVEABLE_POLICY_NAME, DOTS_WITH_NO_BATCH_DIMENSIONS_SAVEABLE_POLICY_NAME, DotsSaveable,
    DotsWithNoBatchDimensionsSaveable, EVERYTHING_SAVEABLE_POLICY_NAME, EverythingSaveable, MemoryTransferStorage,
    NOTHING_SAVEABLE_POLICY_NAME, NothingSaveable, OFFLOAD_DOTS_WITH_NO_BATCH_DIMENSIONS_POLICY_NAME,
    OffloadDotsWithNoBatchDimensions, POLICY_FN_POLICY_NAME, PolicyFn, SAVE_AND_OFFLOAD_ONLY_THESE_NAMES_POLICY_NAME,
    SAVE_ANY_NAMES_BUT_THESE_POLICY_NAME, SAVE_ANYTHING_EXCEPT_THESE_NAMES_POLICY_NAME,
    SAVE_FROM_BOTH_POLICIES_POLICY_NAME, SAVE_ONLY_THESE_NAMES_POLICY_NAME, SaveAndOffloadOnlyTheseNames,
    SaveAnyNamesButThese, SaveAnythingExceptTheseNames, SaveFromBothPolicies, SaveOnlyTheseNames,
};

/// [`PolicyReferences`] of the default [`NothingSaveable`] policy, which every [`Rematerialize`] that does not select
/// a policy shares, so that all of their calls stage operations whose policies compare equal.
static DEFAULT_POLICY_REFERENCES: LazyLock<Arc<PolicyReferences<NothingSaveable>>> =
    LazyLock::new(|| Arc::new(PolicyReferences::new(NothingSaveable)));

/// Residual policy of a [`Rematerialize`] together with its [`ResidualPolicyReference`]s in the type universes that
/// its calls have used so far. [`ResidualPolicyReference`]s compare by the identity of the policy definition that
/// [`ResidualPolicyReference::new`] registers, so creating one per call would make calls of the same function stage
/// unequal operations. Clones of a [`Rematerialize`] share one [`PolicyReferences`], and therefore stage equal
/// operations too.
struct PolicyReferences<P> {
    /// Residual policy.
    policy: P,

    /// [`ResidualPolicyReference`] to `policy` in each type universe that was used so far, keyed by the [`TypeId`] of
    /// the universe.
    references: Mutex<Vec<(TypeId, Box<dyn Any + Send + Sync>)>>,
}

impl<P> PolicyReferences<P> {
    /// Creates new [`PolicyReferences`] for `policy` that hold no references yet.
    #[inline]
    fn new(policy: P) -> Self {
        Self { policy, references: Mutex::new(Vec::new()) }
    }

    /// Returns the [`ResidualPolicyReference`] to the policy in the type universe `T`, which is registered on first use.
    fn reference<T: 'static + Type>(&self) -> ResidualPolicyReference<T>
    where
        P: Clone + ResidualPolicy<T>,
    {
        let mut references = self.references.lock().expect("the residual policy references lock is poisoned");
        let reference = references
            .iter()
            .find(|(universe, _)| *universe == TypeId::of::<T>())
            .and_then(|(_, reference)| reference.downcast_ref::<ResidualPolicyReference<T>>());
        if let Some(reference) = reference {
            return reference.clone();
        }
        let reference = ResidualPolicyReference::new(self.policy.clone());
        references.push((TypeId::of::<T>(), Box::new(reference.clone())));
        reference
    }
}

/// Rematerialized function, which [`rematerialize`] creates from a closure over [`DomainTracer`]s. Refer to the
/// [module documentation](self) for more information.
pub struct Rematerialize<Input, Output, Body, P = NothingSaveable> {
    /// Closure that computes the body of the function.
    body: Body,

    /// Residual policy of the function together with its references in the type universes that were used so far.
    policy: Arc<PolicyReferences<P>>,

    /// Whether the staged calls place an optimization barrier on their inputs when they are differentiated (refer to
    /// [`RematerializeOperation::optimization_barrier`]).
    optimization_barrier: bool,

    /// Input and output types of the closure.
    marker: PhantomData<fn() -> (Input, Output)>,
}

impl<Input, Output, Body, P> Rematerialize<Input, Output, Body, P> {
    /// Returns this function with the provided residual policy, which decides which values of the body differentiation
    /// saves (refer to the [`policies`] module for the built-in policies).
    #[inline]
    pub fn with_policy<Q>(self, policy: Q) -> Rematerialize<Input, Output, Body, Q> {
        Rematerialize {
            body: self.body,
            policy: Arc::new(PolicyReferences::new(policy)),
            optimization_barrier: self.optimization_barrier,
            marker: PhantomData,
        }
    }

    /// Returns this function with the provided optimization-barrier flag, which is `true` by default (refer to
    /// [`RematerializeOperation::optimization_barrier`]). This is the analogue of the `prevent_cse` parameter of
    /// [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html), which can be disabled when
    /// the function is called in a loop body (e.g., of a `scan`), where the loop already keeps the recomputation from
    /// being merged with the original computation.
    #[inline]
    pub fn with_optimization_barrier(mut self, optimization_barrier: bool) -> Self {
        self.optimization_barrier = optimization_barrier;
        self
    }

    /// Stages one call of this function on the provided `input` value and returns its output value. The [`Context`]
    /// `C` that the call is staged into is the [`DispatchDomain`](Value::DispatchDomain) of the values in `input`, so
    /// it is never named at a construction or call site. The body is traced in a fresh trace that is seeded with the
    /// named axes in scope in `C` (refer to [`NamedAxes::named_axes`]), so that it resolves the axes of enclosing
    /// transforms as it would if it were inlined.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `input` has no leaves (in which case [`call_in_context`](Self::call_in_context)
    /// must be used instead), when tracing the body fails, or when the staged [`RematerializeOperation`] rejects the
    /// call.
    pub fn call<
        V: Value<Type = C::Type, DispatchDomain = C>,
        C: Context<Type: 'static, Value = V> + NamedAxes,
        InputValues: Parameterized<V, Family = Input::Family, To<C::Type> = Input::To<C::Type>>,
    >(
        &self,
        input: InputValues,
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<V>, ProgramError>
    where
        C::Operation: From<RematerializeOperation<C::Type>>,
        Body: Fn(Input) -> Result<Output, ProgramError>,
        P: Clone + ResidualPolicy<C::Type>,
        Input: Parameterized<DomainTracer<C>>,
        Input::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant>,
        Input::To<C::Type>: Parameterized<C::Type, Family = Input::Family, To<DomainTracer<C>> = Input>,
        Output: Parameterized<DomainTracer<C>>,
        Output::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant> + ParameterizedFamily<V>,
        Output::To<C::Type>: Parameterized<C::Type, Family = Output::Family, To<DomainTracer<C>> = Output>,
    {
        let mut input_values = Vec::new();
        let input_types = input
            .map_parameters(|value| {
                let r#type = value.r#type().into_owned();
                input_values.push(value);
                r#type
            })
            .map_err(ProgramError::from)?;
        let Some(first) = input_values.first() else {
            return Err(TypeError::invalid(
                "`rematerialize` requires at least one input to recover its context from; use \
                 `Rematerialize::call_in_context` for functions without inputs",
            )
            .into());
        };
        self.stage(&first.dispatch_domain(), input_types, input_values.as_slice())
    }

    /// Stages one call of this function on the provided `input` value in the provided `context` and returns its output
    /// value. This is the explicit-context counterpart of [`call`](Self::call), which supports functions without
    /// inputs and contexts that are not the dispatch domain of the input values.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when tracing the body fails or when the staged [`RematerializeOperation`] rejects
    /// the call.
    pub fn call_in_context<
        C: Context<Type: 'static> + NamedAxes,
        InputValues: Parameterized<C::Value, Family = Input::Family, To<C::Type> = Input::To<C::Type>>,
    >(
        &self,
        context: &C,
        input: InputValues,
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<C::Value>, ProgramError>
    where
        C::Operation: From<RematerializeOperation<C::Type>>,
        Body: Fn(Input) -> Result<Output, ProgramError>,
        P: Clone + ResidualPolicy<C::Type>,
        Input: Parameterized<DomainTracer<C>>,
        Input::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant>,
        Input::To<C::Type>: Parameterized<C::Type, Family = Input::Family, To<DomainTracer<C>> = Input>,
        Output: Parameterized<DomainTracer<C>>,
        Output::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant> + ParameterizedFamily<C::Value>,
        Output::To<C::Type>: Parameterized<C::Type, Family = Output::Family, To<DomainTracer<C>> = Output>,
    {
        let mut input_values = Vec::new();
        let input_types = input
            .map_parameters(|value| {
                let r#type = value.r#type().into_owned();
                input_values.push(value);
                r#type
            })
            .map_err(ProgramError::from)?;
        self.stage(context, input_types, input_values.as_slice())
    }

    /// Traces the body at `input_types` and binds one call of it to `input_values` in `context`.
    fn stage<C: Context<Type: 'static> + NamedAxes>(
        &self,
        context: &C,
        input_types: Input::To<C::Type>,
        input_values: &[C::Value],
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<C::Value>, ProgramError>
    where
        C::Operation: From<RematerializeOperation<C::Type>>,
        Body: Fn(Input) -> Result<Output, ProgramError>,
        P: Clone + ResidualPolicy<C::Type>,
        Input: Parameterized<DomainTracer<C>>,
        Input::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant>,
        Input::To<C::Type>: Parameterized<C::Type, Family = Input::Family, To<DomainTracer<C>> = Input>,
        Output: Parameterized<DomainTracer<C>>,
        Output::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant> + ParameterizedFamily<C::Value>,
        Output::To<C::Type>: Parameterized<C::Type, Family = Output::Family, To<DomainTracer<C>> = Output>,
    {
        let (output_types, body) =
            DomainTracingContext::<C>::trace_with_named_axes(&self.body, input_types, context.named_axes())?;
        let output_structure = output_types.parameter_structure();
        let operation = RematerializeOperation::new(self.policy.reference::<C::Type>())
            .with_optimization_barrier(self.optimization_barrier);
        let outputs = context.bind(operation, vec![body.into_flat_program()], input_values)?;
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

impl<Input, Output, Body: Clone, P> Clone for Rematerialize<Input, Output, Body, P> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            body: self.body.clone(),
            policy: self.policy.clone(),
            optimization_barrier: self.optimization_barrier,
            marker: PhantomData,
        }
    }
}

impl<Input, Output, Body, P: Debug> Debug for Rematerialize<Input, Output, Body, P> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Rematerialize")
            .field("policy", &self.policy.policy)
            .field("optimization_barrier", &self.optimization_barrier)
            .finish_non_exhaustive()
    }
}

/// Creates a [`Rematerialize`] function from a closure `x ↦ y = f(x)` over [`DomainTracer`]s, which saves nothing under
/// differentiation (i.e., it uses the [`NothingSaveable`] policy) and places an optimization barrier on its inputs when
/// it is differentiated. The closure must annotate the type of its tracer input, which determines the context that its
/// body is traced in, and nothing is traced until the function is called. Refer to the [module documentation](self)
/// for more information.
#[inline]
pub fn rematerialize<Input, Output, Body: Fn(Input) -> Result<Output, ProgramError>>(
    body: Body,
) -> Rematerialize<Input, Output, Body> {
    Rematerialize { body, policy: DEFAULT_POLICY_REFERENCES.clone(), optimization_barrier: true, marker: PhantomData }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType};
    use crate::contexts::EagerContext;
    use crate::operations::Sin;
    use crate::programs::Program;
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    type TestTracer = Tracer<TracingContext<Array, ArrayOperation<Array>>>;

    /// Traces `function` into a program over one `f64[]` scalar input.
    fn trace<F: Fn(TestTracer) -> Result<TestTracer, ProgramError>>(
        function: F,
    ) -> Program<Array, ArrayOperation<Array>, Array, Array> {
        TracingContext::<Array, ArrayOperation<Array>>::trace(function, ArrayType::scalar(DataType::F64))
            .unwrap()
            .1
    }

    /// Returns the [`RematerializeOperation`] of the instruction at `index` in `program`.
    fn operation(
        program: &Program<Array, ArrayOperation<Array>, Array, Array>,
        index: usize,
    ) -> RematerializeOperation<ArrayType> {
        match program.instructions()[index].operation() {
            ArrayOperation::Rematerialize(operation) => operation.clone(),
            operation => panic!("expected a `rematerialize` operation but got `{operation}`"),
        }
    }

    #[test]
    fn test_rematerialize_with_policy() {
        let function = rematerialize(|x: TestTracer| Ok(x.sin()?)).with_policy(DotsSaveable);
        let program = trace(|x| function.call(x));
        assert_eq!(operation(&program, 0).policy().name(), "dots_saveable");
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = rematerialize [policy=\"dots_saveable\"] %0 [
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_rematerialize_with_optimization_barrier() {
        let function = rematerialize(|x: TestTracer| Ok(x.sin()?));
        assert!(operation(&trace(|x| function.call(x)), 0).optimization_barrier());
        let function = function.with_optimization_barrier(false);
        assert!(!operation(&trace(|x| function.call(x)), 0).optimization_barrier());
    }

    #[test]
    fn test_rematerialize_call() {
        // Each call stages one `rematerialize` operation, whose body is the traced closure, into the dispatch domain of
        // its inputs. Repeated calls of one function stage equal operations because they share the reference to its
        // policy.
        let function = rematerialize(|x: TestTracer| Ok(x.sin()?));
        let program = trace(|x| function.call(function.call(x)?));
        assert_eq!(
            program.to_string(),
            indoc! {"
            PLACEHOLDER"},
        );
        assert_eq!(operation(&program, 0), operation(&program, 1));
        assert_eq!(program.interpret(Array::scalar(0.5f64).unwrap()), Ok(Array::scalar(0.5f64.sin().sin()).unwrap()),);
    }

    #[test]
    fn test_rematerialize_call_without_inputs() {
        let function = rematerialize(|inputs: Vec<TestTracer>| Ok(inputs));
        assert_eq!(
            function.call(Vec::<TestTracer>::new()),
            Err(ProgramError::from(TypeError::invalid(
                "`rematerialize` requires at least one input to recover its context from; use \
                 `Rematerialize::call_in_context` for functions without inputs",
            ))),
        );
    }

    #[test]
    fn test_rematerialize_call_in_context() {
        // Eager arrays dispatch to a context whose operation family cannot represent `rematerialize`, so eager calls
        // name a context that can.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let function = rematerialize(|inputs: Vec<TestTracer>| -> Result<Vec<TestTracer>, ProgramError> {
            inputs.into_iter().map(|x| Ok(x.sin()?)).collect()
        });
        assert_eq!(
            function.call_in_context(&context, vec![Array::scalar(0.5f64).unwrap()]),
            Ok(vec![Array::scalar(0.5f64.sin()).unwrap()]),
        );

        // Unlike `call`, `call_in_context` supports functions without inputs.
        assert_eq!(function.call_in_context(&context, Vec::<Array>::new()), Ok(Vec::new()));
    }

    #[test]
    fn test_rematerialize_clone() {
        // Clones share the reference to the policy of the function, so their calls stage equal operations.
        let function = rematerialize(|x: TestTracer| Ok(x.sin()?)).with_policy(DotsSaveable);
        let clone = function.clone();
        let program = trace(|x| clone.call(function.call(x)?));
        assert_eq!(operation(&program, 0), operation(&program, 1));

        // Selecting the same policy again defines a new policy, whose calls stage different operations.
        let other = function.clone().with_policy(DotsSaveable);
        let program = trace(|x| other.call(function.call(x)?));
        assert_ne!(operation(&program, 0), operation(&program, 1));
    }

    #[test]
    fn test_rematerialize_debug() {
        let function = rematerialize(|x: TestTracer| Ok(x.sin()?)).with_optimization_barrier(false);
        assert_eq!(
            format!("{function:?}"),
            "Rematerialize { policy: NothingSaveable, optimization_barrier: false, .. }"
        );
    }

    #[test]
    fn test_rematerialize() {
        // Functions that use the default policy share its reference, so their calls stage equal operations.
        let sine = rematerialize(|x: TestTracer| Ok(x.sin()?));
        let identity = rematerialize(|x: TestTracer| Ok(x));
        let program = trace(|x| identity.call(sine.call(x)?));
        assert_eq!(operation(&program, 0), operation(&program, 1));
        assert_eq!(operation(&program, 0).policy().name(), "nothing_saveable");
    }
}
