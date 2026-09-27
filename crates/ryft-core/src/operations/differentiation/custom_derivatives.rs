use std::fmt::Debug;

use crate::contexts::Context;
use crate::differentiation::DifferentiableType;
use crate::operations::differentiation::custom_jvp::{CustomJvpOperation, custom_jvp};
use crate::operations::differentiation::custom_vjp::{CustomVjpOperation, custom_vjp};
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::{ProgramError, Value};
use crate::tracing::DomainTracer;

/// Builder, returned by [`custom_derivative_at`], that stages a custom derivative rule at a known input value. Because
/// the builder captures the input before the rule closures are written, those closures infer their tracer parameter
/// types from the input and need no type annotations.
///
/// [`with_non_differentiated_count`](Self::with_non_differentiated_count) configures the call, and the
/// terminal [`jvp`](Self::jvp) and [`vjp`](Self::vjp) functions stage it as a [`CustomJvpOperation`] or a
/// [`CustomVjpOperation`], respectively.
///
/// Refer to the documentation of the [`custom_jvp`] and [`custom_vjp`] functions for the semantics of the staged calls.
pub struct CustomDerivativeBuilder<Input> {
    /// Input value at which the custom derivative rule is staged.
    input: Input,

    /// Number of leading flattened input leaves that parameterize the call without being differentiated.
    non_differentiated_count: usize,
}

impl<Input> CustomDerivativeBuilder<Input> {
    /// Returns a copy with the provided number of leading flattened input leaves treated as non-differentiated
    /// _plumbing_ inputs.
    ///
    /// This is the [`CustomDerivativeBuilder`] counterpart of
    /// [`CustomJvp::with_non_differentiated_count`](crate::CustomJvp::with_non_differentiated_count)
    /// and [`CustomVjp::with_non_differentiated_count`](crate::CustomVjp::with_non_differentiated_count).
    ///
    /// Refer to the documentation of the [`custom_jvp`] and [`custom_vjp`] functions for the semantics
    /// of non-differentiated inputs.
    #[inline]
    pub fn with_non_differentiated_count(mut self, non_differentiated_count: usize) -> Self {
        self.non_differentiated_count = non_differentiated_count;
        self
    }

    /// Stages a custom Jacobian-Vector Product (JVP) call at the input of this builder and returns its output value.
    /// This is equivalent to `custom_jvp(primal, jvp).with_non_differentiated_count(count).call(input)`, except that
    /// the closures infer their parameter types from the input.
    ///
    /// Refer to the documentation of the [`custom_jvp`] function for the semantics of the staged call.
    ///
    /// # Parameters
    ///
    ///   - `primal`: Closure implementing `f(x) = y`.
    ///   - `jvp`: Closure implementing `(x, ẋ) ↦ (y, ẏ)`, where `ẏ = J_f(x) · ẋ`.
    ///
    /// # Errors
    ///
    /// Returns the [`ProgramError`]s described in the documentation of [`CustomJvp::call`](crate::CustomJvp::call).
    #[inline]
    pub fn jvp<
        V: Value<Type = C::Type, DispatchDomain = C>,
        C: Context<Type: DifferentiableType, Value = V, Operation: From<CustomJvpOperation<C::Type>>>,
        Output: Parameterized<DomainTracer<C>, ParameterStructure: Debug + PartialEq>,
        Primal: Fn(Input::To<DomainTracer<C>>) -> Result<Output, ProgramError>,
        Jvp: Fn(Input::To<DomainTracer<C>>, Input::To<DomainTracer<C>>) -> Result<(Output, Output), ProgramError>,
    >(
        self,
        primal: Primal,
        jvp: Jvp,
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<V>, ProgramError>
    where
        Input: Parameterized<V>,
        Input::Family:
            ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant> + ParameterizedFamily<DomainTracer<C>>,
        Input::To<DomainTracer<C>>:
            Parameterized<DomainTracer<C>, Family = Input::Family, To<C::Type> = Input::To<C::Type>>,
        Input::To<C::Type>:
            Clone + Parameterized<C::Type, Family = Input::Family, To<DomainTracer<C>> = Input::To<DomainTracer<C>>>,
        Output::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant> + ParameterizedFamily<V>,
        Output::To<C::Type>: Parameterized<C::Type, Family = Output::Family, To<DomainTracer<C>> = Output>,
    {
        custom_jvp(primal, jvp)
            .with_non_differentiated_count(self.non_differentiated_count)
            .call(self.input)
    }

    /// Stages a custom Vector-Jacobian Product (VJP) call at the input of this builder and returns its output value.
    /// This is equivalent to `custom_vjp(primal, forward, backward).with_non_differentiated_count(count).call(input)`,
    /// except that the closures infer their parameter types from the input and, for `backward`, from the residuals
    /// that `forward` returns.
    ///
    /// Refer to the documentation of the [`custom_vjp`] function for the semantics of the staged call.
    ///
    /// # Parameters
    ///
    ///   - `primal`: Closure implementing `f(x) = y` for ordinary evaluation.
    ///   - `forward`: Closure implementing `x ↦ (y, r)` for reverse-mode residual production.
    ///   - `backward`: Closure implementing `(r, ȳ) ↦ x̄ = J_f(x)ᵀ · ȳ`.
    ///
    /// # Errors
    ///
    /// Returns the [`ProgramError`]s described in the documentation of [`CustomVjp::call`](crate::CustomVjp::call).
    #[inline]
    pub fn vjp<
        V: Value<Type = C::Type, DispatchDomain = C>,
        C: Context<Type: DifferentiableType, Value = V, Operation: From<CustomVjpOperation<C::Type>>>,
        Output: Parameterized<DomainTracer<C>, ParameterStructure: Debug + PartialEq>,
        Residual: Parameterized<DomainTracer<C>>,
        Primal: Fn(Input::To<DomainTracer<C>>) -> Result<Output, ProgramError>,
        Forward: Fn(Input::To<DomainTracer<C>>) -> Result<(Output, Residual), ProgramError>,
        Backward: Fn(Residual, Output) -> Result<Input::To<DomainTracer<C>>, ProgramError>,
    >(
        self,
        primal: Primal,
        forward: Forward,
        backward: Backward,
    ) -> Result<<Output::To<C::Type> as Parameterized<C::Type>>::To<V>, ProgramError>
    where
        Input: Parameterized<V, ParameterStructure: Debug + PartialEq>,
        Input::Family:
            ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant> + ParameterizedFamily<DomainTracer<C>>,
        Input::To<DomainTracer<C>>:
            Parameterized<DomainTracer<C>, Family = Input::Family, To<C::Type> = Input::To<C::Type>>,
        Input::To<C::Type>:
            Clone + Parameterized<C::Type, Family = Input::Family, To<DomainTracer<C>> = Input::To<DomainTracer<C>>>,
        Output::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant> + ParameterizedFamily<V>,
        Output::To<C::Type>: Clone + Parameterized<C::Type, Family = Output::Family, To<DomainTracer<C>> = Output>,
        Residual::Family: ParameterizedFamily<C::Type> + ParameterizedFamily<C::Constant>,
        Residual::To<C::Type>: Parameterized<C::Type, Family = Residual::Family, To<DomainTracer<C>> = Residual>,
    {
        custom_vjp(primal, forward, backward)
            .with_non_differentiated_count(self.non_differentiated_count)
            .call(self.input)
    }
}

/// Creates a [`CustomDerivativeBuilder`] that stages a custom derivative rule at `input`, which is the input-first
/// counterpart of the [`custom_jvp`] and [`custom_vjp`] functions. Those functions build a reusable function from rule
/// closures before any input is known, so their closures must annotate the tracer type of their input. This function
/// instead receives the input before the rule closures, so the closures infer all of their parameter types from it.
/// Prefer it when a rule is applied where it is defined, and prefer [`custom_jvp`] or [`custom_vjp`] when the same
/// rule is called at several sites.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{Array, Cos, ProgramError, Sin, custom_derivative_at, differentiate_at};
/// # fn main() -> Result<(), ProgramError> {
/// // A custom JVP rule for `sin` that doubles the true derivative, so that its effect is visible.
/// let (value, tangent) = differentiate_at(Array::scalar(0.5f64)?).jvp(Array::scalar(1.0f64)?, |x| {
///     custom_derivative_at(x).jvp(
///         |x| Ok(x.sin()?),
///         |x, tangent| {
///             let tangent = x.cos()? * tangent;
///             Ok((x.sin()?, tangent.clone() + tangent))
///         },
///     )
/// })?;
/// assert_eq!(value, Array::scalar(0.5f64.sin())?);
/// assert_eq!(tangent, Array::scalar(2.0 * 0.5f64.cos())?);
///
/// // A custom VJP rule for `sin` that saves `cos(x)` as its residual and doubles the true gradient.
/// let gradient = differentiate_at(Array::scalar(0.5f64)?).gradient(|x| {
///     custom_derivative_at(x)
///         .vjp(
///             |x| Ok(x.sin()?),
///             |x| Ok((x.sin()?, x.cos()?)),
///             |cosine, cotangent| {
///                 let gradient = cosine * cotangent;
///                 Ok(gradient.clone() + gradient)
///             },
///         )
///         .unwrap()
/// })?;
/// assert_eq!(gradient, Array::scalar(2.0 * 0.5f64.cos())?);
/// # Ok(())
/// # }
/// ```
///
/// # Parameters
///
///   - `input`: [`Parameterized`] value at which the custom derivative rule is staged.
#[inline]
pub fn custom_derivative_at<Input>(input: Input) -> CustomDerivativeBuilder<Input> {
    CustomDerivativeBuilder { input, non_differentiated_count: 0 }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::fmt::Formatter;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayReferenceTransform,
        ArraySliceAxis, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        CotangentAccumulator, CotangentDestination, CotangentDestinationKind, CotangentSeed, DifferentiationError,
        ResidualZeroProvider, TransposableOperation, TranspositionContext, TranspositionDriver, differentiate_at,
    };
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::operations::arithmetic::{AddOperation, MulOperation};
    use crate::operations::manipulation::padding::PadOperation;
    use crate::operations::manipulation::slicing::SliceOperation;
    use crate::operations::references::{ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceWrite};
    use crate::operations::trigonometric::{Cos, Sin};
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{
        Effects, MaybeZero, Operation, OperationProvider, Program, ProgramBuilder, ReferenceAccessDescriptor,
        ReferenceAccessOperation, RegionInterface, TypeError, TypeIdentityRenaming, Typed,
    };
    use crate::specialization::SpecializationCache;
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    /// Concrete callback universe for the retained-rule representation experiment.
    type RetainedRuleTracer = Tracer<TracingContext<ArrayIrValue<Array>, RetainedRuleOperation>>;

    /// An array/composite rule specialized against the current static boundary and destination kind.
    type RetainedRuleCallback = dyn Fn(
            &mut TranspositionContext<ArrayIrValue<Array>, RetainedRuleOperation>,
            &[PartialValue<RetainedRuleTracer>],
            &[MaybeZero<RetainedRuleTracer>],
            &[CotangentAccumulator],
        ) -> Result<(), DifferentiationError>
        + Send
        + Sync;

    /// Base-only programs cannot retain their originating callback or form a cache ownership cycle.
    type RetainedRuleProgram =
        Program<ArrayIrValue<Array>, ArrayIrOperation<Array>, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>;

    /// Immutable definition identity and callback-owned specialization storage for this fixed-universe experiment.
    struct RetainedRuleDefinition {
        /// Unique immutable test declaration identity, included in operation rendering.
        label: &'static str,

        /// User definition, invoked only while transposing a live rule on a specialization miss.
        callback: Arc<RetainedRuleCallback>,

        /// Current input type, materialized seed type, and static destination kind; never runtime buffer identity.
        cache: SpecializationCache<(ArrayIrType, ArrayIrType, CotangentDestinationKind), Arc<RetainedRuleProgram>>,
    }

    impl Debug for RetainedRuleDefinition {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            formatter.debug_struct("RetainedRuleDefinition").field("label", &self.label).finish_non_exhaustive()
        }
    }

    /// Region-free scalar-slice carrier plus the ordinary operations emitted by its retained backward callback.
    /// This deliberately supports direct transposition, not a new general-purpose operation family.
    #[derive(Clone, Debug)]
    enum RetainedRuleOperation {
        Base(ArrayIrOperation<Array>),
        Slice { definition: Arc<RetainedRuleDefinition>, cached: bool },
    }

    impl Operation for RetainedRuleOperation {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            match self {
                Self::Base(operation) => operation.name(),
                Self::Slice { .. } => "retained_slice",
            }
        }

        fn infer_output_types(
            &self,
            inputs: &[Self::Type],
            regions: &[RegionInterface<Self::Type>],
        ) -> Result<Vec<Self::Type>, TypeError> {
            match self {
                Self::Base(operation) => operation.infer_output_types(inputs, regions),
                Self::Slice { .. } => {
                    ArrayIrOperation::<Array>::from(ArrayOperation::Slice(SliceOperation::new(vec![1], vec![2])))
                        .infer_output_types(inputs, regions)
                }
            }
        }

        fn effects(&self) -> Cow<'_, Effects> {
            match self {
                Self::Base(operation) => operation.effects(),
                Self::Slice { .. } => Cow::Borrowed(Effects::empty()),
            }
        }

        fn render(&self, formatter: &mut Formatter<'_>, indentation: usize) -> std::fmt::Result {
            match self {
                Self::Base(operation) => operation.render(formatter, indentation),
                Self::Slice { definition, cached } => {
                    write!(formatter, "retained_slice [definition={}, cached={cached}]", definition.label)
                }
            }
        }
    }

    impl<Request> OperationProvider<ArrayIrType, Request> for RetainedRuleOperation
    where
        ArrayIrOperation<Array>: OperationProvider<ArrayIrType, Request, Operation = ArrayIrOperation<Array>>,
    {
        type Operation = Self;

        fn provide(request: Request, input_types: &[&ArrayIrType]) -> Result<Self, ProgramError> {
            ArrayIrOperation::<Array>::provide(request, input_types).map(Self::Base)
        }
    }

    impl From<AddOperation<ArrayIrType>> for RetainedRuleOperation {
        fn from(operation: AddOperation<ArrayIrType>) -> Self {
            Self::Base(operation.into())
        }
    }

    // Only static array shapes are used where structural cotangent zeros must be materialized.
    impl ResidualZeroProvider<ArrayIrType> for RetainedRuleOperation {}

    impl ReferenceAccessOperation for RetainedRuleOperation {
        type Transform = ArrayReferenceTransform;

        fn base_input_count(&self) -> usize {
            match self {
                Self::Base(operation) => operation.base_input_count(),
                Self::Slice { .. } => 1,
            }
        }

        fn reference_access_descriptor(
            &self,
            input_index: usize,
        ) -> Option<ReferenceAccessDescriptor<'_, Self::Transform>> {
            match self {
                Self::Base(operation) => operation.reference_access_descriptor(input_index),
                Self::Slice { .. } => None,
            }
        }

        fn with_reference_access_transforms(
            &self,
            input_index: usize,
            transforms: Vec<Self::Transform>,
        ) -> Result<Self, ProgramError> {
            match self {
                Self::Base(operation) => {
                    operation.with_reference_access_transforms(input_index, transforms).map(Self::Base)
                }
                Self::Slice { .. } => Err(ProgramError::UnsupportedOperation {
                    message: "retained slice has no reference input".to_string(),
                }),
            }
        }
    }

    impl InterpretableOperation<EagerContext<ArrayIrValue<Array>, Self>> for RetainedRuleOperation {
        fn interpret<D: InterpretationDriver<EagerContext<ArrayIrValue<Array>, Self>>>(
            &self,
            _context: &EagerContext<ArrayIrValue<Array>, Self>,
            _driver: &D,
            inputs: &[ArrayIrValue<Array>],
        ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
            let operation = match self {
                Self::Base(operation) => operation.clone(),
                Self::Slice { .. } => {
                    ArrayIrOperation::from(ArrayOperation::Slice(SliceOperation::new(vec![1], vec![2])))
                }
            };
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new().bind(operation, Vec::new(), inputs)
        }
    }

    impl TransposableOperation<ArrayIrValue<Array>, Self> for RetainedRuleOperation {
        fn transpose<D: TranspositionDriver<ArrayIrValue<Array>, Self>>(
            &self,
            context: &mut TranspositionContext<ArrayIrValue<Array>, Self>,
            _driver: &D,
            inputs: &[PartialValue<RetainedRuleTracer>],
            outputs: &[MaybeZero<RetainedRuleTracer>],
            accumulators: &[CotangentAccumulator],
        ) -> Result<(), DifferentiationError> {
            let Self::Slice { definition, cached } = self else {
                return Err(ProgramError::UnsupportedOperation {
                    message: "experiment only transposes retained slice carriers".to_string(),
                }
                .into());
            };
            if !cached {
                return (definition.callback)(context, inputs, outputs, accumulators);
            }
            let MaybeZero::Value(seed) = &outputs[0] else {
                return Ok(());
            };
            let reference = accumulators[0].reference(context)?;
            let kind = if !accumulators[0].is_needed() {
                CotangentDestinationKind::Ignore
            } else if reference.is_some() {
                CotangentDestinationKind::Reference
            } else {
                CotangentDestinationKind::Return
            };
            let input_type = inputs[0].r#type().into_owned();
            let key = (input_type.clone(), seed.r#type().into_owned(), kind);
            let artifact = definition
                .cache
                .get_or_try_insert_with(key, || {
                    let mut builder = ProgramBuilder::<ArrayIrValue<Array>, Self>::new();
                    let input = builder.add_input(input_type);
                    let output = builder.add_instruction(
                        Self::Slice { definition: definition.clone(), cached: false },
                        Vec::new(),
                        vec![input],
                        None,
                    )?[0];
                    let source = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                        vec![output],
                        vec![Placeholder],
                        vec![Placeholder],
                    )?;
                    let program = source.transpose_with_respect_to(&[0], &[kind])?;
                    let program = program.map_operations(|operation| match operation {
                        Self::Base(operation) => Ok(operation.clone()),
                        Self::Slice { .. } => {
                            Err(ProgramError::MalformedProgram("cached rule retained its callback".to_string()))
                        }
                    })?;
                    Ok::<_, DifferentiationError>(Arc::new(program))
                })
                .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
            let mut arguments = vec![seed.clone()];
            arguments.extend(reference);
            let program = artifact.map_operations(|operation| Ok(Self::Base(operation.clone())))?;
            let values = program.interpret_in_context(&**context, arguments)?;
            if kind == CotangentDestinationKind::Return {
                accumulators[0].accumulate(context, MaybeZero::Value(values[0].clone()))?;
            }
            Ok(())
        }
    }

    /// Stages one call while retaining its callback definition without invoking it.
    fn retained_slice_program(
        definition: Arc<RetainedRuleDefinition>,
        input_type: ArrayType,
    ) -> Program<ArrayIrValue<Array>, RetainedRuleOperation, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, RetainedRuleOperation>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(RetainedRuleOperation::Slice { definition, cached: true }, Vec::new(), vec![input], None)
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_custom_derivative_retained_accumulator_callback() {
        let trace_count = Arc::new(AtomicUsize::new(0));
        let definition = Arc::new(RetainedRuleDefinition {
            label: "slice_at_one",
            callback: Arc::new({
                let trace_count = trace_count.clone();
                move |context, inputs, outputs, accumulators| {
                    trace_count.fetch_add(1, Ordering::SeqCst);
                    if !accumulators[0].is_needed() {
                        return Ok(());
                    }
                    let MaybeZero::Value(seed) = &outputs[0] else {
                        return Ok(());
                    };
                    if let Some(reference) = accumulators[0].reference(context)? {
                        let operation =
                            ReferenceAddUpdateOperation::new().with_transforms(vec![ArrayReferenceTransform::Slice {
                                axes: vec![ArraySliceAxis::new(1, 1, 1)],
                            }]);
                        context.bind(
                            RetainedRuleOperation::Base(operation.into()),
                            Vec::new(),
                            &[reference, seed.clone()],
                        )?;
                    } else {
                        let ArrayIrType::Array(input_type) = inputs[0].r#type().into_owned() else { unreachable!() };
                        let extent = input_type.shape().dimensions()[0].value().unwrap();
                        let zero = context.lift(ArrayIrValue::Array(Array::scalar(0.0f64)?))?;
                        let operation = ArrayIrOperation::from(ArrayOperation::Pad(PadOperation::new(
                            vec![1],
                            vec![extent as i64 - 2],
                            vec![0],
                        )?));
                        let output = context
                            .bind(RetainedRuleOperation::Base(operation), Vec::new(), &[seed.clone(), zero])?
                            .remove(0);
                        accumulators[0].accumulate(context, MaybeZero::Value(output))?;
                    }
                    Ok(())
                }
            }),
            cache: SpecializationCache::new(8),
        });
        let retained = Arc::downgrade(&definition);
        let program = retained_slice_program(definition.clone(), ArrayType::new_static(DataType::F64, [3]));
        drop(definition);
        assert_eq!(
            program.interpret(vec![Array::vector(vec![2.0f64, 4.0, 6.0]).unwrap().into()]),
            Ok(vec![Array::vector(vec![4.0f64]).unwrap().into()]),
        );
        assert_eq!(trace_count.load(Ordering::SeqCst), 0);
        let returned = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap();
        let buffered = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        let ignored = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Ignore]).unwrap();
        assert_eq!(
            buffered.to_string(),
            indoc! {"
                lambda %0:f64[1], %1:ref<f64[3]> .
                let () = reference_add_update [transforms=[slice(axes=[1:2])]] %1 %0
                in ()
            "}
            .trim_end(),
        );
        assert_eq!(
            ignored.to_string(),
            indoc! {"
                lambda %0:f64[1] .
                in ()
            "}
            .trim_end(),
        );
        let seed = ArrayIrValue::Array(Array::vector(vec![3.0f64]).unwrap());
        assert_eq!(
            returned.interpret(vec![seed.clone()]),
            Ok(vec![Array::vector(vec![0.0f64, 3.0, 0.0]).unwrap().into()]),
        );
        let first = ArrayReference::new(Array::vector(vec![10.0f64, 20.0, 30.0]).unwrap());
        let second = ArrayReference::new(Array::vector(vec![40.0f64, 50.0, 60.0]).unwrap());
        assert_eq!(buffered.interpret(vec![seed.clone(), first.clone().into()]), Ok(vec![]));
        assert_eq!(buffered.interpret(vec![seed.clone(), first.clone().into()]), Ok(vec![]));
        assert_eq!(first.read(), Ok(Array::vector(vec![10.0f64, 26.0, 30.0]).unwrap()));
        assert_eq!(buffered.interpret(vec![seed.clone(), second.clone().into()]), Ok(vec![]));
        assert_eq!(second.read(), Ok(Array::vector(vec![40.0f64, 53.0, 60.0]).unwrap()));
        assert_eq!(ignored.interpret(vec![seed.clone()]), Ok(vec![]));
        let zero_seed = ArrayIrValue::Array(Array::vector(vec![0.0f64]).unwrap());
        assert_eq!(ignored.interpret(vec![zero_seed.clone()]), Ok(vec![]));
        assert_eq!(buffered.interpret(vec![zero_seed, first.clone().into()]), Ok(vec![]));
        assert_eq!(first.read(), Ok(Array::vector(vec![10.0f64, 26.0, 30.0]).unwrap()));
        assert_eq!(trace_count.load(Ordering::SeqCst), 2);
        let repeated = program.clone().transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(repeated.to_string(), buffered.to_string());
        assert_eq!(trace_count.load(Ordering::SeqCst), 2);

        // A second current static boundary rebuilds the returned gradient from its current extent, rather than
        // reusing the first trace's three-element padding contract.
        let wider = retained_slice_program(retained.upgrade().unwrap(), ArrayType::new_static(DataType::F64, [4]));
        let wider_pullback = wider.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Return]).unwrap();
        assert_eq!(
            wider_pullback.interpret(vec![seed.clone()]),
            Ok(vec![Array::vector(vec![0.0f64, 3.0, 0.0, 0.0]).unwrap().into()]),
        );
        assert_eq!(trace_count.load(Ordering::SeqCst), 3);

        // The callback receives the current renamed boundary, but existing reference slicing requires a static
        // referent type. Both failures must report their current identity and must not retain failed artifacts.
        let original = DimensionVariable::new("original", DimensionBounds::new(2, Some(6)).unwrap());
        let relocated = DimensionVariable::new("relocated", DimensionBounds::new(2, Some(6)).unwrap());
        let original_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(original.clone())]));
        let relocated_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(relocated.clone())]));
        let symbolic = retained_slice_program(retained.upgrade().unwrap(), original_type.clone());
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(original, relocated).unwrap();
        let renamed = symbolic.clone().rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.input_types(), vec![ArrayIrType::Array(relocated_type.clone())]);
        assert!(matches!(
            symbolic.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "reference slicing requires a static referent type but got `f64[original]`",
        ));
        assert!(matches!(
            renamed.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "reference slicing requires a static referent type but got `f64[relocated]`",
        ));
        assert_eq!(trace_count.load(Ordering::SeqCst), 5);
        assert_eq!(
            retained.upgrade().unwrap().cache.keys(),
            vec![
                (
                    ArrayIrType::Array(ArrayType::new_static(DataType::F64, [4])),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F64, [1])),
                    CotangentDestinationKind::Return,
                ),
                (
                    ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3])),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F64, [1])),
                    CotangentDestinationKind::Reference,
                ),
                (
                    ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3])),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F64, [1])),
                    CotangentDestinationKind::Return,
                ),
            ],
        );
        drop((program, wider, symbolic, renamed));
        assert!(retained.upgrade().is_none());
    }

    #[test]
    fn test_custom_derivative_retained_callback_after_un_projection() {
        /// Array-member payload retaining a callback declared in the canonical composite tracing universe.
        #[derive(Clone, Debug)]
        struct RetainedMemberSlice(Arc<RetainedRuleDefinition>);

        impl Operation for RetainedMemberSlice {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                "retained_member_slice"
            }

            fn infer_output_types(
                &self,
                inputs: &[ArrayType],
                regions: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                SliceOperation::new(vec![1], vec![2]).infer_output_types(inputs, regions)
            }
        }

        impl From<RetainedMemberSlice> for RetainedRuleOperation {
            fn from(operation: RetainedMemberSlice) -> Self {
                Self::Slice { definition: operation.0, cached: true }
            }
        }

        let trace_count = Arc::new(AtomicUsize::new(0));
        let definition = Arc::new(RetainedRuleDefinition {
            label: "converted_slice_at_one",
            callback: Arc::new({
                let trace_count = trace_count.clone();
                move |context, inputs, outputs, accumulators| {
                    trace_count.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(
                        inputs[0].r#type().as_ref(),
                        &ArrayIrType::Array(ArrayType::new_static(DataType::F64, [4])),
                    );
                    let MaybeZero::Value(seed) = &outputs[0] else { unreachable!() };
                    let reference = accumulators[0].reference(context)?.unwrap();
                    let operation =
                        ReferenceAddUpdateOperation::new().with_transforms(vec![ArrayReferenceTransform::Slice {
                            axes: vec![ArraySliceAxis::new(1, 1, 1)],
                        }]);
                    context.bind(
                        RetainedRuleOperation::Base(operation.into()),
                        Vec::new(),
                        &[reference, seed.clone()],
                    )?;
                    Ok(())
                }
            }),
            cache: SpecializationCache::new(8),
        });
        let retained = Arc::downgrade(&definition);
        let mut builder = ProgramBuilder::<Array, RetainedMemberSlice>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [4]));
        let output = builder
            .add_instruction(RetainedMemberSlice(definition.clone()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let member =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(trace_count.load(Ordering::SeqCst), 0);

        // This canonical conversion changes the stored value and type families as well as the operation payload.
        // It carries an already composite-typed callback; it does not make a Rust closure domain-polymorphic.
        let converted = member.into_unprojected::<ArrayIrValue<Array>, RetainedRuleOperation>().unwrap();
        let RetainedRuleOperation::Slice { definition: converted_definition, .. } =
            converted.instructions()[0].operation()
        else {
            unreachable!()
        };
        assert!(Arc::ptr_eq(&definition, converted_definition));
        assert_eq!(trace_count.load(Ordering::SeqCst), 0);
        assert_eq!(
            converted.interpret(vec![Array::vector(vec![2.0f64, 4.0, 6.0, 8.0]).unwrap().into()]),
            Ok(vec![Array::vector(vec![4.0f64]).unwrap().into()]),
        );
        assert_eq!(trace_count.load(Ordering::SeqCst), 0);
        drop(definition);

        let buffered = converted.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(trace_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            buffered.to_string(),
            indoc! {"
                lambda %0:f64[1], %1:ref<f64[4]> .
                let () = reference_add_update [transforms=[slice(axes=[1:2])]] %1 %0
                in ()
            "}
            .trim_end(),
        );
        let seed = ArrayIrValue::Array(Array::vector(vec![3.0f64]).unwrap());
        let buffer = ArrayReference::new(Array::vector(vec![10.0f64, 20.0, 30.0, 40.0]).unwrap());
        assert_eq!(buffered.interpret(vec![seed, buffer.clone().into()]), Ok(vec![]));
        assert_eq!(buffer.read(), Ok(Array::vector(vec![10.0f64, 23.0, 30.0, 40.0]).unwrap()));
        let repeated =
            converted.clone().transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(repeated.to_string(), buffered.to_string());
        assert_eq!(trace_count.load(Ordering::SeqCst), 1);
        drop(converted);
        assert!(retained.upgrade().is_none());
    }

    #[test]
    fn test_custom_derivative_builder_with_non_differentiated_count() {
        // The leading counter is plumbing for a custom JVP rule: it reaches both closures at its usual position
        // and the rule leaves the tangent placeholder of the counter unused.
        let counter = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let counter_tangent = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        assert_eq!(
            differentiate_at((
                ArrayIrValue::Reference(counter.clone()),
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
            ))
            .jvp(
                (
                    ArrayIrValue::Reference(counter_tangent.clone()),
                    ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
                ),
                |input| {
                    custom_derivative_at(input).with_non_differentiated_count(1).jvp(
                        |(counter, x)| {
                            counter.add_update(&x)?;
                            Ok(x)
                        },
                        |(counter, x), (_, tangent)| {
                            counter.add_update(&x)?;
                            Ok((x, tangent))
                        },
                    )
                },
            ),
            Ok((
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()),
            )),
        );
        assert_eq!(counter.read(), Ok(Array::scalar(2.0f32).unwrap()));
        assert_eq!(counter_tangent.read(), Ok(Array::scalar(0.0f32).unwrap()));

        // The leading stash is plumbing for a custom VJP rule: the forward rule forwards it as a residual, and the
        // backward rule writes the incoming cotangent into it while its own cotangent leaf is ignored.
        let stash = ArrayReference::new(Array::scalar(0.0f32).unwrap());
        let (value, pullback) = differentiate_at((
            ArrayIrValue::Reference(stash.clone()),
            ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
        ))
        .vjp(|input| {
            custom_derivative_at(input).with_non_differentiated_count(1).vjp(
                |(_, x)| Ok(x),
                |(stash, x)| Ok((x, stash)),
                |stash, cotangent| {
                    stash.write(&cotangent)?;
                    Ok((stash, cotangent))
                },
            )
        })
        .unwrap();
        assert_eq!(value, ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Ok((None, Some(ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())))),
        );
        assert_eq!(stash.read(), Ok(Array::scalar(3.0f32).unwrap()));
    }

    #[test]
    fn test_custom_derivative_builder_jvp() {
        // The deliberately wrong rule `jvp(x, ẋ) = (sin(x), 2 * cos(x) * ẋ)` doubles the true derivative, which proves
        // that it governs both forward- and reverse-mode differentiation.
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| {
                custom_derivative_at(x).jvp(
                    |x| x.sin(),
                    |x, tangent| {
                        let tangent = x.cos()? * tangent;
                        Ok((x.sin()?, tangent.clone() + tangent))
                    },
                )
            }),
            Ok((Array::scalar(2.0f64.sin()).unwrap(), Array::scalar(2.0 * 2.0f64.cos()).unwrap())),
        );
        assert_eq!(
            differentiate_at(Array::scalar(3.0).unwrap()).value_and_gradient(|x| {
                custom_derivative_at(x)
                    .jvp(
                        |x| x.sin(),
                        |x, tangent| {
                            let tangent = x.cos()? * tangent;
                            Ok((x.sin()?, tangent.clone() + tangent))
                        },
                    )
                    .unwrap()
            }),
            Ok((Array::scalar(3.0f64.sin()).unwrap(), Array::scalar(2.0 * 3.0f64.cos()).unwrap())),
        );
    }

    #[test]
    fn test_custom_derivative_builder_vjp() {
        // Tuple inputs and destructured tuple residuals infer their types from the input and from the forward rule.
        // The deliberately wrong rule doubles the true gradients `(y, x)`.
        assert_eq!(
            differentiate_at((Array::scalar(2.0).unwrap(), Array::scalar(5.0).unwrap())).value_and_gradient(
                |(x, y)| {
                    custom_derivative_at((x, y))
                        .vjp(
                            |(x, y)| Ok(x * y),
                            |(x, y)| Ok((x.clone() * y.clone(), (x, y))),
                            |(x, y), cotangent| {
                                let x_cotangent = y * cotangent.clone();
                                let y_cotangent = x * cotangent;
                                Ok((x_cotangent.clone() + x_cotangent, y_cotangent.clone() + y_cotangent))
                            },
                        )
                        .unwrap()
                },
            ),
            Ok((Array::scalar(10.0).unwrap(), (Array::scalar(10.0).unwrap(), Array::scalar(4.0).unwrap()))),
        );
    }

    #[test]
    fn test_custom_derivative_builder_vjp_effectful_backward_destinations() {
        // This is one custom backward closure with an observable reference effect and an intentionally nonlinear
        // seed formula. Destination specialization must preserve the closure's execution and additive semantics.
        let stash = ArrayReference::new(Array::scalar(-1.0f32).unwrap());
        let (_, pullback) = differentiate_at((
            ArrayIrValue::Reference(stash.clone()),
            ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
        ))
        .vjp(|input| {
            custom_derivative_at(input).with_non_differentiated_count(1).vjp(
                |(_, value)| Ok(value),
                |(stash, value)| Ok((value, stash)),
                |stash, seed| {
                    stash.write(&seed)?;
                    let contribution = seed
                        .context()
                        .bind(
                            ArrayOperation::<Array>::Mul(MulOperation::new()),
                            Vec::new(),
                            &[seed.clone(), seed.clone()],
                        )?
                        .remove(0);
                    Ok((stash, contribution))
                },
            )
        })
        .unwrap();
        let first = ArrayReference::new(Array::scalar(10.0f32).unwrap());
        let second = ArrayReference::new(Array::scalar(20.0f32).unwrap());
        let seed = ArrayIrValue::Array(Array::scalar(3.0f32).unwrap());
        let destinations = [CotangentDestinationKind::Ignore, CotangentDestinationKind::Reference];
        let retained = pullback.transposed_program(&destinations).unwrap();
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(seed.clone()),
                (CotangentDestination::Ignore, CotangentDestination::Reference(ArrayIrValue::Reference(first.clone()))),
            ),
            Ok((None, None)),
        );
        assert_eq!(first.read(), Ok(Array::scalar(19.0f32).unwrap()));
        assert_eq!(stash.read(), Ok(Array::scalar(3.0f32).unwrap()));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(seed.clone()),
                (
                    CotangentDestination::Ignore,
                    CotangentDestination::Reference(ArrayIrValue::Reference(second.clone()))
                ),
            ),
            Ok((None, None)),
        );
        assert_eq!(second.read(), Ok(Array::scalar(29.0f32).unwrap()));
        assert_eq!(first.read(), Ok(Array::scalar(19.0f32).unwrap()));
        assert!(Arc::ptr_eq(&retained, &pullback.transposed_program(&destinations).unwrap()));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(seed),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Ok((None, Some(ArrayIrValue::Array(Array::scalar(9.0f32).unwrap())))),
        );

        // A numerical zero is still a live seed. Dropping every returned gradient must not erase the stash write.
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::scalar(0.0f32).unwrap())),
                (CotangentDestination::Ignore, CotangentDestination::Ignore),
            ),
            Ok((None, None)),
        );
        assert_eq!(stash.read(), Ok(Array::scalar(0.0f32).unwrap()));
    }
}
