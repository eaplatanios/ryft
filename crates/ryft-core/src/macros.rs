/// Checks that `values` contains exactly `expected` entries and, if not, returns an error of the specified type.
#[macro_export]
macro_rules! check_count {
    ("input", $values:expr, $expected:expr, ProgramError $(,)?) => {{
        let values = &$values;
        let expected = $expected;
        if values.len() != expected {
            return Err($crate::ProgramError::InvalidInputCount { expected, actual: values.len() }.into());
        }
    }};
    ("output", $values:expr, $expected:expr, ProgramError $(,)?) => {{
        let values = &$values;
        let expected = $expected;
        if values.len() != expected {
            return Err($crate::ProgramError::InvalidOutputCount { expected, actual: values.len() }.into());
        }
    }};
    ($descriptor:expr, $values:expr, $expected:expr, TypeError $(,)?) => {{
        let values = &$values;
        let expected = $expected;
        if values.len() != expected {
            let count = values.len();
            let descriptor = $descriptor;
            let noun = if expected == 1 { descriptor.to_string() } else { format!("{descriptor}s") };
            return Err($crate::TypeError { message: format!("expected {expected} {noun} but got {count}") });
        }
    }};
}

/// Checks that two flat type signatures are identical and, if not, returns a [`TypeError`](crate::TypeError)
/// whose message names the mismatching descriptor.
///
/// # Parameters
///
///   - `descriptor`: Expression evaluating to a string that names the validated signature in the error message.
///   - `$left`: Expression evaluating to a slice of [`Type`](crate::Type)s.
///   - `$right`: Expression evaluating to a slice of [`Type`](crate::Type)s.
#[macro_export]
macro_rules! check_types {
    ($descriptor:expr, $left:expr, $right:expr $(,)?) => {{
        let left = &$left[..];
        let right = &$right[..];
        if left != right {
            return Err($crate::TypeError {
                message: format!(
                    "{} type signature mismatch: expected [{}] but got [{}]",
                    $descriptor,
                    left.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                    right.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                ),
            });
        }
    }};
}

/// Checks that a concrete [`DeviceMesh`](crate::DeviceMesh) and a [`Sharding`](crate::Sharding) refer to the same
/// [`LogicalMesh`](crate::LogicalMesh). If the logical meshes differ, the macro returns a
/// [`ShardingError::MeshMismatch`](crate::ShardingError::MeshMismatch) converted into the enclosing function's error
/// type using [`Into::into`]. Use this macro in functions that return a [`Result`] whose error type can be constructed
/// from [`ShardingError`](crate::ShardingError).
///
/// # Parameters
///
///   - `$mesh`: Expression evaluating to a [`DeviceMesh`](crate::DeviceMesh) or a reference to one.
///   - `$sharding`: Expression evaluating to a [`Sharding`](crate::Sharding) or a reference to one.
#[macro_export]
macro_rules! check_sharding {
    ($mesh:expr, $sharding:expr $(,)?) => {{
        let mesh = &$mesh;
        let sharding = &$sharding;
        if mesh.logical_mesh() != sharding.mesh() {
            return Err($crate::ShardingError::MeshMismatch {
                expected: mesh.logical_mesh().clone(),
                actual: sharding.mesh().clone(),
            }
            .into());
        }
    }};
}

/// Checks that [`ProgramBuilder`](crate::ProgramBuilder) handles refer to the same builder and returns a
/// [`ProgramError::MismatchedProgramBuilders`](crate::ProgramError::MismatchedProgramBuilders) if they do not.
///
/// # Parameters
///
///   - `$reference`: Expression evaluating to the reference [`ProgramBuilder`](crate::ProgramBuilder) handle.
///   - `$other`: Expression evaluating to a single [`ProgramBuilder`](crate::ProgramBuilder) handle, or bracketed
///     syntax `[$others]` where `$others` evaluates to an iterable of [`ProgramBuilder`](crate::ProgramBuilder)
///     handles.
#[macro_export]
macro_rules! check_builders {
    ($reference:expr, [$others:expr] $(,)?) => {{
        let reference = $reference;
        let mut result = ::std::result::Result::Ok(());
        for other in $others {
            if !::std::rc::Rc::ptr_eq(reference, other) {
                result = ::std::result::Result::Err($crate::ProgramError::MismatchedProgramBuilders);
                break;
            }
        }
        result
    }};
    ($reference:expr, $other:expr $(,)?) => {{
        let reference = $reference;
        let other = $other;
        if ::std::rc::Rc::ptr_eq(reference, other) {
            ::std::result::Result::Ok(())
        } else {
            ::std::result::Result::Err($crate::ProgramError::MismatchedProgramBuilders)
        }
    }};
}

// TODO(eaplatanios): Review this macro.
/// Defines the structural implementations shared by elementwise operations. The generated base
/// includes the unit operation struct, its [`Display`](std::fmt::Display), [`Operation`](crate::Operation),
/// [`ElementwiseOperation`](crate::ElementwiseOperation), [`InterpretableOperation`](crate::InterpretableOperation),
/// and [`PartiallyEvaluatableOperation`](crate::PartiallyEvaluatableOperation) implementations.
///
/// # Parameters
///
///   - `@unary` / `@binary`: Selects the operation arity.
///   - `$(#[$operation_documentation])*`: Documentation attributes attached to the generated operation struct.
///   - `$operation`: Identifier of the generated unit-struct operation (e.g., `SinOperation`).
///   - `$name`: Identifier of an existing operation-name constant (e.g., `SIN_OPERATION_NAME`).
///   - `$capability`: Identifier of the value-level capability trait bound by the generated
///     [`InterpretableOperation`](crate::InterpretableOperation) implementation (e.g., `Sin`).
///   - `$method`: Identifier of the capability trait method used for interpretation (e.g., `sin`).
///   - `validate = $validator`: Optional hook that validates scalar [`DataType`](crate::DataType) inputs before type
///     inference and array element types before array broadcasting.
///   - `validate_array = $array_validator`: Optional hook that validates complete
///     [`ArrayType`](crate::ArrayType) inputs before array broadcasting, for rules that depend on metadata beyond the
///     element type.
#[macro_export]
macro_rules! define_elementwise_operation {
    (
        @unary
        $(#[$operation_documentation:meta])*
        $operation:ident, $name:ident,
        $capability:ident, $method:ident
        $(, validate = $validator:path)?
        $(, validate_array = $array_validator:path)? $(,)?
    ) => {
        $(#[$operation_documentation])*
        #[derive(Clone, Debug, Default)]
        pub struct $operation;

        impl ::std::fmt::Display for $operation {
            #[inline]
            fn fmt(&self, formatter: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                formatter.write_str($name)
            }
        }

        impl $crate::Operation<$crate::DataType> for $operation {
            #[inline]
            fn name(&self) -> &'static str {
                $name
            }

            #[inline]
            fn infer_output_types(
                &self,
                input_types: &[$crate::DataType],
                _region_interfaces: &[$crate::RegionInterface<$crate::DataType>],
            ) -> Result<Vec<$crate::DataType>, $crate::TypeError> {
                $crate::check_count!("input", input_types, 1, TypeError);
                $($validator(input_types, $name)?;)?
                Ok(vec![input_types[0]])
            }
        }

        impl $crate::Operation<$crate::ArrayType> for $operation {
            #[inline]
            fn name(&self) -> &'static str {
                $name
            }

            #[inline]
            fn infer_output_types(
                &self,
                input_types: &[$crate::ArrayType],
                _region_interfaces: &[$crate::RegionInterface<$crate::ArrayType>],
            ) -> Result<Vec<$crate::ArrayType>, $crate::TypeError> {
                $crate::check_count!("input", input_types, 1, TypeError);
                $($validator(&[input_types[0].data_type()], $name)?;)?
                $($array_validator(input_types, $name)?;)?
                $crate::ElementwiseOperation::infer_output_types(self, input_types)
            }
        }

        impl $crate::ElementwiseOperation for $operation {
            #[inline]
            fn input_count(&self) -> usize {
                1
            }
        }

        impl<C: $crate::Domain<Value: $capability>> $crate::InterpretableOperation<C> for $operation
        where
            Self: $crate::Operation<C::Type>,
        {
            #[inline]
            fn interpret<D: $crate::InterpretationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                inputs: &[C::Value],
            ) -> Result<Vec<C::Value>, $crate::ProgramError> {
                $crate::check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].$method()?])
            }
        }

        impl<C: $crate::Context> $crate::PartiallyEvaluatableOperation<C> for $operation where
            C::Operation: ::std::convert::From<$operation>
        {
        }
    };
    (
        @binary
        $(#[$operation_documentation:meta])*
        $operation:ident, $name:ident,
        $capability:ident, $method:ident
        $(, validate = $validator:path)?
        $(, validate_array = $array_validator:path)? $(,)?
    ) => {
        $(#[$operation_documentation])*
        #[derive(Clone, Debug, Default)]
        pub struct $operation;

        impl ::std::fmt::Display for $operation {
            #[inline]
            fn fmt(&self, formatter: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                formatter.write_str($name)
            }
        }

        impl $crate::Operation<$crate::DataType> for $operation {
            #[inline]
            fn name(&self) -> &'static str {
                $name
            }

            #[inline]
            fn infer_output_types(
                &self,
                input_types: &[$crate::DataType],
                _region_interfaces: &[$crate::RegionInterface<$crate::DataType>],
            ) -> Result<Vec<$crate::DataType>, $crate::TypeError> {
                $crate::check_count!("input", input_types, 2, TypeError);
                $($validator(input_types, $name)?;)?
                $crate::Broadcastable::broadcast(&input_types[0], &input_types[1])
                    .map(|output| vec![output])
                    .map_err(|_| $crate::TypeError {
                        message: format!("'{}' input types are not broadcast-compatible", $name),
                    })
            }
        }

        impl $crate::Operation<$crate::ArrayType> for $operation {
            #[inline]
            fn name(&self) -> &'static str {
                $name
            }

            #[inline]
            fn infer_output_types(
                &self,
                input_types: &[$crate::ArrayType],
                _region_interfaces: &[$crate::RegionInterface<$crate::ArrayType>],
            ) -> Result<Vec<$crate::ArrayType>, $crate::TypeError> {
                $crate::check_count!("input", input_types, 2, TypeError);
                $($validator(&[input_types[0].data_type(), input_types[1].data_type()], $name)?;)?
                $($array_validator(input_types, $name)?;)?
                $crate::ElementwiseOperation::infer_output_types(self, input_types)
            }
        }

        impl $crate::ElementwiseOperation for $operation {
            #[inline]
            fn input_count(&self) -> usize {
                2
            }
        }

        impl<C: $crate::Domain<Value: $capability>> $crate::InterpretableOperation<C> for $operation
        where
            Self: $crate::Operation<C::Type>,
        {
            #[inline]
            fn interpret<D: $crate::InterpretationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                inputs: &[C::Value],
            ) -> Result<Vec<C::Value>, $crate::ProgramError> {
                $crate::check_count!("input", inputs, 2, ProgramError);
                Ok(vec![inputs[0].$method(&inputs[1])?])
            }
        }

        impl<C: $crate::Context> $crate::PartiallyEvaluatableOperation<C> for $operation where
            C::Operation: ::std::convert::From<$operation>
        {
        }
    };
}

// TODO(eaplatanios): Review this macro.
/// Defines the value-level capability trait paired with an elementwise operation and its dispatch-domain blanket
/// implementation. This macro is separate from [`define_elementwise_operation!`] so operation modules can place
/// differentiation and transposition implementations before the capability API while retaining shared boilerplate.
///
/// # Parameters
///
///   - `@unary` / `@binary`: Selects whether the capability consumes only `self` or also a right-hand-side value.
///   - `$(#[$capability_documentation])*`: Documentation attributes attached to the generated capability trait.
///   - `$capability`: Identifier of the generated value-level capability trait.
///   - `$method`: Identifier of the generated capability method.
///   - `$operation`: Unit-struct operation bound through the value's dispatch domain.
#[macro_export]
macro_rules! define_elementwise_capability {
    (
        @unary
        $(#[$capability_documentation:meta])*
        $capability:ident, $method:ident, $operation:ident $(,)?
    ) => {
        $(#[$capability_documentation])*
        pub trait $capability: Sized {
            /// Computes this operation elementwise for this value, returning a [`ProgramError`](crate::ProgramError)
            /// if something goes wrong (e.g., when this operation does not support the value's
            /// [`DataType`](crate::DataType)).
            fn $method(&self) -> Result<Self, $crate::ProgramError>;
        }

        impl<V: $crate::Value<DispatchDomain: $crate::Context<Operation: ::std::convert::From<$operation>>>>
            $capability for V
        {
            #[inline]
            fn $method(&self) -> Result<Self, $crate::ProgramError> {
                Ok($crate::Context::bind(
                    &$crate::Value::dispatch_domain(self),
                    $operation,
                    Vec::new(),
                    ::std::slice::from_ref(self),
                )?
                .remove(0))
            }
        }
    };
    (
        @binary
        $(#[$capability_documentation:meta])*
        $capability:ident, $method:ident, $operation:ident $(,)?
    ) => {
        $(#[$capability_documentation])*
        pub trait $capability: Sized {
            /// Computes this operation elementwise for this value and `right`, returning a
            /// [`ProgramError`](crate::ProgramError) if something goes wrong (e.g., when this operation does not
            /// support the values' [`DataType`](crate::DataType)s).
            fn $method(&self, right: &Self) -> Result<Self, $crate::ProgramError>;
        }

        impl<V: $crate::Value<DispatchDomain: $crate::Context<Operation: ::std::convert::From<$operation>>>>
            $capability for V
        {
            #[inline]
            fn $method(&self, right: &Self) -> Result<Self, $crate::ProgramError> {
                Ok($crate::Context::bind(
                    &$crate::Value::dispatch_domain(self),
                    $operation,
                    Vec::new(),
                    &[self.clone(), right.clone()],
                )?
                .remove(0))
            }
        }
    };
}

// TODO(eaplatanios): Review this macro.
/// Implements the [`DifferentiableOperation`](crate::DifferentiableOperation) rule for an operation whose outputs
/// carry no tangent, such as a Boolean-codomain predicate or an explicit gradient barrier: the primal operation is
/// replayed on the input primals and each output is paired with a structural zero tangent, which stays symbolic and
/// stages nothing. Because such a rule stages no live tangent, the operation can never appear on a linear operand in
/// a valid tangent program, so it typically pairs with
/// [`impl_non_transposable_operation!`](crate::impl_non_transposable_operation).
///
/// # Parameters
///
///   - `$operation`: The operation type for which the implementation is generated.
#[macro_export]
macro_rules! impl_non_differentiable_operation {
    ($operation:ty $(,)?) => {
        impl<C: $crate::Context> $crate::DifferentiableOperation<C> for $operation
        where
            C::Type: $crate::DifferentiableType,
            C::Operation: ::std::convert::From<$operation>,
            $operation: $crate::Operation<C::Type>,
        {
            fn jvp<D: $crate::DifferentiationDriver<C>>(
                &self,
                context: &C,
                _driver: &D,
                inputs: &[$crate::DifferentiationDual<C::Value>],
            ) -> Result<Vec<$crate::DifferentiationDual<C::Value>>, $crate::DifferentiationError> {
                // The outputs carry no tangent: replay the primal operation on the input primals and pair each output
                // with a structural zero tangent, which stays symbolic and stages nothing.
                let primal_inputs = inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
                Ok($crate::Context::bind(context, self.clone(), ::std::vec::Vec::new(), &primal_inputs)?
                    .into_iter()
                    .map($crate::DifferentiationDual::new_with_zero_tangent)
                    .collect())
            }
        }
    };
}

// TODO(eaplatanios): Review this macro.
/// Implements the erroring [`TransposableOperation`](crate::TransposableOperation) rule for an operation that is not
/// a linear map on any operand. A valid tangent program never contains such an operation on a linear operand (its
/// forward-mode rule pairs replayed primals with tangents computed by other operations), so the generated rule
/// reports an [`UnsupportedOperation`](crate::ProgramError::UnsupportedOperation) error. The reason non-transposable
/// operations still implement [`TransposableOperation`](crate::TransposableOperation) at all is that transposition is
/// driven through whole operation families: Ryft used to have a separate linear operation family type, but that
/// resulted in overly complicated backend and operation implementations for very little benefit in practice, and so
/// the two families were unified.
///
/// # Parameters
///
///   - `$operation`: The operation type for which the implementation is generated.
#[macro_export]
macro_rules! impl_non_transposable_operation {
    ($operation:ty $(,)?) => {
        impl<T: $crate::Type, V: $crate::Value<Type = T>, O: $crate::Operation<T>> $crate::TransposableOperation<V, O>
            for $operation
        where
            $operation: $crate::Operation<T>,
        {
            fn transpose<D: $crate::TranspositionDriver<V, O>>(
                &self,
                _context: &mut $crate::TracingContext<V, O>,
                _driver: &D,
                _inputs: &[$crate::PartialValue<$crate::Tracer<$crate::TracingContext<V, O>>>],
                _outputs: &[$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<V, O>>>],
            ) -> Result<
                Vec<$crate::MaybeZero<$crate::Tracer<$crate::TracingContext<V, O>>>>,
                $crate::DifferentiationError,
            > {
                // A fully qualified call is required here because operations typically implement `Operation` for both
                // the `DataType` and the `ArrayType` type universes.
                Err($crate::ProgramError::UnsupportedOperation {
                    message: format!("operation `{}` is not transposable", $crate::Operation::<T>::name(self)),
                }
                .into())
            }
        }
    };
}

/// Implements a foreign `std::ops` operator trait as panicking sugar for the four core transform tracer types (i.e.,
/// [`Tracer`](crate::Tracer), [`PartialTracer`](crate::PartialTracer), [`BatchingTracer`](crate::BatchingTracer), and
/// [`DifferentiationTracer`](crate::DifferentiationTracer)) by binding the operation through each tracer's own context.
/// The operator traits (i.e., `std::ops::Add`, `std::ops::Neg`, `std::ops::BitAnd`, etc.) are foreign and so a single
/// `impl<V: Value>` blanket implementation (i.e., the shape used for the fallible in-crate capability traits) is not
/// allowed due to the orphan rule. This macro stamps out the implementations that a blanket would otherwise cover. Note
/// also that because an operator must return `Self`, the two error modes differ by tracer and cannot be collapsed: a
/// staged [`Tracer`](crate::Tracer) records through its [`unary`](crate::Tracer::unary) and
/// [`binary`](crate::Tracer::binary) helpers, which *poison* on a failed bind so that the error surfaces later at
/// tracing boundaries, whereas [`BatchingTracer`](crate::BatchingTracer) and
/// [`DifferentiationTracer`](crate::DifferentiationTracer) have no deferral point and bind directly, panicking with
/// `$message` if the bind fails, and [`PartialTracer`](crate::PartialTracer) follows the same direct-bind shape (i.e.,
/// a mixed known/unknown bind residualizes rather than fails, so a bind error is a genuine type error). Binding
/// directly rather than delegating to the fallible capability trait keeps the eager arms' bounds minimal (e.g.,
/// no `Type = ArrayType` pin is needed on the batching arm).
///
/// # Parameters
///
///   - `@unary` / `@binary`: Selects the operator shape to stamp out. `@unary` produces `fn(self) -> Self` operators
///     and `@binary` produces `fn(self, Self) -> Self` operators.
///   - `$trait`: Path to the foreign `std::ops` operator trait to implement (e.g., `std::ops::Add`).
///   - `$method`: Identifier of the operator trait method to define (e.g., `add`).
///   - `$operation`: Path to the unit-struct operation, used both as the `From` bound target and as the value bound
///     through each tracer's context (e.g., `AddOperation`).
///   - `$message`: Panic message used when an eager tracer's bind fails.
#[macro_export]
macro_rules! define_tracer_operator {
    (@unary $trait:path, $method:ident, $operation:path, $message:literal $(,)?) => {
        impl<C: $crate::StagingContext<Operation: ::std::convert::From<$operation>>> $trait for $crate::Tracer<C> {
            type Output = Self;

            #[inline]
            fn $method(self) -> Self {
                self.unary($operation)
            }
        }

        impl<C: $crate::Context> $trait for $crate::PartialTracer<C>
        where
            $crate::PartialEvaluationContext<C>:
                $crate::Context<Value = $crate::PartialTracer<C>, Operation: ::std::convert::From<$operation>>,
        {
            type Output = Self;

            #[inline]
            fn $method(self) -> Self {
                $crate::Context::bind(self.context(), $operation, Vec::new(), ::std::slice::from_ref(&self))
                    .expect($message)
                    .remove(0)
            }
        }

        impl<C: $crate::Context<Type = $crate::ArrayType>> $trait for $crate::BatchingTracer<C>
        where
            $crate::BatchingContext<C>:
                $crate::Context<Value = $crate::BatchingTracer<C>, Operation: ::std::convert::From<$operation>>,
        {
            type Output = Self;

            #[inline]
            fn $method(self) -> Self {
                $crate::Context::bind(self.context(), $operation, Vec::new(), ::std::slice::from_ref(&self))
                    .expect($message)
                    .remove(0)
            }
        }

        impl<C: $crate::Context> $trait for $crate::DifferentiationTracer<C>
        where
            $crate::DifferentiationContext<C>:
                $crate::Context<Value = $crate::DifferentiationTracer<C>, Operation: ::std::convert::From<$operation>>,
        {
            type Output = Self;

            #[inline]
            fn $method(self) -> Self {
                $crate::Context::bind(self.context(), $operation, Vec::new(), ::std::slice::from_ref(&self))
                    .expect($message)
                    .remove(0)
            }
        }
    };
    (@binary $trait:path, $method:ident, $operation:path, $message:literal $(,)?) => {
        impl<C: $crate::StagingContext<Operation: ::std::convert::From<$operation>>> $trait for $crate::Tracer<C> {
            type Output = Self;

            #[inline]
            fn $method(self, right: Self) -> Self {
                self.binary(&right, $operation)
            }
        }

        impl<C: $crate::Context> $trait for $crate::PartialTracer<C>
        where
            $crate::PartialEvaluationContext<C>:
                $crate::Context<Value = $crate::PartialTracer<C>, Operation: ::std::convert::From<$operation>>,
        {
            type Output = Self;

            #[inline]
            fn $method(self, right: Self) -> Self {
                $crate::Context::bind(self.context(), $operation, Vec::new(), &[self.clone(), right.clone()])
                    .expect($message)
                    .remove(0)
            }
        }

        impl<C: $crate::Context<Type = $crate::ArrayType>> $trait for $crate::BatchingTracer<C>
        where
            $crate::BatchingContext<C>:
                $crate::Context<Value = $crate::BatchingTracer<C>, Operation: ::std::convert::From<$operation>>,
        {
            type Output = Self;

            #[inline]
            fn $method(self, right: Self) -> Self {
                $crate::Context::bind(self.context(), $operation, Vec::new(), &[self.clone(), right.clone()])
                    .expect($message)
                    .remove(0)
            }
        }

        impl<C: $crate::Context> $trait for $crate::DifferentiationTracer<C>
        where
            $crate::DifferentiationContext<C>:
                $crate::Context<Value = $crate::DifferentiationTracer<C>, Operation: ::std::convert::From<$operation>>,
        {
            type Output = Self;

            #[inline]
            fn $method(self, right: Self) -> Self {
                $crate::Context::bind(self.context(), $operation, Vec::new(), &[self.clone(), right.clone()])
                    .expect($message)
                    .remove(0)
            }
        }
    };
}

pub use crate::{
    check_builders, check_count, check_sharding, check_types, define_elementwise_capability,
    define_elementwise_operation, define_tracer_operator, impl_non_differentiable_operation,
    impl_non_transposable_operation,
};
