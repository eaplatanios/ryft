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
            return Err($crate::types::TypeError { message: format!("expected {expected} {noun} but got {count}") });
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
            return Err($crate::types::TypeError {
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
            if !std::rc::Rc::ptr_eq(reference, other) {
                result = ::std::result::Result::Err($crate::ProgramError::MismatchedProgramBuilders);
                break;
            }
        }
        result
    }};
    ($reference:expr, $other:expr $(,)?) => {{
        let reference = $reference;
        let other = $other;
        if std::rc::Rc::ptr_eq(reference, other) {
            ::std::result::Result::Ok(())
        } else {
            ::std::result::Result::Err($crate::ProgramError::MismatchedProgramBuilders)
        }
    }};
}

// TODO(eaplatanios): Review this macro.
/// Defines one elementwise [`Operation`](crate::Operation) together with the structural implementations that every
/// elementwise operation shares: the unit operation struct with a [`Display`](std::fmt::Display) implementation that
/// writes the operation name, type-preserving [`Operation`](crate::Operation) implementations over both
/// [`DataType`](crate::DataType) and [`ArrayType`](crate::ArrayType) metadata, the
/// [`ElementwiseOperation`](crate::ElementwiseOperation) implementation, eager
/// [`InterpretableOperation`](crate::InterpretableOperation) dispatch through the paired value-level capability trait,
/// the [`PartiallyEvaluatableOperation`](crate::PartiallyEvaluatableOperation) fold-or-residualize default, and the
/// capability trait itself with its dispatch-domain staging blanket. The operation's actual semantics stay with its
/// module: concrete value capability implementations, derivative rules, and tests are written by hand next to the
/// macro invocation.
///
/// # Parameters
///
///   - `@unary` / `@binary`: Selects the operation arity to stamp out. `@unary` produces a one-input,
///     type-preserving operation whose generated capability method has the shape
///     `fn(&self) -> Result<Self, ProgramError>`, while `@binary` produces a two-input operation whose
///     [`DataType`](crate::DataType) inference broadcasts/promotes the two operand types and whose generated
///     capability method has the shape `fn(&self, rhs: &Self) -> Result<Self, ProgramError>`.
///   - `$(#[$documentation])*`: Documentation attributes attached to the generated operation struct.
///   - `$operation`: Identifier of the generated unit-struct operation (e.g., `SinOperation`).
///   - `$name`: Identifier of an existing operation-name constant (e.g., `SIN_OPERATION_NAME`).
///   - `$capability`: Identifier of the generated value-level capability trait (e.g., `Sin`).
///   - `$method`: Identifier of the generated capability trait method (e.g., `sin`).
///   - `$(#[$capability_documentation])*`: Documentation attributes attached to the generated capability trait.
#[macro_export]
macro_rules! define_elementwise_operation {
    (
        @unary
        $(#[$documentation:meta])*
        $operation:ident, $name:ident, $capability:ident, $method:ident,
        $(#[$capability_documentation:meta])* $(,)?
    ) => {
        $(#[$documentation])*
        #[derive(Clone, Debug, Default)]
        pub struct $operation;

        impl ::std::fmt::Display for $operation {
            fn fmt(&self, formatter: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                formatter.write_str($name)
            }
        }

        impl $crate::Operation<$crate::DataType> for $operation {
            #[inline]
            fn name(&self) -> &'static str {
                $name
            }

            fn infer_output_types(
                &self,
                input_types: &[$crate::DataType],
                _region_interfaces: &[$crate::RegionInterface<$crate::DataType>],
            ) -> Result<Vec<$crate::DataType>, $crate::types::TypeError> {
                $crate::check_count!("input", input_types, 1, TypeError);
                Ok(vec![input_types[0].clone()])
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
            ) -> Result<Vec<$crate::ArrayType>, $crate::types::TypeError> {
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

        $(#[$capability_documentation])*
        pub trait $capability: Sized {
            /// Computes this operation elementwise for this value, returning a
            /// [`ProgramError`](crate::ProgramError) if something goes wrong (e.g., when this operation does not
            /// support the value's data type).
            fn $method(&self) -> Result<Self, $crate::ProgramError>;
        }

        impl<V: $crate::Value<DispatchDomain: $crate::Context<Operation: ::std::convert::From<$operation>>>>
            $capability for V
        {
            #[inline]
            fn $method(&self) -> Result<Self, $crate::ProgramError> {
                // Fully qualified calls are required here because the `Value` and `Context` traits are not
                // necessarily imported at the macro expansion site.
                let domain = $crate::Value::dispatch_domain(self);
                Ok($crate::Context::bind(&domain, $operation, Vec::new(), &[self.clone()])?.remove(0))
            }
        }
    };
    (
        @binary
        $(#[$documentation:meta])*
        $operation:ident, $name:ident, $capability:ident, $method:ident,
        $(#[$capability_documentation:meta])* $(,)?
    ) => {
        $(#[$documentation])*
        #[derive(Clone, Debug, Default)]
        pub struct $operation;

        impl ::std::fmt::Display for $operation {
            fn fmt(&self, formatter: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                formatter.write_str($name)
            }
        }

        impl $crate::Operation<$crate::DataType> for $operation {
            #[inline]
            fn name(&self) -> &'static str {
                $name
            }

            fn infer_output_types(
                &self,
                input_types: &[$crate::DataType],
                _region_interfaces: &[$crate::RegionInterface<$crate::DataType>],
            ) -> Result<Vec<$crate::DataType>, $crate::types::TypeError> {
                $crate::check_count!("input", input_types, 2, TypeError);
                // The fully qualified call is required here because the `Broadcastable` trait is not necessarily
                // imported at the macro expansion site.
                $crate::Broadcastable::broadcast(&input_types[0], &input_types[1])
                    .map(|output| vec![output])
                    .map_err(|_| $crate::types::TypeError {
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
            ) -> Result<Vec<$crate::ArrayType>, $crate::types::TypeError> {
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

        $(#[$capability_documentation])*
        pub trait $capability: Sized {
            /// Computes this operation elementwise for this value and `rhs`, returning a
            /// [`ProgramError`](crate::ProgramError) if something goes wrong (e.g., when this operation does not
            /// support the values' data types).
            fn $method(&self, rhs: &Self) -> Result<Self, $crate::ProgramError>;
        }

        impl<V: $crate::Value<DispatchDomain: $crate::Context<Operation: ::std::convert::From<$operation>>>>
            $capability for V
        {
            #[inline]
            fn $method(&self, rhs: &Self) -> Result<Self, $crate::ProgramError> {
                // Fully qualified calls are required here because the `Value` and `Context` traits are not
                // necessarily imported at the macro expansion site.
                let domain = $crate::Value::dispatch_domain(self);
                Ok($crate::Context::bind(&domain, $operation, Vec::new(), &[self.clone(), rhs.clone()])?.remove(0))
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
                $crate::Context::bind(self.context(), $operation, Vec::new(), &[self.clone()])
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
                $crate::Context::bind(self.context(), $operation, Vec::new(), &[self.clone()])
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
                $crate::Context::bind(self.context(), $operation, Vec::new(), &[self.clone()])
                    .expect($message)
                    .remove(0)
            }
        }
    };
    (@binary $trait:path, $method:ident, $operation:path, $message:literal $(,)?) => {
        impl<C: $crate::StagingContext<Operation: ::std::convert::From<$operation>>> $trait for $crate::Tracer<C> {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: Self) -> Self {
                self.binary(&rhs, $operation)
            }
        }

        impl<C: $crate::Context> $trait for $crate::PartialTracer<C>
        where
            $crate::PartialEvaluationContext<C>:
                $crate::Context<Value = $crate::PartialTracer<C>, Operation: ::std::convert::From<$operation>>,
        {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: Self) -> Self {
                $crate::Context::bind(self.context(), $operation, Vec::new(), &[self.clone(), rhs.clone()])
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
            fn $method(self, rhs: Self) -> Self {
                $crate::Context::bind(self.context(), $operation, Vec::new(), &[self.clone(), rhs.clone()])
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
            fn $method(self, rhs: Self) -> Self {
                $crate::Context::bind(self.context(), $operation, Vec::new(), &[self.clone(), rhs.clone()])
                    .expect($message)
                    .remove(0)
            }
        }
    };
}

pub use crate::{
    check_builders, check_count, check_sharding, check_types, define_elementwise_operation, define_tracer_operator,
};
